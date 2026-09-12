"""PPO-train a residual head on the frozen current-board neural fusion policy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from bot2048.deep_rl import ConvAgent, FusionPolicy, encode
from bot2048.rl import RLAgent
from scripts.ppo_rl import BatchEnv


class FusionActorCritic(nn.Module):
    def __init__(self, cnn, sparse, fusion, device, scale=1.0, hidden=256, experts=None):
        super().__init__()
        self.cnn, self.sparse, self.device, self.scale = cnn, sparse, device, scale
        self.experts = experts or []
        paths = fusion if isinstance(fusion, (list, tuple)) else [fusion]
        self.base = nn.ModuleList()
        for path in paths:
            artifact = torch.load(path, map_location=device, weights_only=True)
            head = FusionPolicy(artifact["metadata"]["inputs"], artifact["metadata"]["hidden"]).to(device)
            head.load_state_dict(artifact["state_dict"]); head.eval()
            for parameter in head.parameters(): parameter.requires_grad = False
            self.base.append(head)
        inputs = max(head.inputs for head in self.base)
        self.correction = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)
        self.critic = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def features(self, boards, legal):
        variants, mappings = [], []
        for reflection in (False, True):
            for k in range(4):
                variant = np.rot90(boards, k, axes=(1, 2)); mapping = (np.arange(4) - k) % 4
                if reflection: variant = variant[:, :, ::-1]; mapping = (-mapping) % 4
                variants.append(variant.copy()); mappings.append(mapping.tolist())
        with torch.no_grad():
            raw = self.cnn.model(encode(np.concatenate(variants), self.device)).reshape(8, len(boards), 4)
            aligned = torch.stack([raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)
            mask = torch.as_tensor(legal, dtype=torch.bool, device=self.device)
            masked = aligned.masked_fill(~mask[:, None], -1e9)
            choices = masked.argmax(2)
            votes = torch.stack([(choices == action).sum(1) for action in range(4)], 1)
            probabilities = torch.softmax(masked, 2).mean(1)
            expert_features = []
            for expert in self.experts:
                expert_raw = expert.model(encode(np.concatenate(variants), self.device)).reshape(8, len(boards), 4)
                expert_view = torch.stack([expert_raw[i][:, mapping] for i, mapping in enumerate(mappings)], 1)
                expert_view -= expert_view.mean(2, keepdim=True)
                expert_view /= expert_view.std(2, keepdim=True) + 1e-6
                expert_features.append(expert_view.cpu().numpy())
        aligned_np = aligned.cpu().numpy(); aligned_np -= aligned_np.mean(2, keepdims=True)
        packed = np.sum(boards.reshape(-1, 16).astype(np.uint64) << (4 * np.arange(16, dtype=np.uint64)), axis=1, dtype=np.uint64)
        sparse = np.stack([model.values_batch(packed) for model in self.sparse], 1)
        sparse -= sparse.mean(2, keepdims=True); sparse /= sparse.std(2, keepdims=True) + 1e-6
        onehot = torch.nn.functional.one_hot(torch.as_tensor(boards, device=self.device, dtype=torch.long).clamp(0, 15), 16).flatten(1).float()
        rest = np.concatenate([aligned_np.reshape(len(boards), -1), votes.cpu().numpy() / 8.0, probabilities.cpu().numpy(), sparse.reshape(len(boards), -1), *[row.reshape(len(boards), -1) for row in expert_features]], 1).astype(np.float32)
        return torch.cat([onehot, torch.as_tensor(rest, device=self.device)], 1)

    def forward(self, features):
        with torch.no_grad():
            base = torch.stack([head(features[:, :head.inputs]) for head in self.base]).mean(0)
        logits = base + self.scale * torch.tanh(self.correction(features))
        return logits, self.critic(features).squeeze(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cnn", required=True); p.add_argument("--sparse", nargs="+", required=True)
    p.add_argument("--expert", nargs="*", default=[])
    p.add_argument("--fusion", nargs="+", required=True); p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=6); p.add_argument("--envs", type=int, default=128)
    p.add_argument("--rollout", type=int, default=1024); p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2048); p.add_argument("--actor-learning-rate", type=float, default=3e-6)
    p.add_argument("--critic-learning-rate", type=float, default=3e-4); p.add_argument("--temperature", type=float, default=.25)
    p.add_argument("--scale", type=float, default=1.0); p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=1.0); p.add_argument("--win-bonus", type=float, default=50.0)
    p.add_argument("--loss-penalty", type=float, default=50.0); p.add_argument("--seed", type=int, default=99500000)
    p.add_argument("--device", default="mps")
    p.add_argument("--resume", help="Resume correction, critic, and optimizer state")
    args = p.parse_args(); torch.set_num_threads(4); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed); env = BatchEnv(args.envs, args.seed)
    cnn = ConvAgent(args.cnn, args.device); sparse = [RLAgent(path) for path in args.sparse]
    experts = [ConvAgent(path, args.device) for path in args.expert]
    model = FusionActorCritic(cnn, sparse, args.fusion, args.device, args.scale, experts=experts).to(args.device)
    optimizer = torch.optim.Adam([{"params": model.correction.parameters(), "lr": args.actor_learning_rate}, {"params": model.critic.parameters(), "lr": args.critic_learning_rate}], eps=1e-5)
    start_iteration = 0
    if args.resume:
        resumed = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.correction.load_state_dict(resumed["correction_state_dict"])
        model.critic.load_state_dict(resumed["critic_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        for index, group in enumerate(optimizer.param_groups):
            group["lr"] = args.actor_learning_rate if index == 0 else args.critic_learning_rate
        start_iteration = resumed["metadata"]["iteration"]
    started = time.time()
    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        batches=[]; wins=losses=0; model.eval()
        for _ in range(args.rollout):
            masks=env.legal.copy()
            with torch.no_grad():
                features=model.features(env.boards, masks); logits, values=model(features)
                distribution=torch.distributions.Categorical(logits=(logits.masked_fill(~torch.as_tensor(masks,dtype=torch.bool,device=args.device),-1e9)/args.temperature).cpu())
                actions=distribution.sample(); logp=distribution.log_prob(actions)
            env.step(actions.numpy()); rewards=env.rewards.copy()
            rewards += (env.outcomes==1)*(args.win_bonus-10); rewards += (env.outcomes==-1)*(-args.loss_penalty+1)
            wins += int((env.outcomes==1).sum()); losses += int((env.outcomes==-1).sum())
            batches.append((features.cpu().numpy(),masks,actions.numpy(),logp.numpy(),values.cpu().numpy(),rewards,env.done.copy()))
        with torch.no_grad(): _, bootstrap=model(model.features(env.boards,env.legal)); next_value=bootstrap.cpu().numpy()
        advantage=np.zeros(args.envs,dtype=np.float32); advantages=[]
        for row in reversed(batches):
            value,reward,done=row[4],row[5],row[6]; delta=reward+args.gamma*next_value*(1-done)-value
            advantage=delta+args.gamma*args.gae_lambda*(1-done)*advantage; advantages.append(advantage.copy()); next_value=value
        advantages=np.stack(advantages[::-1]).reshape(-1); values=np.stack([r[4] for r in batches]).reshape(-1)
        returns=advantages+values; normalized=(advantages-advantages.mean())/(advantages.std()+1e-8)
        features=np.concatenate([r[0] for r in batches]); masks=np.concatenate([r[1] for r in batches]); actions=np.concatenate([r[2] for r in batches]); old_logp=np.concatenate([r[3] for r in batches]); order=np.arange(len(actions)); metrics=[]; model.train()
        for _ in range(args.epochs):
            rng.shuffle(order)
            for offset in range(0,len(order),args.batch_size):
                ids=order[offset:offset+args.batch_size]; x=torch.as_tensor(features[ids],device=args.device); legal=torch.as_tensor(masks[ids],dtype=torch.bool,device=args.device)
                logits,value=model(x); distribution=torch.distributions.Categorical(logits=logits.masked_fill(~legal,-1e9)/args.temperature)
                selected=torch.as_tensor(actions[ids],device=args.device); ratio=(distribution.log_prob(selected)-torch.as_tensor(old_logp[ids],device=args.device)).exp(); adv=torch.as_tensor(normalized[ids],device=args.device)
                actor=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean(); critic=nn.functional.smooth_l1_loss(value,torch.as_tensor(returns[ids],device=args.device)); loss=actor+.5*critic
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.correction.parameters(),.25); nn.utils.clip_grad_norm_(model.critic.parameters(),.5); optimizer.step(); metrics.append((actor.item(),critic.item(),distribution.entropy().mean().item()))
        row={"iteration":iteration,"steps":iteration*args.envs*args.rollout,"wins":wins,"losses":losses,"actor_loss":float(np.mean([m[0] for m in metrics])),"critic_loss":float(np.mean([m[1] for m in metrics])),"entropy":float(np.mean([m[2] for m in metrics])),"seconds":time.time()-started}
        artifact={"base_fusion":args.fusion,"correction_state_dict":{k:v.cpu() for k,v in model.correction.state_dict().items()},"critic_state_dict":{k:v.cpu() for k,v in model.critic.state_dict().items()},"optimizer_state_dict":optimizer.state_dict(),"metadata":{"inputs":model.correction[0].in_features,"hidden":256,"scale":args.scale,"config":vars(args),**row,"search_at_inference":False}}
        destination=Path(args.output).with_name(f"{Path(args.output).stem}.iteration{iteration}.pt"); torch.save(artifact,str(destination)+".tmp"); Path(str(destination)+".tmp").replace(destination); torch.save(artifact,str(args.output)+".tmp"); Path(str(args.output)+".tmp").replace(args.output)
        print(json.dumps(row),flush=True)
    for sparse_model in sparse: sparse_model.close()
    env.close()


if __name__ == "__main__": main()
