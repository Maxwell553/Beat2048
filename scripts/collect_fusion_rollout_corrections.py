"""Estimate action win rates on states preceding fusion-policy losses."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import ConvAgent, FusionPolicy
from bot2048.rl import RLAgent
from scripts.evaluate_fusion_policy import fusion_actions
from scripts.ppo_rl import BatchEnv
from scripts.train_conv_rl import RECORD


def pack(boards):
    return np.sum(boards.reshape(-1, 16).astype(np.uint64) << (4 * np.arange(16, dtype=np.uint64)), axis=1, dtype=np.uint64)


def unpack(packed):
    return ((packed[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & 15).astype(np.uint8).reshape(-1, 4, 4)


class BatchPolicy:
    def __init__(self, cnn, sparse, fusion, device):
        self.device = device; self.cnn = ConvAgent(cnn, device); self.sparse = [RLAgent(path) for path in sparse]
        artifact = torch.load(fusion, map_location=device, weights_only=True)
        self.fusion = FusionPolicy(artifact["metadata"]["inputs"], artifact["metadata"]["hidden"]).to(device)
        self.fusion.load_state_dict(artifact["state_dict"]); self.fusion.eval()

    def actions(self, boards, legal):
        return fusion_actions(self.cnn, self.sparse, self.fusion, boards, legal, self.device)

    def close(self):
        for model in self.sparse: model.close()


def collect_candidates(policy, envs, seed, losses_needed, history):
    env = BatchEnv(envs, seed); episodes = [deque(maxlen=history) for _ in range(envs)]
    candidates=[]; losses=wins=0; offsets=(16,32,48,64,80,96,112,128)
    while losses < losses_needed:
        for index, board in enumerate(pack(env.boards)): episodes[index].append(int(board))
        env.step(policy.actions(env.boards, env.legal))
        for index in np.flatnonzero(env.outcomes):
            if env.outcomes[index] == -1:
                losses += 1; episode=list(episodes[index])
                for offset in offsets:
                    if len(episode) >= offset: candidates.append(episode[-offset])
            else: wins += 1
            episodes[index].clear()
    env.close()
    return np.asarray(list(dict.fromkeys(candidates)), dtype=np.uint64), wins, losses


def evaluate_chunk(policy, states, rollouts, seed):
    count=len(states); repeated=np.repeat(states,4*rollouts); first=np.tile(np.repeat(np.arange(4),rollouts),count).astype(np.int32)
    group_state=np.repeat(np.arange(count),4*rollouts); group_action=np.tile(np.repeat(np.arange(4),rollouts),count)
    env=BatchEnv(len(repeated),seed); env.set_boards(repeated); legal_initial=env.legal.copy(); valid=legal_initial[np.arange(len(repeated)),first].astype(bool)
    first[~valid]=legal_initial[~valid].argmax(1); env.step(first); finished=~valid; wins=np.zeros(len(repeated),dtype=np.float32)
    ended=(~finished)&(env.outcomes!=0); wins[ended]=env.outcomes[ended]==1; finished|=ended; steps=1
    while not finished.all() and steps < 3000:
        env.step(policy.actions(env.boards,env.legal)); ended=(~finished)&(env.outcomes!=0); wins[ended]=env.outcomes[ended]==1; finished|=ended; steps+=1
    env.close(); q=np.full((count,4),-1e30,dtype=np.float32)
    for state in range(count):
        for action in range(4):
            mask=valid&(group_state==state)&(group_action==action)
            if mask.any(): q[state,action]=wins[mask].mean()
    return q,steps


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--cnn",required=True); p.add_argument("--sparse",nargs="+",required=True); p.add_argument("--fusion",required=True); p.add_argument("--output",required=True)
    p.add_argument("--losses",type=int,default=100); p.add_argument("--candidate-envs",type=int,default=128); p.add_argument("--history",type=int,default=128); p.add_argument("--max-candidates",type=int,default=128); p.add_argument("--rollouts",type=int,default=4); p.add_argument("--chunk-size",type=int,default=16); p.add_argument("--seed",type=int,default=99800000); p.add_argument("--device",default="mps")
    args=p.parse_args(); torch.set_num_threads(2); policy=BatchPolicy(args.cnn,args.sparse,args.fusion,args.device)
    candidates,wins,losses=collect_candidates(policy,args.candidate_envs,args.seed,args.losses,args.history); rng=np.random.default_rng(args.seed); rng.shuffle(candidates); candidates=candidates[:args.max_candidates]
    targets=[]; max_steps=0
    for offset in range(0,len(candidates),args.chunk_size):
        q,steps=evaluate_chunk(policy,candidates[offset:offset+args.chunk_size],args.rollouts,args.seed+100000+offset); targets.append(q); max_steps=max(max_steps,steps); print(json.dumps({"evaluated":offset+len(q),"candidates":len(candidates)}),flush=True)
    q=np.concatenate(targets); records=np.empty(len(candidates),dtype=RECORD); records["board"],records["q"]=candidates,q; records.tofile(args.output)
    boards=unpack(candidates); legal=q>-1e20; base=policy.actions(boards,legal); best=q.argmax(1); base_rate=q[np.arange(len(q)),base]; best_rate=q[np.arange(len(q)),best]
    policy.close(); metadata={"config":vars(args),"candidate_collection_wins":wins,"candidate_collection_losses":losses,"states":len(candidates),"rollouts_per_action":args.rollouts,"best_action_differs":int((best!=base).sum()),"mean_base_win_rate":float(base_rate.mean()),"mean_best_win_rate":float(best_rate.mean()),"mean_estimated_improvement":float((best_rate-base_rate).mean()),"max_rollout_steps":max_steps,"labels":"complete stochastic 2048 outcomes under frozen neural fusion continuation","search_at_inference":False}
    Path(args.output).with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n"); print(json.dumps(metadata),flush=True)


if __name__ == "__main__": main()
