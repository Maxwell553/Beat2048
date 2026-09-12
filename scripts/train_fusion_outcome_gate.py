"""Train a conservative fusion-policy correction from complete-game outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import FusionPolicy, GatedFusionCorrection
from scripts.train_fusion_policy import features


def labels(data, base, minimum_improvement, minimum_best_rate):
    rates=data["targets"]; legal=rates>-1e20; base_action=base.masked_fill(~torch.as_tensor(legal,device=base.device),-1e9).argmax(1).cpu().numpy(); best=rates.argmax(1)
    delta=rates[np.arange(len(rates)),best]-rates[np.arange(len(rates)),base_action]
    positive=(best!=base_action)&(delta>=minimum_improvement)&(rates[np.arange(len(rates)),best]>=minimum_best_rate)
    return base_action,best,delta,positive,legal


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--base",required=True); p.add_argument("--train",required=True); p.add_argument("--validation",required=True); p.add_argument("--output",required=True)
    p.add_argument("--hidden",type=int,default=64); p.add_argument("--epochs",type=int,default=300); p.add_argument("--learning-rate",type=float,default=0.0003); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--minimum-improvement",type=float,default=0.25); p.add_argument("--minimum-best-rate",type=float,default=0.5); p.add_argument("--device",default="mps"); p.add_argument("--seed",type=int,default=99806144)
    args=p.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.train) as z: train={k:z[k] for k in z.files}
    with np.load(args.validation) as z: validation={k:z[k] for k in z.files}
    artifact=torch.load(args.base,map_location=args.device,weights_only=True); base=FusionPolicy(artifact["metadata"]["inputs"],artifact["metadata"]["hidden"]).to(args.device); base.load_state_dict(artifact["state_dict"]); base.eval()
    train_x=features(train,np.arange(len(train["boards"])),args.device); validation_x=features(validation,np.arange(len(validation["boards"])),args.device)
    with torch.no_grad(): train_base=base(train_x); validation_base=base(validation_x)
    train_labels=labels(train,train_base,args.minimum_improvement,args.minimum_best_rate); validation_labels=labels(validation,validation_base,args.minimum_improvement,args.minimum_best_rate)
    model=GatedFusionCorrection(train_x.shape[1],args.hidden).to(args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    gate_target=torch.as_tensor(train_labels[3],device=args.device,dtype=torch.float32); action_target=torch.as_tensor(train_labels[1],device=args.device); positive_weight=torch.tensor([(len(gate_target)-gate_target.sum())/(gate_target.sum()+1e-6)],device=args.device)
    best_score=-1e9; best_row=None
    for epoch in range(1,args.epochs+1):
        model.train(); gate,actions=model(train_x); gate_loss=torch.nn.functional.binary_cross_entropy_with_logits(gate,gate_target,pos_weight=positive_weight); action_loss=torch.nn.functional.cross_entropy(actions[gate_target.bool()],action_target[gate_target.bool()]); loss=gate_loss+action_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch%10: continue
        model.eval()
        with torch.no_grad(): validation_gate,validation_actions=model(validation_x); confidence=torch.sigmoid(validation_gate).cpu().numpy(); proposed=validation_actions.masked_fill(~torch.as_tensor(validation_labels[4],device=args.device),-1e9).argmax(1).cpu().numpy()
        rates=validation["targets"]; base_action=validation_labels[0]; candidate_delta=rates[np.arange(len(rates)),proposed]-rates[np.arange(len(rates)),base_action]
        choices=[]
        for threshold in np.arange(.5,.951,.05):
            selected=confidence>=threshold; gain=float(candidate_delta[selected].sum()/len(rates)); choices.append((gain,float(threshold),int(selected.sum()),float((candidate_delta[selected]>0).mean()) if selected.any() else 0.0))
        gain,threshold,selected,precision=max(choices)
        row={"epoch":epoch,"loss":loss.item(),"gate_loss":gate_loss.item(),"action_loss":action_loss.item(),"train_corrections":int(train_labels[3].sum()),"validation_corrections":int(validation_labels[3].sum()),"threshold":threshold,"selected":selected,"estimated_gain_per_state":gain,"positive_delta_precision":precision}
        if gain>best_score:
            best_score=gain; best_row=row; saved={"state_dict":{k:v.cpu() for k,v in model.state_dict().items()},"metadata":{"inputs":train_x.shape[1],"hidden":args.hidden,"threshold":threshold,"config":vars(args),**row,"search_at_inference":False}}; torch.save(saved,str(args.output)+".tmp"); Path(str(args.output)+".tmp").replace(args.output)
        print(json.dumps(row),flush=True)
    print(json.dumps({"best":best_row}),flush=True)


if __name__=="__main__": main()
