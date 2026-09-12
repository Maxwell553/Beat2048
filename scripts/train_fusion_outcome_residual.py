"""Fit a base-preserving residual head from complete-game outcome labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bot2048.deep_rl import FusionPolicy
from scripts.train_fusion_policy import features


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--base",required=True); p.add_argument("--train",required=True); p.add_argument("--validation",required=True); p.add_argument("--output",required=True)
    p.add_argument("--hidden",type=int,default=256); p.add_argument("--scale",type=float,default=1.0); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=2048); p.add_argument("--learning-rate",type=float,default=0.0001); p.add_argument("--preserve-weight",type=float,default=1.0); p.add_argument("--minimum-improvement",type=float,default=.25); p.add_argument("--minimum-best-rate",type=float,default=.5); p.add_argument("--device",default="mps"); p.add_argument("--seed",type=int,default=103700000)
    args=p.parse_args(); torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    with np.load(args.train) as z: train={k:z[k] for k in z.files}
    with np.load(args.validation) as z: validation={k:z[k] for k in z.files}
    artifact=torch.load(args.base,map_location=args.device,weights_only=True); base=FusionPolicy(artifact["metadata"]["inputs"],artifact["metadata"]["hidden"]).to(args.device);base.load_state_dict(artifact["state_dict"]);base.eval()
    head=nn.Sequential(nn.Linear(artifact["metadata"]["inputs"],args.hidden),nn.ReLU(),nn.Linear(args.hidden,4)).to(args.device);nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias);optimizer=torch.optim.AdamW(head.parameters(),lr=args.learning_rate,weight_decay=.001)
    order=np.arange(len(train["boards"]));best=-1e9
    def targets(data,x):
        rates=data["targets"];legal=torch.as_tensor(rates>-1e20,device=args.device)
        with torch.no_grad(): logits=base(x).masked_fill(~legal,-1e9); base_action=logits.argmax(1).cpu().numpy()
        choice=rates.argmax(1);delta=rates[np.arange(len(rates)),choice]-rates[np.arange(len(rates)),base_action];correction=(choice!=base_action)&(delta>=args.minimum_improvement)&(rates[np.arange(len(rates)),choice]>=args.minimum_best_rate)
        return logits,legal,torch.as_tensor(choice,device=args.device),torch.as_tensor(correction,device=args.device),base_action
    for epoch in range(1,args.epochs+1):
        rng.shuffle(order);head.train();total=0
        for offset in range(0,len(order),args.batch_size):
            ids=order[offset:offset+args.batch_size];x=features(train,ids,args.device);base_logits,legal,choice,correction,_=targets({k:v[ids] for k,v in train.items()},x);out=(base_logits+args.scale*torch.tanh(head(x))).masked_fill(~legal,-1e9)
            preserve=torch.nn.functional.kl_div(torch.log_softmax(out,1),torch.softmax(base_logits,1),reduction="batchmean");correct=torch.nn.functional.cross_entropy(out[correction],choice[correction]) if correction.any() else out.sum()*0;loss=args.preserve_weight*preserve+correct
            optimizer.zero_grad();loss.backward();optimizer.step();total+=loss.item()*len(ids)
        head.eval();base_actions=[];actions=[];corrections=[];choices=[]
        with torch.no_grad():
            for offset in range(0,len(validation["boards"]),args.batch_size):
                ids=np.arange(offset,min(offset+args.batch_size,len(validation["boards"])));x=features(validation,ids,args.device);bl,legal,choice,correction,ba=targets({k:v[ids] for k,v in validation.items()},x);out=(bl+args.scale*torch.tanh(head(x))).masked_fill(~legal,-1e9);base_actions.append(ba);actions.append(out.argmax(1).cpu().numpy());corrections.append(correction.cpu().numpy());choices.append(choice.cpu().numpy())
        ba=np.concatenate(base_actions);action=np.concatenate(actions);correction=np.concatenate(corrections);choice=np.concatenate(choices);labels=np.where(correction,choice,ba);accuracy=float((action==labels).mean());changed=action!=ba;row={"epoch":epoch,"loss":total/len(order),"validation_accuracy":accuracy,"changed":int(changed.sum()),"changed_fraction":float(changed.mean()),"correction_recall":float((action[correction]==choice[correction]).mean()),"changed_precision":float((action[changed]==labels[changed]).mean()) if changed.any() else 0.0}
        score=accuracy-.1*changed.mean()
        if score>best:
            best=score;saved={"base_fusion":args.base,"correction_state_dict":{k:v.cpu() for k,v in head.state_dict().items()},"metadata":{"inputs":artifact["metadata"]["inputs"],"hidden":args.hidden,"scale":args.scale,"config":vars(args),**row,"search_at_inference":False}};torch.save(saved,str(args.output)+".tmp");Path(str(args.output)+".tmp").replace(args.output)
        print(json.dumps(row),flush=True)


if __name__=="__main__":main()
