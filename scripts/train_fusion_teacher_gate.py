"""Train a conservative neural gate to correct high-confidence fusion errors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bot2048.deep_rl import FusionPolicy, GatedFusionCorrection
from scripts.train_fusion_policy import features


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--base",required=True); p.add_argument("--train",required=True); p.add_argument("--validation",required=True); p.add_argument("--output",required=True)
    p.add_argument("--hidden",type=int,default=256); p.add_argument("--epochs",type=int,default=8); p.add_argument("--batch-size",type=int,default=2048); p.add_argument("--learning-rate",type=float,default=0.0003); p.add_argument("--weight-decay",type=float,default=0.001); p.add_argument("--device",default="mps"); p.add_argument("--seed",type=int,default=99913000)
    args=p.parse_args(); torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    with np.load(args.train) as z: train={k:z[k] for k in z.files}
    with np.load(args.validation) as z: validation={k:z[k] for k in z.files}
    artifact=torch.load(args.base,map_location=args.device,weights_only=True); base=FusionPolicy(artifact["metadata"]["inputs"],artifact["metadata"]["hidden"]).to(args.device); base.load_state_dict(artifact["state_dict"]); base.eval()
    model=GatedFusionCorrection(artifact["metadata"]["inputs"],args.hidden).to(args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    order=np.arange(len(train["boards"])); best_accuracy=0
    for epoch in range(1,args.epochs+1):
        rng.shuffle(order); model.train(); total=gate_total=action_total=0
        for offset in range(0,len(order),args.batch_size):
            ids=order[offset:offset+args.batch_size]; x=features(train,ids,args.device); target=train["targets"][ids]; legal=torch.as_tensor(target>-1e20,device=args.device); teacher=torch.as_tensor(target.argmax(1),device=args.device)
            with torch.no_grad(): base_action=base(x).masked_fill(~legal,-1e9).argmax(1)
            wrong=base_action!=teacher; gate,actions=model(x); gate_loss=torch.nn.functional.binary_cross_entropy_with_logits(gate,wrong.float(),pos_weight=torch.tensor(2.5,device=args.device)); action_loss=torch.nn.functional.cross_entropy(actions[wrong],teacher[wrong]); loss=gate_loss+action_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total+=loss.item()*len(ids); gate_total+=gate_loss.item()*len(ids); action_total+=action_loss.item()*len(ids)
        model.eval(); confidences=[]; proposed=[]; bases=[]; teachers=[]
        with torch.no_grad():
            for offset in range(0,len(validation["boards"]),args.batch_size):
                ids=np.arange(offset,min(offset+args.batch_size,len(validation["boards"]))); x=features(validation,ids,args.device); target=validation["targets"][ids]; legal=torch.as_tensor(target>-1e20,device=args.device); teacher=target.argmax(1); base_action=base(x).masked_fill(~legal,-1e9).argmax(1); gate,actions=model(x); action=actions.masked_fill(~legal,-1e9).argmax(1)
                confidences.append(torch.sigmoid(gate).cpu().numpy()); proposed.append(action.cpu().numpy()); bases.append(base_action.cpu().numpy()); teachers.append(teacher)
        confidence=np.concatenate(confidences); proposed=np.concatenate(proposed); base_action=np.concatenate(bases); teacher=np.concatenate(teachers); base_accuracy=float((base_action==teacher).mean()); choices=[]
        for threshold in np.arange(.5,.991,.01):
            selected=confidence>=threshold; final=np.where(selected,proposed,base_action); accuracy=float((final==teacher).mean()); precision=float((proposed[selected]==teacher[selected]).mean()) if selected.any() else 0.0; choices.append((accuracy,float(threshold),int(selected.sum()),precision))
        accuracy,threshold,selected,precision=max(choices); row={"epoch":epoch,"loss":total/len(order),"gate_loss":gate_total/len(order),"action_loss":action_total/len(order),"base_validation_accuracy":base_accuracy,"validation_accuracy":accuracy,"threshold":threshold,"selected":selected,"selected_action_accuracy":precision}
        if accuracy>best_accuracy:
            best_accuracy=accuracy; saved={"state_dict":{k:v.cpu() for k,v in model.state_dict().items()},"metadata":{"inputs":artifact["metadata"]["inputs"],"hidden":args.hidden,"threshold":threshold,"config":vars(args),**row,"search_at_inference":False}}; torch.save(saved,str(args.output)+".tmp"); Path(str(args.output)+".tmp").replace(args.output)
        print(json.dumps(row),flush=True)


if __name__=="__main__": main()
