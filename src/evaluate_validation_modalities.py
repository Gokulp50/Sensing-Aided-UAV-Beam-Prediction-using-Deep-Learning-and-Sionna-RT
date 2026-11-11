"""Validation-only modality ablation for the trained reliability-gated model."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import temporal_reliability_gated_future as model_lib


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,required=True)
    ap.add_argument("--run-dir",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,default=128)
    args=ap.parse_args()
    root,out=args.project_root.resolve(),args.run_dir.resolve()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame,data_root=model_lib.load_joined(root)
    scaled,_,_,_=model_lib.fit_sensor_scaler(frame)
    features,lookup=model_lib.get_image_features(frame,data_root,out,device,24,False)
    windows=model_lib.make_windows(frame,scaled,lookup)
    records=windows["val"]
    loader=DataLoader(model_lib.WindowDataset(records,features),batch_size=args.batch_size,
                      shuffle=False,num_workers=0,pin_memory=(device.type=="cuda"))
    ckpt=torch.load(out/"best_model.pt",map_location=device,weights_only=True)
    model=model_lib.ReliabilityGatedGRU().to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    modes=(("learned_gate",None),("sensor_only","sensor"),("visual_only","visual"),("fixed_equal","equal"))
    collected={name:{"cls":[],"change":[],"gates":[]} for name,_ in modes}
    ys=[]; prevs=[]
    with torch.inference_mode():
        for visual,sensors,y,prev,_ in loader:
            visual=visual.to(device); sensors=sensors.to(device)
            ys.append(y); prevs.append(prev)
            for name,force in modes:
                cls,chg,gate=model(visual,sensors,force_modality=force)
                collected[name]["cls"].append(cls.cpu())
                collected[name]["change"].append(chg.cpu())
                collected[name]["gates"].append(gate.cpu())
    y=torch.cat(ys).long(); prev=torch.cat(prevs).long()
    scenario=pd.read_csv(data_root/"scenario23.csv").set_index("index")
    power_rows=[]
    for r in records:
        rel=str(scenario.loc[r["index"],"unit1_pwr_60ghz"]).replace("\\","/").removeprefix("./")
        power_rows.append(np.maximum(np.atleast_1d(np.loadtxt(data_root/rel,dtype=np.float64)),1e-15))
    powers=torch.as_tensor(np.stack(power_rows),dtype=torch.float64)
    temp=float(ckpt["temperature"]); bias=float(ckpt["bias"])
    metric_rows=[]
    for name,_ in modes:
        cls=torch.cat(collected[name]["cls"])
        change=torch.cat(collected[name]["change"])
        probs=model_lib.make_final_prob(cls,change,prev,temp,bias)
        top=probs.topk(5,dim=1).indices
        row={"modality_policy":name,**model_lib.classification_metrics(probs,y)}
        for k in (1,3,5):
            selected_power=powers.gather(1,top[:,:k]).max(1).values
            regret=10*torch.log10(powers.max(1).values/selected_power)
            row[f"top{k}_mean_power_regret_db"]=float(regret.mean())
            row[f"top{k}_within_0_01db_fraction"]=float((regret<=0.01).double().mean())
        row["mean_last_step_visual_gate"]=float(torch.cat(collected[name]["gates"])[:,0].mean())
        row["mean_last_step_sensor_gate"]=float(torch.cat(collected[name]["gates"])[:,1].mean())
        row["samples"]=len(y)
        metric_rows.append(row)
    metrics=pd.DataFrame(metric_rows)
    metrics.to_csv(out/"validation_modality_ablation.csv",index=False)
    print(metrics.to_string(index=False))
    print(f"Saved validation-only ablation to {out/'validation_modality_ablation.csv'}")


if __name__=="__main__":
    main()
