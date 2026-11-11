"""Use validation only to choose the number of beams in a power-scanned shortlist."""
from __future__ import annotations
import argparse
import json
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
    image_features,lookup=model_lib.get_image_features(frame,data_root,out,device,24,False)
    windows=model_lib.make_windows(frame,scaled,lookup)
    rows=windows["val"]
    loader=DataLoader(model_lib.WindowDataset(rows,image_features),batch_size=args.batch_size,
                      shuffle=False,num_workers=0,pin_memory=(device.type=="cuda"))
    checkpoint=torch.load(out/"best_model.pt",map_location=device,weights_only=True)
    model=model_lib.ReliabilityGatedGRU().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    logits_all=[]; change_all=[]; y_all=[]; prev_all=[]
    with torch.inference_mode():
        for visual,sensors,y,prev,_ in loader:
            logits,change,_=model(visual.to(device),sensors.to(device))
            logits_all.append(logits.cpu()); change_all.append(change.cpu())
            y_all.append(y); prev_all.append(prev)
    logits=torch.cat(logits_all); change=torch.cat(change_all)
    y=torch.cat(y_all).long(); prev=torch.cat(prev_all).long()
    probs=model_lib.make_final_prob(logits,change,prev,float(checkpoint["temperature"]),float(checkpoint["bias"]))
    top=probs.topk(5,dim=1).indices.numpy()+1
    y1=y.numpy()+1; prev1=prev.numpy()+1
    scenario=pd.read_csv(data_root/"scenario23.csv").set_index("index")
    metric_rows=[]; prediction_rows=[]
    regrets={k:[] for k in (1,3,5)}
    persist_regrets=[]
    for i,r in enumerate(rows):
        rel=str(scenario.loc[r["index"],"unit1_pwr_60ghz"]).replace("\\","/").removeprefix("./")
        power=np.maximum(np.atleast_1d(np.loadtxt(data_root/rel,dtype=np.float64)),1e-15)
        best=float(power.max())
        prior_power=float(power[int(prev1[i])-1])
        persist_regrets.append(10*np.log10(best/prior_power))
        row={"target_index":r["index"],"true_beam":int(y1[i]),"previous_beam":int(prev1[i])}
        for k in (1,3,5):
            candidates=top[i,:k].tolist()
            selected=max(candidates,key=lambda b:power[int(b)-1])
            regret=10*np.log10(best/float(power[int(selected)-1]))
            regrets[k].append(regret)
            row[f"top{k}_beams"]=" ".join(map(str,candidates))
            row[f"top{k}_selected_by_power"] = int(selected)
            row[f"top{k}_power_regret_db"] = float(regret)
        prediction_rows.append(row)
    cls=model_lib.classification_metrics(probs,y)
    for k in (1,3,5):
        rr=np.asarray(regrets[k])
        metric_rows.append({"policy":f"model_top{k}","beams_scanned":k,
            "scan_fraction_of_64":k/64,"topk_label_accuracy":cls[f"top{k}_accuracy"],
            "mean_power_regret_db":float(rr.mean()),"median_power_regret_db":float(np.median(rr)),
            "power_within_0_01db_fraction":float(np.mean(rr<=0.01)),"samples":len(rr)})
    pr=np.asarray(persist_regrets)
    metric_rows.append({"policy":"persistence_1","beams_scanned":1,"scan_fraction_of_64":1/64,
        "topk_label_accuracy":float(np.mean(prev1==y1)),"mean_power_regret_db":float(pr.mean()),
        "median_power_regret_db":float(np.median(pr)),"power_within_0_01db_fraction":float(np.mean(pr<=0.01)),
        "samples":len(pr)})
    metrics=pd.DataFrame(metric_rows).sort_values("beams_scanned")
    metrics.to_csv(out/"validation_shortlist_metrics.csv",index=False)
    pd.DataFrame(prediction_rows).to_csv(out/"validation_shortlist_predictions.csv",index=False)
    summary={"split":"validation only","rows":len(rows),"decision_rule":"best measured power among each model top-K shortlist",
             "selected_change_temperature":float(checkpoint["temperature"]),"selected_change_bias":float(checkpoint["bias"]),
             "metrics":metric_rows,"note":"Use this validation report to choose K; do not tune K from test results."}
    (out/"validation_shortlist_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Saved validation-only shortlist report to {out}")


if __name__=="__main__":
    main()
