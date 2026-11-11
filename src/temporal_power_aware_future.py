"""Warm-start and fine-tune the next-beam model with power-aware ranking.

Validation selects the checkpoint and change calibration by top-5 measured-power
regret. The test split is evaluated once after that choice.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import temporal_reliability_gated_future as base


class PowerWindowDataset(Dataset):
    def __init__(self, records, image_features):
        self.records=records
        self.image_features=image_features
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        visual=np.asarray(self.image_features[r["frames"]],dtype=np.float32).copy()
        return (torch.from_numpy(visual),torch.from_numpy(r["sensor"].copy()),
                r["target"],r["previous"],i,torch.from_numpy(r["power_db"].copy()))


def load_feature_cache(frame, cache_dir:Path):
    cache=cache_dir/"resnet50_frame_features.npy"
    ids_path=cache_dir/"resnet50_frame_indices.npy"
    if not cache.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Expected existing ResNet feature cache in {cache_dir}")
    ids=frame["index"].to_numpy(np.int64)
    if not np.array_equal(np.load(ids_path,allow_pickle=False),ids):
        raise ValueError("Feature cache sample order differs from the current project data")
    return np.load(cache,mmap_mode="r"),{int(v):i for i,v in enumerate(ids)}


def add_power_targets(windows,data_root):
    for split,rows in windows.items():
        for r in rows:
            rel=r["power_path"].replace("\\","/").removeprefix("./")
            power=np.maximum(np.atleast_1d(np.loadtxt(data_root/rel,dtype=np.float64)),1e-15)
            if len(power)!=base.N_CLASSES:
                raise ValueError(f"Expected {base.N_CLASSES} beam-power values at {rel}; got {len(power)}")
            db=(10*np.log10(power)).astype(np.float32)
            r["power_db"]=(db-db.max()).astype(np.float32)


def collect(model,loader,device):
    model.eval(); cls=[]; change=[]; y=[]; prev=[]; pwr=[]
    with torch.inference_mode():
        for visual,sensors,targets,previous,_,power_db in loader:
            logits,chg,_=model(visual.to(device),sensors.to(device))
            cls.append(logits.cpu()); change.append(chg.cpu())
            y.append(targets.long()); prev.append(previous.long()); pwr.append(power_db.double())
    return {"cls":torch.cat(cls),"change":torch.cat(change),"y":torch.cat(y),
            "prev":torch.cat(prev),"power_db":torch.cat(pwr)}


def topk_power_metrics(prob,power_db,k):
    candidates=prob.topk(k,dim=1).indices
    available=power_db.max(dim=1).values
    selected=power_db.gather(1,candidates).max(dim=1).values
    regret=(available-selected).clamp_min(0)
    return float(regret.mean()),float((regret<=0.01).double().mean())


def tune_calibration_for_top5(data):
    best=None
    for temp in (0.5,0.75,1.0,1.25,1.5,2.0):
        for bias in (-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5):
            prob=base.make_final_prob(data["cls"],data["change"],data["prev"],temp,bias)
            regret,_=topk_power_metrics(prob,data["power_db"],5)
            top5_acc=float((prob.topk(5,dim=1).indices==data["y"][:,None]).any(1).double().mean())
            candidate=(-regret,top5_acc,-abs(temp-1.0),-abs(bias),temp,bias)
            if best is None or candidate>best: best=candidate
    return best[-2],best[-1],-best[0]


def soft_power_loss(logits,previous,power_db,power_temperature):
    rows=torch.arange(logits.shape[0],device=logits.device)
    conditional_logits=logits.masked_fill(F.one_hot(previous,base.N_CLASSES).bool(),-1e4)
    log_q=F.log_softmax(conditional_logits,dim=1)
    q=log_q.exp()
    relative=power_db.to(logits.device,dtype=logits.dtype)
    regret=(-relative).clamp(0,40)
    target_scores=(relative/power_temperature).masked_fill(F.one_hot(previous,base.N_CLASSES).bool(),-1e4)
    target=F.softmax(target_scores,dim=1)
    listwise=-(target*log_q).sum(dim=1).mean()
    expected_regret=(q*regret).sum(dim=1).mean()
    return listwise,expected_regret,conditional_logits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,required=True)
    ap.add_argument("--feature-cache-dir",type=Path,required=True)
    ap.add_argument("--init-checkpoint",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--patience",type=int,default=5)
    ap.add_argument("--seed",type=int,default=23)
    ap.add_argument("--power-temperature",type=float,default=0.75)
    ap.add_argument("--power-listwise-weight",type=float,default=0.6)
    ap.add_argument("--expected-regret-weight",type=float,default=0.05)
    ap.add_argument("--changed-label-weight",type=float,default=0.25)
    args=ap.parse_args()
    root=args.project_root.resolve(); cache_dir=args.feature_cache_dir.resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any((out/f).exists() for f in ("best_model.pt","metrics.csv","test_predictions.csv")):
        raise FileExistsError(f"Choose a new output folder; completed output already exists: {out}")
    base.seed_everything(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}",flush=True)
    if device.type=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}",flush=True)
    frame,data_root=base.load_joined(root)
    scaled,mean,std,median=base.fit_sensor_scaler(frame)
    features,lookup=load_feature_cache(frame,cache_dir)
    windows=base.make_windows(frame,scaled,lookup)
    add_power_targets(windows,data_root)
    print("Future windows:",{k:len(v) for k,v in windows.items()},flush=True)
    datasets={k:PowerWindowDataset(v,features) for k,v in windows.items()}
    loaders={
        "train":DataLoader(datasets["train"],batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=="cuda"),
        "val":DataLoader(datasets["val"],batch_size=128,shuffle=False,num_workers=0,pin_memory=device.type=="cuda"),
        "test":DataLoader(datasets["test"],batch_size=128,shuffle=False,num_workers=0,pin_memory=device.type=="cuda"),
    }
    model=base.ReliabilityGatedGRU().to(device)
    init=torch.load(args.init_checkpoint.resolve(),map_location=device,weights_only=True)
    model.load_state_dict(init["model"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=2)
    best_val=float("inf"); best_epoch=0; stale=0; history=[]
    started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; seen=0
        for visual,sensors,y,previous,_,power_db in loaders["train"]:
            visual=visual.to(device,non_blocking=True); sensors=sensors.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True).long(); previous=previous.to(device,non_blocking=True).long()
            power_db=power_db.to(device,non_blocking=True,dtype=torch.float32)
            changed=(y!=previous)
            optimizer.zero_grad(set_to_none=True)
            logits,change_logits,_=model(visual,sensors)
            change_loss=F.binary_cross_entropy_with_logits(change_logits,changed.float())
            list_loss,expected_regret,conditional_logits=soft_power_loss(logits,previous,power_db,args.power_temperature)
            if changed.any(): hard=F.cross_entropy(conditional_logits[changed],y[changed])
            else: hard=logits.sum()*0
            loss=change_loss+args.power_listwise_weight*list_loss+args.changed_label_weight*hard+args.expected_regret_weight*expected_regret
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step()
            total+=float(loss.detach())*len(y); seen+=len(y)
        val=collect(model,loaders["val"],device)
        temp,bias,val_regret=tune_calibration_for_top5(val)
        probs=base.make_final_prob(val["cls"],val["change"],val["prev"],temp,bias)
        val_cls=base.classification_metrics(probs,val["y"])
        scheduler.step(val_regret)
        rec={"epoch":epoch,"train_loss":total/max(seen,1),"val_top1":val_cls["top1_accuracy"],
             "val_top3":val_cls["top3_accuracy"],"val_top5":val_cls["top5_accuracy"],
             "val_macro_f1":val_cls["macro_f1"],"val_top5_power_regret_db":val_regret,
             "change_temperature":temp,"change_bias":bias,"learning_rate":optimizer.param_groups[0]["lr"]}
        history.append(rec)
        print(f"Epoch {epoch:02d}/{args.epochs} loss={rec['train_loss']:.4f} val_top5_regret={val_regret:.5f} dB "
              f"top1={rec['val_top1']:.4f} top5={rec['val_top5']:.4f}",flush=True)
        if val_regret<best_val-1e-7:
            best_val=val_regret; best_epoch=epoch; stale=0
            torch.save({"model":model.state_dict(),"epoch":epoch,"temperature":temp,"bias":bias,
                        "sensor_mean":torch.from_numpy(mean),"sensor_std":torch.from_numpy(std),
                        "sensor_median":torch.from_numpy(median),"features":base.FEATURES,"window":base.WINDOW,
                        "selection_metric":"validation top-5 received-power regret"},out/"best_model.pt")
        else:
            stale+=1
            if stale>=args.patience:
                print(f"Early stopping; best epoch {best_epoch}.",flush=True); break
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)

    ckpt=torch.load(out/"best_model.pt",map_location=device,weights_only=True)
    model.load_state_dict(ckpt["model"])
    test=collect(model,loaders["test"],device)
    probs=base.make_final_prob(test["cls"],test["change"],test["prev"],float(ckpt["temperature"]),float(ckpt["bias"]))
    metrics=base.classification_metrics(probs,test["y"])
    top=probs.topk(5,dim=1).indices
    pwr=test["power_db"]
    test_metrics={"split":"test","n":len(test["y"]),"best_epoch":best_epoch,
                  "persistence_top1":float((test["prev"]==test["y"]).double().mean()),
                  "training_seconds":time.perf_counter()-started}
    prior_regret=(pwr.max(1).values-pwr.gather(1,test["prev"].view(-1,1)).squeeze(1)).clamp_min(0)
    test_metrics["persistence_top1_power_regret_db"]=float(prior_regret.mean())
    test_metrics["persistence_within_0_01db_fraction"]=float((prior_regret<=0.01).double().mean())
    test_metrics.update(metrics)
    for k in (1,3,5):
        rg,within=topk_power_metrics(probs,pwr,k)
        test_metrics[f"top{k}_power_regret_db"]=rg
        test_metrics[f"top{k}_within_0_01db_fraction"]=within
    pd.DataFrame([test_metrics]).to_csv(out/"metrics.csv",index=False)
    pred=[]
    for i,r in enumerate(windows["test"]):
        pred.append({"seq_index":r["seq"],"target_index":r["index"],"previous_beam":r["previous"]+1,
                     "true_next_beam":r["target"]+1,"top5_beams":" ".join(map(str,(top[i].numpy()+1).tolist())),
                     "predicted_next_beam":int(top[i,0])+1,"p_change":float(torch.sigmoid(test["change"][i]/float(ckpt["temperature"])+float(ckpt["bias"]))),
                     })
    pd.DataFrame(pred).to_csv(out/"test_predictions.csv",index=False)
    summary={"model":"power-aware fine-tune of reliability-gated camera/sensor GRU",
        "validation_selection":"minimum mean power regret after scanning model top-5 candidates",
        "power_loss":{"listwise_weight":args.power_listwise_weight,"expected_regret_weight":args.expected_regret_weight,
                      "changed_label_weight":args.changed_label_weight,"target_temperature_db":args.power_temperature},
        "best_epoch":best_epoch,"validation_best_top5_regret_db":best_val,
        "test_metrics":test_metrics,"test_evaluated_once":True}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("TEST:",json.dumps(test_metrics,indent=2),flush=True)
    print(f"Saved power-aware run to {out}",flush=True)


if __name__=="__main__":
    main()
