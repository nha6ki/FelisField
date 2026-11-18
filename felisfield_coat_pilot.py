#!/usr/bin/env python3
"""Local cat-mask/coat pilot. No OpenAI API; no human visual review."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import argparse, collections, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

MODEL = "facebook/mask2former-swin-small-coco-instance"
VERSION = "coat-pilot-01"

def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def color_feature(rgb, mask):
    # D65 CIELAB histogram. Only actual foreground pixels contribute.
    x = rgb[mask].astype(np.float64)/255
    x = np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)
    xyz = x @ np.array([[.4124564,.3575761,.1804375],
                        [.2126729,.7151522,.0721750],
                        [.0193339,.1191920,.9503041]]).T
    xyz /= np.array([.95047,1.,1.08883])
    delta=6/29
    f=np.where(xyz>delta**3,np.cbrt(xyz),xyz/(3*delta**2)+4/29)
    lab=np.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],axis=1)
    lab=np.clip(lab,[0,-128,-128],[100,127.999,127.999])
    h=np.histogramdd(lab,bins=(8,8,8),range=((0,100),(-128,128),(-128,128)))[0].ravel()
    return np.sqrt(h/max(h.sum(),1)).astype(np.float32)

def dino_input(rgb, mask):
    ys,xs=np.where(mask)
    crop=rgb[ys.min():ys.max()+1,xs.min():xs.max()+1].copy()
    cm=mask[ys.min():ys.max()+1,xs.min():xs.max()+1]
    crop[~cm]=[123,116,103]
    im=Image.fromarray(crop)
    im.thumbnail((224,224),Image.Resampling.BICUBIC)
    canvas=Image.new("RGB",(224,224),(123,116,103))
    canvas.paste(im,((224-im.width)//2,(224-im.height)//2))
    arr=(np.asarray(canvas,dtype=np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return arr.transpose(2,0,1)

def sample(rows,n):
    groups=collections.defaultdict(list)
    for p in rows: groups[p['source']].append(p)
    n=min(n or len(rows),len(rows))
    quotas={s:int(n*len(g)/len(rows)) for s,g in groups.items()}
    remainder=n-sum(quotas.values())
    for s in sorted(groups,key=lambda s:(-(n*len(groups[s])/len(rows)-quotas[s]),s))[:remainder]:quotas[s]+=1
    result=[]
    for s,g in sorted(groups.items()):
        result.extend(sorted(g,key=lambda p:hashlib.sha256(('coat42:'+p['sha256']).encode()).hexdigest())[:quotas[s]])
    return sorted(result,key=lambda p:p['sha256'])

def extract(args):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoImageProcessor, AutoModel, Mask2FormerForUniversalSegmentation
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback disabled")
    torch.set_num_threads(4);torch.manual_seed(42)
    project=args.project;out=args.output;out.mkdir(parents=True,exist_ok=True)
    rows=sample(json.loads((project/'results/full/points.json').read_text()),args.count)
    config=dict(version=VERSION,model=MODEL,ids=[p['id'] for p in rows],
                detection_score=.8,mask_threshold=.5,min_area_fraction=.02,min_bbox_side=48,
                input_max_side=1280,visual_reviewed=0,multiple_cats="largest predicted area",
                color="sqrt normalized 8x8x8 D65 Lab histogram of mask pixels; no illumination correction",
                dino="masked crop CLS; full crop fit 224; mean fill; L2",
                model_weight_license="upstream Mask2Former model zoo CC BY-NC 4.0; local noncommercial pilot")
    cp=out/'config.json'
    old=json.loads(cp.read_text()) if cp.exists() else None
    if old and any(old.get(k)!=v for k,v in config.items()):raise RuntimeError("Config changed: use a different --output")
    revision=old.get('revision') if old else HfApi().model_info(MODEL).sha
    config['revision']=revision
    config['dino_revision']=(project/'pipeline/model/revision.txt').read_text().strip()
    if old and old.get('dino_revision')!=config['dino_revision']:raise RuntimeError("DINO model changed")
    save(cp,config)
    modelpath=snapshot_download(MODEL,revision=revision,allow_patterns=["*.json","*.safetensors","README.md"])
    processor=AutoImageProcessor.from_pretrained(modelpath,use_fast=False,local_files_only=True)
    segmenter=Mask2FormerForUniversalSegmentation.from_pretrained(modelpath,local_files_only=True,use_safetensors=True).eval().to('cuda')
    cat_ids={int(k) for k,v in segmenter.config.id2label.items() if v.lower()=='cat'}
    if not cat_ids:raise RuntimeError("Model has no cat class")
    dino=AutoModel.from_pretrained(project/'pipeline/model',local_files_only=True).eval().to('cuda')
    records=out/'records';records.mkdir(exist_ok=True)
    masks=out/'masks';masks.mkdir(exist_ok=True)
    started=time.monotonic();archives={}
    try:
        for i,p in enumerate(rows):
            rp=records/(p['sha256']+'.json');fp=records/(p['sha256']+'.npz')
            if rp.exists():
                oldrec=json.loads(rp.read_text())
                if oldrec['status']!='error' and (oldrec['status']!='accepted' or fp.exists()):continue
            record=dict(id=p['id'],sha256=p['sha256'],status='excluded',visual_reviewed=False,
                        cat_detection_performed=True,source=p['source'])
            try:
                if p['archive'] not in archives:archives[p['archive']]=zipfile.ZipFile(project/'data/photo_assets01/packages'/p['archive'])
                raw=archives[p['archive']].read(p['archive_member'])
                if hashlib.sha256(raw).hexdigest()!=p['sha256']:raise ValueError("Image hash mismatch")
                with Image.open(io.BytesIO(raw)) as original:
                    original.seek(0);im=ImageOps.exif_transpose(original).convert('RGB')
                    record['original_size']=list(im.size)
                    im.thumbnail((1280,1280),Image.Resampling.LANCZOS)
                inputs=processor(images=im,return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    outputs=segmenter(**inputs)
                    result=processor.post_process_instance_segmentation(outputs,threshold=.8,mask_threshold=.5,
                               overlap_mask_area_threshold=.8,target_sizes=[(im.height,im.width)])[0]
                seg=result['segmentation'].cpu().numpy()
                cats=[s for s in result['segments_info'] if s['label_id'] in cat_ids and s['score']>=.8]
                cats=[s for s in cats if np.any(seg==s['id'])]
                record['cat_count']=len(cats)
                if not cats:record['reason']='no_cat_above_threshold'
                else:
                    chosen=max(cats,key=lambda s:int((seg==s['id']).sum()))
                    mask=seg==chosen['id'];ys,xs=np.where(mask);area=float(mask.mean())
                    bw=int(xs.max()-xs.min()+1);bh=int(ys.max()-ys.min()+1)
                    record.update(score=float(chosen['score']),area_fraction=area,bbox=[int(xs.min()),int(ys.min()),bw,bh],
                                  processed_size=list(im.size))
                    if area<.02 or min(bw,bh)<48:record['reason']='cat_region_too_small'
                    else:
                        rgb=np.asarray(im)
                        color=color_feature(rgb,mask)
                        with torch.inference_mode():
                            feature=dino(pixel_values=torch.from_numpy(dino_input(rgb,mask)[None]).to('cuda')).last_hidden_state[:,0,:]
                            feature=torch.nn.functional.normalize(feature,dim=1).cpu().numpy()[0]
                        if not np.isfinite(feature).all():raise ValueError("Nonfinite DINO feature")
                        np.savez(fp,color=color,dino=feature)
                        Image.fromarray(mask.astype('uint8')*255).save(masks/(p['sha256']+'.png'))
                        record.update(status='accepted',reason='automated_thresholds_passed')
                del inputs,outputs,result
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:record.update(status='error',reason=str(e))
            save(rp,record)
            if (i+1)%16==0 or i+1==len(rows):print(json.dumps(dict(done=i+1,total=len(rows),seconds=round(time.monotonic()-started,1))),flush=True)
    finally:
        for z in archives.values():z.close()
    allrec=[json.loads((records/(p['sha256']+'.json')).read_text()) for p in rows]
    counts=dict(collections.Counter(r['status'] for r in allrec))
    reasons=dict(collections.Counter(r['reason'] for r in allrec))
    accepted=[p for p,r in zip(rows,allrec) if r['status']=='accepted']
    save(out/'accepted_points.json',accepted)
    summary=dict(total=len(rows),counts=counts,reasons=reasons,visual_reviewed=0,openai_api_calls=0,
                 warning="Confidence/area thresholds are proxies, not proof of correct cat masks.")
    save(out/'extraction_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
    return accepted

def compare(args,rows):
    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import NearestNeighbors
    from umap import UMAP
    n=len(rows);out=args.output
    if n<40:raise RuntimeError("Fewer than 40 accepted images; inspect extraction_summary.json before clustering")
    data=[]
    for p in rows:
        with np.load(out/'records'/(p['sha256']+'.npz')) as a:data.append((a['color'].copy(),a['dino'].copy()))
    colors=np.stack([a[0] for a in data]);dino=np.stack([a[1] for a in data])
    np.save(out/'color_features.npy',colors);np.save(out/'masked_dino_features.npy',dino)
    # PCA denoises DINO only; normalize blocks independently before mixing.
    pca=PCA(n_components=min(50,n-1,dino.shape[1]),random_state=42)
    shape=norm(pca.fit_transform(dino))
    variants={'color':colors,'masked_dino':shape}
    for w in [.25,.5,.75]:
        variants['mixed_color'+str(int(w*100))]=np.concatenate([np.sqrt(w)*colors,np.sqrt(1-w)*shape],axis=1)
    reports=[]
    for name,x in variants.items():
        # These are exploratory hyperparameters, not validated semantic classes.
        size=max(10,min(50,n//20));labels=HDBSCAN(min_cluster_size=size,min_samples=5,metric='euclidean').fit_predict(x)
        keep=labels>=0;idx=np.flatnonzero(keep);m=len(idx)
        variant=out/name;variant.mkdir(exist_ok=True)
        np.save(variant/'labels.npy',labels)
        report=dict(name=name,total=n,retained=m,noise=int((~keep).sum()),clusters=len(set(labels)-{-1}),
                    min_cluster_size=size,min_samples=5,
                    pca_dino_explained_variance=float(pca.explained_variance_ratio_.sum()),
                    note="Cluster counts/retention do not validate coat-pattern semantics.")
        save(variant/'excluded.json',[{'id':p['id'],'reason':'hdbscan_noise'} for p,v in zip(rows,keep) if not v])
        if m>=16:
            y=UMAP(n_components=3,n_neighbors=min(15,m-1),min_dist=.1,metric='euclidean',random_state=42,n_jobs=1).fit_transform(x[keep])
            np.save(variant/'coordinates.npy',y)
            points=[{**rows[i],'x':float(v[0]),'y':float(v[1]),'z':float(v[2]),'cluster':int(labels[i])} for i,v in zip(idx,y)]
            save(variant/'points.json',points)
            k=min(15,m-1);sample_idx=np.random.default_rng(42).choice(m,min(1000,m),replace=False)
            def neighbors(a):
                nn=NearestNeighbors(n_neighbors=k+1).fit(a).kneighbors(a[sample_idx],return_distance=False)
                return [[j for j in r if j!=i][:k] for i,r in zip(sample_idx,nn)]
            hi=neighbors(x[keep]);lo=neighbors(y)
            report['neighbor_recall']=float(np.mean([len(set(a)&set(b))/k for a,b in zip(hi,lo)]))
        else:
            (variant/'points.json').unlink(missing_ok=True)
            report['map_status']='too_few_classified_images'
        reports.append(report);save(out/'comparison.json',reports);print(json.dumps(report),flush=True)
    print('PASS: pilot complete. Review extraction_summary.json and comparison.json; no automatic winner selected.',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path.home()/'projects/FelisField')
    p.add_argument('--count',type=int,default=512,help='0=all; start with 512')
    p.add_argument('--output',type=Path)
    p.add_argument('--compare-only',action='store_true')
    a=p.parse_args()
    if a.count<0:p.error('--count must be nonnegative')
    if a.output is None:a.output=a.project/'results/coat_pilot512'
    rows=json.loads((a.output/'accepted_points.json').read_text()) if a.compare_only else extract(a)
    compare(a,rows)
if __name__=='__main__':main()
