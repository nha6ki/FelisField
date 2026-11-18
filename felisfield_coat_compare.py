#!/usr/bin/env python3
"""Compare saved cat features on one fixed cohort. No images, models, or network."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(name,"4")
import argparse, hashlib, json, collections, time
from pathlib import Path
import numpy as np

VERSION="coat-compare-02"

def save(path,obj):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)

def normalize(x):
    lengths=np.linalg.norm(x,axis=1,keepdims=True)
    if (lengths<1e-10).any():raise ValueError("Zero-length feature vector")
    return x/lengths

def coarsen(x):
    # Saved features are sqrt(probability); restore probability BEFORE summing.
    p=np.square(x.astype(np.float64)).reshape(-1,4,2,4,2,4,2).sum(axis=(2,4,6)).reshape(-1,64)
    p/=p.sum(axis=1,keepdims=True)
    return np.sqrt(p).astype(np.float32)

def load(source):
    rows=json.loads((source/"accepted_points.json").read_text())
    if not 40<=len(rows)<=2000:raise ValueError("This comparison is for a 40–2000-image pilot")
    if len({r["sha256"] for r in rows})!=len(rows):raise ValueError("Duplicate asset hashes")
    color=[];dino=[];digest=hashlib.sha256()
    digest.update(json.dumps([r["id"] for r in rows]).encode())
    for row in rows:
        path=source/"records"/(row["sha256"]+".npz")
        digest.update(path.read_bytes())
        with np.load(path,allow_pickle=False) as z:
            color.append(z["color"].copy());dino.append(z["dino"].copy())
    c=np.stack(color);d=np.stack(dino)
    if c.shape!=(len(rows),512) or d.shape!=(len(rows),384):raise ValueError("Unexpected feature dimensions")
    if not np.isfinite(c).all() or not np.isfinite(d).all() or (c<0).any():raise ValueError("Invalid features")
    return rows,normalize(c),normalize(d),digest.hexdigest()

def run(source,out):
    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN,KMeans
    from sklearn.metrics import pairwise_distances,silhouette_score,adjusted_rand_score
    from sklearn.manifold import trustworthiness
    from umap import UMAP
    rows,c,d,digest=load(source);n=len(rows)
    config=dict(version=VERSION,input_digest=digest,n=n,cohort="all segmentation-accepted pilot images",
                final_exclusion_policy="unchanged; no noise points added to final app",
                purpose="diagnostic comparison; no automatic best model",
                weights=[.25,.5,.75],kmeans_k=[6,10,14],seeds=[42,7])
    out.mkdir(parents=True,exist_ok=True);cp=out/"config.json"
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError("Inputs changed; choose a new output directory")
    save(cp,config)
    save(out/"row_ids.json",[r["id"] for r in rows])
    pc=PCA(n_components=min(50,n-1),random_state=42);p=normalize(pc.fit_transform(d))
    coarse=coarsen(c)
    variants={"color512_baseline":c,"color64":coarse,"masked_dino_pca50":p}
    for w in [.25,.5,.75]:
        variants[f"mixed_color{int(w*100)}"]=np.concatenate([np.sqrt(w)*coarse,np.sqrt(1-w)*p],axis=1)
    k=15
    def distance_neighbors(x):
        distances=pairwise_distances(x,metric="euclidean")
        np.fill_diagonal(distances,np.inf)
        # Stable tie handling and explicit self exclusion.
        return distances,np.argsort(distances,axis=1,kind="stable")[:,:k]
    def overlap(a,b):
        return float(np.mean([len(set(x)&set(y))/k for x,y in zip(a,b)]))
    _,color_neighbors=distance_neighbors(coarse)
    _,dino_neighbors=distance_neighbors(d) # original 384 dimensions, before PCA
    reports=[]
    for name,x in variants.items():
        print("START",name,flush=True);started=time.monotonic()
        folder=out/name;folder.mkdir(exist_ok=True)
        distances,hi=distance_neighbors(x);np.fill_diagonal(distances,0.)
        embedding_neighbors=[];maps=[]
        for seed in (42,7):
            y=UMAP(n_components=3,n_neighbors=15,min_dist=.1,metric="euclidean",random_state=seed,n_jobs=1).fit_transform(x)
            np.save(folder/f"coordinates_seed{seed}.npy",y)
            _,lo=distance_neighbors(y);embedding_neighbors.append(lo)
            maps.append(dict(seed=seed,neighbor_recall_k15=overlap(hi,lo),
                trustworthiness_k15=float(trustworthiness(x,y,n_neighbors=15,metric="euclidean")),
                color_neighbor_recall_k15=overlap(color_neighbors,lo),
                original_dino_neighbor_recall_k15=overlap(dino_neighbors,lo)))
        clusters=[]
        def summarize(labels,method,parameters):
            valid=labels>=0;counts=collections.Counter(int(v) for v in labels[valid])
            entry=dict(method=method,parameters=parameters,retained=int(valid.sum()),noise=int((~valid).sum()),
                       clusters=len(counts),cluster_sizes=sorted(counts.values(),reverse=True))
            if 1<len(counts)<valid.sum():
                entry["silhouette_on_assigned"]=float(silhouette_score(distances[np.ix_(valid,valid)],labels[valid],metric="precomputed"))
            else:entry["silhouette_on_assigned"]=None
            return entry
        for minimum,samples in [(16,5),(10,3),(8,2)]:
            labels=HDBSCAN(min_cluster_size=minimum,min_samples=samples,metric="euclidean").fit_predict(x)
            tag=f"hdbscan_m{minimum}_s{samples}"
            np.save(folder/(tag+".npy"),labels)
            save(folder/(tag+"_excluded.json"),[{"id":r["id"],"reason":"density_noise"} for r,l in zip(rows,labels) if l<0])
            clusters.append(summarize(labels,"HDBSCAN",dict(min_cluster_size=minimum,min_samples=samples)))
        for count in (6,10,14):
            labels=KMeans(n_clusters=count,random_state=42,n_init=20).fit_predict(x)
            other=KMeans(n_clusters=count,random_state=7,n_init=20).fit_predict(x)
            np.save(folder/f"kmeans_k{count}.npy",labels)
            entry=summarize(labels,"KMeans",dict(k=count,n_init=20))
            entry["seed_ARI"]=float(adjusted_rand_score(labels,other));entry["assignment"]="forced assignment; not evidence of natural coat classes"
            clusters.append(entry)
        sources=np.array([r["source"] for r in rows])
        report=dict(name=name,images=n,dimensions=x.shape[1],maps=maps,
            umap_seed_neighbor_overlap=overlap(*embedding_neighbors),
            feature_color_neighbor_overlap=overlap(hi,color_neighbors),
            feature_original_dino_neighbor_overlap=overlap(hi,dino_neighbors),
            same_source_neighbor_fraction=float((sources[hi]==sources[:,None]).mean()),
            clustering=clusters,seconds=round(time.monotonic()-started,2))
        reports.append(report);save(out/"comparison.json",reports)
        print(json.dumps({k:report[k] for k in ("name","images","umap_seed_neighbor_overlap","seconds")}),flush=True)
    summary=dict(status="passed",images=n,variants=len(reports),pca_explained_variance=float(pc.explained_variance_ratio_.sum()),
       input_digest=digest,visual_reviewed=0,openai_api_calls=0,images_read=0,
       final_map_modified=False,automatic_winner=None,
       caveats=["Color metric measures color distributions, not coat-pattern semantics.",
                "DINO features still include pose, shape, and color.",
                "Neighbor recall uses different feature references across variants; do not rank by it alone.",
                "HDBSCAN silhouettes use assigned subsets; KMeans assigns every accepted image.",
                "Two-seed stability is a limited check, not comprehensive validation."])
    save(out/"summary.json",summary)
    # Compact table for copying back to chat; full metrics remain in comparison.json.
    lines=["variant | HDBSCAN retained(m16/m10/m8) | KMeans ARI(k6/k10/k14) | map recall(seed42) | color recall | DINO recall"]
    for r in reports:
        h=[a["retained"] for a in r["clustering"] if a["method"]=="HDBSCAN"]
        km=[round(a["seed_ARI"],3) for a in r["clustering"] if a["method"]=="KMeans"]
        m=r["maps"][0]
        lines.append(f'{r["name"]} | {h} | {km} | {m["neighbor_recall_k15"]:.3f} | {m["color_neighbor_recall_k15"]:.3f} | {m["original_dino_neighbor_recall_k15"]:.3f}')
    (out/"summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines));print("PASS: fixed-cohort comparison complete. Existing app unchanged.",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    project=Path.home()/"projects/FelisField"
    p.add_argument("--input",type=Path,default=project/"results/coat_pilot512")
    p.add_argument("--output",type=Path,default=project/"results/coat_compare02")
    a=p.parse_args()
    if a.input.resolve()==a.output.resolve():p.error("Output must differ from input")
    run(a.input,a.output)
if __name__=="__main__":main()
