#!/usr/bin/env python3
"""Full-cohort 50:50 coat map; reuse pilot records, no OpenAI calls."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(k,"4")
# Pilot already downloaded the pinned segmentation model.
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import argparse,collections,hashlib,json,shutil,time,fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def save(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(path)

def normalized(x):
    v=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or (v<1e-10).any():raise ValueError("Invalid or zero feature")
    return x/v

def coarse(x):
    p=np.square(x.astype(np.float64)).reshape(-1,4,2,4,2,4,2).sum(axis=(2,4,6)).reshape(-1,64)
    return np.sqrt(p/p.sum(axis=1,keepdims=True)).astype(np.float32)

def prepare(project,out,pilot):
    import felisfield_coat_pilot as engine
    rows=engine.sample(json.loads((project/"results/full/points.json").read_text()),0)
    if len({r["id"] for r in rows})!=len(rows) or len({r["sha256"] for r in rows})!=len(rows):
        raise ValueError("Duplicate input IDs/hashes")
    config=json.loads((pilot/"config.json").read_text())
    if not config.get("revision"):raise ValueError("Missing pinned model revision")
    config["ids"]=[r["id"] for r in rows]
    out.mkdir(parents=True,exist_ok=True)
    cp=out/"config.json"
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError("Full-run configuration conflict")
    save(cp,config)
    lookup={r["sha256"]:r["id"] for r in rows};reused=0
    (out/"records").mkdir(exist_ok=True);(out/"masks").mkdir(exist_ok=True)
    for rp in sorted((pilot/"records").glob("*.json")):
        r=json.loads(rp.read_text());sha=r.get("sha256")
        if sha not in lookup or r.get("id")!=lookup[sha] or r.get("status") not in ("accepted","excluded"):continue
        target=out/"records"/rp.name
        if target.exists():continue
        if r["status"]=="accepted":
            feature=rp.with_suffix(".npz");mask=pilot/"masks"/(sha+".png")
            if not feature.is_file() or not mask.is_file():continue
            shutil.copy2(feature,out/"records"/feature.name)
            shutil.copy2(mask,out/"masks"/mask.name)
        # Record last: partial copies are not considered completed.
        shutil.copy2(rp,target);reused+=1
    print(json.dumps({"total":len(rows),"pilot_records_copied":reused,"output":str(out)}),flush=True)
    return rows

def validate_records(out,allrows):
    accepted=[];counts=collections.Counter();excluded=[]
    for p in allrows:
        r=json.loads((out/"records"/(p["sha256"]+".json")).read_text())
        if r.get("id")!=p["id"] or r.get("sha256")!=p["sha256"]:raise ValueError("Record identity mismatch")
        status=r["status"];counts[status]+=1
        if status=="error":raise RuntimeError("Extraction errors remain; rerun the same command to retry")
        if status=="accepted":accepted.append(p)
        elif status=="excluded":excluded.append({"id":p["id"],"sha256":p["sha256"],"reason":r["reason"]})
        else:raise ValueError("Unknown record status")
    save(out/"excluded.json",excluded)
    save(out/"accepted_points.json",accepted)
    return accepted,dict(counts)

def build_map(out,rows,total,counts):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from umap import UMAP
    import importlib.metadata
    if len(rows)<51:raise RuntimeError("At least 51 accepted images required")
    n=len(rows);colors=np.empty((n,512),np.float32);dino=np.empty((n,384),np.float32)
    digest=hashlib.sha256()
    for i,r in enumerate(rows):
        path=out/"records"/(r["sha256"]+".npz");digest.update(path.read_bytes());digest.update(r["id"].encode())
        with np.load(path,allow_pickle=False) as z:colors[i]=z["color"];dino[i]=z["dino"]
    if (colors<0).any():raise ValueError("Negative histogram values")
    colors=coarse(normalized(colors));dino=normalized(dino)
    pca=PCA(n_components=50,random_state=42)
    shape=normalized(pca.fit_transform(dino))
    x=np.concatenate([np.sqrt(.5)*colors,np.sqrt(.5)*shape],axis=1).astype(np.float32)
    assert x.shape==(n,114) and np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
    np.save(out/"combined_features.npy",x)
    np.savez(out/"pca_parameters.npz",mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_)
    print(f"Clustering {n} accepted images into 6 groups (CPU)",flush=True)
    km=KMeans(n_clusters=6,random_state=42,n_init=20).fit(x);labels=km.labels_
    if len(set(labels))!=6:raise RuntimeError("Fewer than 6 distinct clusters; inspect features")
    np.save(out/"cluster_labels.npy",labels);np.save(out/"cluster_centers.npy",km.cluster_centers_)
    print("UMAP 3D (CPU); no repeated parameter sweep",flush=True)
    y=UMAP(n_components=3,n_neighbors=15,min_dist=.1,metric="euclidean",random_state=42,n_jobs=1).fit_transform(x)
    if y.shape!=(n,3) or not np.isfinite(y).all():raise ValueError("Invalid coordinates")
    np.save(out/"coordinates.npy",y)
    # Query at most 500 rows against all accepted images; never allocate N x N.
    sample=np.sort(np.random.default_rng(42).choice(n,min(n,500),replace=False))
    def neighbors(a):
        nn=NearestNeighbors(n_neighbors=16,metric="euclidean",n_jobs=4).fit(a)
        found=nn.kneighbors(a[sample],return_distance=False)
        return [[j for j in row if j!=i][:15] for i,row in zip(sample,found)]
    high=neighbors(x);low=neighbors(y)
    recall=float(np.mean([len(set(a)&set(b))/15 for a,b in zip(high,low)]))
    points=[{**p,"x":float(v[0]),"y":float(v[1]),"z":float(v[2]),"cluster":int(label)}
            for p,v,label in zip(rows,y,labels)]
    save(out/"points.json",points);save(out/"row_ids.json",[p["id"] for p in rows])
    summary=dict(status="passed",total_input=total,accepted=n,excluded=total-n,counts=counts,
        cluster_sizes=dict(sorted(collections.Counter(map(int,labels)).items())),
        features=114,color_dimensions=64,dino_pca_dimensions=50,color_weight=.5,
        pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
        neighbor_recall_k15=recall,evaluation_queries=len(sample),evaluation_reference="all accepted images",
        seed=42,input_feature_digest=digest.hexdigest(),visual_reviewed=0,openai_api_calls=0,
        versions={k:importlib.metadata.version(k) for k in ("numpy","scikit-learn","umap-learn")},
        caveats=["Automatic masks are not visually verified.","Cluster labels are exploratory, not coat-type ground truth.",
                 "Mask2Former upstream model weights: CC BY-NC 4.0; local noncommercial workflow."])
    save(out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print("PASS: full coat map ready.",flush=True)

MAP_HTML = r'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>FelisField | 毛色・柄マップ</title>
<style>
*{box-sizing:border-box}body{margin:0;color:#e9f0ef;background:#101b22;font:14px/1.5 system-ui,sans-serif}#field{position:fixed;inset:0;overflow:hidden;background:radial-gradient(ellipse at 49% 46%,#1b2b32,#0c171e 76%)}canvas{width:100%;height:100%;display:block;touch-action:none;cursor:crosshair}
header,footer{position:absolute;left:max(14px,env(safe-area-inset-left));right:max(14px,env(safe-area-inset-right));display:flex;align-items:center;gap:10px;pointer-events:none}header{top:max(10px,env(safe-area-inset-top))}footer{bottom:max(10px,env(safe-area-inset-bottom));justify-content:space-between}h1{margin:0;font:600 clamp(19px,3vw,26px) Georgia,serif;color:#f4eee2;text-shadow:0 1px 4px #10191e}#count{font-size:12px;font-variant-numeric:tabular-nums}.actions{margin-left:auto;display:flex;gap:5px}button{pointer-events:auto;border:1px solid #45606a;border-radius:20px;background:#15252bea;color:#eaf0ec;padding:8px 12px;font:inherit;cursor:pointer}button:focus-visible,canvas:focus-visible{outline:2px solid #ffc85e}button:hover{background:#30444b}.hint{color:#bacac9;font-size:12px;text-shadow:0 1px 3px #000}#photo{position:absolute;width:auto;height:auto;max-width:min(210px,38vw);max-height:min(176px,30dvh);pointer-events:none;border-radius:5px;box-shadow:0 4px 18px #0009}#ring{position:absolute;width:14px;height:14px;border:1.5px solid white;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;box-shadow:0 0 0 1px #101b22}
dialog{max-width:min(390px,92vw);color:#f0f5f3;background:#16252c;border:1px solid #496369;border-radius:12px;padding:20px}dialog::backdrop{background:#071119aa}dialog h2{font-size:17px;margin:0 0 12px}dl{display:grid;grid-template-columns:auto 1fr;gap:7px 16px}dd{margin:0;overflow-wrap:anywhere}dt{color:#a9bebf}a{color:#a0e2da}.sr{position:absolute;clip-path:inset(50%);height:1px;width:1px;overflow:hidden}@media(max-width:560px){header,footer{left:8px;right:8px;gap:6px}button{font-size:12px;padding:8px}.hint{max-width:60%;font-size:11px}#photo{max-width:36vw}}
#about{width:min(620px,94vw);max-width:94vw;max-height:86dvh;overflow:auto;line-height:1.7}#about h3{margin:1.4em 0 .3em}#about p{margin:.65em 0}#about table{width:100%;border-collapse:collapse}#about td,#about th{padding:4px 0;text-align:left;border-bottom:1px solid #395158}#about td:last-child,#about th:last-child{text-align:right}#about summary{cursor:pointer;color:#a0e2da}#about form{margin-top:16px}footer .actions{flex-shrink:0}@media(max-width:560px){footer{align-items:end}.hint{min-width:0}footer .actions{flex-direction:column;align-items:end}}
.brand,.brand h1{display:flex;align-items:center}.brand{gap:10px}.brand h1{gap:6px}.brand svg{width:1.05em;height:1.05em;flex:none}.brand #count{white-space:nowrap}@media(max-width:560px){.brand{align-items:flex-start;flex-direction:column;gap:0}}
#target{position:absolute;pointer-events:none;transform:translate(-50%,-50%);width:11px;height:11px;border:1.5px solid #ffe7a0;box-shadow:0 0 0 1px #14232d;border-radius:50%}</style>
<main id="field"><canvas id="map" tabindex="0" aria-label="猫写真の3次元マップ"></canvas><i id="target" hidden></i><i id="ring" hidden></i><img id="photo" alt="選択した猫写真" hidden>
<header><div class="brand"><h1><svg aria-hidden="true" focusable="false" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 15 5 5l8 5a16 16 0 0 1 6 0l8-5v10c1 2 1.5 3.4 1.5 5 0 5.2-5.3 8.5-12.5 8.5S3.5 25.2 3.5 20c0-1.6.5-3.4 1.5-5Z"/><path d="M10 18h1m10 0h1m-7.5 3L16 22.5l1.5-1.5"/></svg>FelisField</h1><span id="count">読み込み中…</span></div><div class="actions"><button id="random">ランダム</button><button id="reset">視点を戻す</button></div></header>
<footer><span class="hint">点にカーソルを合わせるかタップで写真表示 · ドラッグで回転 · ホイール／ピンチで拡大</span><div class="actions"><button id="openAbout">このマップについて</button><button id="credit">出典・クレジット</button></div></footer><span id="status" class="sr" role="status"></span></main>
<dialog id="panel"><h2>出典・クレジット</h2><div id="details"></div><p>表示用に縮小・JPEG変換しています。点の色は毛色とDINOv2特徴で分類した6群を示します。</p><form method="dialog"><button>閉じる</button></form></dialog>
<dialog id="about" aria-labelledby="aboutTitle"><h2 id="aboutTitle">このマップについて</h2>
<p>FelisFieldでは、1つの点が1枚の猫写真を表します。気になる点にカーソルを合わせるかタップして、すぐ隣の猫と見比べてみてください。写真の作者表記・利用条件・元ページは「出典・クレジット」から確認できます。</p>
<h3>どこから集めた写真？</h3>
<p>6つの出典から集めた17,303枚を掲載しています。同じ猫が別の写真にも登場することがあり、写っている猫の実個体数は不明です。</p>
<table><thead><tr><th scope="col">出典</th><th scope="col">写真数</th></tr></thead><tbody>
<tr><td>iNaturalist</td><td>16,349枚</td></tr>
<tr><td>WordPress Photo Directory</td><td>388枚</td></tr>
<tr><td>Caltech Camera Traps</td><td>338枚</td></tr>
<tr><td>Internet Archive</td><td>188枚</td></tr>
<tr><td>Wikimedia Commons</td><td>27枚</td></tr>
<tr><td>Flickr</td><td>13枚</td></tr></tbody></table>
<p>全体の94.5%はiNaturalistに由来し、9,138件の観察が含まれます。1件の観察に複数の採用写真がある場合があります。</p>
<h3>点の位置と色はどう決まる？</h3>
<p>猫の輪郭の内側で測った色の分布と、画像モデルDINOv2が捉えた見た目の特徴を等しく重み付けしました。UMAPが、その特徴の似た写真を近くに置くように3次元座標を計算します。軸の向き自体に「明るさ」「縞の多さ」といった決まった意味はありません。</p>
<p>点の色は、元の特徴をKMeansで分けた6グループを示します。同じ色の点にも、さまざまな毛色・柄の猫が含まれます。</p>
<details><summary>猫の抽出条件と計算設定</summary>
<p>出典の記述や注釈から集めた29,046枚に、Mask2Formerを適用しました。検出スコアのしきい値は0.80、マスクのしきい値は0.50です。複数の猫を検出したときは、画像内で面積が最大の猫を解析しました。</p>
<p>検出条件を満たす猫領域が得られなかった7,864枚と、領域が画像面積の2%未満、または外接枠の短辺が48画素未満だった3,879枚を除きました。輪郭は自動推定で、写真ごとの目視確認は行っていません。</p>
<p>色は猫領域のCIELABヒストグラム64次元で表し、DINOv2には背景を一定色に置き換えた猫領域を入力しました。DINOv2特徴は384次元から主成分分析で50次元に圧縮。各特徴を正規化して結合した114次元から、UMAPで3次元に配置しました（近傍数15、min_dist 0.1、ユークリッド距離、乱数シード42）。</p></details>
<h3>地図の距離は何を表す？</h3>
<p>異なる出典の写真に限って比較すると、地図上で最も近い15枚の平均毛色距離は0.462、無作為に選んだ15枚では0.689でした。近くの写真では、猫領域の色分布の違いが平均33.0%小さくなっています。</p>
<p>異なる出典の全15,901,489ペアを3D距離順に10等分すると、平均毛色距離は最も近い区間の0.529から、最も遠い区間の0.781まで順に増えました。ここでの毛色距離は配置にも使った色分布の差（Hellinger距離）です。個々のペアでは、毛色の似た猫が離れていることもあります。</p>
<details><summary>柄について確認できたこと</summary><p>出典の説明から対象の柄名を抽出できた88枚は、すべてWordPress Photo Directory由来でした。このうちトラ柄66枚、三毛14枚、サビ5枚の計85枚の中で近い3枚を調べると、柄の一致率は79.6%でした。無作為に選んだ場合の期待値は62.9%です。タキシード柄は3枚のため比較から除きました。この一致率の対象は、柄名のある85枚に限られます。</p></details>
<form method="dialog"><button>閉じる</button></form></dialog>
<script>
'use strict';
const el=id=>document.getElementById(id), canvas=el('map'),ring=el('ring');let image=el('photo');
const ctx=canvas.getContext('2d'),rgb=[[91,192,235],[246,189,96],[155,222,172],[216,160,229],[242,132,130],[196,211,93]];
const colors=Array.from({length:32},(_,b)=>rgb.map(c=>'rgb('+c.map((v,i)=>Math.round([16,27,34][i]+(v-[16,27,34][i])*(.27+.73*b/31))).join(',')+')'));
let coords=[],labels=[],points=[],px,py,depth,radii,next,chosen=-1,yaw=.45,pitch=-.24,zoom=1,W=1,H=1,queued=false,revision=0;
function redraw(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;draw()})}}
function anchor(){
 if(chosen<0)return;const x=px[chosen],y=py[chosen],inside=x>=0&&x<=W&&y>=0&&y<=H;
 ring.hidden=!inside;ring.style.left=x+'px';ring.style.top=y+'px';
 image.style.visibility=inside?'visible':'hidden';
 if(image.hidden)return;
 const iw=image.offsetWidth,ih=image.offsetHeight;
 image.style.left=Math.max(4,Math.min(W-iw-4,x+12+iw<W?x+12:x-iw-12))+'px';
 image.style.top=Math.max(4,Math.min(H-ih-4,y-ih/2))+'px';
}
function draw(){ctx.clearRect(0,0,W,H);if(!coords.length)return;
 const a=Math.cos(yaw),b=Math.sin(yaw),c=Math.cos(pitch),s=Math.sin(pitch),scale=.48*Math.min(W,H)*zoom;
 const heads=new Int32Array(32*6).fill(-1);
 for(let i=0;i<coords.length;i++){const [x,y,z]=coords[i],u=a*x+b*z,v=-b*x+a*z,d=s*y+c*v;
 const f=3.2/(3.2+d),sx=W/2+scale*f*u,sy=H/2-scale*f*(c*y-s*v);
 px[i]=sx;py[i]=sy;depth[i]=d;const near=Math.max(0,Math.min(1,.5-.47*d)),bin=Math.round(31*near);
 radii[i]=(.85+1.75*near)*Math.min(1.55,Math.sqrt(zoom));
 if(sx< -4||sy< -4||sx>W+4||sy>H+4)continue;
 const key=bin*6+labels[i];next[i]=heads[key];heads[key]=i;}
 for(let k=0;k<heads.length;k++){if(heads[k]<0)continue;ctx.fillStyle=colors[Math.floor(k/6)][k%6];ctx.beginPath();
 for(let i=heads[k];i>=0;i=next[i]){const r=radii[i];ctx.moveTo(px[i]+r,py[i]);ctx.arc(px[i],py[i],r,0,Math.PI*2)}ctx.fill();}
 anchor();}
function resize(){clearHover();W=canvas.clientWidth;H=canvas.clientHeight;const d=Math.min(devicePixelRatio||1,1.5);canvas.width=Math.round(W*d);canvas.height=Math.round(H*d);ctx.setTransform(d,0,0,d,0,0);redraw()}
function details(){const box=el('details');box.replaceChildren();if(chosen<0){box.textContent='点を選ぶと、その写真の出典を表示します。';return}
 const p=points[chosen],dl=document.createElement('dl');
 for(const [name,val,url] of [['写真ID',p.id],['出典',p.source],['クレジット',p.creator],['ライセンス',p.license,p.license_url],['元ページ','開く',p.photo_page||p.source_page],...(p.photo_page&&p.photo_page!==p.source_page?[['観察記録','開く',p.source_page]]:[])]){
 const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=name;dd.textContent=val||'未記録';
 if(url&&/^https?:\/\//i.test(url)){const a=document.createElement('a');a.href=url;a.textContent=val||url;a.target='_blank';a.rel='noopener noreferrer';dd.replaceChildren(a)}else if(name==='元ページ'){dd.textContent='未記録'}dl.append(dt,dd)}box.append(dl)}
function select(i){chosen=i;const token=++revision;image.hidden=true;ring.hidden=i<0;redraw();
 if(i<0){image.removeAttribute('src');return}const fresh=new Image();fresh.onload=()=>{if(token!==revision)return;fresh.id='photo';fresh.alt='選択した猫写真';image.replaceWith(fresh);image=fresh;anchor();el('status').textContent='選択した写真 '+points[i].id};fresh.onerror=()=>{if(token===revision)el('count').textContent='画像を取得できませんでした'};fresh.src='/image/'+i;}

function hit(x,y,touch){let best=-1,bd=Infinity;for(let i=0;i<coords.length;i++){const d=Math.hypot(x-px[i],y-py[i]),radius=radii[i];if(d>radius+(touch?9:4))continue;
 if(best<0||(d<=radius&&bd>radii[best])||(d<=radius&&bd<=radii[best]&&depth[i]<depth[best])||(d>radius&&bd>radii[best]&&d<bd)){best=i;bd=d}}
 return best}
function pick(x,y,touch){select(hit(x,y,touch))}
const fingers=new Map();let moved=false,ox=0,oy=0;
const target=el('target');
let hoverPosition=null,hoverFrame=false;
let hoverCandidate=-1,previewTimer;
function clearHover(){
 hoverPosition=null;target.hidden=true;
 clearTimeout(previewTimer);hoverCandidate=-1;
}
function hover(x,y){
 hoverPosition=[x,y];
 if(hoverFrame)return;
 hoverFrame=true;
 requestAnimationFrame(()=>{
   hoverFrame=false;
   if(!hoverPosition||!px)return;
   const i=hit(...hoverPosition,false);
   if(i!==hoverCandidate){
     clearTimeout(previewTimer);
     hoverCandidate=i;
     if(i>=0&&i!==chosen)previewTimer=setTimeout(()=>{
       if(hoverPosition&&hit(...hoverPosition,false)===i&&chosen!==i)select(i);
     },160);
   }
   target.hidden=i<0;
   if(i>=0){target.style.left=px[i]+'px';target.style.top=py[i]+'px'}
 });
}
canvas.onpointerleave=clearHover;
canvas.onpointerdown=e=>{clearHover();canvas.setPointerCapture(e.pointerId);fingers.set(e.pointerId,[e.clientX,e.clientY]);if(fingers.size===1){moved=false;ox=e.clientX;oy=e.clientY}else moved=true};
canvas.onpointermove=e=>{const old=fingers.get(e.pointerId);if(!old){if(e.pointerType==="mouse"&&!e.buttons)hover(e.clientX,e.clientY);return}const x=e.clientX,y=e.clientY,other=[...fingers].find(([id])=>id!==e.pointerId);
 if(other){const [u,v]=other[1];zoom=Math.max(.25,Math.min(14,zoom*Math.hypot(x-u,y-v)/Math.max(1,Math.hypot(old[0]-u,old[1]-v))));moved=true}
 else if(moved||Math.hypot(x-ox,y-oy)>4){yaw+=(x-old[0])*.006;pitch+=(y-old[1])*.006;moved=true}
 fingers.set(e.pointerId,[x,y]);if(moved)redraw()};
canvas.onpointerup=e=>{fingers.delete(e.pointerId);if(!moved&&coords.length)pick(e.clientX,e.clientY,e.pointerType==='touch')};
canvas.onpointercancel=e=>{fingers.delete(e.pointerId);moved=true};
canvas.addEventListener('wheel',e=>{e.preventDefault();clearHover();zoom=Math.max(.25,Math.min(14,zoom*Math.exp(-e.deltaY*.001)));redraw()},{passive:false});
el('random').onclick=()=>{clearHover();if(points.length)select(Math.floor(Math.random()*points.length))};el('reset').onclick=()=>{clearHover();yaw=.45;pitch=-.24;zoom=1;redraw()};
el('openAbout').onclick=()=>{clearHover();el('about').showModal();el('about').scrollTop=0};
el('credit').onclick=()=>{clearHover();details();el('panel').showModal()};document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!el('panel').open&&!el('about').open){clearHover();select(-1)}});
new ResizeObserver(resize).observe(canvas);
(async()=>{try{const res=await fetch('/data');if(!res.ok)throw Error();const data=await res.json(),v=data.variants.mixed_color50;
 points=data.points;coords=v.coords;labels=v.labels;[px,py,depth,radii]=Array.from({length:4},()=>new Float32Array(points.length));next=new Int32Array(points.length);
 el('count').textContent='猫の写真 '+points.length.toLocaleString('ja-JP')+'枚';redraw()}catch{el('count').textContent='マップを取得できませんでした'}})();
</script></html>
'''

def credit_points(out, rows):
    fields = ("id", "source", "creator", "license", "license_url", "source_page")
    points = [{key: row.get(key) for key in fields} for row in rows]
    path = out / "credit_audit.json"
    if not path.exists():
        print("Credit audit unavailable; displaying original local metadata.", flush=True)
        return points
    records = json.loads(path.read_text(encoding="utf-8"))["photos"]
    by_id = {r["id"]: r for r in records}
    if len(records) != len(by_id) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("Credit audit photo IDs differ from map")
    for point, row in zip(points, rows):
        record = by_id[row["id"]]
        if any(record[key] != row[key] for key in ("sha256", "source", "source_page", "license")):
            raise ValueError("Credit audit mismatch: " + row["id"])
        if record["issues"]:
            continue
        point.update(creator=record["credit"], photo_page=record["photo_page"],
                     license_url=record["license_url"])
    return points


def serve(project,out,port):
    import felisfield_coat_viewer as viewer
    from http.server import ThreadingHTTPServer
    if not (out/"summary.json").exists():raise RuntimeError("Complete the full run before --serve")
    rows=json.loads((out/"points.json").read_text())
    ids=json.loads((out/"row_ids.json").read_text())
    if ids!=[p["id"] for p in rows]:raise ValueError("Point order mismatch")
    for p in rows:
        if not (project/"data/web_previews_v1"/(p["sha256"]+".jpg")).is_file():raise FileNotFoundError("Preview missing: "+p["id"])
    y=np.array([[p["x"],p["y"],p["z"]] for p in rows]);y-=y.mean(axis=0)
    if not np.isfinite(y).all():raise ValueError("Nonfinite map")
    y/=max(float(np.abs(y).max()),1e-10)
    labels=[p["cluster"] for p in rows]
    if any(l not in range(6) for l in labels):raise ValueError("Invalid labels")
    data={"points":credit_points(out, rows),"variants":{"mixed_color50":{"coords":y.tolist(),"labels":labels}}}
    viewer.HTML = MAP_HTML
    server=ThreadingHTTPServer(("127.0.0.1",port),viewer.make_handler(project,rows,data))
    print(f"FelisField full coat map: {len(rows):,} photos. http://localhost:{port}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,default=Path.home()/"projects/FelisField")
    p.add_argument("--serve",action="store_true")
    p.add_argument("--port",type=int,default=8767)
    a=p.parse_args();out=a.project/"results/coat_full50";pilot=a.project/"results/coat_pilot512"
    if a.serve:serve(a.project,out,a.port);return
    out.mkdir(parents=True,exist_ok=True)
    with (out/"run.lock").open("w") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("Another full run is active")
        allrows=prepare(a.project,out,pilot)
        # Invalidate the completion marker while regenerating output.
        (out/"summary.json").unlink(missing_ok=True)
        import felisfield_coat_pilot as engine
        print("Extracting cat regions. Completed records are reused.",flush=True)
        engine.extract(SimpleNamespace(project=a.project,output=out,count=0))
        import torch
        torch.cuda.empty_cache()
        rows,counts=validate_records(out,allrows)
        build_map(out,rows,len(allrows),counts)
if __name__=="__main__":main()
