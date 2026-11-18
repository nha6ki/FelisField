#!/usr/bin/env python3
"""Offline coat comparison viewer; reads saved results and JPEG previews only."""
import argparse,json,re
from pathlib import Path
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from functools import lru_cache
import numpy as np

HTML = '<!doctype html>\n<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>FelisField — Local explorer</title>\n<style>\n:root{font-family:system-ui,sans-serif;color:#e5edf4;background:#101923;color-scheme:dark}*{box-sizing:border-box}body{margin:0}header{padding:18px 24px;border-bottom:1px solid #30404e;display:flex;align-items:center;gap:20px}h1{font-size:24px;margin:0;letter-spacing:.02em}header span{font-size:13px;color:#a7b8c7}main{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(320px,1fr);height:calc(100dvh - 70px)}.map{position:relative;min-height:400px}#plot{width:100%;height:100%;display:block;touch-action:none}.toolbar{flex-wrap:wrap;right:12px}select{background:#24394a;color:white;border:1px solid #486074;border-radius:7px}.toolbar{position:absolute;top:12px;left:16px;z-index:2;display:flex;gap:8px}button{background:#24394a;color:#edf5fb;border:1px solid #486074;border-radius:7px;padding:9px 13px;cursor:pointer}button:hover{background:#36546a}aside{overflow:auto;padding:20px;border-left:1px solid #30404e}.photo{height:44vh;min-height:240px;background:#090f16;border-radius:12px;display:flex;align-items:center;justify-content:center;overflow:hidden}.photo img{display:block;width:100%;height:100%;object-fit:contain}#status{min-height:24px;font-size:14px;color:#b9cbd9;margin:14px 0}h2{font-size:15px;margin-top:24px}dl{display:grid;grid-template-columns:85px 1fr;font-size:13px;line-height:1.7;gap:8px;margin:0}dt{color:#9cb1c1}dd{margin:0;overflow-wrap:anywhere}a{color:#8ed6dc}p.note{font-size:12px;line-height:1.8;color:#9cb1c1}#count{color:#85d9d1}.hidden{display:none!important}@media(max-width:760px){header{padding:14px;gap:12px}header span{font-size:11px}main{display:block;height:auto}.map{height:54dvh;min-height:340px}aside{border-left:0;border-top:1px solid #30404e;padding:16px}.photo{height:40dvh}}\n</style>\n<header><h1>FelisField</h1><span>毛色・柄の比較 · 325枚パイロット</span><span id="count"></span></header>\n<main><section class="map" aria-label="猫写真の3次元マップ"><div class="toolbar"><button id="reset">視点を戻す</button><button id="random">ランダムに選ぶ</button><select id="variant" aria-label="特徴量の条件"><option value="color64">毛色のみ</option><option value="mixed_color50">毛色50%＋DINOv2</option><option value="mixed_color75" selected>毛色75%＋DINOv2</option></select></div><canvas id="plot" aria-label="ドラッグで回転できる3次元マップ"></canvas></section>\n<aside><div class="photo"><img id="photo" class="hidden" alt="選択した猫写真（猫であることは出典の記述に基づきます）"></div><p id="status" role="status">座標データを読み込み中…</p>\n<p id="clusterInfo" class="note"></p><div id="legend" class="note"></div><h2>出典・クレジット</h2><dl><dt>識別子</dt><dd id="pid">—</dd><dt>出典</dt><dd id="source">—</dd><dt>作者</dt><dd id="creator">—</dd><dt>ライセンス</dt><dd id="license">—</dd><dt>元ページ</dt><dd id="origin">—</dd><dt>画像処理</dt><dd>表示用に長辺960px以内へ縮小・JPEG変換（アニメーションは先頭フレーム）。元ファイルは変更しません。</dd></dl>\n<p class="note">ドラッグで回転、ホイール／ピンチで拡大縮小。点をクリック・タップして写真を表示します。写真上には文字を重ねません。</p>\n<p class="note">猫領域を自動抽出できた画像だけを比較しています。色分けは各条件のK-means 6群です。クラスタ番号・色は条件間で同じ意味を表しません。柄や品種の正解ラベルではありません。目視審査は行っていません。</p>\n<p class="note">選択した写真は条件を切り替えても維持されます。座標は比較用に向きを揃えていますが、軸や絶対位置に意味はありません。この版はローカル確認用です。作者が未記録の写真はその旨を表示します。公開用クレジットの完成を保証するものではありません。</p></aside></main>\n<script>\n\'use strict\';\nconst $=id=>document.getElementById(id),canvas=$(\'plot\'),ctx=canvas.getContext(\'2d\');\nlet points=[],order=[],shown=[],screen=[],page=0,chosen=-1,yaw=.5,pitch=-.25,zoom=1,queued=false,requestId=0,controller=null,photoURL=null;\nlet w=1,h=1;let variants={};const palette=[\'#5bc0eb\',\'#f6bd60\',\'#9bdeac\',\'#d8a0e5\',\'#f28482\',\'#c4d35d\'];\nconst pointers=new Map();let moved=false,startX=0,startY=0;\nfunction label(v){return v==null||v===\'\'?\'未記録\':typeof v===\'object\'?JSON.stringify(v):String(v)}\nfunction link(id,url,text){const e=$(id);e.replaceChildren();try{const u=new URL(url);if(![\'http:\',\'https:\'].includes(u.protocol))throw Error();const a=document.createElement(\'a\');a.href=u.href;a.textContent=text;a.target=\'_blank\';a.rel=\'noopener noreferrer\';e.append(a)}catch{e.textContent=text+\'（リンク未記録）\'}}\nfunction schedule(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;draw()})}}\nfunction draw(){\nctx.clearRect(0,0,w,h);const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),scale=Math.min(w,h)*.34*zoom;\nconst ids=chosen>=0&&!shown.includes(chosen)?shown.concat(chosen):shown;\nscreen=ids.map(i=>{const p=points[i],x=p.nx*cy+p.nz*sy,z=-p.nx*sy+p.nz*cy,y=p.ny*cp-z*sp,depth=p.ny*sp+z*cp,f=4/(4+depth);return {i,x:w/2+x*scale*f,y:h/2-y*scale*f,z:depth}}).sort((a,b)=>b.z-a.z);\nfor(const p of screen){ctx.fillStyle=palette[points[p.i].cluster];if(p.x<0||p.x>w||p.y<0||p.y>h)continue;ctx.beginPath();ctx.arc(p.x,p.y,3.3,0,Math.PI*2);ctx.fill()}\nconst p=screen.find(p=>p.i===chosen);if(p){ctx.strokeStyle=\'#ffcb77\';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.stroke()}\n}\nfunction group(){shown=points.map((_,i)=>i);$(\'count\').textContent=points.length.toLocaleString()+\' photos\';schedule()}\nfunction switchVariant(){\nconst v=variants[$(\'variant\').value];if(!v)return;\npoints.forEach((p,i)=>{[p.nx,p.ny,p.nz]=v.coords[i];p.cluster=v.labels[i]});\n$(\'legend\').replaceChildren();\nfor(let k=0;k<6;k++){const el=document.createElement(\'span\');el.style.color=palette[k];el.style.marginRight=\'12px\';el.textContent=\'● 群\'+(k+1)+\'：\'+v.labels.filter(l=>l===k).length+\'枚\';$(\'legend\').append(el)}\nupdateCluster();group()\n}\nfunction updateCluster(){$(\'clusterInfo\').textContent=chosen<0?\'点を選ぶと、所属する群を表示します。\':\'選択中：群\'+(points[chosen].cluster+1)+\' · \'+$(\'variant\').selectedOptions[0].textContent}\nfunction resize(){const r=canvas.getBoundingClientRect();w=r.width;h=r.height;const d=Math.min(devicePixelRatio||1,1.5);canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);ctx.setTransform(d,0,0,d,0,0);schedule()}\nasync function select(i){const p=points[i];if(!p)return;chosen=i;updateCluster();schedule();const token=++requestId;if(controller)controller.abort();controller=new AbortController();$(\'photo\').classList.add(\'hidden\');$(\'photo\').removeAttribute(\'src\');if(photoURL){URL.revokeObjectURL(photoURL);photoURL=null}\n$(\'status\').textContent=\'写真を読み込み中…\';$(\'pid\').textContent=p.id;$(\'source\').textContent=label(p.source);$(\'creator\').textContent=p.creator?label(p.creator):\'未記録（公開前に補完が必要）\';link(\'license\',p.license_url,label(p.license));link(\'origin\',p.source_page,\'元ページを開く\');\ntry{const r=await fetch(\'/image/\'+i,{signal:controller.signal});if(!r.ok)throw Error(\'画像の配信に失敗しました。ターミナルを確認してください。\');const blob=await r.blob();if(token!==requestId)return;const url=URL.createObjectURL(blob);photoURL=url;const img=$(\'photo\');img.onload=()=>{if(token===requestId){img.classList.remove(\'hidden\');$(\'status\').textContent=\'写真 \'+(i+1).toLocaleString()+\' / \'+points.length.toLocaleString()}};img.onerror=()=>{if(token===requestId)$(\'status\').textContent=\'画像を表示できませんでした。\'};img.src=url}catch(e){if(token===requestId&&e.name!==\'AbortError\')$(\'status\').textContent=e.message}}\ncanvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(pointers.size===1){moved=false;startX=e.clientX;startY=e.clientY}else moved=true};\ncanvas.onpointermove=e=>{const old=pointers.get(e.pointerId);if(!old)return;const other=[...pointers.entries()].find(([id])=>id!==e.pointerId);\nif(other){const q=other[1],before=Math.hypot(old.x-q.x,old.y-q.y),after=Math.hypot(e.clientX-q.x,e.clientY-q.y);if(before>0)zoom=Math.max(.2,Math.min(15,zoom*after/before));moved=true}else{yaw+=(e.clientX-old.x)*.006;pitch+=(e.clientY-old.y)*.006;if(Math.hypot(e.clientX-startX,e.clientY-startY)>5)moved=true}\npointers.set(e.pointerId,{x:e.clientX,y:e.clientY});schedule()};\ncanvas.onpointerup=e=>{pointers.delete(e.pointerId);if(!moved){const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=null,dist=144;for(const p of screen){const d=(p.x-x)**2+(p.y-y)**2;if(d<=dist){dist=d;best=p}}if(best)select(best.i)}};\ncanvas.onpointercancel=e=>{pointers.delete(e.pointerId);moved=true};\ncanvas.addEventListener(\'wheel\',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(15,zoom*Math.exp(-e.deltaY*.001)));schedule()},{passive:false});\n$(\'reset\').onclick=()=>{yaw=.5;pitch=-.25;zoom=1;schedule()};\n$(\'variant\').onchange=switchVariant;\n$(\'random\').onclick=()=>{if(points.length)select(Math.floor(Math.random()*points.length))};\nnew ResizeObserver(resize).observe(canvas);\nasync function start(){try{const r=await fetch(\'/data\');if(!r.ok)throw Error(\'比較データの取得に失敗\');const data=await r.json();points=data.points;variants=data.variants;switchVariant();$(\'status\').textContent=\'点を選び、特徴量の条件を切り替えて比較できます。\'}catch(e){$(\'status\').textContent=\'起動エラー: \'+e.message}}\nstart();\n</script></html>'


NAMES=("color64","mixed_color50","mixed_color75")
def load_data(project):
    source=project/"results/coat_pilot512"
    result=project/"results/coat_compare02"
    rows=json.loads((source/"accepted_points.json").read_text())
    ids=json.loads((result/"row_ids.json").read_text())
    if ids != [r["id"] for r in rows]:
        raise ValueError("Comparison row order does not match photo IDs")
    if len(set(ids))!=len(ids) or not rows:raise ValueError("Invalid IDs")
    for row in rows:
        if not re.fullmatch(r"[a-f0-9]{64}",row["sha256"]):raise ValueError("Invalid SHA256")
        if not (project/"data/web_previews_v1"/(row["sha256"]+".jpg")).is_file():
            raise FileNotFoundError("Missing cached preview: "+row["id"])
    variants={};reference=None
    for name in NAMES:
        y=np.load(result/name/"coordinates_seed42.npy",allow_pickle=False).astype(float)
        labels=np.load(result/name/"kmeans_k6.npy",allow_pickle=False)
        if y.shape!=(len(rows),3) or not np.isfinite(y).all():raise ValueError("Invalid coordinates")
        if labels.shape!=(len(rows),) or labels.dtype.kind not in "iu" or not np.isin(labels,np.arange(6)).all():
            raise ValueError("Invalid cluster labels")
        y-=y.mean(axis=0)
        radius=np.sqrt(np.mean(np.sum(y*y,axis=1)))
        if radius<1e-12:raise ValueError("Degenerate coordinates")
        y/=radius
        if reference is None:reference=y.copy()
        else:
            # Orthogonal alignment preserves within-map distances, up to uniform scaling.
            u,_,vt=np.linalg.svd(y.T@reference)
            y=y@(u@vt)
        variants[name]={"coords":y,"labels":labels.tolist()}
    extent=max(float(np.abs(v["coords"]).max()) for v in variants.values())
    for v in variants.values():v["coords"]=(v["coords"]/extent).tolist()
    keys=("id","source","creator","license","license_url","source_page")
    data={"points":[{k:r.get(k) for k in keys} for r in rows],"variants":variants}
    return rows,data

def make_handler(project,rows,data):
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
    @lru_cache(maxsize=64)
    def photo(i):
        return (project/"data/web_previews_v1"/(rows[i]["sha256"]+".jpg")).read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host","").split(":")[0].lower() not in ("localhost","127.0.0.1"):
                self.send_error(403);return
            route=urlsplit(self.path).path
            try:
                if route=="/":body,mime=HTML.encode(),"text/html; charset=utf-8"
                elif route=="/data":body,mime=payload,"application/json; charset=utf-8"
                elif re.fullmatch(r"/image/[0-9]+",route):
                    i=int(route.rsplit("/",1)[1])
                    if i>=len(rows):self.send_error(404);return
                    body,mime=photo(i),"image/jpeg"
                else:self.send_error(404);return
            except OSError:
                self.send_error(500,"Cached image unavailable");return
            self.send_response(200)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
    return Handler

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,default=Path.home()/"projects/FelisField")
    p.add_argument("--port",type=int,default=8766)
    args=p.parse_args();rows,data=load_data(args.project)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(args.project,rows,data))
    print(f"FelisField coat comparison: {len(rows)} images, 3 variants, K-means 6 groups.",flush=True)
    print(f"Open http://localhost:{args.port} — stop with Ctrl+C",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=="__main__":main()
