#!/usr/bin/env python3
"""FelisField local viewer. No API calls; original files unchanged; resized display previews."""
import argparse
import hashlib
import json
import math
import io
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from PIL import Image, ImageOps
import re
import zipfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HTML = r'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FelisField — Local explorer</title>
<style>
:root{font-family:system-ui,sans-serif;color:#e5edf4;background:#101923;color-scheme:dark}*{box-sizing:border-box}body{margin:0}header{padding:18px 24px;border-bottom:1px solid #30404e;display:flex;align-items:center;gap:20px}h1{font-size:24px;margin:0;letter-spacing:.02em}header span{font-size:13px;color:#a7b8c7}main{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(320px,1fr);height:calc(100dvh - 70px)}.map{position:relative;min-height:400px}#plot{width:100%;height:100%;display:block;touch-action:none}.toolbar{flex-wrap:wrap;right:12px}select{background:#24394a;color:white;border:1px solid #486074;border-radius:7px}.toolbar{position:absolute;top:12px;left:16px;z-index:2;display:flex;gap:8px}button{background:#24394a;color:#edf5fb;border:1px solid #486074;border-radius:7px;padding:9px 13px;cursor:pointer}button:hover{background:#36546a}aside{overflow:auto;padding:20px;border-left:1px solid #30404e}.photo{height:44vh;min-height:240px;background:#090f16;border-radius:12px;display:flex;align-items:center;justify-content:center;overflow:hidden}.photo img{display:block;width:100%;height:100%;object-fit:contain}#status{min-height:24px;font-size:14px;color:#b9cbd9;margin:14px 0}h2{font-size:15px;margin-top:24px}dl{display:grid;grid-template-columns:85px 1fr;font-size:13px;line-height:1.7;gap:8px;margin:0}dt{color:#9cb1c1}dd{margin:0;overflow-wrap:anywhere}a{color:#8ed6dc}p.note{font-size:12px;line-height:1.8;color:#9cb1c1}#count{color:#85d9d1}.hidden{display:none!important}@media(max-width:760px){header{padding:14px;gap:12px}header span{font-size:11px}main{display:block;height:auto}.map{height:54dvh;min-height:340px}aside{border-left:0;border-top:1px solid #30404e;padding:16px}.photo{height:40dvh}}
</style>
<header><h1>FelisField</h1><span>猫写真の3次元探索 · 軽量版 v3</span><span id="count"></span></header>
<main><section class="map" aria-label="猫写真の3次元マップ"><div class="toolbar"><button id="reset">視点を戻す</button><button id="random">ランダムに選ぶ</button><select id="limit" aria-label="表示点数"><option value="1000">1,000点</option><option value="3000">3,000点</option><option value="10000">10,000点</option><option value="999999">全点</option></select><button id="next">次の写真群</button></div><canvas id="plot" aria-label="ドラッグで回転できる3次元マップ"></canvas></section>
<aside><div class="photo"><img id="photo" class="hidden" alt="選択した猫写真（猫であることは出典の記述に基づきます）"></div><p id="status" role="status">座標データを読み込み中…</p>
<h2>出典・クレジット</h2><dl><dt>識別子</dt><dd id="pid">—</dd><dt>出典</dt><dd id="source">—</dd><dt>作者</dt><dd id="creator">—</dd><dt>ライセンス</dt><dd id="license">—</dd><dt>元ページ</dt><dd id="origin">—</dd><dt>画像処理</dt><dd>表示用に長辺960px以内へ縮小・JPEG変換（アニメーションは先頭フレーム）。元ファイルは変更しません。</dd></dl>
<p class="note">ドラッグで回転、ホイール／ピンチで拡大縮小。点をクリック・タップして写真を表示します。写真上には文字を重ねません。</p>
<p class="note">近さはDINOv2の特徴量をUMAPで3次元化したものです。距離は近似であり、各軸に決まった意味はありません。猫であることは出典の記述に基づき、目視確認・猫検出は行っていません。</p>
<p class="note">この版はローカル確認用です。作者が未記録の写真はその旨を表示します。公開用クレジットの完成を保証するものではありません。</p></aside></main>
<script>
'use strict';
const $=id=>document.getElementById(id),canvas=$('plot'),ctx=canvas.getContext('2d');
let points=[],order=[],shown=[],screen=[],page=0,chosen=-1,yaw=.5,pitch=-.25,zoom=1,queued=false,requestId=0,controller=null,photoURL=null;
let w=1,h=1;
const pointers=new Map();let moved=false,startX=0,startY=0;
function label(v){return v==null||v===''?'未記録':typeof v==='object'?JSON.stringify(v):String(v)}
function link(id,url,text){const e=$(id);e.replaceChildren();try{const u=new URL(url);if(!['http:','https:'].includes(u.protocol))throw Error();const a=document.createElement('a');a.href=u.href;a.textContent=text;a.target='_blank';a.rel='noopener noreferrer';e.append(a)}catch{e.textContent=text+'（リンク未記録）'}}
function schedule(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;draw()})}}
function draw(){
ctx.clearRect(0,0,w,h);const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),scale=Math.min(w,h)*.34*zoom;
const ids=chosen>=0&&!shown.includes(chosen)?shown.concat(chosen):shown;
screen=ids.map(i=>{const p=points[i],x=p.nx*cy+p.nz*sy,z=-p.nx*sy+p.nz*cy,y=p.ny*cp-z*sp,depth=p.ny*sp+z*cp,f=4/(4+depth);return {i,x:w/2+x*scale*f,y:h/2-y*scale*f,z:depth}}).sort((a,b)=>b.z-a.z);
ctx.fillStyle='#76cbc8';for(const p of screen){if(p.x<0||p.x>w||p.y<0||p.y>h)continue;ctx.fillRect(p.x-1.5,p.y-1.5,3,3)}
const p=screen.find(p=>p.i===chosen);if(p){ctx.strokeStyle='#ffcb77';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.stroke()}
}
function group(){const n=Math.min(Number($('limit').value),order.length),pages=Math.ceil(order.length/n);page%=pages;shown=order.slice(page*n,(page+1)*n);$('count').textContent=shown.length.toLocaleString()+' / '+points.length.toLocaleString()+' · '+(page+1)+'/'+pages;schedule()}
function resize(){const r=canvas.getBoundingClientRect();w=r.width;h=r.height;const d=Math.min(devicePixelRatio||1,1.5);canvas.width=Math.round(w*d);canvas.height=Math.round(h*d);ctx.setTransform(d,0,0,d,0,0);schedule()}
async function select(i){const p=points[i];if(!p)return;chosen=i;schedule();const token=++requestId;if(controller)controller.abort();controller=new AbortController();$('photo').classList.add('hidden');$('photo').removeAttribute('src');if(photoURL){URL.revokeObjectURL(photoURL);photoURL=null}
$('status').textContent='写真を読み込み中…';$('pid').textContent=p.id;$('source').textContent=label(p.source);$('creator').textContent=p.creator?label(p.creator):'未記録（公開前に補完が必要）';link('license',p.license_url,label(p.license));link('origin',p.source_page,'元ページを開く');
try{const r=await fetch('/image/'+i,{signal:controller.signal});if(!r.ok)throw Error('画像の配信に失敗しました。ターミナルを確認してください。');const blob=await r.blob();if(token!==requestId)return;const url=URL.createObjectURL(blob);photoURL=url;const img=$('photo');img.onload=()=>{if(token===requestId){img.classList.remove('hidden');$('status').textContent='写真 '+(i+1).toLocaleString()+' / '+points.length.toLocaleString()}};img.onerror=()=>{if(token===requestId)$('status').textContent='画像を表示できませんでした。'};img.src=url}catch(e){if(token===requestId&&e.name!=='AbortError')$('status').textContent=e.message}}
canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(pointers.size===1){moved=false;startX=e.clientX;startY=e.clientY}else moved=true};
canvas.onpointermove=e=>{const old=pointers.get(e.pointerId);if(!old)return;const other=[...pointers.entries()].find(([id])=>id!==e.pointerId);
if(other){const q=other[1],before=Math.hypot(old.x-q.x,old.y-q.y),after=Math.hypot(e.clientX-q.x,e.clientY-q.y);if(before>0)zoom=Math.max(.2,Math.min(15,zoom*after/before));moved=true}else{yaw+=(e.clientX-old.x)*.006;pitch+=(e.clientY-old.y)*.006;if(Math.hypot(e.clientX-startX,e.clientY-startY)>5)moved=true}
pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});schedule()};
canvas.onpointerup=e=>{pointers.delete(e.pointerId);if(!moved){const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=null,dist=144;for(const p of screen){const d=(p.x-x)**2+(p.y-y)**2;if(d<=dist){dist=d;best=p}}if(best)select(best.i)}};
canvas.onpointercancel=e=>{pointers.delete(e.pointerId);moved=true};
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(15,zoom*Math.exp(-e.deltaY*.001)));schedule()},{passive:false});
$('reset').onclick=()=>{yaw=.5;pitch=-.25;zoom=1;schedule()};
$('limit').onchange=()=>{page=0;group()};$('next').onclick=()=>{page++;group()};
$('random').onclick=()=>{if(points.length)select(Math.floor(Math.random()*points.length))};
new ResizeObserver(resize).observe(canvas);
async function start(){try{const r=await fetch('/points');if(!r.ok)throw Error('座標データの取得に失敗');points=await r.json();const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const p of points)[p.x,p.y,p.z].forEach((v,k)=>{lo[k]=Math.min(lo[k],v);hi[k]=Math.max(hi[k],v)});const center=lo.map((v,k)=>(v+hi[k])/2),span=Math.max(...hi.map((v,k)=>v-lo[k]))||1;points.forEach(p=>{p.nx=(p.x-center[0])*2/span;p.ny=(p.y-center[1])*2/span;p.nz=(p.z-center[2])*2/span});order=points.map((_,i)=>i);let seed=42;for(let i=order.length-1;i>0;i--){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const j=seed%(i+1);[order[i],order[j]]=[order[j],order[i]]}group();$('status').textContent='点を選んでください。「次の写真群」で全写真を順に探索できます。'}catch(e){$('status').textContent='起動エラー: '+e.message}}
start();
</script></html>'''

def load_points(project):
    points = json.loads((project / 'results/full/points.json').read_text(encoding='utf-8'))
    if not isinstance(points, list) or not points:
        raise ValueError('points.json must be a nonempty list')
    packages = project / 'data/photo_assets01/packages'
    for p in points:
        if not all(math.isfinite(float(p[k])) for k in ('x', 'y', 'z')):
            raise ValueError('Non-finite coordinates')
        if not re.fullmatch(r'[a-f0-9]{64}', p['sha256']):
            raise ValueError('Invalid SHA256')
        if not re.fullmatch(r'cat_photo_assets01_part\d{3}\.zip', p['archive']):
            raise ValueError('Invalid archive name')
        if not (packages / p['archive']).is_file():
            raise FileNotFoundError(packages / p['archive'])
        member = p['archive_member']
        if not member.startswith('cat_photo_assets01/objects/') or '..' in Path(member).parts:
            raise ValueError('Invalid archive member')
    return points

def preview_path(project, point):
    return project / 'data/web_previews_v1' / (point['sha256'] + '.jpg')

def build_previews(project, points, workers=4):
    """Build once, read each ZIP once, and resume from atomically completed JPEGs."""
    folder = project / 'data/web_previews_v1'
    folder.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for p in points:
        groups[p['archive']].append(p)
    lock = threading.Lock()
    state = {'total': len(points), 'done': 0, 'created': 0, 'cached': 0, 'errors': []}
    start = time.monotonic()
    def group_job(item):
        archive, rows = item
        pending = [p for p in rows if not preview_path(project, p).is_file()]
        with lock:
            state['cached'] += len(rows) - len(pending)
            state['done'] += len(rows) - len(pending)
        if not pending:
            return
        try:
            with zipfile.ZipFile(project / 'data/photo_assets01/packages' / archive) as z:
                for p in pending:
                    try:
                        data = z.read(p['archive_member'])
                        if hashlib.sha256(data).hexdigest() != p['sha256']:
                            raise ValueError('SHA256 mismatch')
                        with Image.open(io.BytesIO(data)) as im:
                            im.seek(0)
                            # JPEG decoder downsampling reduces full-resolution decoding work.
                            im.draft('RGB', (960,960))
                            im = ImageOps.exif_transpose(im)
                            im.thumbnail((960,960), Image.Resampling.LANCZOS)
                            im = im.convert('RGB')
                            target = preview_path(project, p)
                            temporary = target.with_suffix('.partial')
                            im.save(temporary, format='JPEG', quality=82)
                            os.replace(temporary, target)
                        with lock:
                            state['created'] += 1
                    except Exception as e:
                        with lock:
                            state['errors'].append({'id': p['id'], 'error': str(e)})
                    finally:
                        with lock:
                            state['done'] += 1
                            if state['done'] % 250 == 0 or state['done'] == state['total']:
                                print(json.dumps({k:v for k,v in state.items() if k!='errors'}), flush=True)
        except (OSError, zipfile.BadZipFile) as e:
            with lock:
                state['errors'].append({'archive': archive, 'error': str(e)})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(group_job, item) for item in groups.items()]):
            future.result()
    state['seconds'] = round(time.monotonic()-start, 2)
    state['ready'] = sum(preview_path(project,p).is_file() for p in points)
    state['bytes'] = sum(preview_path(project,p).stat().st_size for p in points if preview_path(project,p).is_file())
    (folder/'preparation.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if state['errors'] or state['ready'] != len(points):
        raise RuntimeError('Preview preparation incomplete. Run --prepare again; completed images are reused.')
    print('PASS: All display previews ready. No ZIP decoding is needed when selecting a photo.', flush=True)

def make_handler(project, points):
    public_keys = ('id','x','y','z','source','creator','license','license_url','source_page')
    payload = json.dumps([{k:p.get(k) for k in public_keys} for p in points], ensure_ascii=False).encode()

    @lru_cache(maxsize=32)
    def get_image(i):
        return preview_path(project, points[i]).read_bytes(), 'image/jpeg'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Loopback binding plus Host validation prevents DNS rebinding access.
            host = self.headers.get('Host','').split(':')[0].lower()
            if host not in ('127.0.0.1','localhost'):
                self.send_error(403); return
            route = urlsplit(self.path).path
            try:
                if route == '/': data, mime = HTML.encode(), 'text/html; charset=utf-8'
                elif route == '/points': data, mime = payload, 'application/json; charset=utf-8'
                elif re.fullmatch(r'/image/\d+', route):
                    i = int(route.rsplit('/',1)[1])
                    if not 0 <= i < len(points): self.send_error(404); return
                    data, mime = get_image(i)
                else: self.send_error(404); return
            except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
                print('Image error:', type(e).__name__, str(e), flush=True)
                self.send_error(500, 'Image unavailable; see terminal log'); return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass
    return Handler

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.home()/'projects/FelisField')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--prepare', action='store_true', help='Prepare persistent display previews and exit')
    parser.add_argument('--workers', type=int, choices=range(1,9), default=4)
    args=parser.parse_args()
    points=load_points(args.project)
    if args.prepare:
        build_previews(args.project, points, args.workers)
        return
    missing_previews=sum(not preview_path(args.project,p).is_file() for p in points)
    if missing_previews:
        raise SystemExit(f'{missing_previews} previews missing. Run this script with --prepare first.')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(args.project,points))
    missing=sum(not p.get('creator') for p in points if p.get('license')!='CC0-1.0')
    print(f'FelisField LIGHT v3: {len(points):,} points; non-CC0 missing creator: {missing:,}',flush=True)
    print(f'Open http://localhost:{args.port} in your Windows browser. Stop: Ctrl+C',flush=True)
    print('Local only. No OpenAI API, no external CDN, no image detection.',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
