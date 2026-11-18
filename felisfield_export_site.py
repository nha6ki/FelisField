"""Export the completed coat map and cached preview JPEGs as a static website."""
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from felisfield_coat_full import MAP_HTML, credit_points


def build(project, folder):
    source = project / "results/coat_full50"
    rows = json.loads((source / "points.json").read_text())
    ids = [p["id"] for p in rows]
    if len(set(ids)) != 17303 or len(rows) != 17303 or ids != json.loads((source / "row_ids.json").read_text()):
        raise ValueError("Unexpected release cohort or row order")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source image hash in release")
    audit = json.loads((source / "credit_audit.json").read_text())
    if any(r["issues"] or r["source_notices"] for r in audit["photos"]):
        raise ValueError("Resolve outstanding credit issues or include additional notices first")
    points = credit_points(source, rows)
    coords = np.array([[p[k] for k in "xyz"] for p in rows], dtype=float)
    coords -= coords.mean(axis=0)
    coords /= np.abs(coords).max()
    labels = [p["cluster"] for p in rows]
    if not np.isfinite(coords).all() or any(label not in range(6) for label in labels):
        raise ValueError("Invalid map coordinates or labels")
    html = MAP_HTML
    for old, new in [("fetch('/data')", "fetch('./map.json')"),
                     ("'/image/'+i", "'./photos/'+i+'.jpg'"),
                     ("表示用に縮小・JPEG変換しています。", "表示用に縮小・JPEG変換しています。表示用の写真にも各写真のライセンスを適用しています。")]:
        if html.count(old) != 1:
            raise ValueError("Unexpected viewer version: " + old)
        html = html.replace(old, new)
    (folder / "index.html").write_text(html, encoding="utf-8")
    data = {"points": points, "variants": {"mixed_color50": {"coords": coords.tolist(), "labels": labels}}}
    (folder / "map.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    photos = folder / "photos"
    photos.mkdir()
    for i, row in enumerate(rows):
        sha = row["sha256"]
        if not re.fullmatch("[a-f0-9]{64}", sha):
            raise ValueError("Invalid photo hash")
        preview = project / "data/web_previews_v1" / (sha + ".jpg")
        with preview.open("rb") as f:
            if f.read(3) != b"\xff\xd8\xff":
                raise ValueError("Invalid JPEG: " + str(preview))
        shutil.copyfile(preview, photos / f"{i}.jpg")
    sizes = [p.stat().st_size for p in folder.rglob("*") if p.is_file()]
    if len(sizes) > 20000 or max(sizes) > 25 * 1024**2:
        raise ValueError("Cloudflare Pages asset limit exceeded")
    return len(sizes), sum(sizes)


def main():
    project = Path(__file__).resolve().parent
    # Keep release files outside the Git checkout and publish only completed builds.
    parent = project.parent / "FelisField-releases"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        folder = Path(tmp) / "site"
        folder.mkdir()
        count, size = build(project, folder)
        dest = parent / "site"
        if dest.exists():
            if not dest.is_dir() or any(p.is_symlink() for p in dest.rglob("*")) or dest.is_symlink():
                raise ValueError("Unexpected destination: " + str(dest))
            from filecmp import cmp
            files = sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())
            existing = sorted(p.relative_to(dest) for p in dest.rglob("*") if p.is_file())
            if files != existing or any(not cmp(folder/p, dest/p, shallow=False) for p in files):
                raise SystemExit("Existing release differs. Rename the existing site directory and rerun: " + str(dest))
        else:
            folder.rename(dest)
        print(f"PASS: {count:,} files, {size / 1024**2:.1f} MiB\n{dest}")


if __name__ == "__main__":
    main()
