"""Build a local credit ledger from the accepted photo records. No network access."""
import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CDLA-Permissive-1.0": "https://cdla.dev/permissive-1-0/",
}


def web(url):
    parts = urlsplit(url or "")
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def audit(p, r):
    problems = []
    license_id = p["license"]
    link = LICENSES.get(license_id)
    if not r:
        problems.append("missing_source_record")
    if not link:
        problems.append("unknown_license")
    if r and (r.get("source") != p["source"] or r.get("license") != license_id
              or r.get("source_page") != p.get("source_page")):
        problems.append("source_record_conflict")
    for raw in (p.get("license_url"), r.get("license_url")):
        if not raw or raw.replace("http://", "https://", 1).rstrip("/") != (link or "").rstrip("/"):
            problems.append("license_url_missing_or_conflicting")
    if not r.get("license_basis"):
        problems.append("missing_license_evidence")
    if r.get("hold_reasons") or r.get("source_accepted_for_acquisition") is False:
        problems.append("source_record_on_hold")
    photo_url = r.get("photo_page") if p["source"] == "inat" else p.get("source_page")
    if not web(photo_url):
        problems.append("missing_source_url")
    if p["source"] == "inat" and (not photo_url or urlsplit(photo_url).path.rstrip("/") != "/photos/"+str(r.get("source_identifier"))
                                   or p["id"] != "inat_"+str(r.get("source_identifier"))):
        problems.append("photo_identity_mismatch")
    credit = r.get("attribution") or r.get("creator") or p.get("creator")
    if license_id != "CC0-1.0" and not (credit or "").strip():
        problems.append("missing_credit")
    return dict(id=p["id"], sha256=p["sha256"], source=p["source"], license=license_id,
                license_url=link, source_page=p.get("source_page"), photo_page=photo_url,
                credit=credit, modifications="表示用に縮小・JPEG変換",
                required_actions=r.get("required_actions") or [],
                source_notices={k:v for k,v in r.items() if v and any(t in k for t in ("notice","disclaimer","copyright"))},
                existing_release_tasks=r.get("remaining_release_tasks") or [],
                license_basis=r.get("license_basis"), issues=sorted(set(problems)))


def main():
    root = Path.home()/"projects/FelisField"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=root/"results/coat_full50/points.json")
    ap.add_argument("--records", type=Path, default=root/"data/photo_assets01/accepted.jsonl")
    ap.add_argument("--out", type=Path, default=root/"results/coat_full50/credit_audit.json")
    args = ap.parse_args()
    points = json.loads(args.points.read_text())
    ids = {p["id"] for p in points}
    if len(ids) != len(points):
        raise ValueError("Duplicate map ID")
    records = {}
    with args.records.open() as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids:
                if r["id"] in records:
                    raise ValueError("Duplicate source ID: "+r["id"])
                records[r["id"]] = r
    ledger = [audit(p, records.get(p["id"], {})) for p in points]
    held = sum(bool(row["issues"]) for row in ledger)
    summary = dict(photos=len(points), matched=len(records), credit_fields_pass=len(points)-held,
                   needs_review=held, credit_restored=sum(not p.get("creator") and bool(row["credit"])
                   for p,row in zip(points,ledger)), issues=Counter(i for row in ledger for i in row["issues"]))
    result = dict(summary=summary, scope="Offline credit field checks only; delivery obligations and live source terms not verified.", photos=ledger)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    tmp.replace(args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
