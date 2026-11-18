"""Source-text-only exploratory assessment of coat patterns on the saved 3D map."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

# Match explicit terms for cats. Colour alone does not establish a pattern.
TERMS = {
    'tabby': (r'\btabb(?:y|ies)\b', r'キジトラ|茶トラ|サバトラ|縞猫|トラ猫'),
    'calico': (r'\bcalicos?\b', r'三毛猫|三毛の猫'),
    'tortoiseshell': (r'\btort(?:oiseshell|ie)s?\b', r'サビ猫|錆猫'),
    'tuxedo': (r'\btuxedo\s+(?:cat|kitten)s?\b', r'タキシード猫'),
}
PHOTO_LEVELS = {'individual_photo_description', 'individual_file_description',
                'individual_photo_title_or_description'}


def label_record(record):
    evidence = record.get('content_evidence') or {}
    if evidence.get('evidence_level') not in PHOTO_LEVELS:
        return None
    caption = ' '.join(filter(None, [record.get('description_evidence'),
                                     evidence.get('description_summary_ja')]))
    labels = [name for name, terms in TERMS.items()
              if any(re.search(term, caption, re.I) for term in terms)]
    if len(labels) == 1 and not re.search(r'\b(?:not|unlike|looks like)\b', caption, re.I):
        return labels[0]
    return None


def evaluate(items):
    # Descriptive within-source check among labelled images only.
    counts = Counter(label for _, label in items)
    eligible = [(p, label) for p, label in items if counts[label] >= 5]
    if len({label for _, label in eligible}) < 2:
        return {'status': 'insufficient_labels'}
    labels = np.array([label for _, label in eligible])
    coords = [[p[k] for k in ('x', 'y', 'z')] for p, _ in eligible]
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    near = np.argsort(d, axis=1, kind='stable')[:, :3]
    return {'status': 'exploratory', 'n': len(labels), 'k': 3,
            'by_pattern': {label: {'n': counts[label],
                            'near_match': float((labels[near[labels == label]] == label).mean()),
                            'random_expected': (counts[label]-1)/(len(labels)-1)}
                           for label in sorted(set(labels))}}


def main():
    root = Path.home() / 'projects/FelisField'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=Path, default=root/'results/coat_full50/points.json')
    parser.add_argument('--records', type=Path, default=root/'data/photo_assets01/accepted.jsonl')
    parser.add_argument('--output', type=Path, default=root/'results/coat_full50/pattern_evaluation.json')
    args = parser.parse_args()
    points = {p['id']: p for p in json.loads(args.points.read_text())}
    matched = set()
    groups = defaultdict(list)
    with args.records.open() as stream:
        for line in stream:
            r = json.loads(line)
            p = points.get(r['id'])
            if p is None:
                continue
            if p['id'] in matched or p['source'] != r['source']:
                raise ValueError(f"Ambiguous source match: {p['id']}")
            matched.add(p['id'])
            label = label_record(r)
            if label:
                source = 'commons' if p['source']=='commons_initial20' else p['source']
                groups[source].append((p, label))
    report = {'map_images':len(points), 'matched_records':len(matched),
              'labeled_images':sum(map(len,groups.values())),
              'by_source':{s:dict(Counter(label for _, label in items)) for s,items in groups.items()},
              'evaluation':{s:evaluate(items) for s,items in groups.items()},
              'labels':[{'id':p['id'],'pattern':label,'url':p['source_page']}
                        for items in groups.values() for p,label in items]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='labels'}, ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
