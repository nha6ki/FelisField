# FelisField

**English** · [日本語](README.ja.md)

FelisField is an interactive 3D map of 17,303 cat photographs from six sources. Each point represents one photograph. Hover over a point or tap it to see the photo, then open **出典・クレジット** (Sources and credits) for its attribution and license.

## How the map works

We ran Mask2Former on 29,046 photographs and kept those that met the detection score and minimum region size thresholds. When a photo contained multiple detected cats, we used the largest predicted cat region. The displayed preview still shows the whole photograph.

Two features contribute equally to the map: a 64-dimensional color histogram of the cat region and a DINOv2 embedding of the cropped cat with the background set to a uniform color. We reduced the embedding to 50 dimensions using PCA, combined the two features, and used UMAP to place the photos in 3D. Point colors indicate six K-means clusters. The pipeline does not identify individual cats, and the predicted masks have not been manually reviewed.

## Run locally with prepared data

Run the commands below from the project directory in WSL. The local viewer and static export use Python 3, NumPy, and the following prepared files. The analysis used Python 3.12, NumPy 2.3.5, scikit-learn 1.8.0, and umap-learn 0.5.9.post2.

| Local path | Contents |
| --- | --- |
| `results/coat_full50/points.json` | Photo IDs, original image hashes, coordinates, cluster assignments, and sources |
| `results/coat_full50/row_ids.json` | Photo IDs in coordinate row order |
| `results/coat_full50/summary.json` | Summary of the completed analysis |
| `results/coat_full50/credit_audit.json` | Attribution and license information for each photo |
| `data/web_previews_v1/` | JPEG previews named by the SHA-256 hash of the original image |

Start the local viewer, then open <http://localhost:8767>.

```bash
./.venv/bin/python felisfield_coat_full.py --serve --project "$PWD"
```

To export the static site, run:

```bash
./.venv/bin/python felisfield_export_site.py
```

The exporter writes to `FelisField-releases/site/` alongside the project directory. The site contains 17,305 files: 17,303 JPEG previews, `index.html`, and `map.json`.

## Code layout

- `felisfield_coat_full.py` runs the full analysis and defines the map interface. `felisfield_coat_viewer.py` provides the local HTTP handler and a viewer for comparing configurations.
- `felisfield_local.py` generates photo previews. `felisfield_coat_pilot.py` extracts cat regions and features, and `felisfield_coat_compare.py` compares pilot configurations.
- `felisfield_coat_eval.py` and `felisfield_pattern_eval.py` evaluate map neighborhoods. `felisfield_credit_audit.py` matches photos to their source records.

This repository contains the source code. Photographs, original source records, extracted features, and model weights must be obtained separately. Each photograph and set of model weights is subject to its provider’s terms.
