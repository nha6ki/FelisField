# FelisField

6つの出典から採用した17,303枚の猫写真を、見た目の近さを手がかりに3D空間で探索するアプリです。点にカーソルを合わせるかタップすると、その写真が開きます。作者表記やライセンスは「出典・クレジット」で確認できます。

## 地図の作り方

29,046枚にMask2Formerを適用し、検出スコアと領域サイズの条件を満たした写真を採用しました。複数の猫が検出された場合は、最も広い領域を使います。猫領域の色分布（64次元）と、背景を一定色にした切り抜きのDINOv2特徴（PCAで50次元）に等しい重みを与え、UMAPで3次元に配置します。点の表示色はKMeansによる6群を示します。写真の個体識別や手作業による輪郭確認は行っていません。

## 既存のデータを使って動かす

以下はWSLのプロジェクトディレクトリで実行します。ローカル表示と書き出しにはPython 3とNumPy、下記の生成済みデータを使います。解析時の環境はPython 3.12、NumPy 2.3.5、scikit-learn 1.8.0、umap-learn 0.5.9.post2でした。

| ローカルファイル | 用途 |
| --- | --- |
| results/coat_full50/points.json | 写真ID、原画像ハッシュ、座標、クラスタ、出典 |
| results/coat_full50/row_ids.json | 座標と写真の順序の検証 |
| results/coat_full50/summary.json | 完了した解析の概要 |
| results/coat_full50/credit_audit.json | 写真別の帰属表示とライセンス情報 |
| data/web_previews_v1/ | 原画像のSHA-256をファイル名にした表示用JPEG |

```bash
./.venv/bin/python felisfield_coat_full.py --serve --project "$PWD"
```

ブラウザで <http://localhost:8767> を開きます。静的サイトは次のコマンドで書き出せます。

```bash
./.venv/bin/python felisfield_export_site.py
```

出力先はプロジェクトと同じ親ディレクトリの `FelisField-releases/site/` です。配信用ファイルは17,303枚のJPEGと、index.html、map.jsonの合計17,305個です。

## コードの構成

- `felisfield_coat_full.py` は全件解析と現行のWeb画面、`felisfield_coat_viewer.py` はローカル配信処理と比較用ビューアーを含みます。
- `felisfield_local.py` は写真プレビューの生成、`felisfield_coat_pilot.py` と `felisfield_coat_compare.py` は領域抽出・予備比較を担当します。
- `felisfield_coat_eval.py` と `felisfield_pattern_eval.py` は近傍評価、`felisfield_credit_audit.py` は出典台帳の照合に使います。

このリポジトリの対象はソースコードです。写真・原台帳・特徴量・学習済み重みは別途必要です。各写真の利用条件はそれぞれの出典に従います。モデルの重みも配布元の条件に従います。
