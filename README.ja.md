# DuoMatching for ComfyUI（日本語概要）

公式の **DuoMatching** モデル（[JohnZhan2023/DuoMatching](https://github.com/JohnZhan2023/DuoMatching)、
論文 [arXiv:2610.03543](https://arxiv.org/abs/2610.03543)）で、ComfyUI 上でテキストから動画を生成するノードパックです。
DuoMatching は Wan2.1-T2V-14B から蒸留されたストリーミング型の Wan2.1-T2V-1.3B で、動画を 1 フレームずつ生成します（最初の潜在
フレームは 4 ステップ、以降は 2 ステップ）。処理はすべて **ComfyUI のプロセス内**で行い、Windows でも flash-attn なしで動きます。
ノードが何かをダウンロードすることはありません。

> **非公式**のインテグレーションです。DuoMatching の著者、Wan チーム（Alibaba）、Comfy Org とは関係がなく、承認も受けていません。
> 詳細は英語版 [README.md](README.md) と [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。

## ノード

| ノード | 内容 |
| --- | --- |
| **DuoMatching Loader** | `models/duomatching` のチェックポイント、テキストエンコーダーの置き場所、常駐の有無を選ぶ |
| **DuoMatching Text to Video** | プロンプト・シード・長さ（潜在フレーム数）・幅・高さ → `IMAGE`（フレーム列）、`fps`（16）、JSON レポート |
| **DuoMatching Unload** | 常駐モデルを解放（GPU と RAM） |

フレームは ComfyUI 標準の **Create Video**（fps はノードの出力）と **Save Video** に接続します。同梱のワークフローもこの構成です。

## 確認した範囲

Windows 11、RTX 5090（32 GB）、ComfyUI v0.38.0：

- **公式コードとビット単位で一致**：上流が固定しているライブラリ（Python 3.10、diffusers 0.38.0、transformers 4.57.1、
  numpy 1.24.4。torch は RTX 5090 に対応しない 2.6.0 の代わりに 2.8.0）の環境で、公式 `inference.py --use_ema --shared_seed` と
  同じ動画（float フレームの SHA-256）を 3 件中 3 件で再現しました（832 x 480・81 フレームの 2 プロンプト、640 x 384・129 フレーム）。
  flash-attn は Windows 版がないため、公式側の実行では同じ計算をする PyTorch SDPA に置き換えています（パックと同じ変更）。
- **実際の ComfyUI キュー**：torch 2.14.1 / diffusers 0.41.0 / transformers 5.20.1 の ComfyUI 環境で、キュー実行はパックの
  プロセス内実行とバイト単位で同じフレームになりました（3 件中 3 件）。もう一方の環境の公式実行と比べると、場面は同じで細部が
  異なります（最初のフレームは PSNR 34〜42 dB、最後のフレームは 17〜26 dB）。bfloat16 の自己回帰モデルでは、ライブラリ間の小さな
  数値差がフレームを追うごとに積み重なるためです。コードの差ではなく環境の差で、同じ環境なら毎回同じ動画になります。
- **実モデルでの実行**：5 秒・8 秒・10 秒、2 種類のサイズ、中断（とその後の実行）、keep_loaded オフ、Unload、16 の倍数でない
  サイズ（生成前にエラー）。
- **クリーンインストール**：新しい ComfyUI v0.38.0、新しい Python 3.12 環境、このリポジトリの `git clone`、`pip install -r
  requirements.txt`、`tools/download_model.py` でのモデル確認（SHA-256）。キューの出力は開発環境と同じフレームになり、同梱
  ワークフローは 2 本ともそのまま動いて、開発環境とバイト単位で同じ MP4 を保存しました。
- 依存関係の注意：diffusers 0.41.0 が `huggingface-hub<2.0` を要求するため、新しい ComfyUI 0.38.0 環境に入れると
  `huggingface-hub` が 2.2.0 から 1.33.0 に下がります（ComfyUI 自体は問題なく動作）。
- **CI**（GitHub Actions、Ubuntu と Windows、Python 3.12 / torch 2.14.1、および上流と同じ diffusers / transformers / tokenizers の
  Python 3.10 / torch 2.8.0）：公式のファイル構成の小さなランダム生成器による CPU テスト、上流の固定コミットとの比較、公式手順との
  比較、CPU での実際の ComfyUI キュー。CI ではモデル（約 23 GB）をダウンロードしません。

**未確認**：Linux / macOS の GPU、RTX 5090（32 GB）以外の GPU（特に AMD / Apple）、他の ComfyUI バージョン、flash-attn を入れた
環境、上流が指定する torch 2.6.0 そのもの、論文ベンチマークでの画質評価（公式出力の再現までを確認しています）。

## インストール

1. V3 ノード API のある ComfyUI（**v0.38.0** で確認）、Python 3.10 以上、NVIDIA GPU（ピーク確保量は 832 x 480・5 秒で 12.7 GiB、
   10 秒で 18.3 GiB。試したのは 32 GB のカードのみ）。空き VRAM が 40 GB 未満なら 11 GB のテキストエンコーダーは RAM に置かれます。
2. このリポジトリを `ComfyUI/custom_nodes/` に clone し、ComfyUI の Python に依存パッケージを入れます：

   ```
   git clone https://github.com/hiroki-abe-58/ComfyUI-DuoMatching custom_nodes/ComfyUI-DuoMatching
   python -m pip install -r custom_nodes/ComfyUI-DuoMatching/requirements.txt
   ```

3. モデルをダウンロードします（約 23 GB）。確認済みのリビジョンに固定し、全ファイルの SHA-256 を検証します：

   ```
   python custom_nodes/ComfyUI-DuoMatching/tools/download_model.py --models-dir models/duomatching
   ```

   `models/duomatching/Wan2.1-T2V-1.3B/`（フォルダー名は固定）と `duomatching.pt` ができます。別の場所は
   `extra_model_paths.yaml` のキー `duomatching` で指定できます。

4. ComfyUI を再起動し、`workflows/gui/t2v_basic.json`（832 x 480、5 秒）または `workflows/gui/t2v_640x384_8s.json`
   （640 x 384、8 秒）を開きます。

flash-attn は不要です。

## 使い方の要点

- **長さ**：`latent_frames` は Wan の潜在フレーム数で、動画フレーム数 = 4 x latent_frames - 3 です。21 で 81 フレーム（16 fps で
  約 5 秒、学習時の長さで公式の既定値）。アテンションキャッシュは全フレームを保持するため、長くするとメモリと時間が増えます。
- **サイズ**：幅と高さは 16 の倍数。学習は 832 x 480 です（640 x 384 も公式スクリプトとの一致を確認）。
- **シード**：同じモデル・プロンプト・シード・設定・環境なら同じ動画になります（確認済み）。ノードは動画ごとにシードを設定します。
  これは公式スクリプトの `--seed N --shared_seed` に当たります（`--shared_seed` なしでは実行全体で 1 回だけ設定されます）。
- **text_encoder**：`auto` は公式の規則どおり、空き VRAM が 40 GB 未満なら 11 GB の umT5 エンコーダーを RAM に置き、使うときだけ
  GPU へ転送します（`dynamic_swap`）。
- **中断**は生成器の呼び出しの合間で止まります（5 秒の動画で 65 回）。モデルは常駐したままで、次のジョブは普通に動きます。

## ライセンス

- このパック独自のコード（`duomatching_comfy/vendor` 以外）：**MIT**（[LICENSE](LICENSE)）。
- `duomatching_comfy/vendor/dm`：上流 DuoMatching のコード、**Apache-2.0**。5 ファイルに記録済みの変更があります
  （[vendor/UPSTREAM.md](duomatching_comfy/vendor/UPSTREAM.md)）。上流の `LICENSE` と `NOTICE` をそのまま同梱しています。
- **由来についての注意**：上流は Causal Forcing と Self Forcing（いずれも Apache-2.0）から派生したと書いています。Self Forcing は
  CausVid を土台にしたと書いており、CausVid のリポジトリは **CC BY-NC-SA 4.0** で公開されています。下流のプロジェクトはこのコードを
  Apache-2.0 で公開しており、このパックもそのライセンスで再配布しますが、その経緯を確定させるものではありません。商用利用の前には
  ご自身で確認してください（[NOTICE](NOTICE)）。
- モデル（同梱しません。`tools/download_model.py` を実行したときだけ取得）：`JohnZhan/DuoMatching`（リポジトリの `LICENSE`
  ファイルは Apache License 2.0、モデルカードにライセンス欄はなし）と `Wan-AI/Wan2.1-T2V-1.3B`（モデルカード上は Apache-2.0）。
