# PLaMo Translator

[PLaMo 3 Translate](https://huggingface.co/collections/pfnet/plamo-3-translate)（Preferred Networks）を手元の GPU で動かす、ローカル翻訳 GUI です。

- **テキスト翻訳**: 文章を貼り付けて翻訳します。翻訳元は自動判定にでき、翻訳先は16言語、文体は常体と敬体から選べます。
  - 画像やコードを含む Markdown（電子書籍の本文や AI とのチャットをコピーしたものなど）は、画像・コードを残したまま文章だけを訳します。
- **PDF・文書翻訳**: 章立てを保ったまま全文を訳し、**HTML（目次付き）と Markdown** で出力します。
  - **論文**: 章番号（`1` / `2.1` / `III.` / `A.` など）を章とみなします。著者・所属と参考文献は原文のまま残します。
  - **書籍・一般文書**（電子書籍の PDF、AI とのチャットの書き出しなど）: 文字の大きさから見出しを判定し、表の中の文章も訳します。欄外にアイコンの付いた囲み（Note・Tip など）は引用として出します。
  - 図と表はそのまま埋め込みます。HTML では図をクリックすると原寸で表示します。
  - 縦長のページに横向き（90度回転）で載っている表は、ページの向きを直してから読み取ります。
  - 全章が1つになった書籍の PDF は、PDF のしおり（目次情報）から章の一覧を出します。訳す章を選べ、章ごとにファイルを分けて出力できます。
  - PDF のほか Markdown（`.md`）・テキスト（`.txt`）も読み込めます。
  - 段落ごとに、折りたたんだ原文を併記します（既定でオン。外すこともできます）。
- **高速**: vLLM と FP8 版のモデルで、6 ページの論文を約 15 秒で訳します（RTX 4060 Laptop、2B の場合）。

翻訳はすべてローカルで行います。PDF や文章を外部に送ることはありません。

測定の詳細（速度・訳の質・メモリ、構成ごとの比較）は [BENCHMARK.md](BENCHMARK.md) にまとめてあります。

## 構成

```
ブラウザ ──> app.py（GUI・PDF処理, .venv） ──HTTP──> vLLM サーバー（serve.sh, .venv-vllm） ──> GPU
```

翻訳エンジンは次の 2 通りから選べます。

| | vLLM + FP8（**推奨**） | Transformers + 4bit（代替） |
|---|---|---|
| 速さ（6 ページの論文、2B） | 約 15 秒 | 約 90 秒 |
| 訳の質（BF16 版との近さ） | 高い | やや下がる |
| 必要なもの | NVIDIA ドライバが CUDA 12.9 に対応していること（下の注を参照）、仮想環境を 2 つ（約 12 GB） | 仮想環境 1 つ（約 3 GB） |
| VRAM | 起動時に確保（既定で約 5.6 GB） | 使う分だけ（約 2.5 GB） |

vLLM が動かない環境では、Transformers の手順を使ってください。アプリは、vLLM サーバーが起動していればそちらを、なければ Transformers を自動で使います。

> **ドライバについての注**: 動作確認したドライバ 566.24 の対応は CUDA 12.7 までですが、CUDA 12.x の互換の仕組みで CUDA 12.9 版の vLLM が動きました。一方、PyPI にある最新の vLLM（CUDA 13 版）は、このドライバでは動きませんでした。

## 動作確認した環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 + WSL2（Ubuntu 24.04） |
| GPU | NVIDIA GeForce RTX 4060 Laptop（VRAM 8 GB）、Windows 側ドライバ 566.24 |
| Python | 3.12 |

Linux（ネイティブ）でも同じ手順で動くはずですが、確認していません。

## リポジトリに含まれないもの

次のものは `.gitignore` でリポジトリから外しています。各自が下の手順で作成してください。

| パス | 中身 | 作り方 |
|---|---|---|
| `.venv/` | アプリの仮想環境 | [手順 2](#2-アプリの環境を作る) |
| `.venv-vllm/` | vLLM サーバーの仮想環境 | [手順 4](#4-vllm-の環境を作る) |
| `model-fp8/`, `model-0.8b-fp8/` | FP8 版のモデル（2B / 0.8B） | [手順 3](#3-モデルのサイズを選んでダウンロードする) |
| `model/`, `model-4bit/` | BF16 版のモデルと、その 4bit 量子化版（Transformers で動かす場合だけ） | [Transformers で動かす](#transformers-で動かすvllm-が使えない場合) |
| `model-lighton/`（推奨）, `.venv-ocr/` と `model-ocr/` | OCR（スキャンした PDF を読む場合だけ）のモデルと仮想環境 | [スキャンした PDF を読む](#スキャンした-pdf-を読むocr任意) |

モデルの重みは [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja) で公開されています。そのため、このリポジトリでは再配布せず、各自が Hugging Face から取得する形にしています。

## セットアップ（推奨: vLLM + FP8）

以下のコマンドは、すべて **WSL2 の Ubuntu のターミナル**で実行します。

### 1. WSL2 から GPU が見えるか確認する

```bash
nvidia-smi
```

GPU 名と、右上の `CUDA Version` を確認してください。

- WSL2 では **Windows 側の NVIDIA ドライバ**を使います。Ubuntu 側に Linux 用のドライバを入れる必要はありません（入れないでください）。
- `nvidia-smi: command not found` になる場合は、Windows 側の NVIDIA ドライバを更新し、PowerShell で `wsl --update` を実行してください。
- 詳しくは [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/) を参照してください。

### 2. アプリの環境を作る

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git build-essential

git clone <このリポジトリの URL> plamo-translator
cd plamo-translator

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

アプリ側には PyTorch は入りません（約 550 MB）。

> **ファイルの置き場所**: リポジトリは `/mnt/c/...`（Windows のドライブ）ではなく、`~/` の下など WSL2 側のファイルシステムに置いてください。`/mnt/c` 経由だとモデルの読み込みが大幅に遅くなります。

### 3. モデルのサイズを選んでダウンロードする

先に [ライセンス](https://plamo.preferredai.jp/info/plamo-community-license-ja) を確認してください。

| サイズ | FP8 版の容量 | 向いている用途 |
|---|---|---|
| **2B（推奨）** | 3.2 GB | 論文の翻訳。固有名詞や専門用語を正しく扱える |
| 0.8B | 1.2 GB | 速さを優先したいとき（2B の約 3 倍速い）、VRAM が少ない GPU |

0.8B は、ツール名などの固有名詞を普通の言葉として訳してしまうことがあります（例: `safe drive and async lib` →「安全なドライブおよび非同期ライブラリ」）。具体例は [BENCHMARK.md](BENCHMARK.md#訳の質) を参照してください。

```bash
# 2B（推奨）
hf download pfnet/plamo-3-translate-2b-FP8 --local-dir ./model-fp8

# 0.8B
hf download pfnet/plamo-3-translate-0.8b-FP8 --local-dir ./model-0.8b-fp8
```

両方ダウンロードしておくと、アプリの画面でモデルを切り替えられます（[モデルの切り替え](#モデルの切り替え)）。保存先のディレクトリ名は `model` で始まっていれば自由です。アプリが自動で見つけます。

### 4. vLLM の環境を作る

vLLM は、アプリと PyTorch・Transformers の版がぶつかるため、別の仮想環境に入れます。ダウンロード量は数 GB、展開後は約 12 GB になります。

```bash
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install --upgrade pip
.venv-vllm/bin/pip install -r requirements-vllm.txt
```

`requirements-vllm.txt` では、CUDA 12.9 版の vLLM 0.32.0 を GitHub のリリースから取得します。なぜこの版にしたかは、ファイルの先頭のコメントと [BENCHMARK.md](BENCHMARK.md#vllm-を動かすまでにつまずいたこと) に書いてあります。

### 5. 起動する

```bash
source .venv/bin/activate
python app.py
```

アプリが vLLM サーバー（`serve.sh`）を自動で起動します。起動後、Windows のブラウザで <http://localhost:7860> を開いてください（WSL2 から Windows への localhost 転送は自動で行われます）。

- 画面上部に「⏳ 起動中…」と表示されている間は、翻訳できません。「✅ … で翻訳できます」になったら使えます。初回は 1〜2 分、2 回目以降は 30 秒ほどかかります。
- アプリを Ctrl+C で終了すると、vLLM サーバーも止まって VRAM が解放されます。
- vLLM サーバーのログは `logs/vllm.log` に出ます。

`./serve.sh` を別のターミナルで先に起動しておくこともできます。その場合、アプリはそのサーバーを使います。ただし、画面からモデルは切り替えられません。

#### serve.sh の設定

`serve.sh` の設定は環境変数で変えられます。アプリから起動する場合も、`python app.py` の前に設定すれば反映されます（例: `GPU_UTIL=0.6 python app.py`）。

| 環境変数 | 既定値 | 内容 |
|---|---|---|
| `MODEL` | `./model-fp8` | 起動するモデル |
| `GPU_UTIL` | `0.7` | 起動時に確保する VRAM の割合。0.8B なら `0.5` でも足ります |
| `PORT` | `8000` | サーバーのポート |
| `MAX_MODEL_LEN` | `8192` | 1 リクエストの最大トークン数（入力と出力の合計） |

例: `GPU_UTIL=0.5 ./serve.sh`

## Transformers で動かす（vLLM が使えない場合）

アプリのプロセス内でモデルを読み込んで翻訳します。vLLM より遅くなりますが、仮想環境は 1 つで済みます。[手順 1・2](#1-wsl2-から-gpu-が見えるか確認する) を済ませてから進めてください。

```bash
source .venv/bin/activate
pip install -r requirements-transformers.txt    # PyTorch など（約 3 GB）

# モデルのサイズを選んでダウンロードする（BF16 版。FP8 版は Transformers では読み込めません）
hf download pfnet/plamo-3-translate-2b   --local-dir ./model   # 2B（推奨、5.4 GB）
hf download pfnet/plamo-3-translate-0.8b --local-dir ./model   # 0.8B（1.7 GB）

# 4bit に量子化する（任意だが推奨。VRAM と読み込み時間が約 1/3 になる）
python quantize.py               # ./model -> ./model-4bit

python app.py --backend transformers
```

- アプリは `./model-4bit` があればそれを、なければ `./model` を使います。
- 8bit にしたい場合は `python quantize.py --bits 8` を実行してください（保存先は `./model-8bit`）。
- 量子化したあと、`./model` は削除してもかまいません。

## スキャンした PDF を読む（OCR、任意）

スキャンした論文や本は、OCR で文字を読み取ってから翻訳できます。OCR のエンジンは 2 つあり、両方入っていれば LightOnOCR-3 を使います。

| エンジン | 大きさ・ライセンス | 動かし方 | 速さ（RTX 4060 Laptop） |
|---|---|---|---|
| **[LightOnOCR-3-0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B)**（推奨） | 0.9B、Apache-2.0 | 翻訳と同じ `.venv-vllm` の vLLM | 1 ページ約 2 秒（8 ページずつ並列） |
| [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 3B、MIT | 別の仮想環境 `.venv-ocr` の Transformers | 1 ページ約 40 秒 |

### LightOnOCR-3（推奨）

`.venv-vllm`（vLLM 0.32）があれば、モデルを取ってくるだけです（約 1.7 GB）。

```bash
.venv/bin/hf download lightonai/LightOnOCR-3-0.8B --local-dir ./model-lighton
```

- 日本語のスキャン本・英語の古い本（ScanSnap で読んだもの）で、見出し・段落・箇条書き・図の位置・キャプション・数式（`$\Gamma$` などの LaTeX）を正しく読めることを確かめました。
- OCR のときだけ、ポート 8001 に OCR 用の vLLM サーバーを起動し、終わったら止めます（起動に約 50 秒）。
- Transformers でも動きますが、Qwen3.5 系の線形アテンションの高速版（`flash-linear-attention`・`causal_conv1d`）が無いと 1 ページ 7 分かかったため、vLLM を使っています。

### Unlimited-OCR（LightOnOCR-3 が使えない場合）

アプリとは transformers の版が合わないため、仮想環境を分けます（約 7 GB）。

```bash
python3 -m venv .venv-ocr
.venv-ocr/bin/pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv-ocr/bin/pip install -r requirements-ocr.txt

# モデル（約 6.7 GB）。同じ重みが 2 つ置かれているので、使う方だけを取る
.venv/bin/hf download baidu/Unlimited-OCR --local-dir ./model-ocr \
  --exclude "model.safetensors" --exclude "wheel/*" --exclude "assets/*" --exclude "*.pdf"
```

vLLM でも動かそうとしましたが、8 GB の GPU には BF16 の重みが収まらず、FP8 にすると繰り返しの抑制が効かずに出力が止まらなくなったため、Transformers を使っています。

### OCR するページ

OCR のエンジンがあれば、PDF・文書翻訳タブで自動で OCR します（オプションはありません）。

- **文字を持たないページ**（画像だけのページ）は OCR します。文字のあるページは、これまでどおり PDF の文字を使います。1 つの PDF に両方が混ざっていてもかまいません。
- **スキャンの上に、すでに文字が載っているページ**（スキャナや Acrobat が OCR した PDF）は、LightOnOCR-3 があるときだけ、その文字の精度を確かめます。
  - 文字の多いページを 3 ページ OCR し、PDF の文字の単語のうち OCR の結果にもあるものの割合（一致率）を出します。
  - 一致率が 95% 未満なら、PDF の文字は精度が低いとみなし、そのページをすべて OCR し直します。95% 以上なら PDF の文字を使います。
  - 実測では、ScanSnap（Hard Real-Time Computing Systems）は約 89%、Canon の My Image Garden（日本語の本）は約 91% で、どちらも OCR し直しました（`guaranteemechanism` のような語のくっつきや「ヤ1様は」→「皆様は」の読み違いが直ります）。
- **OCR のあいだは、翻訳用の vLLM サーバーを止めます。** 8 GB の GPU では、OCR 用と翻訳用の 2 つの vLLM サーバーを同時に動かせません（翻訳側のメモリが足りず起動できませんでした）。OCR が終わると自動で起動し直します。
- 手動で起動した vLLM サーバー（`serve.sh`）や、Transformers で翻訳している場合は、GPU を空けられないので OCR は使えません（文字の載ったスキャンは、そのまま PDF の文字で訳します）。
- 時間の目安（LightOnOCR-3）: 文字の載った英語の本 5 ページで約 3 分、日本語の本 6 ページで約 2 分（OCR サーバーの起動、翻訳エンジンの停止・再起動、翻訳を含む）。ページが増えても、1 ページあたり約 2 秒しか増えません。
- OCR の結果は、見出し・段落・箇条書き・表・数式（LaTeX のまま、訳さない）・図（ページから切り出して埋め込む）に分けて、そのあとは通常の PDF と同じように処理します。ページの上端に毎ページ出る柱（章名など）は見出しにしません。
- OCR のログは `logs/ocr.log` に出ます。

## 使い方

### モデルの切り替え

画面上部の「翻訳モデル」でモデルを選び、「モデルを切り替え」を押します。

- 選択肢には、プロジェクト直下の `model…` ディレクトリのうち、vLLM で動かせるもの（FP8 版・BF16 版）が出ます。`quantize.py` で作った 4bit 版は Transformers 用なので出ません。
- 切り替えは vLLM サーバーの再起動になるため、30 秒〜2 分かかります。その間は「翻訳開始」を押せません。テキスト翻訳は、起動が終わるのを待ってから訳します。
- 切り替えたあとの翻訳速度は、最初からそのモデルで起動した場合と同じです。
- 起動できなかった場合（VRAM 不足など）は、元のモデルに戻します。

### テキスト翻訳タブ

1. 翻訳元・翻訳先・文体を選びます。
2. 原文を入力すると、そのまま訳します（翻訳ボタンはありません）。翻訳元・翻訳先・文体を変えたときも訳し直します。

- 訳している間にさらに入力した場合は、訳し終わったあとに最新の内容で1回だけ訳し直します。
- 原文と訳文の間の ⇄ を押すと、翻訳の向きと原文・訳文が入れ替わります。
- 翻訳元が「自動判定」のときは、訳すたびに原文の言語を判定し、「翻訳元（判定: 日本語）」のように表示します。⇄ を押すと、判定した言語を翻訳元として確定させてから入れ替えます（例: 自動判定・日本語 → 英語 で訳したあとに押すと、英語 → 日本語 になります）。
  - 判定には [lingua](https://github.com/pemistahl/lingua-py)（Apache-2.0）を使います。翻訳そのものは PLaMo の自動判定に任せ、判定結果は表示と入れ替えにだけ使います。
  - "Hello" のような短いラテン文字の文は判定しにくいため、迷ったときは英語とみなします。漢字だけの短い文（「東京」など）は中国語と判定されることがあります。

画像（`![](...)`）やコード（` ``` `）を含む Markdown を入れた場合は、次のように扱います。

- 画像・コード・表の記号はそのまま残し、文章（表の中の文章を含む）だけを訳します。base64 で埋め込まれた画像も翻訳モデルには送りません。
- 訳文欄では画像を *(図)* と表示します。画像入りの訳文は、下に出る Markdown / HTML ファイルに保存します。

### PDF・文書翻訳タブ

1. ファイルを選び（複数可）、出力形式（HTML / Markdown）を選んで「翻訳開始」を押します。
2. 結果のファイルをダウンロードします。複数のファイルを入れた場合は ZIP にもまとめます。

オプションは次のとおりです。

| オプション | 内容 |
|---|---|
| 文書の種類 | **自動判定**（冒頭に Abstract か「1 Introduction」があれば論文、なければ書籍・一般文書）／**論文**／**書籍・一般文書** |
| ページ範囲 | PDF の一部だけを訳します（例: `37-80, 100`）。空欄なら全ページです。指定すると、下の「訳す章」より優先します |
| 訳す章 | しおり付きの書籍の PDF を1つ選んだときだけ出ます。表紙・目次・索引・広告などは最初から外してあります |
| 章ごとにファイルを分ける | 章ごとに HTML / Markdown を出し、ZIP にもまとめます。外すと、選んだ章を1つのファイルにまとめます |
| 図の解像度 | 図を切り出す解像度です。既定は 200dpi です。細かい図が読みにくいときは 300dpi にしてください（ファイルは大きくなります） |
| 段落ごとに原文を併記 | 各段落の下に、折りたたんだ原文を付けます（既定でオン） |
| 図を埋め込む | 図を画像としてファイルに埋め込みます（200dpi で 1 ページあたり 100 KB 程度大きくなります） |
| 参考文献も翻訳する | 論文のとき、既定では参考文献は原文のまま残します |

#### 書籍を1冊まとめて訳す場合

全章が1つになった PDF でも、そのまま入れて訳せます。

- PDF にしおりがあれば、見出しの階層をしおりに合わせます。章タイトルは「Chapter 2: …」の形にそろえます。文字の大きさだけで判定すると、本1冊の中で章ごとに階層がずれるためです。
- 目次のページ（点線とページ番号の表）は訳しません。
- 目安: 827 ページの書籍で、本文の抽出に約 2 分かかります。翻訳は 76 ページ（2 章分）で約 40 秒でした（2B FP8）。1 冊全体では 10 分前後の見込みです。
- 図を 200dpi で埋め込むと、1 冊で 100 MB 程度になります。章ごとにファイルを分けておくと、ブラウザで開きやすくなります。
- しおりの無い PDF では章の一覧は出ません。文字の大きさから見出しを判定し、全体を1つのファイルに出力します。章ごとに分けたいときは、ページ範囲で指定してください。

### app.py の起動オプション

| オプション | 既定値 | 内容 |
|---|---|---|
| `--backend` | `auto` | `vllm` / `transformers` / `auto`（起動済みの vLLM サーバー → アプリが vLLM を起動 → Transformers の順に試す） |
| `--vllm-model PATH` | `./model-fp8` | アプリが vLLM を起動するときに最初に使うモデル。環境変数 `VLLM_MODEL` でも指定できます |
| `--vllm-url URL` | `http://localhost:8000/v1` | vLLM サーバーの URL。環境変数 `VLLM_URL` でも指定できます |
| `--model PATH` | `./model-4bit`（なければ `./model`） | Transformers で使うモデル。環境変数 `PLAMO_MODEL` でも指定できます |
| `--batch-size N` | `8` | Transformers で一度に訳す段落数。VRAM が足りなければ下げてください |
| `--port N` | `7860` | アプリのポート |

## ファイル構成

```
plamo-translator/
├── app.py                        # GUI（Gradio）
├── backends.py                   # 翻訳エンジン（vLLM サーバー / Transformers）
├── paper.py                      # PDF / Markdown → 章立ての抽出 → Markdown / HTML 出力（論文・書籍）
├── ocr.py                        # スキャンしたページの検出・OCR の呼び出し・OCR 結果の Markdown 化
├── ocr_worker.py                 # Unlimited-OCR の本体（.venv-ocr で別プロセスとして動かす）
├── serve.sh                      # vLLM サーバーの起動
├── quantize.py                   # Transformers 用モデルの 4bit / 8bit 量子化
├── requirements.txt              # アプリ（.venv）
├── requirements-vllm.txt         # vLLM サーバー（.venv-vllm）
├── requirements-transformers.txt # Transformers で動かす場合（.venv に追加）
├── requirements-ocr.txt          # OCR（.venv-ocr）
├── BENCHMARK.md                  # 測定結果
├── model-fp8/                    # (git 管理外) FP8 版のモデル
├── model/, model-4bit/           # (git 管理外) Transformers 用のモデル
├── model-lighton/                # (git 管理外) OCR のモデル（LightOnOCR-3）
└── model-ocr/                    # (git 管理外) OCR のモデル（Unlimited-OCR）
```

## 制限事項

- **スキャンした PDF は、OCR を入れたときだけ翻訳できます。** 読み違いもあります（LightOnOCR-3 で「メモリ」→「メモり」、Unlimited-OCR でぼやけたスキャンの `omitted` → `committed` など）。
- **図の中の文字は訳しません。** 図は画像として埋め込み、図の中から取り出した文字は捨てます（「図を埋め込む」をオフにすると、その文字も出力に残りません）。
- **見出しが抜けることがあります。** 章立ては PDF の文字の大きさから判定しています（[pymupdf4llm](https://github.com/pymupdf/pymupdf4llm)）。見出しを取り出せなかった章は、前の章の中に入ります。
- **2カラムの論文は、ページごとに「左カラム → 右カラム」の順に並べ直します。** 横幅いっぱいのタイトル・図表は区切りとして扱います。3カラム以上や、カラムの幅が左右で大きく違うレイアウトには対応していません。
- **数式は崩れることがあります。** PDF から取り出した文字をそのまま翻訳しているためです。
- **訳語が章をまたいで揺れることがあります。** 段落ごとに別々に翻訳しており、前の段落の訳は参照していません。
- **1 語だけの見出し（`Rust`、`GEDF` など）は原文のまま出します。** 文脈なしで訳すと誤訳しやすいためです。
- **書籍モードの囲み（Note など）は、欄外のアイコンを手がかりに見つけています。** アイコンのない囲みは、ふつうの段落として出します。
- **回転したページは、ページ内の文字の 6 割以上が回転しているときだけ直します。** 表が小さく、本文と同じページに横向きで載っている場合は直しません。日本語・中国語の文字が中心の回転したページ（縦書きの本文と区別できないため）も直しません。
- **EPUB と HTML は読み込めません。** PDF に変換するか、本文を Markdown としてテキスト翻訳タブに貼り付けてください。
- **出力は機械翻訳です。** 内容は必ず原文で確認してください。

## トラブルシューティング

### serve.sh が `The NVIDIA driver on your system is too old` で止まる

CUDA 13 版の PyTorch が入っています。`requirements-vllm.txt` を使わずに `pip install vllm` で入れると、こうなります。`.venv-vllm` を削除して、[手順 4](#4-vllm-の環境を作る) をやり直してください。

### serve.sh が `libnvrtc.so.13: cannot open shared object file` で止まる

`torchcodec` が CUDA 13 版になっています。次のコマンドで入れ直してください。

```bash
.venv-vllm/bin/pip install --force-reinstall --no-deps torchcodec==0.16.0+cu129 --index-url https://download.pytorch.org/whl/cu129
```

### serve.sh が `No available memory for the cache blocks` で止まる

確保した VRAM のうち、KV キャッシュに回す分が足りていません。

- ほかに GPU を使っているアプリを閉じてください。
- それでも足りなければ、`GPU_UTIL=0.8 ./serve.sh` のように割合を上げてください。
- BF16 版（`pfnet/plamo-3-translate-2b`）は、8 GB の GPU では vLLM で動きません。FP8 版を使ってください。

### アプリを終了しても、Windows のタスクマネージャーでメモリが減らない

WSL2 は、Linux 側で解放したメモリをすぐには Windows に返しません（タスクマネージャーでは `vmmemWSL` / `Vmmem` として表示されます）。

- すぐに解放したい場合は、PowerShell で `wsl --shutdown` を実行します。
- 恒久的に対策したい場合は、`C:\Users\<ユーザー名>\.wslconfig` に以下を書き、`wsl --shutdown` で WSL を再起動します。

  ```ini
  [wsl2]
  memory=16GB

  [experimental]
  autoMemoryReclaim=gradual
  ```

### Transformers で動かすと、起動しただけで RAM を 8 GB 以上使う

bitsandbytes を import すると、PyTorch が CPU のコア数ぶん `torch.compile` 用のワーカーを起動します。24 コアの環境では、これで約 8 GB を使いました。

`backends.py` と `quantize.py` では、`import torch` の前に `TORCHINDUCTOR_COMPILE_THREADS=1` を設定してこれを止めています。自分のスクリプトから bitsandbytes を使うときも、同じ設定をしてください。

### `CUDA out of memory`（Transformers）

- `--batch-size` を下げてください（例: `--batch-size 2`）。
- BF16 版（`./model`）を使っている場合は、`python quantize.py` で 4bit 版にしてください。

## ライセンス

- このリポジトリのコード: （ライセンスを決めて記載してください）
- PLaMo 3 Translate のモデル: [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja)
  - 個人利用と非商用利用（学術利用を含む）は、登録なしで無償で利用できます。
  - 商用利用には条件があります。詳しくはライセンス本文と [公開時の告知](https://www.preferred.jp/ja/blog/tech/plamo-3-translate-weight-release) を確認してください。
