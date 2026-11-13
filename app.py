import argparse
import atexit
import os
import re
import signal
import sys
import threading
import time
import zipfile
import tempfile

import gradio as gr

import ocr
from detect_lang import detect_language
import paper
from backends import TransformersBackend, VllmBackend, VllmServer, find_vllm_models

# ------------------------------------------------------------
# 翻訳エンジンの選択（--backend auto のとき、上から順に試す）
#   1. vLLM サーバーがすでに起動している（./serve.sh を手動で起動した） -> それを使う
#   2. .venv-vllm と vLLM 用のモデル（./model-fp8 など）がある
#        -> アプリが vLLM サーバーを起動する。GUI でモデルを切り替えられる
#   3. それ以外 -> Transformers（model-4bit があればそれ、無ければ model）
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="auto")
parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8000/v1"),
                    help="vLLM サーバーの URL")
parser.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL"),
                    help="アプリが vLLM サーバーを起動するときに最初に使うモデル（既定: ./model-fp8）")
parser.add_argument("--model", default=os.environ.get("PLAMO_MODEL"),
                    help="Transformers で使うモデルのディレクトリ")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--batch-size", type=int, default=8,
                    help="Transformers で一度に訳す段落数（VRAMが足りなければ下げる）")
args = parser.parse_args()

# アプリが起動・管理する vLLM サーバー（手動起動のサーバーや Transformers のときは None）
vllm_server = None
vllm_models = []          # [(表示名, パス)]
_backend = None


def create_backend():
    global vllm_server, vllm_models

    if args.backend != "transformers":
        if VllmBackend.is_available(args.vllm_url):
            print(f"起動済みの vLLM サーバー（{args.vllm_url}）を使います。モデルの切り替えはできません。")
            return VllmBackend(args.vllm_url)

        vllm_models = find_vllm_models(".")
        if os.access(".venv-vllm/bin/vllm", os.X_OK) and vllm_models:
            paths = [p for _, p in vllm_models]
            initial = args.vllm_model or ("./model-fp8" if "./model-fp8" in paths else paths[0])
            vllm_server = VllmServer(args.vllm_url)
            vllm_server.model_path = initial   # 画面の初期表示用（起動はスレッドで行う）
            # 終了時（Ctrl+C を含む）に vLLM サーバーも止めて VRAM を解放する
            atexit.register(vllm_server.stop)
            threading.Thread(target=_start_vllm, args=(initial,), daemon=True).start()
            return None   # 起動が終わったら _start_vllm がセットする

        if args.backend == "vllm":
            raise SystemExit(
                "vLLM を使えません。.venv-vllm と ./model-fp8 を用意するか、./serve.sh を起動してください。"
            )

    model_path = args.model or next(
        (p for p in ("./model-4bit", "./model") if os.path.isdir(p)), None
    )
    if model_path is None:
        raise SystemExit(
            "翻訳エンジンが見つかりません。README の手順で vLLM の環境とモデルを用意するか、"
            "「Transformers で動かす」の手順で ./model を用意してください。"
        )
    return TransformersBackend(model_path, batch_size=args.batch_size)


def _start_vllm(model_path):
    global _backend
    print(f"vLLM サーバーを起動中: {model_path}")
    try:
        vllm_server.start(model_path)
    except Exception as e:
        print(f"vLLM サーバーを起動できませんでした: {e}")
        return
    _backend = VllmBackend(args.vllm_url)
    print("backend:", _backend.name)


def get_backend():
    """翻訳エンジンを返す。vLLM サーバーの起動中・切り替え中はエラーにする"""
    if vllm_server is not None and not vllm_server.ready:
        raise gr.Error("翻訳モデルを起動中です。画面上部の表示が「翻訳できます」になってから実行してください。")
    return _backend


_ocr_status = ""      # OCR 中の表示（空なら翻訳エンジンの状態を出す）


def engine_status():
    if _ocr_status:
        return _ocr_status
    if vllm_server is not None:
        return vllm_server.status_text()
    return f"✅ `{_backend.name}` で翻訳できます"


def can_free_gpu():
    """OCR のために翻訳エンジンを止めて GPU を空けられるか（アプリが vLLM を起動したときだけ）"""
    return vllm_server is not None and not isinstance(_backend, TransformersBackend)


def run_ocr_freeing_gpu(path, pages, on_progress, recheck=()):
    """翻訳エンジンを止めて GPU を空け、OCR してから翻訳エンジンを起動し直す

    8 GB の GPU では、OCR のモデルと翻訳用の vLLM サーバーを同時に動かせない
    （Unlimited-OCR は約 6.8 GiB、LightOnOCR-3 は小さいが vLLM 同士でメモリを分け合えない）。
    recheck は、すでに文字が載っているスキャンのページ（精度を確かめ、低ければ OCR し直す）。
    """
    global _backend, _ocr_status
    if not pages and not can_free_gpu():
        # 文字の載ったスキャンだけなら、OCR をあきらめて既存の文字で訳す
        return {}
    if isinstance(_backend, TransformersBackend):
        raise gr.Error("OCR は vLLM で翻訳しているときだけ使えます"
                       "（Transformers の翻訳モデルが GPU を使ったままになるため）。")
    restart = vllm_server.model_path if vllm_server is not None and vllm_server.process else None
    _ocr_status = f"📷 OCR（{ocr.engine_name()}）のため、翻訳エンジンを一時停止しています"
    try:
        if restart:
            _backend = None
            vllm_server.stop()
        problem = ocr.check_gpu()
        if problem:
            raise gr.Error(problem + "手動で起動した vLLM サーバー（serve.sh）などを止めてから実行してください。")

        def on_message(msg):
            global _ocr_status
            _ocr_status = f"📷 {msg}（そのあいだ翻訳エンジンを一時停止しています）"
            on_progress(0, 0, msg)

        try:
            return ocr.run_ocr(path, pages, lambda done, total: on_progress(done, total, None),
                               recheck=recheck, on_message=on_message)
        except RuntimeError as e:
            raise gr.Error(str(e))
    finally:
        _ocr_status = ""
        if restart:
            try:
                vllm_server.start(restart)
                _backend = VllmBackend(args.vllm_url)
            except Exception as e:
                gr.Warning(f"翻訳エンジンを起動し直せませんでした: {e}")


def switch_model(model_path):
    """GUI からのモデル切り替え（vLLM サーバーの再起動）"""
    global _backend
    if vllm_server is None:
        raise gr.Error("この起動方法ではモデルを切り替えられません。")
    if model_path == vllm_server.model_path and vllm_server.ready:
        return engine_status(), gr.update(value=model_path)
    _backend = None
    try:
        vllm_server.switch(model_path)
    except Exception as e:
        gr.Warning(vllm_server.message or str(e))
    if vllm_server.ready:
        _backend = VllmBackend(args.vllm_url)
    return engine_status(), gr.update(value=vllm_server.model_path)


def _exit_on_sigterm(signum, frame):
    sys.exit(0)   # atexit を走らせて vLLM サーバーを止める


signal.signal(signal.SIGTERM, _exit_on_sigterm)
# Ctrl+C（SIGINT）で KeyboardInterrupt を起こし、atexit で vLLM サーバーを止める
# （バックグラウンド起動などで SIGINT が無視に設定されている場合も元に戻す）
signal.signal(signal.SIGINT, signal.default_int_handler)

_initial_backend = create_backend()
if _initial_backend is not None:
    _backend = _initial_backend
    print("backend:", _backend.name)


# 言語コードは model/README_ja.md の「対応言語」を参照
LANGUAGES = [
    ("自動判定", "auto"),
    ("英語", "en"),
    ("日本語", "ja"),
    ("やさしい日本語", "ja(easy)"),
    ("中国語（簡体字）", "zh-hans"),
    ("中国語（繁体字）", "zh-hant"),
    ("韓国語", "ko"),
    ("フランス語", "fr"),
    ("ドイツ語", "de"),
    ("スペイン語", "es"),
    ("イタリア語", "it"),
    ("ポルトガル語", "pt"),
    ("ロシア語", "ru"),
    ("ベトナム語", "vi"),
    ("タイ語", "th"),
    ("インドネシア語", "id"),
]
TARGET_LANGUAGES = [lang for lang in LANGUAGES if lang[1] != "auto"]

FORMS = [("常体（だ・である）", "plain"), ("敬体（です・ます）", "polite")]


def split_text(text, max_chars=2200):
    """段落を維持しながら文章を分割する"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]

        if len(current) + len(paragraph) > max_chars:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current += "\n\n" + paragraph if current else paragraph

    if current:
        chunks.append(current)

    return chunks


# ------------------------------------------------------------
# テキスト翻訳
# ------------------------------------------------------------
def is_markdown_document(text):
    """画像（![](...)）やコード（```）を含む Markdown か。電子書籍の本文やチャットの書き出しなど"""
    return bool(re.search(r"!\[[^\]\n]*\]\(|^\s*```", text, re.MULTILINE))


def _hide_images(md_text):
    """画面に出すときは、埋め込み画像（base64）を目印に置き換えて軽くする"""
    # 途中で切れた（閉じ括弧のない）画像も置き換える
    return re.sub(r"!\[[^\]\n]*\]\(data:[^)\s]*\)?", "*(図)*", md_text)


def translate_input_text(text, source_lang, target_lang, form,
                         progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("翻訳する文章を入力してください。")
    if source_lang == target_lang:
        raise gr.Error("翻訳元と翻訳先が同じ言語です。")

    backend = get_backend()

    def on_progress(done, total):
        progress(done / total, desc=f"翻訳中 ({done}/{total})")

    if not is_markdown_document(text):
        chunks = split_text(text)
        results = backend.translate_many(chunks, source_lang, target_lang, form,
                                         progress=on_progress)
        return "\n\n".join(results), None

    # Markdown: 画像・コード・表の記号はそのまま残し、文章だけを訳す
    blocks = paper.parse_blocks(text, mode=paper.DOCUMENT)
    paper.translate_blocks(
        blocks,
        lambda texts, cb: backend.translate_many(texts, source_lang, target_lang, form, progress=cb),
        progress=on_progress,
    )
    md_text = paper.render_markdown(blocks, "")
    output_dir = tempfile.mkdtemp(prefix="plamo_")
    files = []
    for ext, content in (("md", md_text), ("html", paper.render_html(md_text, "翻訳結果"))):
        path = os.path.join(output_dir, f"translated.{ext}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        files.append(path)
    return _hide_images(md_text), files


LANGUAGE_NAMES = dict((code, name) for name, code in LANGUAGES)


def source_label(detected=None):
    """翻訳元の欄の見出し。自動判定のときは判定した言語を添える"""
    if detected:
        return f"翻訳元（判定: {LANGUAGE_NAMES[detected]}）"
    return "翻訳元"


def auto_translate_text(text, source_lang, target_lang, form, progress=gr.Progress()):
    """入力・言語・文体が変わるたびに呼ぶ（翻訳ボタンは無い）

    空欄なら訳文も消す。翻訳エンジンの起動中は、起動が終わるまで待ってから訳す
    （入力のたびにエラーを出さないため）。
    翻訳元が「自動判定」のときは原文の言語を判定し、翻訳元の見出しに出して、⇄ の入れ替えに使う。
    戻り値: 訳文, 画像入りの訳文ファイル, 翻訳元の欄の更新, 判定した言語
    """
    detected = detect_language(text) if source_lang == "auto" and text else None
    label = gr.update(label=source_label(detected))
    if not text or not text.strip():
        return "", None, label, None
    if source_lang == target_lang:
        raise gr.Error("翻訳元と翻訳先が同じ言語です。")
    if vllm_server is not None:
        deadline = time.time() + vllm_server.startup_timeout
        while vllm_server.state == "starting" and time.time() < deadline:
            progress(0, desc="翻訳モデルの起動を待っています")
            time.sleep(1)
    translated, files = translate_input_text(text, source_lang, target_lang, form, progress)
    return translated, files, label, detected


def swap_languages(source_lang, target_lang, text, translated, detected):
    """翻訳の向きと原文・訳文を入れ替える。自動判定のときは判定した言語を翻訳元として確定させる"""
    if source_lang == "auto":
        if not detected:
            # まだ判定していない（原文が空など）ときは入れ替えない
            return gr.update(), target_lang, text, translated, None
        source_lang = detected
    return gr.update(value=target_lang, label=source_label()), source_lang, translated, text, None


# ------------------------------------------------------------
# PDF・文書翻訳（章ごとの全文訳を Markdown / HTML で出力）
# ------------------------------------------------------------
DOC_MODES = [("自動判定", "auto"), ("論文", paper.PAPER), ("書籍・一般文書", paper.DOCUMENT)]
IMAGE_DPI = [("標準（150dpi・軽い）", 150), ("高画質（200dpi）", 200), ("最高画質（300dpi・重い）", 300)]


def parse_page_range(text, page_count):
    """ "37-80, 100" -> 0 始まりのページ番号のリスト。空欄なら None（全ページ）"""
    if not text or not text.strip():
        return None
    pages = []
    for part in re.split(r"[,、\s]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)?\s*[-~〜]\s*(\d+)?|(\d+)", part)
        if not m:
            raise gr.Error(f"ページ範囲を読み取れません: {part}（例: 37-80, 100）")
        if m.group(3):
            first = last = int(m.group(3))
        else:
            first = int(m.group(1) or 1)
            last = int(m.group(2) or page_count)
        if first < 1 or last > page_count or first > last:
            raise gr.Error(f"ページ範囲が文書の外です: {part}（全 {page_count} ページ）")
        pages += [p for p in range(first - 1, last) if p not in pages]
    return pages


def _is_pdf(path):
    return path.lower().endswith(".pdf")


def book_chapters(path, mode):
    """書籍として扱う PDF なら、しおりの章の一覧を返す（論文・しおりの無い PDF は空）"""
    if not _is_pdf(path):
        return []
    chapters = paper.pdf_chapters(path)
    if len(chapters) < 3:
        return []
    if mode == paper.PAPER or (mode == "auto" and paper.detect_pdf_mode(path) == paper.PAPER):
        return []
    return chapters


def show_chapters(files, mode):
    """ファイルを選んだときに、訳す章の選択肢を出す（PDF 1つのときだけ）"""
    hidden = (gr.update(choices=[], value=[], visible=False), gr.update(visible=False))
    if not files or len(files) != 1:
        return hidden
    chapters = book_chapters(files[0], mode)
    if not chapters:
        return hidden
    choices = [(f"{c.title}（{c.start + 1}〜{c.end} ページ）", str(i))
               for i, c in enumerate(chapters)]
    selected = [str(i) for i, c in enumerate(chapters) if paper.is_body_chapter(c)]
    return (gr.update(choices=choices, value=selected, visible=True),
            gr.update(visible=True))


def pdf_markdown(path, pages, mode, with_images, image_dpi, ocr_texts):
    """PDF のページを Markdown にする。OCR したページは OCR の結果を、それ以外は PDF の文字を使う"""
    import pymupdf

    with pymupdf.open(path) as doc:
        page_list = list(pages) if pages is not None else list(range(doc.page_count))
        if not any(p in ocr_texts for p in page_list):
            return paper.pdf_to_markdown(path, with_images=with_images, mode=mode,
                                         pages=pages, image_dpi=image_dpi)
        parts, run = [], []
        skip_titles = ocr.running_titles(ocr_texts)

        def flush():
            if run:
                parts.append(paper.pdf_to_markdown(path, with_images=with_images, mode=mode,
                                                   pages=list(run), image_dpi=image_dpi))
                run.clear()

        for p in page_list:
            if p in ocr_texts:
                flush()
                parts.append(ocr.ocr_to_markdown(ocr_texts[p], doc[p], with_images, image_dpi,
                                                 skip_titles))
            else:
                run.append(p)
        flush()
    return "\n\n".join(parts)


def load_units(path, mode, page_range, chapter_ids, with_images, image_dpi, stem,
               use_ocr=False, on_ocr_progress=None):
    """ファイルを読み、(出力の単位のリスト, 文書の種類) を返す

    出力の単位は (ファイル名に付ける名前, ブロック列)。書籍の PDF で章を選んだときは章ごと、
    それ以外はファイル全体で1つ。
    use_ocr なら、文字を持たないページ（スキャン）を先に OCR しておく。
    """
    if not _is_pdf(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            md_source = f.read()
        doc_mode = paper.detect_mode(md_source) if mode == "auto" else mode
        return [("", paper.parse_blocks(md_source, mode=doc_mode, title=stem))], doc_mode

    import pymupdf
    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        # PDF に題名が入っていればそれを文書の題にする（無ければファイル名）
        stem = (doc.metadata or {}).get("title", "").strip() or stem
    pages = parse_page_range(page_range, page_count)
    chapters = book_chapters(path, mode) if pages is None else []
    selected = []
    if chapters:
        selected = [chapters[int(i)] for i in (chapter_ids or []) if int(i) < len(chapters)]
        if not selected:
            raise gr.Error("訳す章を1つ以上選んでください。")
        targets = [p for ch in selected for p in range(ch.start, ch.end)]
    else:
        targets = pages if pages is not None else list(range(page_count))

    ocr_texts = {}
    if use_ocr and ocr.ocr_available():
        scanned = ocr.find_scanned_pages(path, targets)
        # スキャンの上にすでに文字（スキャナの OCR など）が載っているページ。
        # LightOnOCR-3 があれば、その文字の精度を確かめ、低ければ OCR し直す
        recheck = ocr.find_text_layer_scans(path, targets) if ocr.lighton_available() else []
        if scanned or recheck:
            ocr_texts = run_ocr_freeing_gpu(path, scanned, on_ocr_progress, recheck)

    if chapters:
        units = []
        for n, ch in enumerate(selected):
            md_source = pdf_markdown(path, range(ch.start, ch.end), paper.DOCUMENT,
                                     with_images, int(image_dpi), ocr_texts)
            blocks = paper.parse_blocks(md_source, mode=paper.DOCUMENT, title=stem)
            paper.apply_outline(blocks, ch)
            units.append((f"{n + 1:02d}_{_slug(ch.title)}", blocks))
        return units, paper.DOCUMENT

    if mode != "auto":
        doc_mode = mode
    elif any(p in ocr_texts for p in targets[:3]):
        # スキャンしたページを含むときは、冒頭3ページの文字（OCR したページは OCR の結果）で判定する
        with pymupdf.open(path) as doc:
            head = [ocr.ocr_to_markdown(ocr_texts[p], with_images=False,
                                        skip_titles=ocr.running_titles(ocr_texts)) if p in ocr_texts
                    else doc[p].get_text() for p in targets[:3]]
        doc_mode = paper.detect_mode("\n\n".join(head))
    else:
        doc_mode = paper.detect_pdf_mode(path, pages)
    md_source = pdf_markdown(path, pages, doc_mode, with_images, int(image_dpi), ocr_texts)
    return [("", paper.parse_blocks(md_source, mode=doc_mode, title=stem))], doc_mode


def _slug(title, max_len=40):
    """ファイル名に使える短い名前にする（"Chapter 2: Building ..." -> "Chapter_2_Building"）"""
    slug = re.sub(r"[^\w]+", "_", title).strip("_")
    return slug[:max_len].rstrip("_") or "part"


def translate_files(files, source_lang, target_lang, form, formats,
                    bilingual, with_images, translate_refs, mode, page_range, image_dpi,
                    chapter_ids, split_chapters, progress=gr.Progress()):
    if not files:
        raise gr.Error("ファイルを選択してください。")
    if not formats:
        raise gr.Error("出力形式を1つ以上選んでください。")
    backend = get_backend()

    output_dir = tempfile.mkdtemp(prefix="plamo_")
    output_files = []
    preview = ""

    for n, path in enumerate(files):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        prefix = f"[{n + 1}/{len(files)}] {name}"

        progress(0, desc=f"{prefix}: 本文を抽出中（書籍1冊では数分かかります）")
        ocr_step = {"msg": "スキャンしたページを OCR 中"}

        def on_ocr_progress(done, total, message=None):
            if message:
                ocr_step["msg"] = message
            if total:
                progress(done / total, desc=f"{prefix}: {ocr_step['msg']} ({done}/{total} ページ、"
                                            f"1 ページ約 {ocr.seconds_per_page()} 秒)")
            else:
                progress(0, desc=f"{prefix}: {ocr_step['msg']}")

        units, doc_mode = load_units(path, mode, page_range, chapter_ids if len(files) == 1 else None,
                                     with_images, image_dpi, stem, True, on_ocr_progress)
        if not split_chapters and len(units) > 1:
            # 選んだ章を1つのファイルにまとめる（題は先頭の1つだけ残す）
            merged = units[0][1] + [b for _, blocks in units[1:] for b in blocks if b.kind != "title"]
            units = [("", merged)]

        all_blocks = [b for _, blocks in units for b in blocks]
        if not any(b.text for b in all_blocks if b.kind not in ("title", "heading", "table", "image")):
            hint = ("" if ocr.ocr_available()
                    else "スキャンした PDF は OCR が必要です（README の「スキャンした PDF を読む」を参照）。")
            raise gr.Error(f"{name}: テキストを抽出できませんでした。{hint}")

        def on_progress(done, total):
            progress(done / total, desc=f"{prefix}: 翻訳中 ({done}/{total} 段落)")

        # 章ごとに分けて出す場合も、翻訳は全章まとめて投げる（vLLM で並列に処理できる）
        paper.translate_blocks(
            all_blocks,
            lambda texts, cb: backend.translate_many(
                texts, source_lang, target_lang, form, progress=cb
            ),
            translate_references=translate_refs,
            progress=on_progress,
        )

        for unit_name, blocks in units:
            md_text = paper.render_markdown(blocks, name, bilingual=bilingual)
            title = next((b.translated or b.text for b in blocks if b.kind == "title"), stem)
            if unit_name:
                chapter = next((b for b in blocks if b.kind == "heading"), None)
                if chapter:
                    title = f"{chapter.translated or chapter.text} - {title}"
            base = f"{stem}_{unit_name}_ja" if unit_name else f"{stem}_ja"

            if "Markdown" in formats:
                out = os.path.join(output_dir, f"{base}.md")
                with open(out, "w", encoding="utf-8") as f:
                    f.write(md_text)
                output_files.append(out)

            if "HTML" in formats:
                out = os.path.join(output_dir, f"{base}.html")
                with open(out, "w", encoding="utf-8") as f:
                    f.write(paper.render_html(md_text, title))
                output_files.append(out)

            if not preview:
                # 画面のプレビューは画像を除いて軽くする
                kind = "論文" if doc_mode == paper.PAPER else "書籍・一般文書"
                preview = f"*（{kind}として処理）*\n\n" + _hide_images(md_text)

    if len(output_files) > 2:
        zip_path = os.path.join(output_dir, "translated_files.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for out in output_files:
                z.write(out, os.path.basename(out))
        output_files = [zip_path] + output_files

    return output_files, preview


def language_controls(default_source="en", default_target="ja"):
    with gr.Row():
        source = gr.Dropdown(LANGUAGES, value=default_source, label="翻訳元")
        target = gr.Dropdown(TARGET_LANGUAGES, value=default_target, label="翻訳先")
        form = gr.Radio(FORMS, value="plain", label="文体（翻訳先が日本語のとき）")
    return source, target, form


with gr.Blocks(title="PLaMo Translator") as demo:

    gr.Markdown(
        """
        # PLaMo 3 Translate
        ### Local Translator

        ローカル環境で文章・PDF（論文・書籍など）を翻訳します。
        """
    )

    with gr.Row(equal_height=True):
        can_switch = vllm_server is not None and len(vllm_models) > 1
        model_select = gr.Dropdown(
            vllm_models,
            value=vllm_server.model_path if vllm_server else None,
            label="翻訳モデル（切り替えには 1〜2 分かかります）",
            interactive=can_switch,
            visible=vllm_server is not None,
            scale=3,
        )
        switch_button = gr.Button(
            "モデルを切り替え", interactive=can_switch,
            visible=vllm_server is not None, scale=1,
        )
    engine_md = gr.Markdown(engine_status())
    # 起動・切り替えの進み具合を表示する（翻訳の処理を待たずに更新できるよう同時実行の制限を外す）
    gr.Timer(2).tick(engine_status, outputs=engine_md,
                     concurrency_limit=None, show_progress="hidden")

    with gr.Tab("テキスト翻訳"):
        t_source, t_target, t_form = language_controls("auto", "ja")
        t_detected = gr.State(None)     # 自動判定で判定した原文の言語（⇄ の入れ替えに使う）

        with gr.Row(elem_id="text-row"):
            text_input = gr.Textbox(
                label="原文",
                lines=14,
                placeholder="翻訳したい文章を入力すると、そのまま訳します"
            )
            # 原文と訳文の間に置く入れ替えボタン（言語と文章を入れ替える）
            swap_button = gr.Button("⇄", elem_id="swap-button", scale=0, min_width=48)
            text_output = gr.Textbox(
                label="訳文",
                lines=14,
                buttons=["copy"]
            )

        gr.Markdown(
            "画像（`![](...)`）やコード（` ``` `）を含む Markdown"
            "（電子書籍の本文やAIとのチャットをコピーしたものなど）は、画像・コードを残したまま"
            "文章だけを訳します。訳文欄では画像を *(図)* と表示し、画像入りの訳文は下の"
            "ファイル（Markdown / HTML）に保存します。"
        )
        text_files = gr.File(label="画像入りの訳文", file_count="multiple", visible=True)

        # 入力したらすぐ訳す。翻訳中にさらに入力したら、終わったあとに最新の内容で1回だけ訳し直す
        # （.input はユーザーの操作だけで起きるので、入れ替えボタンでは訳し直さない）
        for event in (text_input.input, t_source.input, t_target.input, t_form.input):
            event(
                fn=auto_translate_text,
                inputs=[text_input, t_source, t_target, t_form],
                outputs=[text_output, text_files, t_source, t_detected],
                trigger_mode="always_last",
                show_progress="minimal",
            )
        swap_button.click(
            fn=swap_languages,
            inputs=[t_source, t_target, text_input, text_output, t_detected],
            outputs=[t_source, t_target, text_input, text_output, t_detected]
        )

    with gr.Tab("PDF・文書翻訳"):
        gr.Markdown(
            "章立てを保ったまま全文訳を HTML / Markdown で出力します。\n\n"
            "- **論文**: 章番号（1 / 1.1 / I. / A. など）の付いた見出しを章とみなし、"
            "著者・所属と参考文献は原文のまま残します。\n"
            "- **書籍・一般文書**（電子書籍・AIとのチャットの書き出しなど）: 文字の大きさから見出しを判定し、"
            "表の中の文章も含めて全文を訳します。欄外にアイコンの付いた囲み（Note など）は引用として出します。\n"
            "- PDF のほか、Markdown（.md）・テキスト（.txt）も読み込めます。"
        )
        p_source, p_target, p_form = language_controls("en", "ja")

        with gr.Row():
            p_mode = gr.Radio(DOC_MODES, value="auto", label="文書の種類")
            p_pages = gr.Textbox(
                label="ページ範囲（PDFのみ・空欄で全ページ。指定すると章の選択より優先）",
                placeholder="例: 37-80, 100",
            )
            p_dpi = gr.Dropdown(IMAGE_DPI, value=200, label="図の解像度")

        # 書籍の PDF（しおり付き）を1つ選んだときだけ出す
        p_chapters = gr.CheckboxGroup(
            [], label="訳す章（PDFのしおりから。表紙・目次・索引などは最初から外しています）",
            visible=False,
        )
        p_split = gr.Checkbox(True, label="章ごとにファイルを分ける（ZIPにもまとめます）",
                              visible=False)
        with gr.Row():
            p_formats = gr.CheckboxGroup(
                ["HTML", "Markdown"], value=["HTML", "Markdown"], label="出力形式"
            )
            p_bilingual = gr.Checkbox(True, label="段落ごとに原文を併記（折りたたみ）")
            p_images = gr.Checkbox(True, label="図を埋め込む")
            p_refs = gr.Checkbox(False, label="参考文献も翻訳する（論文）")

        pdf_input = gr.File(
            label="翻訳するファイルを選択（PDF / Markdown / テキスト）",
            file_count="multiple",
            file_types=[".pdf", ".md", ".markdown", ".txt"],
            type="filepath"
        )

        translate_button = gr.Button(
            "翻訳開始",
            variant="primary"
        )

        output_file = gr.File(
            label="翻訳結果",
            file_count="multiple"
        )

        with gr.Accordion("プレビュー（1本目）", open=False):
            preview_md = gr.Markdown()

        for event in (pdf_input.change, p_mode.change):
            event(show_chapters, inputs=[pdf_input, p_mode], outputs=[p_chapters, p_split])

        translate_button.click(
            fn=translate_files,
            inputs=[pdf_input, p_source, p_target, p_form, p_formats,
                    p_bilingual, p_images, p_refs, p_mode, p_pages, p_dpi,
                    p_chapters, p_split],
            outputs=[output_file, preview_md]
        )


    # モデルの切り替え: 実行中は翻訳ボタンを押せないようにして、終わったら戻す
    _locked = [model_select, switch_button, translate_button]

    def _set_interactive(flag):
        return [gr.update(interactive=flag and can_switch)] * 2 + [gr.update(interactive=flag)]

    switch_button.click(
        lambda: _set_interactive(False), outputs=_locked, queue=False,
    ).then(
        switch_model, inputs=model_select, outputs=[engine_md, model_select],
    ).then(
        lambda: _set_interactive(True), outputs=_locked, queue=False,
    )


# Transformers はGPUを同時に使うと VRAM が足りなくなるので、リクエストは1件ずつ処理する
# （vLLM の場合も、1件の中で段落を並列に投げるのでこれで十分）
demo.queue(default_concurrency_limit=1)

# 入れ替えボタン（⇄）を原文と訳文の間の縦中央に置く
CSS = """
#text-row { align-items: center; }
#swap-button { font-size: 1.5rem; padding: 0.5rem; }
"""

demo.launch(
    css=CSS,
    server_name="0.0.0.0",
    server_port=args.port
)
