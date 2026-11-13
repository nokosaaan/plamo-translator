"""スキャンした PDF を OCR で読み取る

OCR のエンジンは 2 つ。使える方を選び、両方あれば LightOnOCR-3 を使う。

- **LightOnOCR-3-0.8B**（ `model-lighton/` 、Apache-2.0）: 翻訳と同じ `.venv-vllm` の vLLM で
  OCR 用のサーバーを一時的に起動し、ページを並列に送る。1 ページ約 2 秒。日本語も読める
- **Unlimited-OCR**（ `model-ocr/` 、MIT）: `.venv-ocr` の別プロセス（ `ocr_worker.py` ）で
  Transformers を使う。1 ページ約 40 秒

ここでは
- OCR が必要なページを見つける（find_scanned_pages / find_text_layer_scans）
- OCR して結果を受け取る（run_ocr）
- OCR の出力を paper.py が読める Markdown に直す（ocr_to_markdown）
を行う。

LightOnOCR-3 の出力（grounding モード）は、ブロックごとに種類と位置（0〜1000 に正規化した座標）が付いた形:

    ![title](80,70,920,140) # Company report 2025
    ![text](80,175,920,225) Revenue rose 20%.
    ![image](80,285,460,545) A solar-powered factory.

Unlimited-OCR の出力は、座標が 0〜999 の次の形:

    <|det|>title [137, 65, 844, 132]<|/det|>Deadlock- and Starvation-free ...
    <|det|>text [75, 244, 492, 336]<|/det|>Abstract—Robot middleware is ...
    <|det|>table [244, 106, 681, 913]<|/det|><table><tr><td>...</td></tr></table>
    <|det|>equation [580, 742, 845, 760]<|/det|>\\[ ... \\]
    <|det|>image [510, 66, 920, 86]<|/det|>
"""

import base64
import html
import json
import os
import re
import signal
import subprocess
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent
OCR_PYTHON = _ROOT / ".venv-ocr" / "bin" / "python"
OCR_MODEL = _ROOT / "model-ocr"
WORKER = _ROOT / "ocr_worker.py"

LIGHTON_MODEL = _ROOT / "model-lighton"
VLLM_BIN = _ROOT / ".venv-vllm" / "bin" / "vllm"
LIGHTON_PORT = 8001
LIGHTON_PARALLEL = 8          # 同時に送るページ数
LIGHTON_LONGEST_PX = 2048     # モデルカードの推奨: 長辺 2048px
# 既存の文字（スキャナの OCR など）とこのモデルの結果の、単語の一致率がこれ未満なら、
# 既存の文字は精度が低いとみなして OCR で上書きする（実測: ScanSnap・My Image Garden の文字は約 0.9）
REOCR_AGREEMENT = 0.95
REOCR_SAMPLES = 3

# OCR に必要な GPU の空きメモリ
#   Unlimited-OCR: base 設定で測ったピーク 6.8 GiB に少し余裕を持たせた値
#   LightOnOCR-3: vLLM に GPU の 6 割を渡す（3 割では KV キャッシュがほぼ残らない）
REQUIRED_FREE_MIB_UNLIMITED = 7000
LIGHTON_GPU_UTIL = 0.6
LIGHTON_MIN_GPU_UTIL = 0.35


def lighton_available() -> bool:
    return VLLM_BIN.exists() and (LIGHTON_MODEL / "config.json").is_file()


def unlimited_available() -> bool:
    return OCR_PYTHON.exists() and (OCR_MODEL / "config.json").is_file()


def ocr_available() -> bool:
    return lighton_available() or unlimited_available()


def engine_name() -> str:
    return "LightOnOCR-3" if lighton_available() else "Unlimited-OCR"


def seconds_per_page() -> int:
    return 2 if lighton_available() else 40


def _is_scanned(page) -> bool:
    import pymupdf

    if len(page.get_text().strip()) < 30:
        return bool(page.get_images() or page.get_drawings())
    # 出版社のダウンロード透かし（余白の "Downloaded from ... by [IP]" など）だけ文字を持ち、
    # 本文はページ全体を覆う画像、というスキャン PDF がある。大きな画像の上に文字がほとんど
    # 無ければスキャンしたページとみなす
    area = page.rect.width * page.rect.height
    big = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()
           if pymupdf.Rect(info["bbox"]).get_area() >= area * 0.3]
    if not big:
        return False
    on_image = off_image = 0
    for w in page.get_text("words"):
        center = pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
        if any(center in r for r in big):
            on_image += len(w[4])
        else:
            off_image += len(w[4])
    # 画像の外に本文がある（大きな図と本文が並んだページ）なら、文字のあるページとして扱う
    return on_image < 30 and off_image < 200


def _is_text_layer_scan(page) -> bool:
    """ページ全体が 1 枚の画像で、その上に文字（スキャナなどの OCR の結果）が載っているページ"""
    import pymupdf

    if len(page.get_text().strip()) < 30:
        return False
    area = page.rect.width * page.rect.height
    return any(pymupdf.Rect(info["bbox"]).get_area() >= area * 0.9
               for info in page.get_image_info())


def find_scanned_pages(pdf_path: str, pages: Optional[Sequence[int]] = None) -> List[int]:
    """本文が画像になっているページ（スキャンしたページ）を返す"""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [i for i in (pages if pages is not None else range(doc.page_count))
                if _is_scanned(doc[i])]


def find_text_layer_scans(pdf_path: str, pages: Optional[Sequence[int]] = None) -> List[int]:
    """スキャンした画像の上に、すでに OCR の文字が載っているページを返す"""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [i for i in (pages if pages is not None else range(doc.page_count))
                if not _is_scanned(doc[i]) and _is_text_layer_scan(doc[i])]


def gpu_free_mib() -> Optional[int]:
    return _gpu_query("memory.free")


def gpu_total_mib() -> Optional[int]:
    return _gpu_query("memory.total")


def _gpu_query(field: str) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def check_gpu() -> Optional[str]:
    """OCR に必要な GPU メモリが空いていなければ、その理由を返す"""
    free, total = gpu_free_mib(), gpu_total_mib()
    if free is None or total is None:
        return None
    if lighton_available():
        need = int(total * LIGHTON_MIN_GPU_UTIL) + 256
    else:
        need = REQUIRED_FREE_MIB_UNLIMITED
    if free < need:
        return f"OCR に必要な GPU メモリが空いていません（空き {free} MiB、必要 約 {need} MiB）。"
    return None


# ------------------------------------------------------------
# 既存の文字の精度
# ------------------------------------------------------------
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def _tokens(text: str) -> Counter:
    """比べるための単語（ラテン文字は 2 文字以上の語、日中韓は 2 文字ずつ）"""
    text = re.sub(r"!\[[^\]]*\]\([\d,\s]*\)", " ", text)
    text = unicodedata.normalize("NFKC", text).lower()
    words = re.findall(r"[a-zà-ɏ]{2,}", text)
    cjk = "".join(_CJK.findall(text))
    return Counter(words) + Counter(cjk[i:i + 2] for i in range(len(cjk) - 1))


def text_layer_agreement(layer_text: str, ocr_text: str) -> float:
    """既存の文字の単語のうち、OCR の結果にもある割合（読み違い・語のくっつきで下がる）"""
    a, b = _tokens(layer_text), _tokens(ocr_text)
    n = sum(a.values())
    return sum((a & b).values()) / n if n else 1.0


# ------------------------------------------------------------
# OCR の実行
# ------------------------------------------------------------
def run_ocr(pdf_path: str, pages: Sequence[int],
            progress: Optional[Callable[[int, int], None]] = None,
            log_path: str = "logs/ocr.log", recheck: Sequence[int] = (),
            on_message: Optional[Callable[[str], None]] = None) -> Dict[int, str]:
    """{ページ番号: OCR の出力} を返す。GPU の空きは呼び出し側で用意する

    recheck は、すでに文字が載っているスキャンのページ。LightOnOCR-3 のときだけ、
    そのうち数ページを OCR して既存の文字と比べ、精度が低ければ全部を OCR し直す。
    """
    if lighton_available():
        return _run_lighton(pdf_path, pages, progress, log_path, recheck, on_message)
    return _run_unlimited(pdf_path, pages, progress, log_path)


def _run_unlimited(pdf_path, pages, progress, log_path, dpi: int = 200) -> Dict[int, str]:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    results: Dict[int, str] = {}
    if not pages:
        return results
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [str(OCR_PYTHON), str(WORKER), str(OCR_MODEL), pdf_path,
             json.dumps(list(pages)), "--dpi", str(dpi)],
            stdout=subprocess.PIPE, stderr=log, text=True,
        )
        try:
            if progress:
                progress(0, len(pages))
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "page":
                    results[event["page"]] = event["text"]
                    if progress:
                        progress(len(results), len(pages))
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if proc.returncode != 0 or len(results) != len(pages):
        tail = Path(log_path).read_text(errors="replace").splitlines()[-5:]
        raise RuntimeError(f"OCR が途中で終了しました（詳細: {log_path}）\n" + "\n".join(tail))
    return results


class LightOnServer:
    """LightOnOCR-3 を載せた vLLM サーバー（OCR のあいだだけ起動する）"""

    def __init__(self, log_path: str, startup_timeout: int = 600):
        self.log_path = Path(log_path)
        self.startup_timeout = startup_timeout
        self.url = f"http://127.0.0.1:{LIGHTON_PORT}"
        self.process = None

    def start(self):
        free, total = gpu_free_mib(), gpu_total_mib()
        util = LIGHTON_GPU_UTIL
        if free is not None and total:
            util = min(util, round((free - 256) / total, 2))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w")
        env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
        self.process = subprocess.Popen(
            [str(VLLM_BIN), "serve", str(LIGHTON_MODEL),
             "--served-model-name", "lighton-ocr",
             "--host", "127.0.0.1", "--port", str(LIGHTON_PORT),
             "--gpu-memory-utilization", str(util),
             "--max-model-len", "10240", "--max-num-seqs", str(LIGHTON_PARALLEL),
             "--limit-mm-per-prompt", '{"image": 1}',
             # 入力画像は長辺 2048px（約 300 万画素）までにする。上限を決めないと、vLLM が
             # 最大の画像（約 1,600 万画素）を前提にメモリを見積もり、KV キャッシュが残らない
             "--mm-processor-kwargs", '{"size": {"longest_edge": 3211264, "shortest_edge": 65536}}',
             "--mm-processor-cache-gb", "0",
             "--default-chat-template-kwargs", '{"enable_thinking": false}'],
            cwd=str(_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        import httpx
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"OCR サーバーが起動中に終了しました（詳細: {self.log_path}）\n"
                                   + self._log_tail())
            try:
                if httpx.get(self.url + "/v1/models", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise RuntimeError(f"OCR サーバーが {self.startup_timeout} 秒以内に起動しませんでした（詳細: {self.log_path}）")

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
        # GPU メモリが解放されるのを少し待つ
        time.sleep(3)

    def _log_tail(self, lines: int = 5) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def ocr_page(self, page) -> str:
        import httpx
        import pymupdf

        zoom = LIGHTON_LONGEST_PX / max(page.rect.width, page.rect.height)
        png = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": "grounding"},   # 図や見出しの位置も出させる
        ]
        # モデルカードの推奨値（temperature 0.2, top_p 0.9）
        r = httpx.post(self.url + "/v1/chat/completions", timeout=600, json={
            "model": "lighton-ocr", "messages": [{"role": "user", "content": content}],
            "max_tokens": 6144, "temperature": 0.2, "top_p": 0.9,
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _run_lighton(pdf_path, pages, progress, log_path, recheck, on_message) -> Dict[int, str]:
    import pymupdf

    pages, recheck = list(pages), [p for p in recheck if p not in set(pages)]
    results: Dict[int, str] = {}
    server = LightOnServer(log_path)
    lock = threading.Lock()   # pymupdf の Document は同時に触らない

    with pymupdf.open(pdf_path) as doc:
        def one(p):
            with lock:
                page = doc[p]
            return p, server.ocr_page(page)

        def run(targets, label):
            done = 0
            if progress:
                progress(0, len(targets))
            with ThreadPoolExecutor(LIGHTON_PARALLEL) as ex:
                for p, text in ex.map(one, targets):
                    results[p] = text
                    done += 1
                    if progress:
                        progress(done, len(targets))

        try:
            if on_message:
                on_message(f"{engine_name()} を起動しています")
            server.start()
            if recheck:
                # 文字の多いページを数ページ選んで OCR し、既存の文字と比べる
                samples = sorted(recheck, key=lambda p: -len(doc[p].get_text()))[:REOCR_SAMPLES]
                if on_message:
                    on_message("PDF に入っている文字の精度を確かめています")
                run(samples, "check")
                layer = "\n".join(doc[p].get_text() for p in samples)
                found = "\n".join(results[p] for p in samples)
                agreement = text_layer_agreement(layer, found)
                if agreement < REOCR_AGREEMENT:
                    if on_message:
                        on_message(f"PDF の文字は精度が低い（一致率 {agreement:.0%}）ので OCR し直します")
                    pages += [p for p in recheck if p not in results]
                else:
                    # 既存の文字で十分なので、確かめに使ったページも既存の文字を使う
                    for p in samples:
                        results.pop(p, None)
            if on_message and pages:
                on_message(f"{len(pages)} ページを OCR しています")
            run([p for p in pages if p not in results], "ocr")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"OCR に失敗しました: {e}（詳細: {log_path}）") from e
        finally:
            server.stop()
    return results


# ------------------------------------------------------------
# OCR の出力 -> Markdown
# ------------------------------------------------------------
_BLOCK_RE = re.compile(r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]*\])?\s*<\|/det\|>")
_DROP = {"header", "page_number", "footer", "page_header", "page_footer"}


def _split_blocks(text: str):
    """[(種類, [x0, y0, x1, y1] or None, 中身), ...]"""
    matches = list(_BLOCK_RE.finditer(text))
    blocks = []
    for m, nxt in zip(matches, matches[1:] + [None]):
        body = text[m.end():nxt.start() if nxt else len(text)].strip()
        try:
            box = json.loads(m.group(2)) if m.group(2) else None
        except ValueError:
            box = None
        blocks.append((m.group(1), box, body))
    if not matches and text.strip():
        blocks.append(("text", None, text.strip()))
    return blocks


def html_table_to_markdown(table_html: str) -> str:
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.DOTALL | re.IGNORECASE)
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("|", "\\|").replace("\n", " ").strip()
                 for c in cells]
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["|" + "|".join(rows[0]) + "|", "|" + "|".join(["---"] * width) + "|"]
    lines += ["|" + "|".join(r) + "|" for r in rows[1:]]
    return "\n".join(lines)


def _looks_like_code(body: str) -> bool:
    lines = body.splitlines()
    if len(lines) < 3:
        return False
    signs = sum(1 for l in lines if re.match(r"^\s{2,}\S", l) or re.search(r"[{};]\s*$", l))
    return signs >= len(lines) / 3


def _crop_image(page, box, dpi: int, scale: float = 999, pad: float = 0) -> Optional[str]:
    """正規化座標（0〜scale）の範囲をページから切り出し、data URI にする。pad は余白（pt）"""
    import pymupdf

    r = page.rect
    x0, y0, x1, y1 = box
    clip = pymupdf.Rect(r.x0 + r.width * x0 / scale, r.y0 + r.height * y0 / scale,
                        r.x0 + r.width * x1 / scale, r.y0 + r.height * y1 / scale)
    clip = (clip + (-pad, -pad, pad, pad)) & r
    if clip.width < 8 or clip.height < 8:
        return None
    png = page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
    return "data:image/png;base64," + base64.b64encode(png).decode()


# LightOnOCR-3 の grounding 出力: "![種類](x0,y0,x1,y1) 中身"（座標は 0〜1000）
_LIGHTON_RE = re.compile(r"!\[([a-z_]+\+?)\]\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)[ \t]*")
_LIGHTON_DROP = {"header", "footer", "page_number", "header_image", "footer_image"}
_LIGHTON_IMAGES = {"image", "chart"}


def _split_lighton(text: str):
    """[(種類, [x0, y0, x1, y1], 中身, 続きか), ...]"""
    matches = list(_LIGHTON_RE.finditer(text))
    blocks = []
    for m, nxt in zip(matches, matches[1:] + [None]):
        body = text[m.end():nxt.start() if nxt else len(text)].strip()
        label = m.group(1)
        blocks.append((label.rstrip("+"), [int(m.group(i)) for i in range(2, 6)], body,
                       label.endswith("+")))
    if not matches and text.strip():
        blocks.append(("text", None, text.strip(), False))
    return blocks


def _norm_title(body: str) -> str:
    return re.sub(r"[\s#*_\d]+", "", unicodedata.normalize("NFKC", body)).lower()


def running_titles(ocr_texts: Dict[int, str]) -> set:
    """ページ上端の見出しのうち、2 ページ以上に出るもの（柱。本の章名などを各ページの上に刷ったもの）

    LightOnOCR-3 は柱を header ではなく title とすることがあり、そのままでは
    ページごとに見出しが増えてしまう。
    """
    counts = Counter()
    for text in ocr_texts.values():
        if not is_lighton_output(text):
            continue
        for kind, box, body, _ in _split_lighton(text):
            if kind == "title" and box and box[1] < 80 and _norm_title(body):
                counts[_norm_title(body)] += 1
    return {t for t, n in counts.items() if n >= 2}


def lighton_to_markdown(text: str, page=None, with_images: bool = True, image_dpi: int = 200,
                        skip_titles: Optional[set] = None) -> str:
    out = []
    prev_kind = None
    for kind, box, body, cont in _split_lighton(text):
        if kind in _LIGHTON_DROP:
            continue
        if kind == "title" and skip_titles and box and box[1] < 80 and _norm_title(body) in skip_titles:
            continue
        if kind in _LIGHTON_IMAGES:
            # 図はページから切り出す（モデルが付ける短い説明文は使わない）
            if with_images and page is not None and box:
                uri = _crop_image(page, box, image_dpi, scale=1000, pad=3)
                if uri:
                    out.append(f"![]({uri})")
            prev_kind = kind
            continue
        if not body:
            continue
        if kind == "table" and "<table" in body.lower():
            md = html_table_to_markdown(body)
        elif kind == "formula":
            # 数式は訳さず、LaTeX のまま残す
            latex = body.strip().strip("$").strip()
            md = "```latex\n" + latex + "\n```"
        elif kind == "title" and re.match(r"^#*\s*[-*•・]\s", body):
            # 箇条書きの頭（「- Static vs. Dynamic.」など）を見出しにしない
            md = re.sub(r"^#+\s*", "", body)
        elif kind == "title":
            md = body if body.startswith("#") else "# " + " ".join(body.split())
        elif kind == "list":
            # 箇条書きの項目は 1 つずつ段落にする（paper.py は空行で項目を分ける）
            md = re.sub(r"\n(?=\s*([-*•・]|\d+[.)])\s)", "\n\n", body)
        else:
            md = body
        if cont and out and prev_kind == kind and kind not in ("table", "formula", "title"):
            # 段組み・ページをまたいで続く段落はつなげる
            out[-1] = re.sub(r"-$", "", out[-1]) + ("" if out[-1].endswith("-") else " ") + md
        else:
            out.append(md)
        prev_kind = kind
    return "\n\n".join(out)


def is_lighton_output(text: str) -> bool:
    return bool(_LIGHTON_RE.search(text)) and "<|det|>" not in text


def ocr_to_markdown(text: str, page=None, with_images: bool = True, image_dpi: int = 200,
                    skip_titles: Optional[set] = None) -> str:
    """OCR の出力を Markdown にする。page（pymupdf の Page）があれば図を切り出して埋め込む

    skip_titles は running_titles() で求めた柱（各ページの上端に繰り返し出る見出し）
    """
    if is_lighton_output(text):
        return lighton_to_markdown(text, page, with_images, image_dpi, skip_titles)
    out = []
    for kind, box, body in _split_blocks(text):
        if kind in _DROP:
            continue
        if kind == "image":
            if with_images and page is not None and box and len(box) == 4:
                uri = _crop_image(page, box, image_dpi)
                if uri:
                    out.append(f"![]({uri})")
            continue
        if not body:
            continue
        if kind == "title":
            out.append("# " + " ".join(body.split()))
        elif kind == "table" and "<table" in body.lower():
            out.append(html_table_to_markdown(body))
        elif kind == "equation":
            # 数式は訳さず、LaTeX のまま残す
            out.append("```latex\n" + body.strip() + "\n```")
        elif _looks_like_code(body):
            out.append("```\n" + body.rstrip() + "\n```")
        else:
            # 段落の中の改行（行の折り返し）はつなげる。ハイフンで切れた単語は戻す
            para = re.sub(r"-\n(?=[a-z])", "", body)
            out.append(re.sub(r"\s*\n\s*", " ", para).strip())
    return "\n\n".join(out)
