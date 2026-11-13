"""PDF論文を章立てを保ったまま翻訳し、Markdown / HTML に書き出す

流れ:
    PDF --pymupdf4llm--> Markdown --parse_blocks--> ブロック列
        --translate(バッチ)--> 訳文付きブロック --render_markdown / render_html--> 出力

翻訳モデルには依存せず、`translate_many(list[str]) -> list[str]` を外から受け取る。

文書の種類（mode）:
    PAPER    論文。章番号付きの見出しだけを章とみなし、著者・所属と参考文献は原文のまま残す
    DOCUMENT 書籍の電子版・AIとのチャットの書き出しなど。見出しは文字の大きさの順に従い、
             全文を訳す。表の文章も訳し、欄外のアイコン付きの囲み（Note など）は引用にする
"""

import copy
import html
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import pymupdf4llm
from markdown_it import MarkdownIt

PAPER = "paper"
DOCUMENT = "document"

# 欄外アイコン付きの囲みの始まりと終わりの目印（Markdown の画像の場所に入れておく）
NOTE_START = "plamo:note-start"
NOTE_END = "plamo:note-end"


# ------------------------------------------------------------
# ブロック
# ------------------------------------------------------------
@dataclass
class Block:
    kind: str                  # title / heading / para / list / quote / table / image
    text: str = ""             # 原文（装飾を除いたもの）。table / image では Markdown そのまま
    level: int = 0             # heading の深さ (2=章, 3=節, 4=項)
    number: str = ""           # heading の番号 ("1", "2.1", "III", "A")
    marker: str = ""           # list の記号 ("-", "1.")
    translate: bool = True     # False なら原文のまま出す（前付け・参考文献など）
    in_refs: bool = False      # 参考文献の章に属するか
    segments: List[str] = field(default_factory=list)  # 翻訳単位に分割した原文
    translated: Optional[str] = None


# ------------------------------------------------------------
# PDF -> Markdown
# ------------------------------------------------------------
def pdf_to_markdown(
    pdf_path: str,
    with_images: bool = True,
    mode: str = PAPER,
    pages: Optional[Sequence[int]] = None,
    image_dpi: int = 150,
) -> str:
    """pages は 0 始まりのページ番号（None なら全ページ）。image_dpi は図を切り出す解像度"""
    options = dict(
        use_ocr=False,        # テキストを持つPDFならOCR不要（遅い）
        embed_images=with_images,
        show_progress=False,
        pages=list(pages) if pages is not None else None,
    )
    import pymupdf

    # 表などが90度回して載っているページは、向きを直してから解析する（ファイルは書き換えない）
    doc = pymupdf.open(pdf_path)
    upright_rotated_pages(doc, options["pages"])
    try:
        # pymupdf4llm の解析結果（ページごとのブロック）を受け取り、
        # 2カラムの読み順を直してから Markdown にする
        from pymupdf4llm.helpers import document_layout

        parsed = document_layout.parse_document(
            doc, force_text=True, image_dpi=image_dpi, **options
        )
        for page in parsed.pages:
            fix_scanned_pictures(doc[page.page_number - 1], page, image_dpi, with_images)
            page.boxes = order_two_columns(page.boxes, page.width)
            if mode == DOCUMENT:
                page.boxes = mark_side_notes(page.boxes)
        return parsed.to_markdown(
            header=False,     # ページヘッダー・フッターを除く
            footer=False,
            embed_images=with_images,
            show_progress=False,
        )
    except (ImportError, AttributeError, TypeError):
        # pymupdf4llm の内部APIが変わった場合は、読み順の補正なしで変換する
        return pymupdf4llm.to_markdown(
            doc, header=False, footer=False, dpi=image_dpi, **options
        )
    finally:
        doc.close()


def upright_rotated_pages(doc, pages: Optional[Sequence[int]] = None) -> List[int]:
    """本文の大半が90度回転しているページ（縦長のページに横長の表を載せたものなど）の向きを直す

    pymupdf4llm は回転した文字を1行ずつの細切れとして読むため、表の行と列が崩れる。
    ページの表示の向き（/Rotate）を変えると、表を横向きのまま正しく読み取れる。
    ページ上端の柱（ページ番号・章名）は回転していないことが多いので、文字数の多数決で決める。
    向きを直したページの番号（0 始まり）を返す。
    """
    fixed = []
    for i in (pages if pages is not None else range(doc.page_count)):
        page = doc[i]
        counts = {}
        cjk = 0
        for block in page.get_text("dict", flags=0)["blocks"]:
            for line in block.get("lines", []):
                dx, dy = (round(v) for v in line["dir"])
                text = "".join(sp["text"] for sp in line["spans"]).strip()
                counts[(dx, dy)] = counts.get((dx, dy), 0) + len(text)
                if (dx, dy) != (1, 0):
                    cjk += len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
        total = sum(counts.values())
        if total < 50:
            continue
        rotated = total - counts.get((1, 0), 0)
        if rotated and cjk > rotated * 0.5:
            continue    # 日本語・中国語の縦書きの本文は、回転した表ではないので触らない
        # dir は回転前の座標での文字の進む向き。(0,-1) は下から上 → 時計回りに90度回すと読める
        for direction, rotation in (((0, -1), 90), ((0, 1), 270), ((-1, 0), 180)):
            if counts.get(direction, 0) > total * 0.6:
                if page.rotation != rotation:
                    page.set_rotation(rotation)
                    fixed.append(i)
                break
    return fixed


def detect_mode(text: str) -> str:
    """冒頭に Abstract（要旨）か「1 Introduction」があれば論文、なければ書籍・一般文書とみなす"""
    head = text[:12000]
    flags = re.IGNORECASE | re.MULTILINE
    if re.search(r"^[#*_\s]*(abstract|要旨|概要)\b", head, flags):
        return PAPER
    if re.search(r"^[#*_\s]*(1|I)\.?\s+introduction\b", head, flags):
        return PAPER
    return DOCUMENT


def detect_pdf_mode(pdf_path: str, pages: Optional[Sequence[int]] = None) -> str:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        numbers = list(pages) if pages is not None else range(doc.page_count)
        text = "\n".join(doc[i].get_text() for i in list(numbers)[:3])
    return detect_mode(text)


# ------------------------------------------------------------
# PDF のしおり（目次情報）
# ------------------------------------------------------------
@dataclass
class Chapter:
    title: str
    start: int                 # 0 始まりのページ番号
    end: int                   # この直前のページまで
    entries: list = field(default_factory=list)   # [(深さ, 見出し)]（章の中のしおり）


# 既定では訳さない、本文以外のページ
_FRONT_BACK_MATTER = re.compile(
    r"^(cover|title page|copyright|table of contents|contents|index|"
    r"other books|packt page|about the publisher|advertisement|目次|索引|奥付)\b",
    re.IGNORECASE,
)


def pdf_chapters(pdf_path: str) -> List[Chapter]:
    """しおりの一番上の階層を章として返す（しおりの無い PDF なら空）"""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        toc = [(lv, title.strip(), page - 1) for lv, title, page in doc.get_toc()
               if page >= 1 and title.strip()]
        page_count = doc.page_count
    if not toc:
        return []
    top = min(lv for lv, _, _ in toc)
    chapters: List[Chapter] = []
    for lv, title, page in toc:
        if lv == top:
            if chapters and chapters[-1].start == page:
                chapters[-1].title += f" / {title}"     # 同じページから始まる章はまとめる
            else:
                chapters.append(Chapter(title, page, page_count))
        elif chapters:
            chapters[-1].entries.append((lv - top + 1, title))
        else:
            chapters.append(Chapter(title, page, page_count, []))
    for prev, nxt in zip(chapters, chapters[1:]):
        prev.end = max(nxt.start, prev.start + 1)
    return chapters


def is_body_chapter(chapter: Chapter) -> bool:
    return not _FRONT_BACK_MATTER.match(chapter.title)


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def apply_outline(blocks: List[Block], chapter: Chapter, offset: int = 1):
    """見出しの階層を、PDF のしおりに合わせる

    pymupdf4llm は文字の大きさから見出しの階層を決めるため、本1冊を通すと
    章によって階層がずれたり、小節がすべて同じ深さになったりする。
    しおりに載っている見出しはしおりの深さを使い、載っていない見出しは直前の見出しの1段下にする。
    章タイトル（しおりの一番上）は、しおりの書き方（"Chapter 2: ..."）にそろえる。
    """
    entries = [(1, chapter.title)] + chapter.entries
    keys = [_norm(t) for _, t in entries]
    pos = 0
    last_level = offset + 1
    chapter_heading_done = False

    for b in blocks:
        if b.kind != "heading":
            continue
        key = _norm(b.text)
        match = None
        if len(key) >= 3:
            # 同じ名前の見出し（各章の "Technical requirements" など）を取り違えないよう、前から順に探す
            for i in range(pos, min(pos + 40, len(entries))):
                if keys[i] == key or (len(key) >= 8 and keys[i].endswith(key)):
                    match = i
                    break
        if match is not None:
            depth, title = entries[match]
            pos = match + 1
            b.level = min(depth + offset, 5)
            if match == 0:
                b.text = title.split(" / ")[0]
                chapter_heading_done = True
        else:
            b.level = min(last_level + 1, 5)
        last_level = b.level

    if not chapter_heading_done:
        # 章タイトルが本文から取れなかったとき（章の扉が画像など）は、先頭に入れる
        first = next((i for i, b in enumerate(blocks) if b.kind != "title"), len(blocks))
        blocks.insert(first, Block("heading", chapter.title.split(" / ")[0],
                                   level=min(1 + offset, 5)))


_MARGIN_CLASSES = ("page-header", "page-footer")


def mark_side_notes(boxes):
    """欄外の小さなアイコンと、その右に並ぶ囲み（Note・Tip など）を見つける

    書籍では、囲みの左の余白にアイコンが置かれていることが多い。pymupdf4llm はこれを
    図として切り出すため、アイコン（と囲みの文字の切れ端）の画像が大量に混ざってしまう。

    - 本文の左端より左にはみ出した、幅の小さい図で、
    - その右に高さの重なる文章のブロックがあるもの

    をアイコンとみなし、画像の代わりに囲みの始まり・終わりの目印を入れる。
    """
    texts = [b for b in boxes if b.boxclass not in _MARGIN_CLASSES + ("picture", "formula")]
    if not texts:
        return boxes
    left_margin = min(b.x0 for b in texts)
    body_width = max(b.x1 for b in texts) - left_margin

    out = list(boxes)
    for icon in [b for b in boxes if b.boxclass == "picture"]:
        if icon.x1 - icon.x0 > body_width * 0.2 or icon.x0 >= left_margin - 2:
            continue
        beside = [b for b in texts
                  if b.x0 >= icon.x1 - 10 and b.y0 < icon.y1 and b.y1 > icon.y0]
        if not beside:
            continue
        icon.image = NOTE_START
        icon.textlines = None
        end = copy.copy(icon)
        end.image = NOTE_END
        # 目印で囲みの文章を挟む（同じ値のブロックがあり得るので、位置は is で探す）
        out = [b for b in out if b is not icon]
        positions = [i for i, b in enumerate(out) if any(b is x for x in beside)]
        out.insert(max(positions) + 1, end)
        out.insert(min(positions), icon)
    return out


def _is_scanned_with_text(page) -> bool:
    """ページ全体が1枚の画像で、その上に（スキャナの OCR などの）文字が載っているページか"""
    import pymupdf

    area = page.rect.width * page.rect.height
    return any(pymupdf.Rect(info["bbox"]).get_area() >= area * 0.9
               for info in page.get_image_info())


def _ink_bbox(page, region, dpi: int = 100, threshold: int = 180):
    """region の中で、インク（暗い画素）のある範囲を返す。無ければ None"""
    import numpy as np
    import pymupdf

    if region.is_empty or region.width < 4 or region.height < 4:
        return None
    pix = page.get_pixmap(clip=region, dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    dark = img < threshold
    # スキャンのごみ（点）を拾わないよう、暗い画素が 2 つ以上ある行・列だけを数える
    rows = np.where(dark.sum(axis=1) >= 2)[0]
    cols = np.where(dark.sum(axis=0) >= 2)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    sx, sy = region.width / pix.width, region.height / pix.height
    return pymupdf.Rect(region.x0 + cols[0] * sx, region.y0 + rows[0] * sy,
                        region.x0 + (cols[-1] + 1) * sx, region.y0 + (rows[-1] + 1) * sy)


def fix_scanned_pictures(page, layout, image_dpi: int, with_images: bool, pad: float = 4) -> None:
    """スキャンした本（全面画像＋OCR の文字）で、図の範囲を画素から求め直す

    pymupdf4llm のレイアウト解析は、ページの文字と線（ベクター図形）から図の範囲を決める。
    スキャンした PDF では図の線は画像の一部なので見えず、OCR が拾えた図中のラベルの範囲だけが
    「図」になる（例: Hard Real-Time Computing Systems の図 2.14 で、右半分だけが切り出される）。
    そこで、図の上下にある本文・キャプションの間を画像として調べ、インクのある範囲まで図を広げる。
    """
    import pymupdf

    pictures = [b for b in layout.boxes if b.boxclass == "picture"]
    if not pictures or not _is_scanned_with_text(page):
        return
    others = [b for b in layout.boxes if b.boxclass not in ("picture",) + _MARGIN_CLASSES]
    body = [b for b in others if b.boxclass == "text"] or others
    col_x0 = min((b.x0 for b in body), default=page.rect.x0 + 36)
    col_x1 = max((b.x1 for b in body), default=page.rect.x1 - 36)
    margins = [b for b in layout.boxes if b.boxclass in _MARGIN_CLASSES]
    page_top = max((b.y1 for b in margins if b.y1 < page.rect.height / 2), default=page.rect.y0)
    page_bottom = min((b.y0 for b in margins if b.y0 > page.rect.height / 2), default=page.rect.y1)

    # レイアウト解析がキャプションや本文を見落とすことがあるので、文字の行からも境界を拾う:
    # 「Figure 2.4 …」「図 3 …」のようなキャプションの行と、本文の幅の半分を超える行
    caption_re = re.compile(r"^\s*(Fig(ure|\.)|Table|図|表)\s*\d", re.IGNORECASE)
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            r = pymupdf.Rect(line["bbox"])
            if text and (caption_re.match(text) or r.width > (col_x1 - col_x0) * 0.5):
                lines.append(r)

    # キャプションのかたまり（「Figure 2.9 Two tasks …」の 1〜2 行）
    captions = [pymupdf.Rect(bl[:4]) for bl in page.get_text("blocks")
                if bl[6] == 0 and caption_re.match(bl[4])]

    def crop(rect):
        if not with_images:
            return None
        pix = page.get_pixmap(clip=rect, dpi=image_dpi)
        return None if pymupdf.IRect(pix.irect).is_empty else pix.tobytes("png")

    def caption_box(rect):
        from pymupdf4llm.helpers.document_layout import LayoutBox, get_raw_lines
        box = LayoutBox(rect.x0, rect.y0, rect.x1, rect.y1, "caption")
        box.textlines = [{"bbox": l[0], "spans": l[1]} for l in get_raw_lines(
            textpage=None, blocks=layout.fulltext, clip=rect, ignore_invisible=False)]
        return box

    done = []
    replace = {}      # 図の箱 -> 置き換える箱の列（キャプションで分けたとき）
    for pic in pictures:
        rect = pymupdf.Rect(pic.x0, pic.y0, pic.x1, pic.y1)
        # 図の上下で、いちばん近い本文・キャプション・表などの端までを調べる範囲にする
        bounds = [pymupdf.Rect(b.x0, b.y0, b.x1, b.y1) for b in others]
        bounds += [r for r in lines if not r.intersects(rect)]
        above = [r.y1 for r in bounds if r.y1 <= rect.y0 + 2]
        below = [r.y0 for r in bounds if r.y0 >= rect.y1 - 2]
        top = max(above, default=page_top) + 1
        bottom = min(below, default=page_bottom) - 1
        region = pymupdf.Rect(min(col_x0, rect.x0) - pad, top, max(col_x1, rect.x1) + pad, bottom)
        region &= page.rect
        ink = _ink_bbox(page, region)
        if ink is None:
            continue
        new = (rect | ink) + (-pad, -pad, pad, pad)
        new &= region | rect
        # 同じ図の一部として検出された別の図の範囲は、1つにまとめる
        if any(abs(new & d) >= 0.8 * abs(new) for d in done):
            pic.boxclass = "_merged"
            continue
        done.append(new)
        # 広げた図の中にある、図のラベルの文字（本文として訳されてしまう）を除く
        for b in layout.boxes:
            if b is not pic and b.boxclass in ("text", "picture") and \
                    pymupdf.Rect(b.x0, b.y0, b.x1, b.y1) in new:
                b.boxclass = "_merged"

        # レイアウト解析が、キャプションごと（ときには上下 2 つの図をまとめて）1 つの図に
        # することがある。図の中にあるキャプションで上下に分け、キャプションは文章として訳す
        inner = sorted((c for c in captions if abs(c & new) >= 0.8 * abs(c)), key=lambda c: c.y0)
        if not inner:
            pic.x0, pic.y0, pic.x1, pic.y1 = new.x0, new.y0, new.x1, new.y1
            pic.image = crop(new)
            continue
        parts, y = [], new.y0
        for cap in inner + [None]:
            seg = pymupdf.Rect(new.x0, y, new.x1, cap.y0 - 1 if cap else new.y1)
            seg_ink = _ink_bbox(page, seg)
            if seg_ink is not None and seg_ink.height > 8:
                fig = (seg_ink + (-pad, -pad, pad, pad)) & seg
                from pymupdf4llm.helpers.document_layout import LayoutBox
                box = LayoutBox(fig.x0, fig.y0, fig.x1, fig.y1, "picture")
                box.image = crop(fig)
                box.textlines = []
                parts.append(box)
            if cap:
                parts.append(caption_box(cap))
                y = cap.y1 + 1
        replace[id(pic)] = parts

    boxes = []
    for b in layout.boxes:
        if b.boxclass == "_merged":
            continue
        boxes.extend(replace.get(id(b), [b]))
    layout.boxes = boxes


def order_two_columns(boxes, page_width):
    """2カラムのページで、ブロックを「左カラム → 右カラム」の順に並べ直す

    pymupdf4llm は、左右のカラムの段落を高さ順に交互に並べてしまうことがある
    （論文の1ページ目で、左にAbstract、右にIntroductionがあるときなど）。

    - 横幅いっぱいのブロック（タイトル・著者欄・2カラムにまたがる図表）と、
      それに高さが重なるブロックは「帯」として、上から順・左から順に置く
    - 帯と帯のあいだは、左カラムを上から下へ、続いて右カラムを上から下へ置く
    - 2カラムと判定できないページは元の順番のまま返す
    """
    mid = page_width / 2
    tol = page_width * 0.05

    def is_left(b):
        return b.x1 <= mid + tol

    def is_right(b):
        return b.x0 >= mid - tol

    body = [b for b in boxes if b.boxclass not in _MARGIN_CLASSES]
    headers = [b for b in boxes if b.boxclass == "page-header"]
    footers = [b for b in boxes if b.boxclass == "page-footer"]

    # 左右それぞれに2つ以上ブロックがあるページだけを2カラムとみなす
    if sum(map(is_left, body)) < 2 or sum(map(is_right, body)) < 2:
        return boxes

    # 横幅いっぱいのブロックから「帯」（高さの範囲）を作る
    bands = []
    for b in sorted((b for b in body if not is_left(b) and not is_right(b)),
                    key=lambda b: b.y0):
        if bands and b.y0 <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], b.y1)
        else:
            bands.append([b.y0, b.y1])

    def band_of(b):
        """ブロックが帯と高さで半分以上重なれば、その帯の番号を返す"""
        height = max(b.y1 - b.y0, 1e-6)
        for i, (y0, y1) in enumerate(bands):
            if min(b.y1, y1) - max(b.y0, y0) >= height / 2:
                return i
        return None

    # 帯（i）と、帯 i の直前の区間（"gap", i）にブロックを振り分ける。
    # 最後の帯より下は ("gap", len(bands))
    groups = {}
    for b in body:
        i = band_of(b)
        if i is not None:
            key = (i, 1)
        else:
            gap = sum(1 for y0, _ in bands if y0 <= b.y0)
            key = (gap, 0)
        groups.setdefault(key, []).append(b)

    ordered = []
    for key in sorted(groups):
        members = groups[key]
        if key[1] == 1:   # 帯: 上から順、同じ高さなら左から
            ordered += sorted(members, key=lambda b: (round(b.y0), b.x0))
        else:             # 帯のあいだ: 左カラム → 右カラム
            ordered += sorted((b for b in members if is_left(b)), key=lambda b: b.y0)
            ordered += sorted((b for b in members if not is_left(b)), key=lambda b: b.y0)

    return headers + ordered + footers


# ------------------------------------------------------------
# 装飾の除去
# ------------------------------------------------------------
_TAG_RE = re.compile(r"</?(?:mark|u|b|i|em|strong|span|sub)[^>]*>")
_SUP_RE = re.compile(r"<sup>(.*?)</sup>")


def clean_inline(text: str) -> str:
    """pymupdf4llm が付ける太字・斜体・下線などを除いてプレーンテキストにする"""
    text = text.replace("�", "")                 # ORCIDアイコンなどの化け
    text = _SUP_RE.sub(r"\1", text)
    text = _TAG_RE.sub("", text)
    text = text.replace("**", "").replace("~~", "")
    # _斜体_ だけを外す（snake_case の _ は残す）
    text = re.sub(r"(?<![\w\\])_(?=\S)(.+?)(?<=\S)_(?!\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ------------------------------------------------------------
# 見出しの判定
# ------------------------------------------------------------
_KNOWN_SECTIONS = re.compile(
    r"^(abstract|introduction|related works?|background|preliminar(y|ies)|"
    r"conclusions?|discussion|evaluation|experiments?|references|bibliography|"
    r"acknowledge?ments?|appendix|appendices|future work|limitations)\b",
    re.IGNORECASE,
)
# 1語だけの見出しは原則として訳さない（固有名詞の誤訳を防ぐ）が、
# 論文でよく使われる一般的な章名は訳す
_COMMON_SINGLE_WORD_SECTIONS = re.compile(
    r"^(results?|methods?|methodology|implementation|design|overview|motivation|"
    r"summary|analysis|setup|evaluation|discussion|conclusions?|introduction|"
    r"background|preliminaries|approach|experiments?|preface|foreword|exercises?|"
    r"questions|glossary|contents|index|acknowledge?ments?|appendix|references|"
    r"prerequisites|requirements|installation|usage|examples?|notes?|tips?)$",
    re.IGNORECASE,
)
_REFERENCES = re.compile(r"^(references|bibliography|参考文献)\s*$", re.IGNORECASE)

_ROMAN = r"(?:X{0,3})(?:IX|IV|V?I{0,3})"


def classify_heading(text: str):
    """見出しなら (level, number, title)、章・節でなければ None"""
    if len(text) > 120:
        return None

    m = re.match(rf"^({_ROMAN})\.\s+(\S.*)$", text)          # IEEE: "III. METHOD"
    if m and m.group(1):
        return 2, m.group(1), m.group(2)

    m = re.match(r"^([A-Z])\.\s+(\S.*)$", text)               # IEEE: "A. Model"
    if m:
        return 3, m.group(1), m.group(2)

    m = re.match(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Za-z].*)$", text)  # "2.1 Model"
    if m:
        number = m.group(1)
        return min(2 + number.count("."), 5), number, m.group(2)

    m = re.match(r"^(appendix\s+[A-Z0-9]+)[.:]?\s*(.*)$", text, re.IGNORECASE)
    if m:
        return 2, m.group(1), m.group(2) or m.group(1)

    if _KNOWN_SECTIONS.match(text):
        return 2, "", text

    return None


# ------------------------------------------------------------
# Markdown -> ブロック列
# ------------------------------------------------------------
_LIST_RE = re.compile(r"^\s*([-*+•]|\d{1,3}[.)])\s+(.*)$", re.DOTALL)
_END_OF_SENTENCE = re.compile(r"[.!?:;。．！？)\]”\"']\s*$")


def _split_raw_blocks(md: str) -> List[str]:
    """空行区切り。ただし表（| で始まる行の連続）とコード（``` から ``` まで）は1ブロックにまとめる"""
    blocks, current = [], []
    in_code = False

    def flush():
        if current:
            blocks.append("\n".join(current))
            current.clear()

    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            if not in_code:
                flush()
            current.append(line)
            in_code = not in_code
            if not in_code:
                flush()
            continue
        if in_code:
            current.append(line)    # コードの中の空行は区切りにしない
            continue
        if not line.strip():
            flush()
            continue
        if current and line.lstrip().startswith("|") != current[-1].lstrip().startswith("|"):
            flush()
        current.append(line)
    flush()
    return blocks


_PICTURE_TEXT_RE = re.compile(
    r"(<!--|-{3,})\s*Start of picture text\s*(-->|-{3,}).*?(<!--|-{3,})\s*End of picture text\s*(-->|-{3,})",
    re.DOTALL,
)
_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^)\s]*\)?")   # 閉じ括弧のない（途中で切れた）画像も含む
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _image_src(markdown_image: str) -> str:
    m = re.search(r"\(([^)\s]*)\)\s*$", markdown_image)
    return m.group(1) if m else ""


def parse_blocks(md: str, mode: str = PAPER, title: Optional[str] = None) -> List[Block]:
    """Markdown をブロック列にする

    mode=DOCUMENT では title（ファイル名など）を文書の題として先頭に置く。
    """
    blocks: List[Block] = []
    document = mode == DOCUMENT
    in_note = False
    chapter_number = ""

    # pymupdf4llm が図の中の文字を書き出した部分。図は画像として埋め込むので訳さずに除く
    # （訳すと囲みの印まで訳され、図のラベルが細切れの文として並んでしまう）
    md = _PICTURE_TEXT_RE.sub("\n\n", md)

    if document:
        if title:
            blocks.append(Block("title", title, translate=False))
        # 文中に埋め込まれた画像（base64 を含む）を独立させ、翻訳モデルに送らないようにする
        md = _IMAGE_RE.sub(lambda m: f"\n\n{m.group(0)}\n\n", md)

    for raw in _split_raw_blocks(md):
        stripped = raw.strip()

        if stripped.startswith("|"):
            if document:
                blocks.append(_table_block(stripped))
            else:
                blocks.append(Block("table", stripped, translate=False))
            continue
        if stripped.startswith("!["):
            src = _image_src(stripped)
            if src in (NOTE_START, NOTE_END):
                in_note = src == NOTE_START
                continue
            blocks.append(Block("image", stripped, translate=False))
            continue
        if stripped.startswith("```"):
            prev = blocks[-1] if blocks else None
            if document and prev and prev.kind == "table" and prev.text.startswith("```"):
                # ページの境目で切れたコードを1つにつなげる
                prev.text = prev.text.rstrip()[:-3].rstrip("\n") + "\n" + \
                    re.sub(r"^```[^\n]*\n?", "", stripped)
            else:
                blocks.append(Block("table", stripped, translate=False))  # コードはそのまま
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped, re.DOTALL)
        if m and document:
            text = clean_inline(m.group(2).replace("\n", " "))
            if not text or text.lower() == "blank":
                continue
            if re.fullmatch(r"(chapter\s+)?(\d{1,3}|[IVXLC]{1,6})", text, re.IGNORECASE):
                # 章の扉の大きな章番号（"# 2"）は、続く章タイトルの頭に付ける
                chapter_number = text
                continue
            if not re.search(r"[^\W\d_]{2}", _INLINE_CODE_RE.sub("", text)):
                # "# `*DOCUMENTATION*`" のような、コードの行が見出しと誤認されたもの
                blocks.append(Block("para", text, translate=False))
                continue
            if re.match(r"^(table|fig(ure)?\.?|listing|表|図)\s*[\dA-Z]+([.\-]\d+)*\b", text,
                        re.IGNORECASE):
                # 表・図のキャプションが大きな文字で見出しと誤認されたもの
                _append_paragraph(blocks, text)
                continue
            if chapter_number:
                text = f"{chapter_number} {text}"
                chapter_number = ""
            # 題（title）を # で出すので、題があるときは見出しを1段下げる
            offset = 1 if title else 0
            blocks.append(Block("heading", text, level=min(len(m.group(1)) + offset, 5),
                                translate=_translatable_title(text)))
            continue
        if m:
            text = clean_inline(m.group(2))
            if not text or text.lower() == "blank":
                continue
            if len(m.group(1)) == 1 and not any(b.kind == "title" for b in blocks):
                blocks.append(Block("title", text))
                continue
            heading = classify_heading(text)
            if heading:
                level, number, title = heading
                # 1語だけの見出し（Rust, GEDF, MRM など）は文脈なしだと誤訳しやすい
                # （例: "Rust" -> 「錆」）ので原文のままにする
                blocks.append(Block("heading", title, level=level, number=number,
                                    translate=_translatable_title(title)))
            else:
                # 著者名・「TABLE II」などは見出し扱いしない
                blocks.append(Block("para", text))
            continue

        if stripped.startswith(">"):
            text = clean_inline(re.sub(r"^>\s?", "", stripped, flags=re.MULTILINE))
            if text:
                blocks.append(Block("quote", text))
            continue

        m = _LIST_RE.match(stripped)
        if m:
            text = clean_inline(m.group(2).replace("\n", " "))
            if text:
                marker = "-" if not m.group(1)[0].isdigit() else m.group(1)
                blocks.append(Block("list", text, marker=marker))
            continue

        text = clean_inline(stripped.replace("\n", " "))
        if text:
            if in_note:
                blocks.append(Block("quote", text))
            else:
                _append_paragraph(blocks, text)

    if not document:
        _mark_untranslated(blocks)
    return blocks


def _translatable_title(text: str) -> bool:
    """1語だけの見出し（Rust, GEDF, MRM など）は文脈なしだと誤訳しやすい
    （例: "Rust" -> 「錆」）ので原文のままにする。よく使われる一般的な章名は訳す"""
    if len(text.split()) != 1 or _CJK_CHARS.search(text):
        # 日中韓の見出しは空白で語に分かれないので、1語かどうかでは判断しない
        return True
    return bool(_KNOWN_SECTIONS.match(text) or _COMMON_SINGLE_WORD_SECTIONS.match(text))


_CJK_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


# ------------------------------------------------------------
# 表（書籍・一般文書のみ、文章の入ったセルを訳す）
# ------------------------------------------------------------
def _split_row(line: str) -> List[str]:
    return line.strip().strip("|").split("|")


def _cell_text(cell: str) -> str:
    return clean_inline(re.sub(r"<br\s*/?>", " ", cell))


def _is_prose_cell(cell: str) -> bool:
    """コードや数値ではなく、2語以上の文章が入ったセル"""
    words = re.findall(r"[^\W\d_]{2,}", _INLINE_CODE_RE.sub("", _cell_text(cell)))
    return len(words) >= 2


def _table_block(md_table: str) -> Block:
    b = Block("table", md_table)
    for line in md_table.splitlines():
        if re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line):     # |---|---|
            continue
        b.segments += [_cell_text(c) for c in _split_row(line) if _is_prose_cell(c)]
    # 書籍の目次（"Preface ....... xxiii"、"Cloning a Git tree • 22"）は訳さない
    leaders = len(re.findall(r"\.{5,}|•\s*\d+", md_table))
    if leaders >= max(3, md_table.count("\n") // 2):
        b.segments = []
    b.translate = bool(b.segments)
    return b


def _render_table(b: Block, translated_cells: List[str]) -> str:
    cells = iter(translated_cells)
    lines = []
    for line in b.text.splitlines():
        if re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line):
            lines.append(line)
            continue
        row = []
        for c in _split_row(line):
            if _is_prose_cell(c):
                t = next(cells, None)
                c = t.replace("|", "\\|").replace("\n", " ") if t else c
            row.append(c)
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def _append_paragraph(blocks: List[Block], text: str):
    """ページ・段組みの境目で切れた段落を前の段落につなげる"""
    if text[0].islower():
        # 図・表・脚注を挟んで切れていることがあるので、それらは飛ばして探す
        for prev in reversed(blocks):
            if prev.kind in ("image", "table", "quote"):
                continue
            if prev.kind in ("para", "list") and not _END_OF_SENTENCE.search(prev.text):
                prev.text = f"{prev.text} {text}"
                return
            break
    blocks.append(Block("para", text))


def _mark_untranslated(blocks: List[Block]):
    """前付け（著者・所属）と参考文献は原文のまま残す"""
    has_sections = sum(b.kind == "heading" for b in blocks) >= 2
    in_front = has_sections        # 論文でなければ全文を訳す
    in_refs = False
    in_abstract = False

    for b in blocks:
        if b.kind == "heading":
            in_front = False
            if b.level <= 2:
                in_refs = bool(_REFERENCES.match(b.text))
            if in_refs and not _REFERENCES.match(b.text):
                b.translate = False
            continue
        if b.kind in ("table", "image"):
            continue
        if b.kind == "title":
            continue
        if in_front:
            # アブストラクト（以降の段落も含む）とキーワードは訳す。著者・所属は原文のまま
            if re.match(r"^(abstract|index terms|keywords)\b", b.text, re.IGNORECASE):
                in_abstract = True
            b.translate = in_abstract
        elif in_refs:
            b.translate = False
            b.in_refs = True


# ------------------------------------------------------------
# 翻訳
# ------------------------------------------------------------
def split_long(text: str, max_chars: int = 1500) -> List[str]:
    """長すぎる段落を文単位で分割する"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?。])\s+(?=[A-Z0-9\[(“\"])", text)
    parts, current = [], ""
    for s in sentences:
        while len(s) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(s[:max_chars])
            s = s[max_chars:]
        if current and len(current) + len(s) + 1 > max_chars:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}" if current else s
    if current:
        parts.append(current)
    return parts


def translate_blocks(
    blocks: List[Block],
    translate_many: Callable[[List[str], Callable[[int, int], None]], List[str]],
    translate_references: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
):
    targets = []
    for b in blocks:
        if b.kind == "image":
            continue
        if b.kind == "table":
            # 書籍・一般文書の表は、_table_block で文章のセルを segments に入れてある
            if b.translate and b.segments:
                targets.append(b)
            continue
        if not b.translate and not (translate_references and b.in_refs):
            continue
        b.segments = split_long(b.text)
        targets.append(b)

    texts = [s for b in targets for s in b.segments]
    results = translate_many(texts, progress) if texts else []

    i = 0
    for b in targets:
        n = len(b.segments)
        if b.kind == "table":
            b.translated = _render_table(b, results[i:i + n])
        else:
            b.translated = "".join(results[i:i + n]) if n > 1 else results[i]
        i += n


# ------------------------------------------------------------
# 出力
# ------------------------------------------------------------
def _heading_line(b: Block) -> str:
    title = b.translated or b.text
    if not b.number:
        number = ""
    elif b.number.isdigit() or "." in b.number or b.number.lower().startswith("appendix"):
        number = f"{b.number} "        # "1 ", "2.1 "
    else:
        number = f"{b.number}. "       # IEEE形式 "III. ", "A. "
    # 章名は原文も残しておく（論文を読むときに対応が取りやすい）
    orig = f"（{b.text}）" if b.translated and b.translated != b.text else ""
    return f"{'#' * b.level} {number}{title}{orig}"


def _original_details(text: str) -> str:
    return f"<details><summary>原文</summary>\n\n{text}\n\n</details>"


def render_markdown(blocks: List[Block], source_name: str, bilingual: bool = False) -> str:
    out: List[str] = []
    pending_originals: List[str] = []   # 箇条書きの原文はリストの後ろにまとめる
    prev_kind = ""

    def flush_list_originals():
        if pending_originals:
            out.append(_original_details("\n".join(f"- {t}" for t in pending_originals)))
            pending_originals.clear()

    for b in blocks:
        if b.kind != "list":
            flush_list_originals()

        if b.kind == "title":
            out.append(f"# {b.translated or b.text}")
            if b.translated:
                out.append(f"*{b.text}*")
            out.append(f"> 翻訳元: `{source_name}`（PLaMo 3 Translate による機械翻訳）")
        elif b.kind == "heading":
            out.append(_heading_line(b))
        elif b.kind in ("table", "image"):
            out.append(b.translated or b.text)
            if b.kind == "table" and bilingual and b.translated:
                out.append(_original_details(b.text))
        elif b.kind == "list":
            line = f"{b.marker} {b.translated or b.text}"
            # 連続する箇条書きは空行を挟まず1つのリストにする
            if prev_kind == "list":
                out[-1] += "\n" + line
            else:
                out.append(line)
            if bilingual and b.translated:
                pending_originals.append(b.text)
        elif b.kind == "quote":
            line = "> " + (b.translated or b.text)
            # 連続する引用（書籍の囲みなど）は1つの引用にまとめる（原文を併記するときは分ける）
            if prev_kind == "quote" and not bilingual:
                out[-1] += "\n>\n" + line
            else:
                out.append(line)
            if bilingual and b.translated:
                out.append(_original_details(b.text))
        else:
            out.append(b.translated or b.text)
            if bilingual and b.translated:
                out.append(_original_details(b.text))
        prev_kind = b.kind

    flush_list_originals()
    return "\n\n".join(out) + "\n"


_HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #fdfdfc; --fg: #1f2328; --muted: #656d76; --line: #d8dee4;
  --accent: #0a5bd3; --side: #f5f6f8; --code: #f0f2f4;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #15171a; --fg: #e3e5e8; --muted: #9aa1a9; --line: #30363d;
    --accent: #6ea8fe; --side: #1c1f23; --code: #23272d;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--fg);
  font-family: "Hiragino Kaku Gothic ProN", "Noto Sans JP", "Yu Gothic UI", Meiryo, sans-serif;
  line-height: 1.9; font-size: 16px;
}}
.layout {{ display: flex; align-items: flex-start; }}
nav {{
  position: sticky; top: 0; height: 100vh; overflow-y: auto; width: 300px; flex: none;
  background: var(--side); border-right: 1px solid var(--line); padding: 24px 16px; font-size: 13.5px;
  line-height: 1.6;
}}
nav .toc-title {{ font-weight: 700; margin-bottom: 8px; color: var(--muted); }}
nav a {{ display: block; color: var(--fg); text-decoration: none; padding: 3px 6px; border-radius: 4px; }}
nav a:hover {{ background: var(--line); }}
nav a.l3 {{ padding-left: 20px; color: var(--muted); }}
nav a.l4, nav a.l5 {{ padding-left: 34px; color: var(--muted); font-size: 12.5px; }}
main {{ flex: 1; min-width: 0; max-width: 860px; margin: 0 auto; padding: 32px 24px 96px; }}
h1 {{ font-size: 1.7em; line-height: 1.4; margin: 0 0 4px; }}
h1 + p em {{ color: var(--muted); font-style: normal; }}
h2 {{ font-size: 1.35em; margin-top: 2.4em; padding-bottom: 6px; border-bottom: 2px solid var(--line); }}
h3 {{ font-size: 1.15em; margin-top: 2em; }}
h4, h5 {{ font-size: 1.02em; margin-top: 1.6em; }}
h2, h3, h4, h5 {{ scroll-margin-top: 16px; }}
p {{ margin: 0.9em 0; text-align: justify; }}
blockquote {{ margin: 1em 0; padding: 4px 14px; border-left: 4px solid var(--line); color: var(--muted); font-size: 0.92em; }}
img {{ max-width: 100%; height: auto; display: block; margin: 16px auto; background: #fff; border-radius: 4px;
  cursor: zoom-in; }}
pre {{ background: var(--code); padding: 10px 14px; border-radius: 6px; overflow-x: auto; line-height: 1.5; }}
pre code {{ background: none; padding: 0; font-size: 0.85em; }}
.zoom {{ position: fixed; inset: 0; background: rgba(0,0,0,.85); display: none; overflow: auto;
  cursor: zoom-out; z-index: 10; }}
.zoom.open {{ display: block; }}
.zoom img {{ max-width: none; margin: 24px auto; cursor: zoom-out; }}
table {{ border-collapse: collapse; margin: 16px 0; font-size: 0.85em; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid var(--line); padding: 4px 8px; }}
th {{ background: var(--side); }}
code {{ background: var(--code); padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
details {{ margin: -0.4em 0 1.2em; font-size: 0.88em; color: var(--muted); }}
details summary {{ cursor: pointer; user-select: none; width: fit-content; }}
details[open] {{ border-left: 3px solid var(--line); padding-left: 12px; }}
.toggle {{ position: fixed; right: 16px; bottom: 16px; padding: 8px 12px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--side); color: var(--fg); cursor: pointer; font-size: 13px; }}
@media (max-width: 900px) {{
  nav {{ display: none; }}
  main {{ padding: 20px 16px 80px; }}
}}
</style>
</head>
<body>
<div class="layout">
<nav><div class="toc-title">目次</div>{toc}</nav>
<main>
{body}
</main>
</div>
{toggle}
<div class="zoom" onclick="this.classList.remove('open')"><img alt=""></div>
<script>
// 図をクリックすると原寸で表示する（細かい図を読むため）
document.querySelectorAll('main img').forEach(img => img.addEventListener('click', () => {{
  const z = document.querySelector('.zoom');
  z.querySelector('img').src = img.src;
  z.classList.add('open');
}}));
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.querySelector('.zoom').classList.remove('open');
}});
</script>
</body>
</html>
"""

_TOGGLE = """<button class="toggle" onclick="
  const d=document.querySelectorAll('details'); const open=![...d].every(x=>x.open);
  d.forEach(x=>x.open=open);">原文を全て開閉</button>"""


def render_html(markdown_text: str, title: str) -> str:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = md.parse(markdown_text)

    toc = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag in ("h2", "h3", "h4", "h5"):
            anchor = f"sec-{len(toc)}"
            tok.attrSet("id", anchor)
            text = tokens[i + 1].content
            toc.append(
                f'<a class="l{tok.tag[1]}" href="#{anchor}">{html.escape(text)}</a>'
            )

    body = md.renderer.render(tokens, md.options, {})
    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        toc="\n".join(toc),
        body=body,
        toggle=_TOGGLE if "<details>" in markdown_text else "",
    )
