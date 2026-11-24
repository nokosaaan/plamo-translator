"""ブロック列（paper.parse_blocks の結果）を LaTeX のプロジェクトにする

PDF -> ブロック列 -> 翻訳 のあとに、Markdown / HTML と並ぶ出力形式の 1 つとして使う。

    files = render_latex_project(blocks, "paper.pdf", bilingual=True, split=True)
    # files は {ZIP の中のパス: 中身（str か bytes）}

出力（split=True のとき。False なら本文を main.tex に全部入れる）:

    main.tex            プリアンブルと \\input{src/...}、\\bibliography{reference}
    src/<章>.tex        章（いちばん上の見出し）ごとの本文。前付け（著者・概要）は src/front.tex
    reference.bib       参考文献（PDF の参考文献の章を番号で分けて @misc にしたもの）
    figures/figNNN.png  図

コンパイルは pdflatex（日本語は bxcjkjatype）でも lualatex（luatexja）でも通る:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

本文中の Markdown の書き方（**太字**、*斜体*、`コード`、[リンク](url)、$数式$）は LaTeX の命令に、
参考文献の番号（[1,2]、[10–12]）は \\cite{ref1,ref2} に置き換え、そのほかの特殊文字はエスケープする。
"""

import base64
import re
from typing import Dict, List, Optional, Tuple, Union

from paper import Block, _split_row

PREAMBLE = r"""\documentclass[a4paper,11pt]{article}
%% pdflatex でも lualatex でもコンパイルできる:
%%   pdflatex main && bibtex main && pdflatex main && pdflatex main
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{textcomp}
  \usepackage[whole]{bxcjkjatype}   % 日本語（IPAex フォント）
\else
  \usepackage{luatexja}
\fi
\usepackage[margin=22mm]{geometry}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{fancyvrb}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\setlength{\parskip}{0.4em}
\setlength{\LTpre}{0.6em}
\setlength{\LTpost}{0.6em}
% 原文の併記（訳文の下に小さく灰色で出す）
\newenvironment{original}{\begin{quote}\small\color{gray}}{\end{quote}}
"""

_SECTION = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection", 5: "paragraph"}

_SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
    "#": r"\#", "^": r"\textasciicircum{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}",
}

# pdflatex（bxcjkjatype の IPAex フォント）に字形の無い記号を LaTeX の命令にする。
# ギリシャ文字・合字（ﬁ）・日本語は IPAex フォントで出るので、そのままにする
_UNICODE = {
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "≡": r"\equiv", "∼": r"\sim",
    "≪": r"\ll", "≫": r"\gg", "∈": r"\in", "∉": r"\notin", "∋": r"\ni", "⊂": r"\subset",
    "⊆": r"\subseteq", "⊃": r"\supset", "⊇": r"\supseteq", "∪": r"\cup", "∩": r"\cap",
    "∅": r"\emptyset", "∞": r"\infty", "∀": r"\forall", "∃": r"\exists", "¬": r"\neg",
    "∧": r"\wedge", "∨": r"\vee", "⊕": r"\oplus", "⊗": r"\otimes", "∘": r"\circ",
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∂": r"\partial", "∇": r"\nabla",
    "√": r"\surd", "∝": r"\propto", "±": r"\pm", "∓": r"\mp", "×": r"\times", "÷": r"\div",
    "⋅": r"\cdot", "·": r"\cdot", "∗": r"\ast", "⋯": r"\cdots", "∣": r"\mid", "‖": r"\|",
    "⌈": r"\lceil", "⌉": r"\rceil", "⌊": r"\lfloor", "⌋": r"\rfloor", "⟨": r"\langle",
    "⟩": r"\rangle", "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "↦": r"\mapsto",
    "↑": r"\uparrow", "↓": r"\downarrow", "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "−": "-", "′": "'", "″": "''",
    "✓": r"\checkmark", "□": r"\square", "■": r"\blacksquare",
}
_SPECIAL_RE = re.compile(r"[\\{}$&#^_%~]|" + "|".join(map(re.escape, _UNICODE)))


def escape(text: str) -> str:
    """LaTeX の特殊文字をエスケープする（Markdown の記法は見ない）"""
    def sub(m):
        ch = m.group(0)
        if ch in _SPECIAL:
            return _SPECIAL[ch]
        cmd = _UNICODE[ch]
        return cmd if not cmd.startswith("\\") else r"\ensuremath{" + cmd + "}"
    return _SPECIAL_RE.sub(sub, text)


# ------------------------------------------------------------
# 参考文献（番号 -> BibTeX）
# ------------------------------------------------------------
_cite_numbers: set = set()     # render_latex_project の間だけ、\cite にしてよい番号を入れる

_CITE_RE = re.compile(r"[\[［](\d{1,3}(?:\s*[,，、–—\-~〜]\s*\d{1,3})*)[\]］]")


def _expand_cite(body: str) -> List[int]:
    nums = []
    for part in re.split(r"\s*[,，、]\s*", body):
        m = re.fullmatch(r"(\d+)\s*[–—\-~〜]\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and b - a < 200:
                nums.extend(range(a, b + 1))
                continue
            return []
        if part.isdigit():
            nums.append(int(part))
        else:
            return []
    return nums


def split_references(texts: List[str]) -> Dict[int, str]:
    """参考文献の章の文章を {番号: 1 件の文献} に分ける

    「1 Takahiro …」「[1] A. Smith, …」「1. …」のどれでも、1, 2, 3 … と順に番号を探して区切る。
    1 つのブロックに 2 件がつながっている（「92 … 93 DAG GEDF …」）場合も分けられる。
    """
    whole = "\n".join(t.strip() for t in texts if t.strip())
    found, pos, n = [], 0, 1
    while True:
        m = re.compile(r"(?:^|(?<=[\s.]))[\[［]?" + str(n) + r"[\]］]?\.?\s+(?=\S)").search(whole, pos)
        if not m:
            break
        found.append((n, m.start(), m.end()))
        pos, n = m.end(), n + 1
    if len(found) < 2:
        # 番号が見つからなければ、ブロックを 1 件ずつの文献とみなす
        return {i + 1: t.strip() for i, t in enumerate(t for t in texts if t.strip())}
    refs = {}
    for i, (num, _, end) in enumerate(found):
        stop = found[i + 1][1] if i + 1 < len(found) else len(whole)
        refs[num] = re.sub(r"\s+", " ", whole[end:stop]).strip()
    return refs


def _split_authors(segment: str) -> Optional[str]:
    """著者の部分を BibTeX の "A and B and C" にする。著者らしくなければ None"""
    if len(segment) > 400 or not re.search(r"[A-Za-z\u3040-\u9fff]", segment):
        return None
    names = [n.strip() for n in re.split(r",\s*(?:and\s+)?|\s+and\s+|、", segment) if n.strip()]
    if len(names) == 1 and len(names[0].split()) > 4:
        return "{" + escape(names[0]) + "}"        # 団体名（"Open source robotics foundation (OSRF)"）
    if any(len(n.split()) > 5 for n in names):
        return None
    return " and ".join(escape(n) for n in names)


def bib_entry(num: int, text: str) -> str:
    """1 件の文献の文章を @misc にする（著者・題・掲載先・年・URL を文章から推定する）"""
    text = text.strip()
    url = None
    m = re.search(r"(?:URL:\s*)?(https?://\S+?)[.,;]?(?=\s|$)", text)
    if m:
        url, end = m.group(1), m.end()
        # PDF の改行で URL が切れている（"…/awkernel_ async_lib/src/gedf.rs."）ときはつなぐ
        cont = re.match(r"\s+([\w.\-]*/[\w./\-#?=%~]+?)[.,;]?(?=\s|$)", text[end:])
        if cont and url[-1] in "_-/." :
            url += cont.group(1)
            end += cont.end()
        text = (text[:m.start()] + text[end:]).strip()
    year = None
    years = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", text)
    if years:
        year = years[-1]
    fields = []
    # 著者と題を、イニシャル（"J."）ではない最初の ". " で分ける。IEEE 形式（“題,”）も見る
    quoted = re.search(r"[“\"](.+?)[,.]?[”\"]", text)
    if quoted:
        authors, title, rest = text[:quoted.start()].rstrip(" ,"), quoted.group(1), text[quoted.end():]
    else:
        # "J." や "Jr." のようなイニシャル・略語では切らない（"API." では切る）
        cuts = [m.end() for m in re.finditer(r"(?<!\b[A-Z])(?<!\b[A-Z][a-z])\.\s+", text)]
        if len(cuts) >= 2:
            authors, title, rest = text[:cuts[0]].rstrip(". "), text[cuts[0]:cuts[1]].rstrip(". "), text[cuts[1]:]
        elif cuts:
            authors, title, rest = "", text[:cuts[0]].rstrip(". "), text[cuts[0]:]
        else:
            authors, title, rest = "", text, ""
    author_field = _split_authors(authors) if authors else None
    if authors and author_field is None:
        title = f"{authors}. {title}"
    if author_field:
        fields.append(f"  author = {{{author_field}}}")
    fields.append(f"  title = {{{{{escape(title.strip(' .,'))}}}}}")
    rest = rest.strip(" .,")
    if year:
        # 掲載先の末尾の年は year と重なる（unsrturl は両方を出す）ので除く
        rest = re.sub(r"[,\s]*\b" + year + r"\.?$", "", rest).strip(" .,")
    if rest:
        fields.append(f"  howpublished = {{{escape(rest)}}}")
    if year:
        fields.append(f"  year = {{{year}}}")
    if url:
        fields.append(f"  url = {{{url}}}")
    return f"@misc{{ref{num},\n" + ",\n".join(fields) + "\n}\n"


# ------------------------------------------------------------
# 行内の Markdown -> LaTeX
# ------------------------------------------------------------
_INLINE_RE = re.compile(
    r"(?P<math>(?<!\\)\$(?!\s)(?P<math_body>[^$\n]+?)(?<!\s)\$)"
    r"|(?P<code>`(?P<code_body>[^`\n]+)`)"
    r"|(?P<link>\[(?P<link_text>[^\]\n]+)\]\((?P<link_url>[^)\s]+)\))"
    r"|(?P<cite>" + _CITE_RE.pattern + r")"
    r"|(?P<bold>\*\*(?P<bold_body>.+?)\*\*)"
    r"|(?P<italic>(?<![\w*])[*_](?P<italic_body>[^*_\n]+?)[*_](?![\w*]))"
    r"|(?P<br><br\s*/?>)"
)


def _looks_like_math(body: str) -> bool:
    """$...$ が数式か（"$5 and $10" のような金額を数式にしない）"""
    return bool(re.search(r"[\\^_=<>{}]", body)) or bool(re.fullmatch(r"[A-Za-z](\w|')?", body.strip()))


def inline(text: str) -> str:
    """Markdown の行内の記法を LaTeX にする"""
    out, pos = [], 0
    for m in _INLINE_RE.finditer(text):
        if m.group("math") and not _looks_like_math(m.group("math_body")):
            continue
        if m.group("cite"):
            nums = _expand_cite(m.group("cite")[1:-1])
            if not nums or not _cite_numbers or not all(n in _cite_numbers for n in nums):
                continue
        out.append(escape(text[pos:m.start()]))
        if m.group("math"):
            out.append(f"${m.group('math_body')}$")
        elif m.group("code"):
            out.append(r"\texttt{" + escape(m.group("code_body")) + "}")
        elif m.group("link"):
            url = m.group("link_url").replace("%", r"\%").replace("#", r"\#")
            out.append(r"\href{" + url + "}{" + inline(m.group("link_text")) + "}")
        elif m.group("cite"):
            out.append(r"\cite{" + ",".join(f"ref{n}" for n in nums) + "}")
        elif m.group("bold"):
            out.append(r"\textbf{" + inline(m.group("bold_body")) + "}")
        elif m.group("italic"):
            out.append(r"\emph{" + inline(m.group("italic_body")) + "}")
        elif m.group("br"):
            out.append(r"\newline{}")
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)


# ------------------------------------------------------------
# ブロック -> LaTeX
# ------------------------------------------------------------
def _heading(b: Block) -> str:
    title = b.translated or b.text
    number = f"{b.number} " if b.number and (b.number.isdigit() or "." in b.number) else (
        f"{b.number}. " if b.number else "")
    orig = f"（{b.text}）" if b.translated and b.translated != b.text else ""
    cmd = _SECTION.get(b.level, "paragraph")
    # 番号は原文のものをそのまま使うので、LaTeX の自動番号は付けない（目次には載せる）
    text = inline(number + title + orig)
    toc = cmd if cmd != "paragraph" else "subsubsection"
    return f"\\{cmd}*{{{text}}}\n\\addcontentsline{{toc}}{{{toc}}}{{{text}}}"


def _table(md_table: str) -> str:
    rows = []
    for line in md_table.strip().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in _split_row(line)]
        if all(re.fullmatch(r":?-{2,}:?", c) or not c for c in cells):
            continue        # 区切りの行（|---|---|）
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    col = f"p{{\\dimexpr({0.98 / width:.3f}\\linewidth-2\\tabcolsep)\\relax}}"
    lines = [r"{\small", r"\begin{longtable}{|" + "|".join([col] * width) + "|}", r"\hline"]
    for i, r in enumerate(rows):
        r = r + [""] * (width - len(r))
        lines.append(" & ".join(inline(c) for c in r) + r" \\ \hline")
        if i == 0:
            lines.append(r"\endhead")
    lines += [r"\end{longtable}", "}"]
    return "\n".join(lines)


def _code(fenced: str) -> str:
    m = re.match(r"^\s*```(\w*)\n(.*?)\n?```\s*$", fenced, re.DOTALL)
    lang, body = (m.group(1), m.group(2)) if m else ("", fenced)
    if lang == "latex":
        # OCR が LaTeX のまま残した数式
        body = body.strip().strip("$").strip()
        if body.startswith(r"\[") or body.startswith(r"\begin{"):
            return body
        return "\\[\n" + body + "\n\\]"
    return "\\begin{Verbatim}[fontsize=\\small,frame=single]\n" + body + "\n\\end{Verbatim}"


_IMAGE_SRC_RE = re.compile(r"!\[[^\]\n]*\]\(data:image/(png|jpe?g);base64,([A-Za-z0-9+/=]+)\)")


def _blocks_to_latex(blocks: List[Block], bilingual: bool, note: Optional[str],
                     figures: List[Tuple[str, bytes]]) -> List[str]:
    """ブロック列を LaTeX の断片の列にする（図は figures に足していく）"""
    out: List[str] = []
    list_items: List[str] = []
    list_originals: List[str] = []

    def flush_list():
        if list_items:
            out.append("\\begin{itemize}\n" + "\n".join(list_items) + "\n\\end{itemize}")
            list_items.clear()
        if list_originals:
            out.append("\\begin{original}\n\\begin{itemize}\n" + "\n".join(list_originals)
                       + "\n\\end{itemize}\n\\end{original}")
            list_originals.clear()

    def original(text):
        if bilingual:
            out.append("\\begin{original}\n" + inline(text) + "\n\\end{original}")

    for b in blocks:
        if b.kind != "list":
            flush_list()
        if b.kind == "title":
            lines = [r"\begin{center}", r"{\LARGE\bfseries " + inline(b.translated or b.text) + r"\par}"]
            if b.translated and b.translated != b.text:
                lines.append(r"\vspace{0.5em}{\large\itshape " + inline(b.text) + r"\par}")
            if note:
                lines.append(r"\vspace{0.5em}{\small " + inline(note) + r"\par}")
            lines.append(r"\end{center}")
            out.append("\n".join(lines))
        elif b.kind == "heading":
            out.append(_heading(b))
        elif b.kind == "image":
            for m in _IMAGE_SRC_RE.finditer(b.text):
                ext = "jpg" if m.group(1).startswith("j") else "png"
                name = f"fig{len(figures) + 1:03d}.{ext}"
                figures.append((name, base64.b64decode(m.group(2))))
                out.append("\\begin{center}\n\\includegraphics[width=0.9\\linewidth,height=0.75\\textheight,"
                           f"keepaspectratio]{{{name}}}\n\\end{{center}}")
        elif b.kind == "table":
            content = b.translated or b.text
            if content.lstrip().startswith("```"):
                out.append(_code(content))
            else:
                out.append(_table(content))
                if bilingual and b.translated:
                    out.append("\\begin{original}\n" + _table(b.text) + "\n\\end{original}")
        elif b.kind == "list":
            marker = b.marker if re.match(r"\d", b.marker or "") else None
            head = f"\\item[{escape(marker)}] " if marker else "\\item "
            list_items.append(head + inline(b.translated or b.text))
            if bilingual and b.translated:
                list_originals.append(head + inline(b.text))
        elif b.kind == "quote":
            out.append("\\begin{quote}\n" + inline(b.translated or b.text) + "\n\\end{quote}")
            if b.translated:
                original(b.text)
        else:
            out.append(inline(b.translated or b.text))
            if b.translated:
                original(b.text)
    flush_list()
    return [x for x in out if x]


def _slug(text: str, used: set) -> str:
    s = re.sub(r"^\s*([A-Z]|\d+(\.\d+)*|[IVX]+)[.)]?\s+", "", text)      # 先頭の番号を除く
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()[:30].strip("_") or "section"
    name, i = s, 2
    while name in used:
        name, i = f"{s}_{i}", i + 1
    used.add(name)
    return name


def render_latex_project(blocks: List[Block], source_name: str, bilingual: bool = False,
                         note: Optional[str] = None, split: bool = False
                         ) -> Dict[str, Union[str, bytes]]:
    """LaTeX のプロジェクト（main.tex・src/・reference.bib・figures/）を {パス: 中身} で返す

    note は題の下に入れる一言（翻訳元の表示など）。split なら章ごとに src/<章>.tex に分ける。
    """
    global _cite_numbers
    files: Dict[str, Union[str, bytes]] = {}
    figures: List[Tuple[str, bytes]] = []

    # 参考文献の章は本文から外し、reference.bib にする
    ref_blocks = [b for b in blocks if b.in_refs and b.kind != "heading"]
    body_blocks = [b for b in blocks if not b.in_refs]
    if body_blocks and ref_blocks and body_blocks[-1].kind == "heading":
        body_blocks = body_blocks[:-1]     # 「References」の見出し
    refs = split_references([b.text for b in ref_blocks]) if ref_blocks else {}
    if refs:
        files["reference.bib"] = "\n".join(bib_entry(n, t) for n, t in refs.items())

    _cite_numbers = set(refs)
    try:
        if split:
            headings = [b.level for b in body_blocks if b.kind == "heading"]
            top = min(headings) if headings else None
            parts: List[Tuple[str, List[Block]]] = [("front", [])]
            title_blocks = [b for b in body_blocks if b.kind == "title"]
            used = {"front"}
            for b in body_blocks:
                if b.kind == "title":
                    continue
                if b.kind == "heading" and b.level == top:
                    parts.append((_slug(b.text, used), []))
                parts[-1][1].append(b)
            body = "\n\n".join(_blocks_to_latex(title_blocks, bilingual, note, figures))
            inputs = []
            for name, part in parts:
                if not part:
                    continue
                files[f"src/{name}.tex"] = "\n\n".join(_blocks_to_latex(part, bilingual, None, figures)) + "\n"
                inputs.append(f"\\input{{src/{name}}}")
            body += "\n\n" + "\n\n".join(inputs)
        else:
            body = "\n\n".join(_blocks_to_latex(body_blocks, bilingual, note, figures))
            if note and not any(b.kind == "title" for b in body_blocks):
                body = r"{\small " + inline(note) + r"\par}" + "\n\n" + body
    finally:
        _cite_numbers = set()

    tail = ""
    if refs:
        # 本文で引用していない文献も載せる（\nocite{*}）。番号は引用順（unsrturl）
        tail = "\n\n\\nocite{*}\n\\bibliographystyle{unsrturl}\n\\bibliography{reference}"
    files["main.tex"] = PREAMBLE + "\n\\begin{document}\n\n" + body + tail + "\n\n\\end{document}\n"
    for name, data in figures:
        files[f"figures/{name}"] = data
    return files
