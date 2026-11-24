"""ブロック列（paper.parse_blocks の結果）を LaTeX にする

PDF -> ブロック列 -> 翻訳 のあとに、Markdown / HTML と並ぶ出力形式の 1 つとして使う。
出力は lualatex（luatexja の ltjsarticle）でコンパイルする前提。日本語の訳文も、
翻訳しない原文（英語など）も同じ雛形で通る。

    tex, figures = render_latex(blocks, "paper.pdf", bilingual=True)
    # figures は [(ファイル名, PNG のバイト列), ...]。tex からは figures/<名前> で参照する

ブロックの中身は Markdown の書き方（**太字**、*斜体*、`コード`、[リンク](url)、$数式$）を
含むので、それを LaTeX の命令に置き換え、そのほかの特殊文字はエスケープする。
"""

import base64
import re
from typing import List, Optional, Tuple

from paper import Block, _split_row

PREAMBLE = r"""\documentclass[a4paper,11pt]{ltjsarticle}
% lualatex でコンパイルする:  lualatex %(name)s.tex（目次や表の幅を合わせるため 2 回）
\usepackage[margin=22mm]{geometry}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{fancyvrb}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.4em}
\setlength{\LTpre}{0.6em}
\setlength{\LTpost}{0.6em}
% 原文の併記（訳文の下に小さく灰色で出す）
\newenvironment{original}{\begin{quote}\small\color{gray}}{\end{quote}}

\begin{document}
"""

_SECTION = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection", 5: "paragraph"}

_SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
    "#": r"\#", "^": r"\textasciicircum{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}",
}
_SPECIAL_RE = re.compile(r"[\\{}$&#^_%~]")


def escape(text: str) -> str:
    """LaTeX の特殊文字をエスケープする（Markdown の記法は見ない）"""
    return _SPECIAL_RE.sub(lambda m: _SPECIAL[m.group(0)], text)


# 行内の Markdown の記法。上から順に試す
_INLINE_RE = re.compile(
    r"(?P<math>(?<!\\)\$(?!\s)(?P<math_body>[^$\n]+?)(?<!\s)\$)"
    r"|(?P<code>`(?P<code_body>[^`\n]+)`)"
    r"|(?P<link>\[(?P<link_text>[^\]\n]+)\]\((?P<link_url>[^)\s]+)\))"
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
        out.append(escape(text[pos:m.start()]))
        if m.group("math"):
            out.append(f"${m.group('math_body')}$")
        elif m.group("code"):
            out.append(r"\texttt{" + escape(m.group("code_body")) + "}")
        elif m.group("link"):
            url = m.group("link_url").replace("%", r"\%").replace("#", r"\#")
            out.append(r"\href{" + url + "}{" + inline(m.group("link_text")) + "}")
        elif m.group("bold"):
            out.append(r"\textbf{" + inline(m.group("bold_body")) + "}")
        elif m.group("italic"):
            out.append(r"\emph{" + inline(m.group("italic_body")) + "}")
        elif m.group("br"):
            out.append(r"\newline{}")
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def _heading(b: Block) -> str:
    title = b.translated or b.text
    number = f"{b.number} " if b.number and (b.number.isdigit() or "." in b.number) else (
        f"{b.number}. " if b.number else "")
    orig = f"（{b.text}）" if b.translated and b.translated != b.text else ""
    cmd = _SECTION.get(b.level, "paragraph")
    # 番号は原文のものをそのまま使うので、LaTeX の自動番号は付けない（目次には載せる）
    text = inline(number + title + orig)
    return f"\\{cmd}*{{{text}}}\n\\addcontentsline{{toc}}{{{cmd if cmd != 'paragraph' else 'subsubsection'}}}{{{text}}}"


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


def render_latex(blocks: List[Block], source_name: str, bilingual: bool = False,
                 note: Optional[str] = None) -> Tuple[str, List[Tuple[str, bytes]]]:
    """(tex の本文, 図のファイル) を返す。note は題の下に入れる一言（翻訳元の表示など）"""
    out: List[str] = []
    figures: List[Tuple[str, bytes]] = []
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

    has_title = False
    for b in blocks:
        if b.kind != "list":
            flush_list()
        if b.kind == "title":
            has_title = True
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
                           f"keepaspectratio]{{figures/{name}}}\n\\end{{center}}")
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
            item = (f"\\item[{escape(marker)}] " if marker else "\\item ") + inline(b.translated or b.text)
            list_items.append(item)
            if bilingual and b.translated:
                list_originals.append((f"\\item[{escape(marker)}] " if marker else "\\item ") + inline(b.text))
        elif b.kind == "quote":
            out.append("\\begin{quote}\n" + inline(b.translated or b.text) + "\n\\end{quote}")
            original(b.text) if b.translated else None
        else:
            out.append(inline(b.translated or b.text))
            if b.translated:
                original(b.text)
    flush_list()

    head = PREAMBLE.replace("%(name)s", re.sub(r"\.\w+$", "", source_name))
    if not has_title and note:
        out.insert(0, r"{\small " + inline(note) + r"\par}")
    body = "\n\n".join(x for x in out if x)
    return head + body + "\n\n\\end{document}\n", figures
