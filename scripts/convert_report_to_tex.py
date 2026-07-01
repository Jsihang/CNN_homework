import re
from pathlib import Path


SRC = Path("EXPERIMENT_REPORT.md")
DST = Path("EXPERIMENT_REPORT.tex")


PREAMBLE = r"""\documentclass[UTF8,a4paper,12pt]{ctexart}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{float}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{caption}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\graphicspath{{./}}
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  columns=fullflexible,
  frame=single,
  backgroundcolor=\color{gray!6}
}

\title{基于 GAN 的 CIFAR-10 图像生成实验报告}
\author{}
\date{}

\begin{document}
\maketitle
"""


POSTAMBLE = r"""
\end{document}
"""


SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def protect_math(text):
    blocks = []

    def repl_block(match):
        blocks.append(match.group(0))
        return f"@@MATH{len(blocks)-1}@@"

    text = re.sub(r"\$\$.*?\$\$", repl_block, text, flags=re.S)
    text = re.sub(r"\$[^$\n]+\$", repl_block, text)
    return text, blocks


def restore_math(text, blocks):
    for i, block in enumerate(blocks):
        text = text.replace(f"@@MATH{i}@@", block)
    return text


def escape_plain(text):
    out = []
    for ch in text:
        out.append(SPECIALS.get(ch, ch))
    return "".join(out)


def convert_inline(text):
    text, math_blocks = protect_math(text)
    code_spans = []

    def repl_code(match):
        code_spans.append(match.group(1))
        return f"@@CODE{len(code_spans)-1}@@"

    text = re.sub(r"`([^`]+)`", repl_code, text)

    bold_spans = []

    def repl_bold(match):
        bold_spans.append(convert_inline(match.group(1)))
        return f"@@BOLD{len(bold_spans)-1}@@"

    text = re.sub(r"\*\*(.+?)\*\*", repl_bold, text)
    text = escape_plain(text)

    for i, code in enumerate(code_spans):
        text = text.replace(f"@@CODE{i}@@", r"\texttt{" + escape_plain(code) + "}")
    for i, bold in enumerate(bold_spans):
        text = text.replace(f"@@BOLD{i}@@", r"\textbf{" + bold + "}")

    return restore_math(text, math_blocks)


def split_table_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def is_table_start(lines, i):
    return (
        i + 1 < len(lines)
        and lines[i].strip().startswith("|")
        and re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1])
    )


def convert_table(lines, i):
    headers = split_table_row(lines[i])
    rows = []
    i += 2
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append(split_table_row(lines[i]))
        i += 1

    col_count = len(headers)
    col_spec = "l" + "X" * max(0, col_count - 1)
    out = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\linewidth}{" + col_spec + "}",
        r"\toprule",
        " & ".join(convert_inline(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        row = row + [""] * (col_count - len(row))
        out.append(" & ".join(convert_inline(c) for c in row[:col_count]) + r" \\")
    out.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    return out, i


def image_latex(path, caption):
    width = "0.75\\linewidth" if path.endswith("GAN.png") else "\\linewidth"
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width={width}]{{{path}}}",
        rf"\caption{{{convert_inline(caption)}}}",
        r"\end{figure}",
        "",
    ]


def convert(md):
    lines = md.splitlines()
    out = [PREAMBLE]
    i = 0
    in_code = False
    code_lang = ""

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                code_lang = stripped[3:].strip()
                out.append(r"\begin{lstlisting}[language=bash]" if code_lang == "bash" else r"\begin{lstlisting}")
                in_code = True
            else:
                out.append(r"\end{lstlisting}")
                out.append("")
                in_code = False
            i += 1
            continue

        if in_code:
            out.append(line)
            i += 1
            continue

        if stripped == "":
            out.append("")
            i += 1
            continue

        if is_table_start(lines, i):
            table_out, i = convert_table(lines, i)
            out.extend(table_out)
            continue

        html_img = re.match(r'<img\s+[^>]*src="([^"]+)"[^>]*alt="([^"]*)"[^>]*/?>', stripped)
        if html_img:
            out.extend(image_latex(html_img.group(1), html_img.group(2) or "image"))
            i += 1
            continue

        md_img = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if md_img:
            out.extend(image_latex(md_img.group(2), md_img.group(1) or "image"))
            i += 1
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title == "基于 GAN 的 CIFAR-10 图像生成实验报告":
                i += 1
                continue
            out.append(r"\section*{" + convert_inline(title) + "}")
            i += 1
            continue
        if stripped.startswith("## "):
            out.append(r"\section{" + convert_inline(stripped[3:].strip()) + "}")
            i += 1
            continue
        if stripped.startswith("### "):
            out.append(r"\subsection{" + convert_inline(stripped[4:].strip()) + "}")
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            out.append(r"\begin{enumerate}")
            for item in items:
                out.append(r"\item " + convert_inline(item))
            out.append(r"\end{enumerate}")
            out.append("")
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            out.append(r"\begin{itemize}")
            for item in items:
                out.append(r"\item " + convert_inline(item))
            out.append(r"\end{itemize}")
            out.append("")
            continue

        if stripped.startswith("$$"):
            block = [stripped]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip().endswith("$$"):
                    i += 1
                    break
                i += 1
            out.extend(block)
            out.append("")
            continue

        out.append(convert_inline(stripped) + "\n")
        i += 1

    out.append(POSTAMBLE)
    return "\n".join(out)


def main():
    md = SRC.read_text(encoding="utf-8")
    tex = convert(md)
    DST.write_text(tex, encoding="utf-8")
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
