import json
from pathlib import Path


SPEC_PATH = Path("figures/specs/method_training_pipeline.json")
OUT_PATH = Path("figures/method_training_pipeline.svg")


def esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def node_anchor(node, target):
    dx = target["x"] - node["x"]
    dy = target["y"] - node["y"]
    if abs(dx) / max(node["w"], 1) > abs(dy) / max(node["h"], 1):
        x = node["x"] + (node["w"] / 2 if dx >= 0 else -node["w"] / 2)
        y = node["y"]
    else:
        x = node["x"]
        y = node["y"] + (node["h"] / 2 if dy >= 0 else -node["h"] / 2)
    return x, y


def text_block(label, x, y, size=14, color="#111827", weight="500"):
    lines = str(label).split("\\n")
    line_h = size * 1.25
    start = y - line_h * (len(lines) - 1) / 2
    out = []
    for idx, line in enumerate(lines):
        out.append(
            f'<text x="{x:.1f}" y="{start + idx * line_h:.1f}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{esc(line)}</text>'
        )
    return "\n".join(out)


def render(spec):
    width = spec["canvas"]["width"]
    height = spec["canvas"]["height"]
    nodes = {n["id"]: n for n in spec["nodes"]}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>',
        "</marker>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="1.5" stdDeviation="1.5" flood-color="#0f172a" flood-opacity="0.12"/>',
        "</filter>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        text_block(spec["title"], width / 2, 28, size=18, color="#0f172a", weight="700"),
    ]

    for group in spec.get("groups", []):
        parts.append(
            f'<rect x="{group["x"]}" y="{group["y"]}" width="{group["w"]}" height="{group["h"]}" '
            f'rx="14" fill="{group["fill"]}" stroke="{group["stroke"]}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{group["x"] + 16}" y="{group["y"] + 24}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" '
            f'fill="#475569">{esc(group["label"])}</text>'
        )

    for edge in spec["edges"]:
        a = nodes[edge["from"]]
        b = nodes[edge["to"]]
        x1, y1 = node_anchor(a, b)
        x2, y2 = node_anchor(b, a)
        dash = ' stroke-dasharray="6 5"' if edge.get("style") == "dashed" else ""
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#475569" stroke-width="1.6"{dash} marker-end="url(#arrow)"/>'
        )
        if edge.get("label"):
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            parts.append(
                f'<rect x="{mx - 44:.1f}" y="{my - 10:.1f}" width="88" height="18" '
                f'rx="9" fill="#ffffff" stroke="#e2e8f0" stroke-width="0.8"/>'
            )
            parts.append(
                f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="10.5" fill="#475569">{esc(edge["label"])}</text>'
            )

    for node in spec["nodes"]:
        x = node["x"] - node["w"] / 2
        y = node["y"] - node["h"] / 2
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{node["w"]}" height="{node["h"]}" '
            f'rx="12" fill="{node["fill"]}" stroke="{node["stroke"]}" stroke-width="1.8" filter="url(#shadow)"/>'
        )
        parts.append(text_block(node["label"], node["x"], node["y"], size=13.5))

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    spec = json.loads(SPEC_PATH.read_text())
    OUT_PATH.write_text(render(spec))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
