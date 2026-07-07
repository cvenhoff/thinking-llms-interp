#!/usr/bin/env python3
"""
Figure 3: Hybrid Model in Action — ORZ-7B.
Publication-quality version: word-level pastel highlights, calibrated
font metrics, auto-sized figure.
"""
import json, os, re
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA    = ("/workspace-vast/constantinv/thinking-llms-interp"
           "/mlp_pipeline/figures/orz32b_example371_token_info.json")
OUT_DIR = "/workspace-vast/constantinv/thinking-llms-interp/mlp_pipeline/figures"
OUT_PDF = os.path.join(OUT_DIR, "hybrid_example_orz32b.pdf")
OUT_PNG = os.path.join(OUT_DIR, "hybrid_example_orz32b.png")

# ── Load ───────────────────────────────────────────────────────────────────────
with open(DATA) as f:
    d = json.load(f)
tli      = d["token_latent_info"]
question = d["question"]

def _fmt_gold(s):
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)
    return re.sub(r'\\[a-zA-Z]+', '', s).strip()
gold_disp = _fmt_gold(d.get("gold_answer", ""))

def clean_question(q):
    q = re.sub(r'\$([^$]*)\$', r'\1', q)
    q = re.sub(r'\\\(([^)]*)\\\)', r'\1', q)
    q = re.sub(r'\\\[([^\]]*)\\\]', r'\1', q)   # inline display math \[...\]
    q = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', q)
    q = re.sub(r'\[asy\].*?\[/asy\]', '[figure]', q, flags=re.DOTALL)
    q = q.replace('\\leq', '\u2264').replace('\\le', '\u2264')
    q = q.replace('\\geq', '\u2265').replace('\\ge', '\u2265')
    q = re.sub(r'\\[a-zA-Z]+', '', q)
    q = re.sub(r'[{}]', '', q)
    q = re.sub(r' +', ' ', q)
    return q.strip()

# ── Palette ────────────────────────────────────────────────────────────────────
# Pre-blend highlight backgrounds against white at alpha=0.82, so the PDF has
# zero transparency (avoids pdflatex transparency-group rendering failures).
def _blend(hex_col, a=0.82):
    r,g,b = int(hex_col[1:3],16)/255, int(hex_col[3:5],16)/255, int(hex_col[5:7],16)/255
    return f"#{round((r*a+(1-a))*255):02X}{round((g*a+(1-a))*255):02X}{round((b*a+(1-a))*255):02X}"

# Color pool assigned dynamically to the categories present in this example,
# ordered by steered-position frequency (most frequent gets the first color).
_POOL = [
    ("#B4D8F0", "#1A5276"),  # blue
    ("#B4E8C8", "#1E8449"),  # green
    ("#FCDEA4", "#CA6F1E"),  # orange
    ("#CEB4E8", "#5B2C8F"),  # purple
    ("#FAC8CC", "#C0392B"),  # red
    ("#F8ECA4", "#7D6608"),  # yellow
    ("#D8C4EC", "#7D3C98"),  # violet
]
from collections import Counter as _Counter
_freq = _Counter(t["latent_key"] for t in tli
                 if t.get("coefficient", 0) > 0 and t.get("selection") == "steered")
_present = [k for k, _ in _freq.most_common()]
PALETTE = {k: (_blend(bg), fg) for k, (bg, fg) in zip(_present, _POOL)}
# Titles: prefer descriptions block, fall back to per-token latent_title.
_titles = {}
for _t in tli:
    if _t.get("latent_key"):
        _titles.setdefault(_t["latent_key"], _t.get("latent_title", _t["latent_key"]))
for _k, _v in (d.get("descriptions") or {}).items():
    _titles[_v.get("key", f"idx{_k}")] = _v.get("title", _titles.get(_v.get("key"), _k))
CAT_LABELS = {k: _titles.get(k, k) for k in PALETTE}

# ── apply_sub: regex substitution preserving char-level color tracking ─────────
def apply_sub(text, cc, pattern, replacement):
    out_chars, out_keys = [], []
    prev = 0
    for m in re.finditer(pattern, text):
        for ch, k in cc[prev:m.start()]:
            out_chars.append(ch); out_keys.append(k)
        rep_col = cc[m.start()][1] if m.start() < len(cc) else None
        for ch in m.expand(replacement):
            out_chars.append(ch); out_keys.append(rep_col)
        prev = m.end()
    for ch, k in cc[prev:]:
        out_chars.append(ch); out_keys.append(k)
    return "".join(out_chars), list(zip(out_chars, out_keys))

# ── Build char-level (char, key) list ──────────────────────────────────────────
char_cols = []
for tok in tli:
    ck = tok["latent_key"] if tok.get("coefficient", 0) > 0 else None
    for ch in tok["token"]:
        char_cols.append((ch, ck))

text = "".join(c[0] for c in char_cols)
cc   = list(char_cols)

# Pass 1: \frac{a}{b} → (a)/(b)  [handles nested braces separately]
i, rc, rk = 0, [], []
fp = re.compile(r'\\frac\{([^{}]*)\}\{([^{}]*)\}')
while i < len(text):
    m = fp.match(text, i)
    if m:
        repl = f"({m.group(1)})/({m.group(2)})"
        col = cc[i][1] if i < len(cc) else None
        for ch in repl: rc.append(ch); rk.append(col)
        i = m.end()
    else:
        rc.append(text[i]); rk.append(cc[i][1] if i < len(cc) else None)
        i += 1
text, cc = "".join(rc), list(zip(rc, rk))

# Pass 2: simple single-pass substitutions
SUBS = [
    # Strip special tokens (BPE end-of-text etc.)
    (r'<\|[^|>]*\|>', ''),
    (r'(?s)```output\s*.*?```', ' [tool output] '),  # collapse tool-output blocks
    (r'(?s)```[a-zA-Z]*\s*.*?```', ' [runs code] '),   # collapse code blocks
    (r'\$', ''),   # drop inline-math delimiters ($...$)
    (r'\\leq\b','≤'), (r'\\geq\b','≥'), (r'\\neq\b','≠'),
    (r'\\le\b','≤'), (r'\\ge\b','≥'),
    (r'\\cdot\b','·'), (r'\\times\b','×'), (r'\\pm\b','±'),
    (r'\\ldots\b','…'), (r'\\infty\b','∞'),
    (r'\\alpha\b','α'), (r'\\beta\b','β'), (r'\\gamma\b','γ'),
    (r'\\pi\b','π'), (r'\\theta\b','θ'), (r'\\lambda\b','λ'),
    (r'\\sqrt\{([^}]*)\}',r'√(\1)'),
    (r'\\text\{([^}]*)\}',r'\1'),(r'\\mathrm\{([^}]*)\}',r'\1'),
    (r'\\mathbf\{([^}]*)\}',r'\1'),(r'\\mathbb\{([^}]*)\}',r'\1'),
    (r'\\boxed\{([^}]*)\}',r'\1'),
    (r'\\left\b',''),(r'\\right\b',''),(r'\\bigg?\b',''),(r'\\Big\b',''),
    (r'\\\(',''),(r'\\\)',''),(r'\\\[',''),(r'\\\]',''),
    (r'\\\\','\n'),
    (r'\\[a-zA-Z]+\b',''),
    (r'[{}]',''),
    (r'###\s*',''),
    (r' {2,}',' '),
]
for pat, repl in SUBS:
    text, cc = apply_sub(text, cc, pat, repl)

# Strip leading whitespace
while cc and cc[0][0] in ' \n':
    cc = cc[1:]; text = text[1:]

# ── Build word list  [(word_str, color_key | None)] ─────────────────────────
# Strategy: group into runs of non-space/non-newline; assign dominant color
words = []
i = 0
while i < len(cc):
    ch, key = cc[i]
    if ch == '\n':
        words.append(('\n', None)); i += 1
    elif ch == ' ':
        j = i
        while j < len(cc) and cc[j][0] == ' ': j += 1
        words.append((' ', None)); i = j   # collapse all spaces to one
    else:
        j = i
        wchars = []
        while j < len(cc) and cc[j][0] not in (' ', '\n'):
            wchars.append(cc[j]); j += 1
        wstr = "".join(c[0] for c in wchars)
        keys = [c[1] for c in wchars if c[1] is not None]
        wkey = Counter(keys).most_common(1)[0][0] if keys else None
        words.append((wstr, wkey)); i = j

# ── Font calibration (one-time measurement) ────────────────────────────────────
FIG_W = 8.5
FONT  = "DejaVu Sans"
FS    = 7.2
LM    = 0.040   # left margin  (figure fraction)
RM    = 0.040   # right margin
TW    = 1.0 - LM - RM  # usable text width in figure fraction

_cf = plt.figure(figsize=(FIG_W, 1.0))
_cf.canvas.draw()
_ren  = _cf.canvas.get_renderer()
_fw   = _cf.get_window_extent(_ren).width   # figure width in pixels

# Measure width of 'n' (average-width letter)
_t = _cf.text(LM, 0.5, "n"*200, fontsize=FS, family=FONT, va='center')
_cf.canvas.draw()
_bb = _t.get_window_extent(_ren)
CW = (_bb.width / 200) / _fw               # avg char width (figure fraction)
_t.remove()

# Measure actual space width
_ts = _cf.text(LM, 0.3, "x x"*100, fontsize=FS, family=FONT, va='center')
_tf = _cf.text(LM, 0.3, "xx"*100,  fontsize=FS, family=FONT, va='center')
_cf.canvas.draw()
_bs = _ts.get_window_extent(_ren)
_bf = _tf.get_window_extent(_ren)
SPACE_W = max(CW * 0.35, (_bs.width - _bf.width) / 100 / _fw)  # space width
_ts.remove(); _tf.remove()
plt.close(_cf)

CPL = max(60, int(TW / CW) - 6)   # used only for question-text wrapping

# ── Batch-measure actual word widths (one draw call for all unique words) ─────
# This gives accurate proportional-font widths instead of 'n'-width estimates.
def _batch_measure_words(word_set, fontsize, font, fig_w):
    """Return {word: width_as_fig_fraction} using a single canvas.draw()."""
    wlist = list(word_set)
    # Layout: stack words at different y positions in a tall temp fig
    rows = len(wlist)
    if rows == 0:
        return {}
    _bf = plt.figure(figsize=(fig_w, max(1.0, rows * 0.12)))
    _br = _bf.canvas.get_renderer()
    _fw = _bf.get_window_extent(_br).width
    texts = {}
    for i, w in enumerate(wlist):
        texts[w] = _bf.text(0.01, (i + 0.5) / rows, w,
                            fontsize=fontsize, family=font, va='center')
    _bf.canvas.draw()
    result = {}
    for w, t in texts.items():
        bb = t.get_window_extent(_br)
        result[w] = bb.width / _fw
    plt.close(_bf)
    return result

_unique_words = {wstr for wstr, _ in words if wstr not in (' ', '\n') and wstr.strip()}
WORD_W = _batch_measure_words(_unique_words, FS, FONT, FIG_W)

# ── Word-wrap into lines ────────────────────────────────────────────────────────
# Wrap based on cumulative MEASURED width (figure fractions), not char count.
MAX_W = TW - CW * 1.5    # leave a tiny right margin buffer
lines = []
curr, curr_w = [], 0.0
for wstr, wkey in words:
    if wstr == '\n':
        lines.append(curr); curr, curr_w = [], 0.0
    elif wstr == ' ':
        if curr:
            curr.append((' ', None, SPACE_W)); curr_w += SPACE_W
    else:
        ww = WORD_W.get(wstr, len(wstr) * CW)
        if curr_w + ww > MAX_W and curr_w > 0:
            # strip trailing space from previous line
            if curr and curr[-1][0] == ' ':
                curr.pop()
            lines.append(curr); curr, curr_w = [], 0.0
        curr.append((wstr, wkey, ww)); curr_w += ww
if curr:
    lines.append(curr)

# Collapse runs of >1 empty line to a single blank
clean_lines = []
prev_empty = False
for l in lines:
    empty = (len(l) == 0 or all(e[0] == ' ' for e in l))
    if empty and prev_empty: continue
    clean_lines.append(l)
    prev_empty = empty
lines = clean_lines

# ── Compute figure height ──────────────────────────────────────────────────────
TITLE_FS = 9.5
META_FS  = 6.8
ANNOT_FS = 7.2
LEG_FS   = 7.0

# All heights in inches
def pt2in(pt): return pt / 72.0

LH_IN   = FS * 1.50 / 72        # text line height
BLANK_H = LH_IN * 0.55          # paragraph break height
Q_FS    = 8.0                    # question font size
Q_LH    = Q_FS * 1.4 / 72

# Wrap question text
q_clean = clean_question(question)
q_words = q_clean.split()
q_lines_text = []
cl, clen = [], 0
for w in q_words:
    wl = len(w)+1
    if clen + wl > CPL and clen > 0:
        q_lines_text.append(' '.join(cl)); cl, clen = [w], len(w)+1
    else:
        cl.append(w); clen += wl
if cl: q_lines_text.append(' '.join(cl))
n_q_lines = len(q_lines_text)

Q_PAD_IN  = 0.085
Q_H_IN    = Q_PAD_IN + n_q_lines * Q_LH + pt2in(ANNOT_FS)*1.5 + Q_PAD_IN + 0.015

# Count non-blank / blank lines in response
n_blank = sum(1 for l in lines if not l or all(e[0]==' ' for e in l))
n_text  = len(lines) - n_blank
RESP_H_IN = n_text * LH_IN + n_blank * BLANK_H

# Total height
TITLE_H = pt2in(TITLE_FS) * 1.50
META_H  = pt2in(META_FS)  * 1.40
GAP1    = 0.095           # after subtitle
GAP2    = 0.085           # after question box
ANNOT_H = pt2in(ANNOT_FS) * 1.40  # "Hybrid model response:" label
GAP3    = 0.040           # after label
GAP4    = 0.065           # before legend
LEG_H   = 0.58
BOT_PAD = 0.06

FIG_H = (TITLE_H + META_H + GAP1
         + Q_H_IN + GAP2
         + ANNOT_H + GAP3
         + RESP_H_IN + GAP4
         + LEG_H + BOT_PAD)

# Figure-fraction helpers
def iy(y_in):  return 1.0 - y_in / FIG_H     # inches-from-top → fig fraction
def ih(h_in):  return h_in / FIG_H            # height in inches → fig fraction
LH   = LH_IN  / FIG_H
BLKH = BLANK_H / FIG_H

print(f"CPL={CPL}  n_lines={len(lines)} (blank={n_blank})  FIG_H={FIG_H:.2f}\"")

# ── Create figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor('white')

y_in = 0.0  # cursor: inches from top

# ── Title ──────────────────────────────────────────────────────────────────────
y_in += TITLE_H * 0.82
fig.text(0.5, iy(y_in),
         "Hybrid Model in Action:  ORZ-32B",
         fontsize=TITLE_FS, family=FONT, ha='center', va='baseline',
         fontweight='bold', color='#111111')

y_in += META_H * 1.0
fig.text(0.5, iy(y_in),
         "Qwen2.5-32B base   ·   Open-Reasoner-Zero-32B thinking vectors   ·   MATH500",
         fontsize=META_FS, family=FONT, ha='center', va='baseline',
         color='#666666')

y_in += GAP1

# ── Question box ──────────────────────────────────────────────────────────────
Q_BOX_TOP = y_in
Q_BOX_BOT = y_in + Q_H_IN
BX, BW = LM, TW

# Box background
qbox = FancyBboxPatch(
    (BX, iy(Q_BOX_BOT)),
    BW, ih(Q_H_IN),
    boxstyle="round,pad=0.008",
    facecolor="#F7F8FA", edgecolor="#CCCCCC", linewidth=0.7,
    transform=fig.transFigure, zorder=0
)
fig.add_artist(qbox)

# Left accent bar
bar = FancyBboxPatch(
    (BX, iy(Q_BOX_BOT)),
    0.005, ih(Q_H_IN),
    boxstyle="square,pad=0",
    facecolor="#4472C4", edgecolor='none',
    transform=fig.transFigure, zorder=1
)
fig.add_artist(bar)

# Question text (indented after bar)
qx = BX + 0.010
y_in += Q_PAD_IN
for ql in q_lines_text:
    fig.text(qx, iy(y_in), ql,
             fontsize=Q_FS, family=FONT, va='baseline',
             color='#222222', transform=fig.transFigure, zorder=2)
    y_in += Q_LH

# Divider
y_in += 0.012
div_y = iy(y_in)
fig.add_artist(mlines.Line2D(
    [BX + 0.010, BX + BW - 0.005], [div_y, div_y],
    color='#DDDDDD', linewidth=0.6, transform=fig.transFigure
))
y_in += 0.005

# Answer / outcome line
y_in += pt2in(ANNOT_FS) * 1.45
CORRECT_COL = "#1B7E34"
WRONG_COL   = "#C0392B"
fig.text(qx, iy(y_in),
         f"Correct answer:  {gold_disp}",
         fontsize=ANNOT_FS, family=FONT, va='baseline',
         color='#222222', fontweight='bold', transform=fig.transFigure, zorder=2)
fig.text(0.38, iy(y_in),
         "│   Base model:  incorrect",
         fontsize=ANNOT_FS, family=FONT, va='baseline',
         color=WRONG_COL, fontweight='bold', transform=fig.transFigure, zorder=2)
fig.text(0.63, iy(y_in),
         "│   Hybrid model:  correct",
         fontsize=ANNOT_FS, family=FONT, va='baseline',
         color=CORRECT_COL, fontweight='bold', transform=fig.transFigure, zorder=2)

y_in = Q_BOX_BOT + GAP2

# ── "Hybrid model response:" label ────────────────────────────────────────────
y_in += ANNOT_H * 0.85
fig.text(LM, iy(y_in),
         "Hybrid model response  "
         "(steered token positions highlighted by reasoning category):",
         fontsize=7.0, family=FONT, va='baseline',
         color='#555555', style='italic', transform=fig.transFigure)
y_in += GAP3

# ── Response text ─────────────────────────────────────────────────────────────
P_BELOW = 0.17 * LH   # patch below baseline (descenders)
P_ABOVE = 0.64 * LH   # patch above baseline (cap height)
P_H     = P_BELOW + P_ABOVE
HPAD    = CW * 0.30   # horizontal inset inside each merged span

def _ww(entry):
    return entry[2] if len(entry) > 2 else WORD_W.get(entry[0], len(entry[0]) * CW)

for line_words in lines:
    empty = not line_words or all(e[0] == ' ' for e in line_words)
    if empty:
        y_in += BLANK_H
        continue

    y_in += LH_IN
    baseline_frac = iy(y_in)

    # Phase 1: compute x positions for every token in the line
    positions = []   # (wstr, wkey, x_start, x_end)
    x = LM
    for entry in line_words:
        wstr, wkey = entry[0], entry[1]
        if wstr == ' ':
            positions.append((' ', None, x, x + SPACE_W))
            x += SPACE_W
        else:
            ww = _ww(entry)
            positions.append((wstr, wkey, x, x + ww))
            x += ww

    # Phase 2: merge consecutive same-category words (with spaces between them)
    # into single highlight spans.  Isolated highlights on ≤2-char words are
    # skipped unless adjacent to a longer highlighted word.
    spans = []   # (x0, x1, color_key)
    i = 0
    while i < len(positions):
        wstr, wkey, x0, x1 = positions[i]
        if wkey and wkey in PALETTE and wstr.strip():
            # Expand span rightward: include spaces + next word if same key
            span_x0, span_x1 = x0, x1
            j = i + 1
            while j < len(positions):
                nw, nk, nx0, nx1 = positions[j]
                if nw == ' ':
                    # peek: is the word after this space the same key?
                    if j + 1 < len(positions) and positions[j+1][1] == wkey \
                            and positions[j+1][0].strip():
                        span_x1 = positions[j+1][3]
                        j += 2  # consume space + word
                    else:
                        break
                elif nk == wkey and nw.strip():
                    span_x1 = nx1
                    j += 1
                else:
                    break
            # Only draw if span covers enough content
            # (skip tiny isolated highlights on very short single words)
            span_chars = sum(len(positions[k][0]) for k in range(i, j)
                             if positions[k][0].strip())
            if span_chars >= 2:
                spans.append((span_x0, span_x1, wkey))
            i = j
        else:
            i += 1

    # Phase 3: draw merged patches
    for sx0, sx1, skey in spans:
        bg, _ = PALETTE[skey]
        patch = FancyBboxPatch(
            (sx0 - HPAD, baseline_frac - P_BELOW),
            (sx1 - sx0) + 2 * HPAD, P_H,
            boxstyle="round,pad=0.003",
            facecolor=bg, edgecolor='none',
            transform=fig.transFigure, zorder=1, clip_on=False
        )
        fig.add_artist(patch)

    # Phase 4: draw text on top
    for wstr, wkey, x0, _ in positions:
        if wstr == ' ':
            continue
        fig.text(x0, baseline_frac, wstr,
                 fontsize=FS, family=FONT, va='baseline', ha='left',
                 color='#0D0D0D', transform=fig.transFigure,
                 zorder=2, clip_on=False)

# ── Legend ────────────────────────────────────────────────────────────────────
y_in += GAP4

# Collect only categories present in this example (sorted by frequency)
present = sorted(
    [k for k in PALETTE if any(
        t.get("latent_key") == k and t.get("coefficient", 0) > 0
        for t in tli
    )],
    key=lambda k: -sum(
        1 for t in tli
        if t.get("latent_key") == k and t.get("coefficient", 0) > 0
    )
)
N = len(present)
NCOLS = 3
nrows = -(-N // NCOLS)

TITLE_IN  = pt2in(LEG_FS) * 1.6
ROW_IN    = pt2in(LEG_FS) * 1.75
VPAD_IN   = 0.055
LEG_TOTAL = VPAD_IN + TITLE_IN + nrows * ROW_IN + VPAD_IN

leg_bg = FancyBboxPatch(
    (LM, iy(y_in + LEG_TOTAL)),
    TW, ih(LEG_TOTAL),
    boxstyle="round,pad=0.006",
    facecolor="#F7F8FA", edgecolor="#DDDDDD", linewidth=0.6,
    transform=fig.transFigure, zorder=0
)
fig.add_artist(leg_bg)

# Title row
y_in += VPAD_IN + TITLE_IN * 0.85
fig.text(0.5, iy(y_in),
         "Reasoning categories  (steered token positions only)",
         fontsize=LEG_FS, family=FONT, ha='center', va='baseline',
         color='#444444', transform=fig.transFigure)

SWATCH_W  = 0.013
SWATCH_H  = ih(pt2in(LEG_FS) * 1.1)
GAP_TEXT  = 0.008
COL_W = TW / NCOLS
LABEL_W = 0.22  # approx label width

for idx, ck in enumerate(present):
    row = idx // NCOLS
    col = idx % NCOLS
    bg, fg = PALETTE[ck]

    row_baseline_in = y_in + (row + 1) * ROW_IN
    entry_y = iy(row_baseline_in)   # baseline

    # Centre entry in column
    entry_total = SWATCH_W + GAP_TEXT + LABEL_W
    entry_x = LM + col * COL_W + (COL_W - entry_total) / 2

    swatch = FancyBboxPatch(
        (entry_x, entry_y - SWATCH_H * 0.55),
        SWATCH_W, SWATCH_H,
        boxstyle="round,pad=0.002",
        facecolor=bg, edgecolor=fg, linewidth=0.6,
        transform=fig.transFigure, zorder=2
    )
    fig.add_artist(swatch)
    fig.text(entry_x + SWATCH_W + GAP_TEXT, entry_y,
             CAT_LABELS[ck],
             fontsize=LEG_FS - 0.3, family=FONT, va='baseline',
             color='#1A1A1A', transform=fig.transFigure, zorder=2)

plt.savefig(OUT_PNG, dpi=180, bbox_inches='tight',
            pad_inches=0.05, facecolor='white')
plt.close()

from PIL import Image as _PIL
_im = _PIL.open(OUT_PNG).convert('RGB')
# PNG: clean RGB (no alpha channel — pdflatex chokes on RGBA PNGs)
_im.save(OUT_PNG, format='PNG')
# PDF: PIL-generated raster PDF — no matplotlib vector quirks that break pdflatex
_im.save(OUT_PDF, resolution=180)

print(f"Saved: {OUT_PDF} (PIL raster PDF)")
print(f"Saved: {OUT_PNG} (RGB)")
