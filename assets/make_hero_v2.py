"""Compose hero image: TikZ concept comparison (top) + paper figure (bottom)"""
from PIL import Image
import subprocess, os

# ── Step 1: Convert TikZ concept comparison PDF → PNG ──
tikz_pdf = r'D:\PRISM\assets\hero_concept_v2.pdf'
tikz_png = r'D:\PRISM\assets\_tmp_tikz.png'

subprocess.run([
    'pdftoppm', '-png', '-r', '600', '-singlefile',
    tikz_pdf, tikz_png.replace('.png', '')
], check=True, capture_output=True)
print('TikZ concept → PNG')

# ── Step 2: Use paper figure as-is ──
chart_png = r'D:\PRISM\paper\figures\fig_scaling_projection.png'

# ── Step 3: Load and compose ──
top = Image.open(tikz_png)
bot = Image.open(chart_png)

# Scale chart to match top width
tw, th = top.size
bw, bh = bot.size
scale = tw / bw
bot_resized = bot.resize((tw, int(bh * scale)), Image.LANCZOS)

# Stack with gap
gap = 40
combined = Image.new('RGB', (tw, th + gap + bot_resized.size[1]), 'white')
combined.paste(top, (0, 0))
combined.paste(bot_resized, (0, th + gap))

out = r'D:\PRISM\assets\hero_concept_v2.png'
combined.save(out, dpi=(300, 300))
print(f'Saved: {out} ({combined.size[0]}x{combined.size[1]})')

# Cleanup
os.remove(tikz_png)
