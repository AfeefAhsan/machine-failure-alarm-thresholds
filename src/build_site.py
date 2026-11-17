"""Assemble the self-contained index.html (CSS, JS, data and figures inlined) from site_src/.
Run after src/pipeline.py and src/export_tree.py:  python src/build_site.py"""
import base64, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "site_src"
h = (S / "template.html").read_text(encoding="utf-8")
h = h.replace('<link rel="stylesheet" href="style.css">', "<style>\n" + (S / "style.css").read_text(encoding="utf-8") + "\n</style>")
safe = lambda p: p.read_text(encoding="utf-8").strip().replace("</", "<\\/")
data = f"<script>window.RESULTS={safe(ROOT/'site_data/results.json')};window.TREE={safe(ROOT/'site_data/tree.json')};</script>"
h = h.replace('<script src="app.js"></script>', data + "\n<script>\n" + (S / "app.js").read_text(encoding="utf-8") + "\n</script>")
h = h.replace('<script src="explore.js"></script>', "<script>\n" + (S / "explore.js").read_text(encoding="utf-8") + "\n</script>")
def img(m):
    b = base64.b64encode((ROOT / m.group(1)).read_bytes()).decode()
    return f'src="data:image/png;base64,{b}"'
h = re.sub(r'src="(figures/[^"]+\.png)"', img, h)
(ROOT / "index.html").write_text(h, encoding="utf-8")
print("wrote index.html", round(len(h) / 1024), "KB")
