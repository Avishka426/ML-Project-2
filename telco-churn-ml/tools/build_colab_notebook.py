"""
build_colab_notebook.py
-----------------------
Creates notebooks/telco_churn_analysis_colab.ipynb: an exact copy of the main
notebook (every cell and every saved output) with ONE extra cell just below the title.

Colab only receives the .ipynb file, not the project folder, so that first
cell recreates src/ and the raw CSV from a compressed copy embedded in it
(the CSV's SHA-256 is checked, so it is byte-identical). The cell is collapsed
in Colab (#@title form) and in Jupyter / VS Code (source_hidden). After it has
run, the helper files can be opened from Colab's Files panel.

Re-run after changing the main notebook or src/:
    python tools/build_colab_notebook.py
"""

import base64
import copy
import hashlib
import io
import tarfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "notebooks" / "telco_churn_analysis.ipynb"
OUT = ROOT / "notebooks" / "telco_churn_analysis_colab.ipynb"
SRC_FILES = ["__init__.py", "config.py", "data_cleaning.py", "eda_plots.py", "stats_tests.py",
             "modeling.py", "model_plots.py", "interpretation.py"]
CSV = "data/raw/Telco_Customer_Churn.csv"
CSV_SHA256 = hashlib.sha256((ROOT / CSV).read_bytes()).hexdigest()

# Bundle src/ and the raw CSV into one compressed, base64-encoded tar archive
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
    for rel in [f"src/{name}" for name in SRC_FILES] + [CSV]:
        info = tar.gettarinfo(ROOT / rel, arcname=rel)
        info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""   # reproducible archive
        with open(ROOT / rel, "rb") as handle:
            tar.addfile(info, handle)
blob = base64.b64encode(buffer.getvalue()).decode()
wrapped = "\n".join(blob[i:i + 100] for i in range(0, len(blob), 100))

setup = nbf.v4.new_code_cell(f'''#@title ▶ Colab setup – run first (recreates the project's src/ folder and dataset)
# Colab only receives this .ipynb file, so this cell rebuilds the rest of the project
# (helper code in src/ and the raw CSV) from the compressed copy below.
# Everything after this cell is identical to telco_churn_analysis.ipynb.
import base64, hashlib, io, os, sys, tarfile
from pathlib import Path

BASE = Path("/content/telco-churn-ml") if "google.colab" in sys.modules else Path.cwd() / "telco-churn-ml-colab"
for sub in ["notebooks", "data/processed", "outputs/figures", "outputs/tables", "outputs/models"]:
    (BASE / sub).mkdir(parents=True, exist_ok=True)

PROJECT_FILES_B64 = """
{wrapped}
"""
with tarfile.open(fileobj=io.BytesIO(base64.b64decode("".join(PROJECT_FILES_B64.split()))), mode="r:gz") as tar:
    # "data" filter: only plain files inside BASE (and no warning on newer Pythons)
    tar.extractall(BASE, **({{"filter": "data"}} if hasattr(tarfile, "data_filter") else {{}}))

csv_hash = hashlib.sha256((BASE / "{CSV}").read_bytes()).hexdigest()
assert csv_hash == "{CSV_SHA256}", "dataset copy is corrupted"
os.chdir(BASE / "notebooks")          # run the analysis from notebooks/, exactly like the full project
print(f"Project recreated in {{BASE}} – the analysis below can now run.")''')
setup.metadata = {"cellView": "form", "jupyter": {"source_hidden": True}}

nb = nbf.read(MAIN, as_version=4)
colab = copy.deepcopy(nb)
# Setup cell goes right below the title, so the notebook opens exactly like the main one
at = 1 if nb.cells and nb.cells[0].cell_type == "markdown" else 0
colab.cells = colab.cells[:at] + [setup] + colab.cells[at:]
colab.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
colab.metadata["colab"] = {"provenance": []}
nbf.write(colab, OUT)
print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB, "
      f"{len(nb.cells)} cells copied from the main notebook + 1 setup cell)")
