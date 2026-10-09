"""Execute the training notebook using this environment's registered kernel."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JUPYTER_PATH", str(ROOT / "venv/share/jupyter"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / "venv/.jupyter-runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / "venv/.ipython"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "venv/.matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

import nbformat
from nbclient import NotebookClient

path = ROOT / "keypoint_classification.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook, timeout=3600, kernel_name="sign-language",
    resources={"metadata": {"path": str(ROOT)}},
    on_cell_start=lambda cell, cell_index: print(f"cell {cell_index}: running", flush=True),
)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
for cell in notebook.cells:
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            lines = output.text.strip().splitlines()
            print("\n".join(lines[-8:]))
print("NOTEBOOK PASS", flush=True)
