import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[2]
runtime_dir = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(runtime_dir / "matplotlib")
os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir / "jupyter")
os.environ["IPYTHONDIR"] = str(runtime_dir / "ipython")

for name in sys.argv[1:]:
    path = root / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    print("Executing", name, flush=True)
    client = NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print("Completed", name, flush=True)
