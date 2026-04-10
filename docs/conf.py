import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

project = "VAD"
copyright = "2026, Antje Farnier"
author = "Antje Farnier"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "imported-members": False,
}

autosummary_generate = True
autosummary_imported_members = False

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_preprocess_types = False

add_module_names = True
python_use_unqualified_type_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_mock_imports = [
    "torchmetrics",
]

suppress_warnings = [
    "ref.ref",
    "image.not_readable",
]

myst_enable_extensions = ["colon_fence"]
html_theme = "furo"
html_title = "VAD"
