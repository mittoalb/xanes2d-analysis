# Sphinx configuration — xanes2d-analysis user guide.
# These pages are the source; tools/sync_docs.sh copies them into the
# 32-ID TXM Software User Guide (32id-gui-docs, section xanes2d_analysis).
project = "xanes2d-analysis"
author = "APS 32-ID"
copyright = "2026, APS 32-ID"

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_title = "xanes2d-analysis — User Guide"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
