import os
import sys

# Support both `python -m biohacking_research.src` (relative import works)
# and `python path/to/__main__.py` (direct script execution, relative import fails).
if __package__:
    from .batch_search import main
else:
    # Add the repo root (two levels up from this file) to sys.path so that
    # `biohacking_research.src.batch_search` can be found as an absolute import.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from biohacking_research.src.batch_search import main

if __name__ == "__main__":
    raise SystemExit(main())
