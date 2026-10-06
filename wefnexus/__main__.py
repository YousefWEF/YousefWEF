"""Run the *wefnexus* command-line interface: ``python -m wefnexus <command> ...``.

See :mod:`wefnexus.cli` for the commands (``run``, ``allocate``,
``negotiate``, ``compare``, ``pareto``, ``report``, ``export-basin``) and the
exit codes (0 success, 1 no result, 2 usage error).
"""
from __future__ import annotations

import sys

from wefnexus.cli import main

if __name__ == "__main__":
    sys.exit(main())
