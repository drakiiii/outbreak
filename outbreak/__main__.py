"""Enable ``python -m outbreak ...`` to invoke the command-line interface."""

import sys

from .cli import main

sys.exit(main())
