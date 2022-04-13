set -euo pipefail

# formatting python
black . \
  --check

# formatting python
flake8 src/ \
  --max-line-length=88 # match black's line limit

# data static
mypy src/ \
  --ignore-missing-imports
