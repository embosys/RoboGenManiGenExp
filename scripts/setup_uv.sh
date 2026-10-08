#!/usr/bin/env bash
# Bootstrap Gym's legacy metadata before uv sync, then build OMPL locally.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.9 --managed-python --seed
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 9), "Existing .venv must use Python 3.9"'
if ! .venv/bin/python -c 'from importlib.metadata import version; assert version("gym") == "0.21.0"' 2>/dev/null; then
  uv pip install --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    pip==23.0.1 setuptools==65.5.0 wheel==0.38.4
  .venv/bin/python -m pip --isolated install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-deps --no-build-isolation gym==0.21.0
fi
# This fixes only Gym's invalid optional dependency version, not its runtime code.
.venv/bin/python - <<'PYCODE'
from pathlib import Path
import sysconfig
metadata = Path(sysconfig.get_path("purelib")) / "gym-0.21.0.dist-info/METADATA"
metadata.write_text(metadata.read_text().replace("opencv-python (>=3.)", "opencv-python (>=3)"))
PYCODE
bash scripts/build_ompl_bindings.sh
uv run python run_external_scene.py --help
