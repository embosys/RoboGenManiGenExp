#!/usr/bin/env bash
# Build the author's OMPL version without installing or upgrading system packages.
set -euo pipefail
cd "$(dirname "$0")/.."
case "$(uname -s):$(uname -m)" in
  Darwin:arm64) task_platform=osx-arm64; task_compiler=/usr/bin/c++ ;;
  Linux:x86_64) task_platform=linux-64; task_compiler=/usr/bin/g++ ;;
  *) echo "Unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac
if [[ "$task_platform" == osx-arm64 ]]; then
  uv sync --group ompl-build
  # The torch 2.0 ARM wheel has an x86_64 tag internally despite its ARM binary.
  .venv/bin/python - <<'PYTORCH'
from pathlib import Path
import sysconfig
wheel = Path(sysconfig.get_path('purelib')) / 'torch-2.0.0.dist-info/WHEEL'
wheel.write_text(wheel.read_text().replace('macosx_11_0_x86_64', 'macosx_11_0_arm64'))
PYTORCH
else
  .venv/bin/python -m pip install pygccxml==2.2.1 pyplusplus==1.8.5
fi
mkdir -p output/tools
if [[ ! -x output/tools/bin/micromamba ]]; then
  curl -L --fail "https://micro.mamba.pm/api/micromamba/$task_platform/latest" -o output/tools/micromamba.tar.bz2
  tar -xjf output/tools/micromamba.tar.bz2 -C output/tools bin/micromamba
fi
task_ompl_action=create
[[ ! -d output/ompl_env/conda-meta ]] || task_ompl_action=install
output/tools/bin/micromamba "$task_ompl_action" -y -r output/mamba -p output/ompl_env --override-channels -c conda-forge \
  python=3.9 ompl=1.5.2 libboost=1.82 libboost-headers=1.82 libboost-python-devel=1.82 eigen=3.4.0 cmake make
task_castxml_action=create
[[ ! -d output/castxml_env/conda-meta ]] || task_castxml_action=install
output/tools/bin/micromamba "$task_castxml_action" -y -r output/mamba -p output/castxml_env --override-channels -c conda-forge castxml=0.7.0
if [[ ! -d output/tools/ompl-1.5.2 ]]; then
  curl -L --fail https://github.com/ompl/ompl/archive/refs/tags/1.5.2.tar.gz -o output/tools/ompl-1.5.2.tar.gz
  tar -xzf output/tools/ompl-1.5.2.tar.gz -C output/tools
fi
task_prefix="$PWD/output/ompl_env"
# pygccxml exposes std::string differently with libc++ and libstdc++; adjust selectors only.
.venv/bin/python - <<'PY'
from pathlib import Path
import sys
path = Path('output/tools/ompl-1.5.2/py-bindings/generate_bindings.py')
source = path.read_text()
if sys.platform == 'darwin':
    source = source.replace('std::basic_string<char>', 'std::string')
else:
    source = source.replace('std::string', 'std::basic_string<char>')
path.write_text(source.replace("decl.name.startswith('SpecificParam<std::basic_string')",
                               "decl.name.startswith(('SpecificParam<std::basic_string', 'SpecificParam<std::string'))")
                .replace('ScopedState< ompl::base::StateSpace >', 'ScopedState<>')
                .replace("planner.name != 'QRRT'", "planner.alias != 'QRRT'"))
PY
output/ompl_env/bin/cmake -S output/tools/ompl-1.5.2 -B output/tools/ompl-1.5.2/build \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_COMPILER="$task_compiler" -DCMAKE_PREFIX_PATH="$task_prefix" -DBOOST_ROOT="$task_prefix" \
  -DPYTHON_EXEC="$PWD/.venv/bin/python" -DCASTXML="$PWD/output/castxml_env/bin/castxml" \
  -DOMPL_BUILD_DEMOS=OFF -DOMPL_BUILD_TESTS=OFF -DOMPL_BUILD_PYTESTS=OFF -DOMPL_BUILD_PYBINDINGS=ON \
  -DOMPL_REGISTRATION=OFF
output/ompl_env/bin/cmake --build output/tools/ompl-1.5.2/build --target util.h base.h geometric.h -j 2
task_python="$PWD/.venv/bin/python"
(
  cd output/tools/ompl-1.5.2/build/py-bindings
  for task_module in util base geometric; do
    "$task_python" ../../py-bindings/generate_bindings.py "$task_module"
  done
)
for task_module in util base geometric; do
  test -d "output/tools/ompl-1.5.2/build/py-bindings/bindings/$task_module"
done
output/ompl_env/bin/cmake -S output/tools/ompl-1.5.2 -B output/tools/ompl-1.5.2/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
output/ompl_env/bin/cmake --build output/tools/ompl-1.5.2/build --target py_ompl_util py_ompl_base py_ompl_geometric -j 2
.venv/bin/python -c 'import site; from pathlib import Path; Path(site.getsitepackages()[0], "ompl.pth").write_text(str(Path("output/tools/ompl-1.5.2/py-bindings").resolve())+"\n")'
.venv/bin/python -c 'from ompl import base, geometric; print("OMPL 1.5.2 binding import OK")'
