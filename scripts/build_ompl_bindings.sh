#!/usr/bin/env bash
# Build the author's OMPL version without installing or upgrading system packages.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/tools
if [[ ! -x output/tools/bin/micromamba ]]; then
  curl -L --fail https://micro.mamba.pm/api/micromamba/linux-64/latest -o output/tools/micromamba.tar.bz2
  tar -xjf output/tools/micromamba.tar.bz2 -C output/tools bin/micromamba
fi
task_ompl_action=create
[[ ! -d output/ompl_env/conda-meta ]] || task_ompl_action=install
output/tools/bin/micromamba "$task_ompl_action" -y -r output/mamba -p output/ompl_env --override-channels -c conda-forge \
  python=3.9 ompl=1.5.2 libboost=1.82 libboost-headers=1.82 libboost-python-devel=1.82 eigen=3.4.0 cmake make
task_castxml_action=create
[[ ! -d output/castxml_env/conda-meta ]] || task_castxml_action=install
output/tools/bin/micromamba "$task_castxml_action" -y -r output/mamba -p output/castxml_env --override-channels -c conda-forge castxml=0.7.0
.venv/bin/python -m pip install pygccxml==2.2.1 pyplusplus==1.8.5
if [[ ! -d output/tools/ompl-1.5.2 ]]; then
  curl -L --fail https://github.com/ompl/ompl/archive/refs/tags/1.5.2.tar.gz -o output/tools/ompl-1.5.2.tar.gz
  tar -xzf output/tools/ompl-1.5.2.tar.gz -C output/tools
fi
task_prefix="$PWD/output/ompl_env"
# pygccxml 2 resolves std::string to its canonical type; only the binding selector changes.
.venv/bin/python - <<'PY'
from pathlib import Path
path = Path('output/tools/ompl-1.5.2/py-bindings/generate_bindings.py')
path.write_text(path.read_text().replace('std::string', 'std::basic_string<char>')
                .replace('ScopedState< ompl::base::StateSpace >', 'ScopedState<>')
                .replace("planner.name != 'QRRT'", "planner.alias != 'QRRT'"))
PY
output/ompl_env/bin/cmake -S output/tools/ompl-1.5.2 -B output/tools/ompl-1.5.2/build \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_PREFIX_PATH="$task_prefix" -DBOOST_ROOT="$task_prefix" \
  -DPYTHON_EXEC="$PWD/.venv/bin/python" -DCASTXML="$PWD/output/castxml_env/bin/castxml" \
  -DOMPL_BUILD_DEMOS=OFF -DOMPL_BUILD_TESTS=OFF -DOMPL_BUILD_PYTESTS=OFF -DOMPL_BUILD_PYBINDINGS=ON
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
