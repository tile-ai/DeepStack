#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${DEEPSTACK_AE_ENV:-deepstack-ae}"
cd "${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniconda/Anaconda and rerun." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update --name "${ENV_NAME}" --file "${ROOT_DIR}/environment.yml" --prune
else
  conda env create --name "${ENV_NAME}" --file "${ROOT_DIR}/environment.yml"
fi

conda run --name "${ENV_NAME}" python -m pip install --no-deps --editable "${ROOT_DIR}"
conda run --name "${ENV_NAME}" python -m ae.cli doctor

echo "Environment '${ENV_NAME}' is ready."
