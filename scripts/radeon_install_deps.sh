#!/bin/bash
# Inside the container. The image already has the ROCm PyTorch.
# A venv created from /opt/venv does not see that PyTorch unless the image
# site-packages path is added back.
set -euo pipefail
if [ ! -x /workspace/venv/bin/python ]; then
  /opt/venv/bin/python -m venv --system-site-packages /workspace/venv
fi
echo /opt/venv/lib/python3.12/site-packages > /workspace/venv/lib/python3.12/site-packages/rocm-torch.pth
/workspace/venv/bin/python -m pip install -r /workspace/src/requirements-rocm.txt
/workspace/venv/bin/python -c 'import torch, transformers; print(torch.__version__, torch.version.hip, transformers.__version__, "devices", torch.cuda.device_count())'
