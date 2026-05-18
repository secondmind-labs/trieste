python -m pip install --upgrade pip
pip install -e .
python3 -m pip install 'tensorflow[and-cuda]'
pip install -U tensorboard_plugin_profile
pip install .[qhsri] -r tests/latest/requirements.txt -c tests/latest/constraints.txt

export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_ORDER=PCI_BUS_ID