# tinyml

## Install

```bash
conda create -n tinyml python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate tinyml

pip install -e ./llm-awq
pip install ./smoothquant

cd llm-awq/awq/kernels
pip install .
```