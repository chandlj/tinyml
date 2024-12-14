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

## Run

To run the experiments, you need to download the scales and quantize the model.

### Download scales

```bash
bash download_scales.sh
```

### Quantize

```bash
python entry.py --mode quantize --model [model_name]
```

### Test

```bash
python entry.py --mode test --model [model_name]
```

### Run All

All models can be quantized and tested with the following commands.

```bash
python entry.py --mode quantize --model all
python entry.py --mode test --model all
```
