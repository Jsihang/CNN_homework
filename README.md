# CIFAR-10 GAN Project

This project implements image generation on CIFAR-10 with PyTorch. It supports a
DCGAN baseline and a WGAN-GP variant, saves generated samples and loss curves,
and provides an evaluation script for FID and Inception Score.

## Environment

Use the existing conda environment:

```bash
conda activate nern
```

The code reads the extracted CIFAR-10 files directly from:

```text
/data1/nHome1/xieqihu/dataset/CV/CIFAR10_data/cifar-10-batches-py/
```

No dataset download is required.

## Quick Smoke Test

Run a short CPU/GPU pipeline test:

```bash
conda run -n nern python train.py --fast-dev-run --epochs 1 --batch-size 32 --num-workers 0
```

Expected outputs:

```text
outputs/dcgan/config.json
outputs/dcgan/history.csv
outputs/dcgan/loss_curve.png
outputs/dcgan/samples/real_cifar10.png
outputs/dcgan/samples/epoch_001.png
outputs/dcgan/checkpoints/latest.pt
```

## Train Models

Train the DCGAN baseline:

```bash
conda run -n nern python train.py \
  --gan-type dcgan \
  --out-dir outputs/dcgan \
  --epochs 50 \
  --batch-size 128
```

Train the WGAN-GP improved model:

```bash
conda run -n nern python train.py \
  --gan-type wgan-gp \
  --out-dir outputs/wgan_gp \
  --epochs 50 \
  --batch-size 128 \
  --n-critic 5 \
  --gp-lambda 10
```

Train the stronger class-conditional ResNet GAN with DiffAugment and EMA:

```bash
conda run -n nern python train_conditional.py \
  --arch resnet \
  --diffaugment \
  --ema \
  --out-dir outputs/conditional_resnet_aug_ema \
  --epochs 200 \
  --batch-size 128 \
  --g-channels 128 \
  --d-channels 128 \
  --device cuda
```

If the current node has no CUDA device, reduce `--epochs`, `--batch-size`, or use
`--max-train-items` for debugging. For final report numbers, run on a GPU if
available.

## Generate Images

```bash
conda run -n nern python sample.py \
  --checkpoint outputs/dcgan/checkpoints/latest.pt \
  --out outputs/dcgan/final_grid.png \
  --num-images 64
```

Generate a labeled class-conditional grid:

```bash
conda run -n nern python sample_conditional.py \
  --checkpoint outputs/conditional_resnet_aug_ema/checkpoints/latest.pt \
  --out outputs/conditional_resnet_aug_ema/labeled_grid.png \
  --samples-per-class 8 \
  --device cuda
```

## Evaluate

FID and Inception Score should be reported with pretrained Inception weights.
If the weights are already cached in the environment:

```bash
conda run -n nern python evaluate.py \
  --checkpoint outputs/dcgan/checkpoints/latest.pt \
  --out outputs/dcgan/metrics.json \
  --num-samples 5000
```

On a machine that can access the network, add `--allow-download` if torchvision
needs to fetch the weights. For code-path debugging only, use
`--untrained-inception-ok`; numbers produced in that mode are not valid final
metrics.

## Suggested Report Structure

1. Task definition: CIFAR-10 generation with GAN.
2. Method: DCGAN architecture, adversarial objective, WGAN-GP improvement.
3. Experimental setup: dataset path, preprocessing to `[-1, 1]`, optimizer,
   learning rate, batch size, epochs, device.
4. Results: generated samples across epochs, final sample grid, loss curves,
   FID and IS table.
5. Analysis: DCGAN stability, WGAN-GP effect, visual quality, failure cases such
   as blurry samples or mode collapse.
6. Conclusion: summarize whether WGAN-GP improves stability and image quality.
