# AN-Swin-Transformer for Image Classification

This folder contains the implementation of the AN-Swin-Transformer for image classification.

## Usage

### Install

- Create a conda virtual environment and activate it:

```bash
conda create -n swin python=3.9 -y
conda activate swin
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

- Install `timm==0.4.12`:

```bash
conda install -c conda-forge timm
```

- Install other requirements:

```bash
opencv-python==4.4.0.46
termcolor==1.1.0
yacs==0.1.8
pyyaml
scipy
```

- Install fused window process for acceleration, activated by passing `--fused_window_process` in the running script
```bash
cd kernels/window_process
python setup.py install #--user
```

- Install apex to use fused layernorm for acceleration, activated by passing `--fused_layernorm` in the running script
```bash
gcc>=9.2.0
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cuda_ext" ./
```
Note: Without gcc, apex installation will likely cause issues

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```
- For ImageNet-22K dataset, make a folder named `fall11_whole` and move all images to labeled sub-folders in this
  folder. Then download the train-val split
  file ([ILSVRC2011fall_whole_map_train.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_train.txt)
  & [ILSVRC2011fall_whole_map_val.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt))
  , and put them in the parent directory of `fall11_whole`. The file structure should look like:

  ```bash
    $ tree imagenet22k/
    imagenet22k/
    ├── ILSVRC2011fall_whole_map_train.txt
    ├── ILSVRC2011fall_whole_map_val.txt
    └── fall11_whole
        ├── n00004475
        ├── n00005787
        ├── n00006024
        ├── n00006484
        └── ...
  ```

### Evaluation

To evaluate a pre-trained `AN-Swin-Transformer` on ImageNet val, run:

For example, to evaluate the `Swin-B-AN` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main_adjoined.py --eval \
--cfg configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml --pretrained swin_base_patch4_window7_224_22k.pth \
--resume Swin-B-AN.pth --data-path path/to/data --disable_amp
```
The results look like this:
```
INFO Accuracy of the network on the 50000 test images: 85.23%
INFO Adjoined Accuracy of the network on the 50000 test images: 85.05%
```
For example, to evaluate the `Swin-B-DAN` with a single GPU:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main_adjoined.py --eval \
--cfg configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml --pretrained swin_base_patch4_window7_224_22k.pth \
--resume Swin-B-DAN.pth --data-path path/to/data --disable_amp opts DAN_TRAINING True
```
The results look like this:
```
INFO Accuracy of the network on the 50000 test images: 85.3%
INFO Adjoined Accuracy of the network on the 50000 test images: 85.27%
```
### Training on ImageNet-1K pretrained Swin-B

To train a `AN-Swin-Transformer` on ImageNet, run:

For example, to train the `Swin-B-AN`:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main_adjoined.py \
--cfg configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml --pretrained swin_base_patch4_window7_224_22k.pth \
--data-path path/to/data --disable_amp --batch_size <batch-size> \
--opts TRAIN.BASE_LR 1e-03 TRAIN.MIN_LR 1e-03 TRAIN.WARMUP_LR 1e-03
```

For example, to train the `Swin-B-DAN`:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main_adjoined.py \
--cfg configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml --pretrained swin_base_patch4_window7_224_22k.pth \
--data-path path/to/data --disable_amp --batch_size <batch-size> \
--opts TRAIN.BASE_LR 1e-03 TRAIN.MIN_LR 1e-03 TRAIN.WARMUP_LR 1e-03 DAN_TRAINING True
```


**Notes**:

- To use zipped ImageNet instead of folder dataset, add `--zip` to the parameters.
    - To cache the dataset in the memory instead of reading from files every time, add `--cache-mode part`, which will
      shard the dataset into non-overlapping pieces for different GPUs and only load the corresponding one for each GPU.
- When GPU memory is not enough, you can try the following suggestions:
    - Use gradient accumulation by adding `--accumulation-steps <steps>`, set appropriate `<steps>` according to your need.
    - Use gradient checkpointing by adding `--use-checkpoint`, e.g., it saves about 60% memory when training `Swin-B-AN`.
      Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
    - We recommend using multi-node with more GPUs for training very large models, a tutorial can be found
      in [this page](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
- To change config options in general, you can use `--opts KEY1 VALUE1 KEY2 VALUE2`, e.g.,
  `--opts TRAIN.EPOCHS 100 TRAIN.WARMUP_EPOCHS 5` will change total epochs to 100 and warm-up epochs to 5.
- For additional options, see [config_adjoined](config_adjoined.py) and run `python main_adjoined.py --help` to get detailed message.