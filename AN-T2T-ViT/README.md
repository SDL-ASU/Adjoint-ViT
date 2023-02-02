## 1. Requirements
### Pip

[timm](https://github.com/rwightman/pytorch-image-models), pip install timm

pip3 install torch torchvision torchaudio

pip install PyYAML

###  Conda
[timm] conda install -c conda-forge timm

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

conda install -c conda-forge pyyaml


### Data prepare
ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## 2. Validation

Download the [T2T-ViT-AN], then test it by running:

```
bash distributed_train_adjoined.sh 4 path/to/data/ --model t2t_vit_14 -b 128 --lr 1e-3 --weight-decay .05 --img-size 224 --num-classes 1000 --epochs 1 --output /output --initial-checkpoint path/to/checkpoint --eval_checkpoint --compression_factor 2 --block_depth 2
```
The results look like:

```
Top-1 accuracy of the large model is: 81.0%
Top-1 accuracy of the small/compressed model is: 80.4%

```

Download the [T2T-ViT-DAN], then test it by running:

```
bash distributed_train_adjoined.sh 4 path/to/data/ --model t2t_vit_14 -b 128 --lr 1e-3 --weight-decay .05 --img-size 224 --num-classes 1000 --epochs 1 --output /output --initial-checkpoint path/to/checkpoint --eval_checkpoint --compression_factor 1 1 1 1 3 2 3 1 3 1 1 1 1 1 --block_depth 4 --DAN_training
```


The results look like:

```
Top-1 accuracy of the large model is: 81.5%
Top-1 adjoined accuracy of the small model is: 81.3%

```

## 3. Train

Train the T2T-ViT-14 (AN) (run on 4 GPUs) (from scratch):

```
bash distributed_train_adjoined.sh 4 path/to/data --model t2t_vit_14 -b 128 --lr 1e-3 --weight-decay .05 --img-size 224 --num-classes 1000 --epochs 300 --output ./output --compression_factor 2 --block_depth 2
```

Train the T2T-ViT-14 (AN) (run on 4 GPUs) (from T2T-ViT-14 pretrained weights):

```
bash distributed_train_adjoined.sh 3 path/to/data --model t2t_vit_14 -b 128 --lr 1e-3 --weight-decay .05 --img-size 224 --num-classes 1000 --epochs 300 --pretrained_normal path/to/checkpoint --output ./output --compression_factor 2 --block_depth 2
```

Train the T2T-ViT-14 (DAN) (run on 4 GPUs) (from scratch):

```
bash distributed_train_adjoined.sh 4 path/to/data --model t2t_vit_14 -b 128 --lr 1e-3 --weight-decay .05 --img-size 224 --num-classes 1000 --epochs 300 --output ./output --compression_factor 1 1 1 1 3 2 3 1 3 1 1 1 1 1 --block_depth 4 --DAN_training
```

Train the T2T-ViT-14 (DAN) (run on 4 GPUs) (from T2T-ViT-14 pretrained weights):

```
bash distributed_train_adjoined.sh 4 path/to/data --model t2t_vit_14 -b 128 --lr 1e-3 --weight-decay .05 --img-size 224 --num-classes 1000 --epochs 300 --pretrained_normal path/to/checkpoint --output ./output --compression_factor 1 1 1 1 3 2 3 1 3 1 1 1 1 1 --block_depth 4 --DAN_training
```


If you want to train our T2T-ViT on images with 384x384 resolution, please use '--img-size 384'.



