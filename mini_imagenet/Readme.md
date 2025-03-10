# Pytorch DALI demo with
Source: https://github.com/yaysummeriscoming/DALI_pytorch_demo

How to run 

Split mini image net data
```
python split.py --input ../../Datasets/mini_imagenet/ --output ../../Datasets/split_mini_imagenet/
```

Train base model
```
python train.py model=mini_imnet_resnet hparams=resnet_baseline
```


Train model with dali loader
```
python main_dali.py /workspace/Datasets/split_mini_imagenet/
```