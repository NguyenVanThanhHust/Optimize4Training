# Pytorch DALI demo with
Source: https://github.com/yaysummeriscoming/DALI_pytorch_demo

How to run 

Split mini image net data
```
python split.py --input ../../Datasets/mini_imagenet/ --output ../../Datasets/split_mini_imagenet/
```

```
python main.py /workspace/Datasets/split_mini_imagenet/
```