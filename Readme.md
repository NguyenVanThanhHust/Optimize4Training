# Optimize for training

## TODO 

## How to run
Run mnist with DALI and pytorch lightning
```
python mnist_dali.py --learning_rate 0.001 --hidden_dim 128 --accelerator cuda
```

```
python train_dali.py --learning_rate 0.001 --accelerator cuda
```

## Result
| Data      | Device      | Method | Train time | Total Time |
| ------------- | ------------- |------------- |------------- |------------- |
| MNIST | GeForce RTX 3060 Mobile / Max-Q | Baseline pytorch | fad | dsfa| 
| Cifar | GeForce RTX 3060 Mobile / Max-Q | Baseline pytorch | fad | dsfa| 

## Reference
[ffcv_docker](https://github.com/kschuerholt/pytorch_cuda_opencv_ffcv_docker)
[ffcv with cifar](https://github.com/libffcv/ffcv/tree/main/examples/cifar)
[94% on CIFAR-10 in 3.29 Seconds on a Single GPU](https://arxiv.org/abs/2404.00498)
[COCO DALI Reader](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/coco_reader.html)
