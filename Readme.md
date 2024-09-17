# Optimize for training

Run mnist with DALI and pytorch lightning
```
python mnist_dali.py --learning_rate 0.001 --hidden_dim 128 --accelerator cuda
```

```
python train_dali.py --learning_rate 0.001 --accelerator cuda
```

## Install
Build docker image
```
docker build -t ffcv_img -f ./dockers/ffcv.Dockerfile ./dockers/
```
Run docker container
```
docker run --rm --name opt4train_ctn -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -w /workspace -it ffcv_img:latest python baseline.py --lr 0.05
```
Or debug
```
docker run --name opt4train_ctn -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -w /workspace/ -it ffcv_img:latest /bin/bash
```

## Reference
[ffcv_docker](https://github.com/kschuerholt/pytorch_cuda_opencv_ffcv_docker)
[ffcv with cifar](https://github.com/libffcv/ffcv/tree/main/examples/cifar)
[94% on CIFAR-10 in 3.29 Seconds on a Single GPU](https://arxiv.org/abs/2404.00498)