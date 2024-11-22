## Install
Build base training docker image 
```
docker build -t pytorch_img -f ./dockers/pytorch.Dockerfile ./dockers/
```

Build docker image for ffcv
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