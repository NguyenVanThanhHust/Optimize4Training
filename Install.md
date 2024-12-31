## Install
Build base training docker image 
```
docker build -t dali_img -f ./dockers/dali.Dockerfile ./dockers/
```

Build docker image for ffcv
```
docker build -t ffcv_img -f ./dockers/ffcv.Dockerfile ./dockers/
```

Run docker container
```
docker run --name opt4train_ctn -it --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -w /workspace -it dali_img:latest python baseline.py --lr 0.05
```
Or debug
```
docker run --name opt4train_ctn -it --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -w /workspace/ -it dali_img:latest /bin/bash
```
Then start with command
```
docker start opt4train_ctn && docker exec -it opt4train_ctn /bin/bash 
```
