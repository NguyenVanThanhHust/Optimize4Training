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
If you want to use jupyter in local machine
```
docker run --rm --name op4train_jupyter_ctn --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace -w /workspace dali_img jupyter notebook --NotebookApp.token='' --NotebookApp.password='' --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```