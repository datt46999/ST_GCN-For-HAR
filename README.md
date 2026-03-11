## 📖 Introduction 
Model using Spatio-Temporal Graph Convolutional Networks (STGCN) for human action recognite.
This model implementation in this repository is inspired by the architecture and training pipeline used in [MMAction2](https://github.com/open-mmlab/mmaction2) which data was using NTURGB+D [2D Skeleton]

## 🛠️ Installation
```shell
conda create --name myenv python=3.10 -y
conda activate myenv
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install chumpy --no-build-isolation
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
mim install mmdet  
mim install mmpose  
git clone https://github.com/datt46999/ST_GCN-For-HAR.git
cd mmaction2
pip install -v -e .
```
## 👨‍🏫 Get Started

### Download dataset:[NTURGB+D_2d](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md)
### Train:
```shell
python tools/train.py configs/config.py 
```

### Test:
```shell 
python tools/test.py configs/config.py ${CHECKPOINT_FILE} [ARGS]
```
## 👀 Model pretrained :
Download: [Model](https://drive.google.com/drive/folders/1Y2UPp4tRIQCRlbGpkXU1KfZJYeo9vs6d?usp=sharing)

## 👨‍🏫 Deploy and results:

Val acc after traning in 16 epochs with RepeatDataset = 5 and each epoch: iter = 12529

<img width="620" height="487" alt="image" src="https://github.com/user-attachments/assets/03a26a31-f346-496f-a1e2-9dfed42d6c6c" />

Loss function:

<img width="641" height="514" alt="image" src="https://github.com/user-attachments/assets/5845ffc1-1c8a-4059-b152-771d77a0ca9b" />

Test in video: 

https://github.com/user-attachments/assets/bff59e4e-f630-473c-9cc7-b46cd1f7d8ac



https://github.com/user-attachments/assets/6f3dbf2c-1faf-4370-b172-843e0a6675a1





