## 📖 Introduction 
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

### Download dataset: https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md
### Train:
```shell
python tools/train.py configs/config.py 
```

### Test:
'''shell 
python tools/test.py configs/config.py ${CHECKPOINT_FILE} [ARGS]
'''
## 👀 Model pretrained :
Download: https://drive.google.com/file/d/1gnpbIvQQ-VzlxPK-0MC0hCiaA24AOjco/view?usp=sharing
Deploy results:
Val acc after traning in 16 epochs with RepeatDataset = 5 and each epoch: iter = 12529

<img width="620" height="487" alt="image" src="https://github.com/user-attachments/assets/03a26a31-f346-496f-a1e2-9dfed42d6c6c" />

Loss function:

<img width="611" height="474" alt="image" src="https://github.com/user-attachments/assets/64e660c1-2885-4bf9-aeaf-7e798d60e210" />




