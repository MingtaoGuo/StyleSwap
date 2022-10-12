# StyleSwap
Unofficial implementation of the paper: StyleSwap: Style-Based Generator Empowers Robust Face Swapping

## Description   
--------------

This repo is mainly to re-implement the face swap paper based on stylegan
- StyleSwap: [StyleSwap: Style-Based Generator Empowers Robust Face Swapping](https://arxiv.org/abs/2209.13514)

## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/MingtaoGuo/StyleSwap.git
cd StyleSwap
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment.yaml`.

### Train
- Download the face parsing model from [faceparsing](https://github.com/zllrunning/face-parsing.PyTorch)
- Download the face recognition model arcface from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) Go to  Baidu Drive **->** arcface_torch **->** glint360k_cosface_r50_fp16_0.1 **->** backbone.pth

- Generate face mask by faceparsing
``` 
python face_parsing/face_parsing.py --img_dir face_dataset --res_dir face_mask_dataset --resolution 256 --model_path saved_models/79999_iter.pth 
```
- Train StyleSwap 
``` 
python train_simple.py --img_path face_dataset --mask_path face_mask_dataset --arcface saved_models/backbone.pth 
```
### Inference
- Pretrained model is coming soon
## Author 
Mingtao Guo
E-mail: gmt798714378@hotmail.com

## Reference
[1]. Xu, Zhiliang, et al. "StyleSwap: Style-Based Generator Empowers Robust Face Swapping." arXiv preprint arXiv:2209.13514 (2022).
