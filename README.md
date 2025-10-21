# Privacy-Preserving Person Re-Identification through Identity Retrieval and Hierarchical Latent Code Protection.


## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (Not mandatory bur recommended)
- Python 3


### Installation
- Dependencies:
	1. lpips
	2. wandb
	3. pytorch
	4. torchvision
	5. matplotlib
	6. dlib
- All dependencies can be installed using *pip install* and the package name


## Pretrained Models
Please download the pretrained models from the following link:

|[StyleGAN3](https://drive.google.com/file/d/1dhiPc29Leqq4d1MEx-ZDIFwxJHwgQzn_/view?usp=drive_link) | 
Pretrained generator model. Fine-tuned on the Market-1501, MSMT17, and CUHK03 datasets.

|[e4e encoder](https://drive.google.com/file/d/1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm/view?usp=sharing) | Pretrained e4e encoder. Used for StyleCLIP editing.

|[ResNet50 re-ID](https://drive.google.com/file/d/1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV/view) | Pretrained ResNet50 re-ID encoder. Used to extract the identity vector of an image.

|[Vector gallery](https://drive.google.com/file/d/165pxM2xUcahf85Xdv4_VF5MCBr1vojA2/view?usp=drive_link) | This is a vector gallery containing 19,732 identity features and latent codes.


This folder includes:
- **StyleGAN3_ada_reID** (used for image generation from latent codes)  
- **e4e_reID** (used to extract latent codes that will be stored in the vector gallery)  
- **ResNet50-reID** (used to extract identity features from original images and for vector gallery construction)
- **Vector gallery** (It is used for latent code retrieval based on identity features)

Note: The StyleGAN model is used directly from the official [stylegan3-ada-pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch).

By default, it is assumed that all pretrained models are downloaded and stored in the `pretrained_model` directory. 
However, you can specify your own paths by modifying the relevant values in `configs/path_configs.py`. 


## Running ProtectReID
The primary training script is `scripts/run.py`. It takes aligned and cropped images from the paths specified in the "Input info" subsection of `configs/paths_config.py`.

The results, including inversion latent codes and optimized generators, are saved to the directories listed under "Dirs for output files" in `configs/paths_config.py`.

The hyperparameters for the inversion task are defined in `configs/hyperparameters.py`, initialized with the default values used in the paper.
