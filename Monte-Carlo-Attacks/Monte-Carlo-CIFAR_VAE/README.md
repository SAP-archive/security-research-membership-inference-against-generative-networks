# VAE for Image Generation
Variational AutoEncoder - Keras implementation on mnist and cifar10 datasets
Taken from https://github.com/chaitanya100100/VAE-for-Image-Generation 

## Train a model 
python cifar10_train.py

## Attack
python mc_attack.py 1

source activate tensorflow_p36 && pip install opencv-python && pip install sklearn && python mc_attack.py 5 && python mc_attack.py 5 && python mc_attack.py 5 && python mc_attack.py 5 && sudo shutdown -P now

## Reconstruction Attack

```
source activate tensorflow_p36 && python reconstruction_attack.py 5 && python reconstruction_attack.py 5 && sudo shutdown -P now
```