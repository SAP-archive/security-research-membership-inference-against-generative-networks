# DCGAN-CIFAR10

```
source activate tensorflow_p36
```

# train a model
```
python main.py --gan_type GAN --dataset cifar10 --epoch 2500 && sudo shutdown -P now
```

# attack
```
python mc_attack.py 5 && python mc_attack.py 5 && python mc_attack.py 5 && python mc_attack.py 5 && sudo shutdown -P now
```