# Conditional Variational Auto-Encoder for MNIST ATTACK

## To train and save a CVAE on MNIST run
Adjust number of epochs both in here and the attacks!
Due to numerical errors sometimes loss functions become NAN. Just restart in this case!
```
source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 --percentage_train_data 0.1 --PRR_n_img_x 14 --PRR_n_img_y 14
```

```
source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 --percentage_train_data 0.4 --PRR_n_img_x 14 --PRR_n_img_y 14
```

```
source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 --percentage_train_data 0.1 --keep_prob 0.5
```
```
source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 --percentage_train_data 0.1 --keep_prob 0.7
```
```
source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 400 --percentage_train_data 0.05
```
```
source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 --percentage_train_data 0.2
```

## To run monte carlo attacks
For the attacks you always have to specify epochs of trained model (first param) and no of experiments (2nd param). The calc_rec_error.py module has the epochs hard-coded. Must be manually corrected.

```
source activate tensorflow_p36 && python mc_attack_cvae.py 149 5 && python mc_attack_cvae.py 149 5 && python mc_attack_cvae.py 149 5 && python mc_attack_cvae.py 149 5 && sudo shutdown -P now
```
```
python mc_attack_cvae.py 399 5 && python mc_attack_cvae.py 399 5 && sudo shutdown -P now
```
```
python mc_attack_cvae.py 149 5 && python mc_attack_cvae.py 149 5 && sudo shutdown -P now
```
```
source activate tensorflow_p36 && python mc_attack_bagging.py 149 5 && python mc_attack_bagging.py 149 5 && python mc_attack_bagging.py 149 5 && python mc_attack_bagging.py 149 5 && sudo shutdown -P now
```

## To run ais attack
Currently only works with 10%
```
source activate tensorflow_p36 && for i in $(seq 0 15); do echo $i; python ais_attack.py 299; done; sudo shutdown -P now
```

## To run reconstruction attack
```
source activate tensorflow_p36 && python reconstruction_attack.py 249 5 && python reconstruction_attack.py 249 5 && python reconstruction_attack.py 249 5 && python reconstruction_attack.py 249 5 && python reconstruction_attack.py 249 5 && sudo shutdown -P now
```
```
source activate tensorflow_p36 && python reconstruction_attack.py 299 5 && python reconstruction_attack.py 299 5 && python reconstruction_attack.py 299 5 && python reconstruction_attack.py 299 5 && python reconstruction_attack.py 299 5 && sudo shutdown -P now
```

## To evaluate n of reconstruction attack
```
source activate tensorflow_p36 && python reconstruction_attack_param_n.py 299 5 && python reconstruction_attack_param_n.py 299 5 && python reconstruction_attack_param_n.py 299 5 && python reconstruction_attack_param_n.py 299 5 && sudo shutdown -P now
```

## To run black box attack
Currently only works with 10%
1. dependent on: empty /bb_vis folder, empty /bb_models folder 
2. sample from existing vae:
```
python gen_samples.py
```
3. train gan on these samples: 
```
python train_small.py
```
4. black box attack:
```
for i in $(seq 0 10); do echo $i; python bb_attack.py 1 100; done; sudo shutdown -P now
```
5. combined:
```
mkdir bb_vis && mkdir bb_models && source activate tensorflow_p36 && pip install pillow && pip install scikit-image && pip install scikit-learn && python run_main.py --dim_z 10 --num_epochs 300 && python gen_samples.py && python train_small.py && for i in $(seq 0 10); do echo $i; python bb_attack.py 1 100; done && sudo shutdown -P now
```
