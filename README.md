# Membership Inference Attacks against Generative Models

## Description
SAP Security Research sample code to reproduce the research done in our paper "Monte Carlo and Reconstruction Membership Inference Attacks against Generative Models" [1].

### Folder: Membership Inference Attacks against Generative Models
Code for Monte Carlo attacks and reconstruction attacks on different datasets. The code for a specific dataset is in the corresponding folder. Though the code base varies there is a general structure for the experiments. First, you have to train a model which is then saved in some folder such as /models. The models are always trained on a random subset of the entire dataset. For the attacks, it is necessary to know which records were selected. Hence, when loading the data the train_inds.csv and percentage.csv files are saved in the directory. They fully specify the training and test data. Consequently, the attacks need these files and the corresponding models. It is crucial that the files and models match. The attacks will generate csv files reporting the success rates of the attacks. These have to be copied into the evaluation folder. Afterward, the notebook can be run to reproduce the corresponding plots. To summarize, the general workflow is:

1. Train a model. After training, the model, the train_inds.csv and percentage.csv files are saved. (In case you want to change the parameters of the model such as dropout keep rates, you either have to provide them as parameters or change them manually in the python files.)
2. Attack the model. The python files require the saved model and the train_inds.csv and percentage.csv files. Again, they have to match. Otherwise, your success rates will constantly be 50%. (In case you have changed parameters, check that these changes are also made in the corresponding python files).

The commands for training and attacking are listed in the README.md files in the subfolders. In the following, we explain how the different tables and diagrams of the paper can be reproduced. For the first plot, we give a detailed explanation demonstrating the above principles.

**Fig. 3 MC attack accuracy on MNIST with PCA based distance against VAEs depending on sample size.** Everything you need is in the folder /Monte-Carlo-Attacks/Monte-Carlo-MNIST_CVAE. To train a model with 10 latent dimensions for 300 epochs run ```python run_main.py --dim_z 10 --num_epochs 300 ```. To evaluate the parameter n (number of samples) in five experiments for a model trained for 300 epochs run ```python mc_attack_cvae_sample_size.py 299 5```. A csv file will be generated. Copy this file into the folder /Monte-Carlo-Attacks/Monte-Carlo-MNIST_CVAE/Evaluation_Sample_Size and rename it to Param_Change.csv. Run the notebook Evaluation.ipynb to generate the plots used in the paper. Alternatively, you can run the mc_attack_cvae_sample_size.py script several times and merge the csv files into one. With Powershell this can be done via ```cmd /c copy  ((gci "*.csv" -Name) -join '+') "Param_Change.csv"```.

**Fig. 6/ Fig. 7 Generated samples of the trained models.** The MNIST and Fashion MNIST models output samples automatically during training. For CIFAR-10 there are notebooks (Create Images.ipynb and Sample From Generator.ipynb)

**Fig. 5 Average attack accuracy (differing scales) for single and set MI on the datasets.** The evaluation notebooks of the GANs and VAEs automatically save .npy files, such as y_set_cgan.npy. Copy the .npy files of the GANs into the evaluation folder of the VAE and run the comparison script. It will produce the diagrams.

**Single and set MI accuracy tables.** In contrast to the diagrams, the creation of the tables is not fully automated. However, the data can be extracted from the corresponding evaluation notebooks.

**Single and set MI accuracy tables for dropout on MNIST.** As stated above, the only change is that you have to specify the dropout either manually in the code or via parameters. For example, for VAEs on MNIST and Fashion MNIST, you can specify the parameter by running ```run_main.py --dim_z 10 --num_epochs 300 --percentage_train_data 0.1 --keep_prob 0.5 ``` for a keep rate of 50%. The attacks are run as usual.

**Single and set MI accuracy tables for varying training data sizes on MNIST.** You can also specify the parameter. For example, run ```run_main.py --dim_z 10 --num_epochs 300 --percentage_train_data 0.2 ``` to train a VAE on MNIST or Fashion MNIST on 20% of the training data. **Important:** If you change the percentage_train_data or num_epochs parameter they have to be passed to the attack manually. You have to change the source code manually. 



## Requirements
- [Python](https://www.python.org/) 3
- [Jupyter](https://jupyter.org/)
- [Tensorflow](https://github.com/tensorflow) version 1.13
- numpy, argparse
- Check further dependencies in the Jupyter notebooks...

## Downloads
Once the requirements (see above) are fulfilled the Python sample code can be run directly resp. in Jupyter notebooks on a Jupyter Server.


## Authors / Contributors

 - Benjamin Hilprecht
 - Daniel Bernau
 - Martin Härterich

 
## Known Issues
There are no known issues.


## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.


## Citations
If you use this code in your research, please cite:

```
@article{Hilprecht2019MonteCA,
  title={Monte Carlo and Reconstruction Membership Inference Attacks against Generative Models},
  author={Benjamin Hilprecht and Martin H{\"a}rterich and Daniel Bernau},
  journal={PoPETs},
  year={2019},
  volume={2019},
  pages={232-249}
}
```

## References
[1] Benjamin Hilprecht, Martin Härterich, and Daniel Bernau:
Monte Carlo and Reconstruction Membership Inference Attacks against Generative Models.
Proceedings on Privacy Enhancing Technologies; 2019 (4):232–249
https://petsymposium.org/2019/files/papers/issue4/popets-2019-0067.pdf


## License

This project is licensed under SAP Sample Code License Agreement except as noted otherwise in the [LICENSE file](SAP%20Sample%20Code%20License%20Agreement%20v1.0.pdf).




----------------


