# nst2046-yay
Repo for our NST2046 group project
## Training Stable Audio Model
Before starting your training run, you'll need a model config file, as well as a dataset config file. For more information about those, refer to the Configurations section below

The training code also requires a Weights & Biases account to log the training outputs and demos. Create an account and log in with:
```
$ wandb login
```
To start a training run, run the `NST Project Stable Audio Model Training.py` script in the repo root with:
```
$ python3 ./train.py --dataset-config /path/to/dataset/config --model-config /path/to/model/config --name harmonai_train
```
