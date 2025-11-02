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

## Streamlit for interface

create a .env file in your project directory and add
```
GEMINI_API_KEY=<your_key>
```
create a venv (using at least python3.10)
```
py -3.10 -m venv venv
```
activate the venv (this is different for mac and windows so I'm not putting a command)
install requirements
```
pip install -r requirements.txt
```
from your terminal cd into the project directory and run
```
streamlit run streamlit_app.py
```
You can follow the localhost url it dislays to see the app
