<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

## Setup
```
COMPETITION_NAME='XXX'

git clone https://github.com/Ynakatsuka/$COMPETITION_NAME
cd $COMPETITION_NAME

# Credentials
cp .env.template .env
vim .env  # Set your credentials

# Download data
cd data/input/ && kaggle competitions download -c $COMPETITION_NAME && unzip $COMPETITION_NAME.zip  && cd ../..

# Create Docker container and execute
./bin/docker.sh
```

## How to run
```yaml
# train
python run.py
# train with specified config
python run.py augmentation=default
# train with specified parameters
python run.py trainer.model.params.backbone.name=densenet121

# inference
python run.py run=inference
```

## Others
<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```yaml
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python run.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```
> ⚠️ Currently sweeps aren't failure resistant (if one job crashes than the whole sweep crashes), but it will be supported in future Hydra release.

</details>

<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.<br>
Simply install pre-commit package with:
```yaml
pip install pre-commit
```
Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):
```yaml
pre-commit install
```
After that your code will be automatically reformatted on every new commit.<br>
Currently template contains configurations of **black** (python code formatting), **isort** (python import sorting), **flake8** (python code analysis) and **prettier** (yaml formating). <br>

To reformat all files in the project use command:
```yaml
pre-commit run -a
```

</details>

<details>
<summary><b>Version control your data and models with DVC</b></summary>

Use [DVC](https://dvc.org) to version control big files, like your data or trained ML models.<br>
To initialize the dvc repository:
```yaml
dvc init
```
To start tracking a file or directory, use `dvc add`:
```yaml
dvc add data/MNIST
```
DVC stores information about the added file (or a directory) in a special .dvc file named data/MNIST.dvc, a small text file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the original data:
```yaml
git add data/MNIST.dvc data/.gitignore
git commit -m "Add raw data"
```

</details>

<details>
<summary><b>Support installing project as a package</b></summary>

It allows other people to easily use your modules in their own projects.
Change name of the `src` folder to your project name and add `setup.py` file:
```python
from setuptools import find_packages, setup

setup(
    name="src",  # you should change "src" to your project name
    version="0.0.0",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    # replace with your own github project link
    url="https://github.com/ashleve/lightning-hydra-template",
    install_requires=["pytorch-lightning>=1.2.0", "hydra-core>=1.0.6"],
    packages=find_packages(),
)
```
Now your project can be installed from local files:
```yaml
pip install -e .
```
Or directly from git repository:
```yaml
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```
So any file can be easily imported into any other file like so:
```python
from project_name.models.mnist_model import MNISTLitModel
from project_name.datamodules.mnist_datamodule import MNISTDataModule
```

</details>


## References
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
- [kvt](https://github.com/pudae/kaggle-understanding-clouds)

<br>
