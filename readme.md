# Inferring realistic binary parameters from stellar isochrones using machine learning

This repository is related to the master's thesis in computer science with the title stated above. I contains all the python files and Jupyter notebooks related to this thesis.

## Using the developped method

To use the method developed in the thesis (see image below), you need to run the file "first_method.py" in the "src\ML\first_method\" folder using the models in "model\final_models".

More explanation on how to use the method can be found in the file itself.

![alt text](https://github.com/anpirott/thesis/blob/master/img/mes_images/diagramme_2_ML.png?raw=True)

## Folders description

Inside each folder, a readme containing more information can be found, but a short description will be given here for each of them.

- data\ contains the two datasets which are used in this thesis.
- img\ contains the images which were used in the thesis.
- model\ contains the machine learning models which were trained for this  thesis.
- notebooks\ contains all the Jupyter notebooks which were created to test things and run the experiments on the models.
- predictions\ contains the saved predictions of all the experiments we performed on the models
- results\ contains the saved results (metrics and plots) of all the experiments we performed on the models
- src\ contains all the python files which are used to prepare the data, train the models, evaluate the models, ...

## Requirements

The requirements for the repository can be found in "requirement.txt" and "pyproject.toml".

How to install the requirement : 

- Using pip : `pip install -r requirements.txt`
- Using uv : `uv pip sync requirements.txt` or `uv pip sync pylock.toml`
