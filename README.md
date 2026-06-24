## environment
python: 3.11.15
OS: ubuntu

## how to make .venv
run this command in terminal. Don't forget installing python 3.11

python3.11 -m venv .venv

## attention!!!
don't deploy experiments folder

## how to run
You can start whole etu-ML function with shell commands below

cd [path to etu-ML]
python -m experiments.CharSeg.protetypes.app

## how to tune parameters

- related to debugging
parameters is all assosiated in app.py
Please change valuables inside a class 'CharacterSegmentationConfig'.

- related to each steps
parameters are separated into each files containing its function.
Please change valuables inside classes '*Config'.

ex) AstarNormalConfig