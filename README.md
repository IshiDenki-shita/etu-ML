## environment
python: 3.11.15
OS: ubuntu

## how to make .venv
run this command in terminal. Don't forget installing python 3.11

python3.11 -m venv .venv

## how to run

- CharSeg
cd [path to etu-ML]
python -m CharSeg.app

## how to tune parameters (CharSeg)

- related to debugging
parameters is all assosiated in app.py
Please change valuables inside a class 'CharacterSegmentationConfig'.

- related to each steps
parameters are separated into each files containing its function.
Please change valuables inside classes '*Config'.

ex) AstarNormalConfig
