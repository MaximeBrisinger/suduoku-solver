# Sudoku solver
This project aims at solving a Sudoku from a picture, which might have a poor quality (blurred, not upright, small...).

## How to run the project
Run the main file. The input `.jpg` file has to be in `data/` folder.
```bash
python main.py --input_file test4.jpg
```

## Prerequisites and Installation
Run the following command
```bash
conda create -n sudokuenv
```
```bash
conda activate sudokuenv
```
```bash
conda install pip
```
```bash
pip install -r requirements.txt
```

## Sources
Digits dataset found at : https://www.kaggle.com/datasets/preatcher/standard-fonts-digit-dataset-09?resource=download

## Contributors
Maxime Brisinger