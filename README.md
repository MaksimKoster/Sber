# Diabets
## Тестовое задание для команды "AI и исследование данных" в ДРКБ.   

## Data
`diabets/data/raw/diabetes_train_analysis.csv` - датасет для обучения, инфа о здоровье  
`diabets/data/raw/diabetes_train_info.csv` - датасет для обучения, инфа о пациенте    
`diabets/data/raw/diabetes_test_analysis.csv` - датасет для тестирования, инфа о здоровье  
`diabets/data/raw/diabetes_test_info.csv` - датасет для тестирования, инфа о пациенте  

## Techs
- [MLflow](https://mlflow.org) - An open source platform for the machine learning lifecycle.
- [DVC](https://dvc.org) - Git-based data science. 
- [Hydra](https://hydra.cc) - Framework for elegantly configuring complex applications
- [Anaconda](https://www.anaconda.com) -  Python data science framework

## Installation

- Install conda from https://conda.io/docs/user-guide/install/

Install environment and libs.
```sh
bash create.sh
```
Activate environment
```sh
conda activate SberTask
```

## DVC (Get Data)
```sh
dvc pull
```

## Run
Run preprocess, tune, predict, pipeline
#### You can change parameters in make file
```sh
make preprocess
make tune
make predict
```
Run all three
```sh
make pipeline
```
#### MLflow
Run ML flow UI
```sh
mlflow ui
```
Access UI on **http://127.0.0.1:5000**
