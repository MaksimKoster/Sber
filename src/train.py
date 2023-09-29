import csv
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import mlflow
import mlflow.sklearn

import warnings
import hydra
from hydra import utils

import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

import os
from train_pipeline import pipeline

@hydra.main(config_path='../configs', config_name='parameters')
def main(config):
	warnings.filterwarnings("ignore")
	np.random.seed(40)

	X_train = pd.read_csv(utils.get_original_cwd() + '/' + config.processed_data.features.train)
	X_test = pd.read_csv(utils.get_original_cwd() + '/' + config.processed_data.features.test)
	y_train = pd.read_csv(utils.get_original_cwd() + '/' + config.processed_data.target.train)
	y_test = pd.read_csv(utils.get_original_cwd() + '/' + config.processed_data.target.test)
	class_pipeline = pipeline(config)

	mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
	mlflow.set_experiment(config.mlflow.experiment_name)

	with mlflow.start_run(
							tags={
                             "mlflow.note.content": config['model']
                            },
							run_name='best-{}'.format(config['model']['model'])
							):

		mlflow.log_input(mlflow.data.from_pandas(X_train), context="train_features")
		mlflow.log_input(mlflow.data.from_pandas(X_test), context="test_features")
		mlflow.log_input(mlflow.data.from_pandas(y_train), context="train_target")
		mlflow.log_input(mlflow.data.from_pandas(y_test), context="test_target")

		param_grid = dict(config['model']['hyperparameters'])

		for key, value in param_grid.items():
			if isinstance(value, str):
				param_grid[key] = eval(param_grid[key])

		param_grid = {ele: (list(param_grid[ele])) for ele in param_grid}			
		
		grid_search = GridSearchCV(class_pipeline, param_grid=param_grid, scoring=config.GridSearchCV.scoring, cv=config.GridSearchCV.cv, n_jobs=config.GridSearchCV.n_jobs)

		grid_search.fit(X_train, y_train.values.ravel())

		labels_pred = grid_search.predict(X_test)

		print(confusion_matrix(y_test, labels_pred))
		print(metrics.f1_score(y_test, labels_pred, average='macro'))

		mlflow.log_params(grid_search.best_params_)
		mlflow.log_artifacts(utils.to_absolute_path("configs"))
		mlflow.log_metric('f1_macro', eval(config.metrics.score)(y_test, labels_pred, average = config.metrics.average))

		mlflow.sklearn.log_model(grid_search, 'best-{}'.format(config['model']['model']))
		
		#mlflow.sklearn.save_model(grid_search, utils.to_absolute_path('models/best-{}'.format(config['model']['model'])))

if __name__== "__main__":
	main()