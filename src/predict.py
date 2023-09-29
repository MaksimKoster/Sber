import mlflow
from mlflow import sklearn

import hydra
from hydra import utils

import warnings
import pickle
import os

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing

@hydra.main(config_path='../configs',config_name='parameters')
def main(config):
	warnings.filterwarnings("ignore")
	np.random.seed(40)

	X_test = pd.read_csv(utils.get_original_cwd() + '/' + config.processed_data.features.test)
	y_test = pd.read_csv(utils.get_original_cwd() + '/' + config.processed_data.target.test)

	mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')

	current_experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
	experiment_id = current_experiment.experiment_id

	req = 'tags.mlflow.runName = ' + '\'best-' + (config['model']['model']) + '\''
	res = mlflow.search_runs([experiment_id], filter_string=req)

	mod_name, run_id = res.iloc[0]['tags.mlflow.runName'], res.iloc[0]['run_id']
	model = mlflow.sklearn.load_model(utils.to_absolute_path(f"mlruns/{experiment_id}/{run_id}/artifacts/{mod_name}"))
		
	labels_pred = model.predict(X_test)

	print('------------')
	print('\nconfusion_matrix:') 
	print(confusion_matrix(y_test, labels_pred))
	print('\nf1_score: ', metrics.f1_score(y_test, labels_pred, average='weighted'))
	print('------------')

if __name__ == '__main__':
	main()