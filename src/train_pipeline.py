import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def pipeline(config):

	class_pipeline = Pipeline(
		[
			('classifier', eval(config.model['model'])())
		]
	)

	return class_pipeline