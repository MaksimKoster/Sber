preprocess:
	python3 src/preprocess.py
tune:
	python3 src/train.py --multirun model=svc,randomforestclassifier,decisiontreeclassifier,kneighborsclassifier
predict:
	python3 src/predict.py --multirun model=svc,randomforestclassifier,decisiontreeclassifier,kneighborsclassifier
pipeline: preprocess tune predict