import numpy as np 
import pandas as pd 

import hydra
from hydra import utils

import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import warnings

def get_age(x):
    x = str(x)
    if len(x) < 5:
        return int(x)
    else:
        res = int(round(int(x) / 365))
    return res

def upsample(features, target, repeat, upsampled_сlass):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    
    if upsampled_сlass == 0:
        features_upsampled = pd.concat([features_zeros]* repeat + [features_ones] )
        target_upsampled = pd.concat([target_zeros]* repeat + [target_ones] )
        features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
        
    elif upsampled_сlass == 1:
        features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
        features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    else:
        features_upsampled = 0
        target_upsampled = 0  
       
    return features_upsampled, target_upsampled

def process_data(df_test_analysis, df_test_info, 
                 df_train_analysis, df_train_info,
                 config):
    
    df_test = pd.merge(df_test_analysis, df_test_info, on="id")
    df_train = pd.merge(df_train_analysis, df_train_info, on="id")
    
    weight_na_const = -100
    
    df_train['weight'].fillna(weight_na_const, inplace=True)
    df_test['weight'].fillna(weight_na_const, inplace=True)
    
    df_train_no_weight = df_train[df_train['weight'] == weight_na_const]
    df_test_no_weight = df_test[df_test['weight'] == weight_na_const]
    
    df_train.drop(df_train_no_weight.index, inplace = True)
    df_test.drop(df_test_no_weight.index, inplace = True)
    
    df_train = df_train.drop(['id'], axis=1)
    df_test = df_test.drop(['id'], axis=1)
    
    df_train.loc[df_train.gender.isin(['female', 'f']), 'gender'] = 0
    df_train.loc[df_train.gender.isin(['male', 'm']), 'gender'] = 1
    df_train['gender'] = df_train['gender'].astype(int)
    
    df_test.loc[df_test.gender.isin(['female', 'f']), 'gender'] = 0
    df_test.loc[df_test.gender.isin(['male', 'm']), 'gender'] = 1
    df_test['gender'] = df_test['gender'].astype(int)
    
    cholesterol_gluc_enc_dict = {'low':0, 'medium':1, 'high':2}
    
    df_train['cholesterol'].replace(cholesterol_gluc_enc_dict, inplace=True)
    df_train['gluc'].replace(cholesterol_gluc_enc_dict, inplace=True)
    
    df_test['cholesterol'].replace(cholesterol_gluc_enc_dict, inplace=True)
    df_test['gluc'].replace(cholesterol_gluc_enc_dict, inplace=True)
    
    df_train['age'] = df_train['age'].apply(get_age)
    df_test['age'] = df_test['age'].apply(get_age)
    
    pressure = df_train['pressure'].str.split(r'[^0-9a-zA-Z-]+', expand=True)
    df_train["high pressure"]= pressure[0].astype(int) 
    df_train["low pressure"]= pressure[1].astype(int) 
    df_train.drop(columns =['pressure'], inplace = True)
    
    pressure = df_test['pressure'].str.split(r'[^0-9a-zA-Z-]+', expand=True)
    df_test["high pressure"]= pressure[0].astype(int) 
    df_test["low pressure"]= pressure[1].astype(int) 
    df_test.drop(columns =['pressure'], inplace = True)
    
    df_train['high pressure'] = df_train['high pressure'].apply(lambda x: abs(x))
    
    df_drop = df_train[(df_train["high pressure"] >= 200) | (df_train["high pressure"] <= 80)]
    df_train.drop(df_drop.index, inplace = True)
    
    df_drop = df_train[(df_train["low pressure"] >= 110) | (df_train["low pressure"] <= 50)]
    df_train.drop(df_drop.index, inplace = True)
    
    df_train.drop(df_train[(df_train["height"] == 250) | (df_train["height"] < 146)].index, inplace=True)
    df_train.drop(df_train[(df_train["weight"] < 47)].index, inplace=True)
    
    numeric = ['ket', 'age', 'height', 'weight', 'high pressure', 'low pressure']
    
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()
    
    scaler_train.fit(df_train[numeric])
    scaler_test.fit(df_test[numeric])
    
    df_train[numeric] = scaler_train.transform(df_train[numeric])
    df_test[numeric] = scaler_test.transform(df_test[numeric])
    
    features_train_upsampled, target_train_upsampled = upsample(df_train.loc[:, df_train.columns != 'diabetes'], 
                                                            df_train['diabetes'], 4, 1)
    
    features_train_upsampled['diabetes'] = target_train_upsampled
    
    df_test_target = df_test['diabetes']
    df_test_features = df_test.drop(['diabetes'], axis=1)
    
    df_train_target = features_train_upsampled['diabetes']
    df_train_features = features_train_upsampled.drop(['diabetes'], axis=1)
    
    df_test_target.to_csv(utils.get_original_cwd() + '/' + config.processed_data.test.target, index=False)
    df_test_features.to_csv(utils.get_original_cwd() + '/' + config.processed_data.test.features, index=False)
    
    df_train_target.to_csv(utils.get_original_cwd() + '/' + config.processed_data.train.target, index=False)
    df_train_features.to_csv(utils.get_original_cwd() + '/' + config.processed_data.train.features, index=False)


@hydra.main(config_path='../configs',config_name='data.yaml')
def main(config):
    warnings.filterwarnings("ignore")
    train_analysis = pd.read_csv(utils.get_original_cwd() + '/' + config.data.train.analysis)
    train_info = pd.read_csv(utils.get_original_cwd() + '/' + config.data.train.info)
    test_analysis = pd.read_csv(utils.get_original_cwd() + '/' + config.data.test.analysis)
    test_info = pd.read_csv(utils.get_original_cwd() + '/' + config.data.test.info)

    process_data(test_analysis, test_info, 
                 train_analysis, train_info,
                 config)

if __name__== '__main__':
    main()