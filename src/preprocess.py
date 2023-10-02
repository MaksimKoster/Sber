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
import os

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

def parse_pressure(df):
    pressure = df['pressure'].str.split(r'[^0-9a-zA-Z-]+', expand=True)
    
    df["high pressure"]= pressure[0].astype(int) 
    df["low pressure"]= pressure[1].astype(int) 
    
    df.drop(columns =['pressure'], inplace = True)
    return df

def get_age(x):
    x = str(x)
    if len(x) < 5:
        return int(x)
    else:
        res = int(round(int(x) / 365))
    return res

def scale(df, numeric ,scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[numeric])
    new_df = scaler.transform(df[numeric])
    return new_df

def code_gluc_chol(df):
    cholesterol_gluc_enc_dict = {'low':0, 'medium':1, 'high':2}
    df['cholesterol'].replace(cholesterol_gluc_enc_dict, inplace=True)
    df['gluc'].replace(cholesterol_gluc_enc_dict, inplace=True)
    return df

def code_gen(df):
    df.loc[df.gender.isin(['female', 'f']), 'gender'] = 0
    df.loc[df.gender.isin(['male', 'm']), 'gender'] = 1
    df['gender'] = df['gender'].astype(int)
    return df


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
    
    df_train_no_weight = df_train_no_weight.drop(['id'], axis=1)
    df_test_no_weight = df_test_no_weight.drop(['id'], axis=1)
    
    df_train = code_gen(df_train)
    df_test = code_gen(df_test)
    
    df_train_no_weight = code_gen(df_train_no_weight)
    df_test_no_weight = code_gen(df_test_no_weight)
    
    df_train = code_gluc_chol(df_train)
    df_test = code_gluc_chol(df_test)
    
    df_train_no_weight = code_gluc_chol(df_train_no_weight)
    df_test_no_weight = code_gluc_chol(df_test_no_weight)
    
    df_train['age'] = df_train['age'].apply(get_age)
    df_test['age'] = df_test['age'].apply(get_age)
    
    df_train_no_weight['age'] = df_train_no_weight['age'].apply(get_age)
    df_test_no_weight['age'] = df_test_no_weight['age'].apply(get_age)
    
    df_test = parse_pressure(df_test)
    df_train = parse_pressure(df_train)
    
    df_train_no_weight = parse_pressure(df_train_no_weight)
    df_test_no_weight = parse_pressure(df_test_no_weight)
    
    df_train['high pressure'] = df_train['high pressure'].apply(lambda x: abs(x))
    df_train_no_weight['high pressure'] = df_train_no_weight['high pressure'].apply(lambda x: abs(x))
    
    df_drop = df_train[(df_train["high pressure"] >= 200) | (df_train["high pressure"] <= 80)]
    df_train.drop(df_drop.index, inplace = True)
    
    df_drop = df_train_no_weight[(df_train_no_weight["high pressure"] >= 200) | (df_train_no_weight["high pressure"] <= 80)]
    df_train_no_weight.drop(df_drop.index, inplace = True)
    
    df_drop = df_train[(df_train["low pressure"] >= 110) | (df_train["low pressure"] <= 50)]
    df_train.drop(df_drop.index, inplace = True)
    
    df_drop = df_train_no_weight[(df_train_no_weight["low pressure"] >= 110) | (df_train_no_weight["low pressure"] <= 50)]
    df_train_no_weight.drop(df_drop.index, inplace = True)
    
    df_train.drop(df_train[(df_train["height"] == 250) | (df_train["height"] < 146)].index, inplace=True)
    df_train.drop(df_train[(df_train["weight"] < 47)].index, inplace=True)
    
    df_train_no_weight.drop(df_train_no_weight[(df_train_no_weight["height"] == 250) | (df_train_no_weight["height"] < 146)].index, inplace=True)

    numeric = ['ket', 'age', 'height', 'weight', 'high pressure', 'low pressure']
    df_train[numeric] = scale(df_train, numeric)
    df_test[numeric] = scale(df_test, numeric)

    df_train_no_weight[numeric] = scale(df_train_no_weight, numeric)
    df_test_no_weight[numeric] = scale(df_test_no_weight, numeric)
    
    features_train_upsampled, target_train_upsampled = upsample(df_train.loc[:, df_train.columns != 'diabetes'], 
                                                            df_train['diabetes'], 4, 1)
    
    features_train_upsampled['diabetes'] = target_train_upsampled
    features_train_upsampled


    df_test_target = df_test['diabetes']
    df_test_features = df_test.drop(['diabetes'], axis=1)

    df_train_target = features_train_upsampled['diabetes']
    df_train_features = features_train_upsampled.drop(['diabetes'], axis=1)

    df_test_no_weight_target = df_test_no_weight['diabetes']
    df_test_no_weight_features = df_test_no_weight.drop(['diabetes'], axis=1)

    df_train_no_weight_target = df_train_no_weight['diabetes']
    df_train_no_weight_features = df_train_no_weight.drop(['diabetes'], axis=1)

    df_test_target.to_csv(utils.get_original_cwd() + '/' + config.processed_data.test.target, index=False)
    df_test_features.to_csv(utils.get_original_cwd() + '/' + config.processed_data.test.features, index=False)

    df_train_target.to_csv(utils.get_original_cwd() + '/' + config.processed_data.train.target, index=False)
    df_train_features.to_csv(utils.get_original_cwd() + '/' + config.processed_data.train.features, index=False)

    df_test_no_weight_target.to_csv(utils.get_original_cwd() + '/' + config.processed_data_no_weight.test.target, index=False)
    df_test_no_weight_features.to_csv(utils.get_original_cwd() + '/' + config.processed_data_no_weight.test.features, index=False)

    df_train_no_weight_target.to_csv(utils.get_original_cwd() + '/' + config.processed_data_no_weight.train.target, index=False)
    df_train_no_weight_features.to_csv(utils.get_original_cwd() + '/' + config.processed_data_no_weight.train.features, index=False)


@hydra.main(config_path='../configs',config_name='data.yaml')
def main(config):
    warnings.filterwarnings("ignore")

    out_dir = utils.get_original_cwd() + '/data/processed'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_analysis = pd.read_csv(utils.get_original_cwd() + '/' + config.data.train.analysis)
    train_info = pd.read_csv(utils.get_original_cwd() + '/' + config.data.train.info)
    test_analysis = pd.read_csv(utils.get_original_cwd() + '/' + config.data.test.analysis)
    test_info = pd.read_csv(utils.get_original_cwd() + '/' + config.data.test.info)

    process_data(test_analysis, test_info, 
                 train_analysis, train_info,
                 config)

if __name__== '__main__':
    main()