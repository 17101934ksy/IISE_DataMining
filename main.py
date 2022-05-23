from modular import *

import pickle
import matplotlib as plt

from loader import *
from loader.loader import DataLoader
from func.processing import train_preprocessing, test_preprocessing, lgb_predict_threshold
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    print("GPU 실행: 1 입력 그 외 숫자 cpu 실행")
    
    cuda = int(input())
    
    train_df = DataLoader('./data/KDDTrain+.txt').data
    test_df = DataLoader('./data/KDDTest+.txt').data

    X_train_full, y_train_full, service, flag, oh_encoder = train_preprocessing(train_df)
    X_test, y_test = test_preprocessing(test_df, service, flag, oh_encoder)
    

    if cuda == 1:
        with open('./save_model/save_model.pickle', 'rb') as f:
            lgb_model = pickle.load(f) 

    else:
        lgb_model = LGBMClassifier(colsample_bytree=0.8, learning_rate=0.01, max_depth=1, objective='binary', random_state=41, subsample=0.75)


    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, train_size=0.8, stratify=y_train_full, random_state = 41)
    
    # X_train, y_train으로 최적의 파라미터를 찾음
    lgb_model.fit(X_train, y_train)

    y_test_pred = lgb_predict_threshold(lgb_model, X_test)
    print(confusion_matrix(y_test, y_test_pred))