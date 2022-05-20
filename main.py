from modular import *

import pickle
import matplotlib as plt

from loader import *
from loader.loader import DataLoader
from func.processing import train_processing, test_processing
from sklearn.metrics import plot_confusion_matrix

if __name__ == '__main__':
    train_df = DataLoader('./data/KDDTrain+.txt').data
    test_df = DataLoader('./data/KDDTest+.txt').data

    X_train_full, y_train_full, service, flag, oh_encoder = train_processing(train_df)
    X_test, y_test = test_processing(test_df, service, flag, oh_encoder)
    
    with open('./save_model/save_model.pickle', 'rb') as f:
        lgb_model = pickle.load(f) 

    print(f'lgb_model: {lgb_model}')

    lgb_model.fit(X_train_full, y_train_full)
    y_test_pred = lgb_model.predict(X_test)

    label=['Normal', 'Anomaly']

    plot = plot_confusion_matrix(lgb_model,
                                X_test, y_test,
                                display_labels=label,
                                cmap=plt.cm.Reds,
                                normalize=None)

    plot.ax_.set_title('Confusion Matrix')
