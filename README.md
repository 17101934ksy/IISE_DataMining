# IISE_DataMining

## Anomaly Detection NSL-KDD

### Quickstart

구글 Colab에서 클론하기

```python
from google.colab import drive
drive.mount('/content/drive')

!ls
!pwd

%cd "/content/drive/My Drive/git"

!git clone https://github.com/17101934ksy/IISE_DataMining.git
```

### Environment(version)
```python
python 3.7.13
numpy 1.21.6
pandas 1.3.5
matplotlib 3.2.2
seaborn 0.11.2
requests 2.23.0
PIL.image 7.1.2
sklearn 1.0.2
xgboost
lightgbm

```

### Fix Randomseed
```python
CFG = {
    'PATH': '/content/drive/MyDrive/Colab Notebooks/데이터마이닝/project',
    'SEED': 41
    }

def set_env(path, seed):
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  os.chdir(path)

set_env(CFG['PATH'],CFG['SEED'])
```
### Model Metrics
주제특성에 맞도록 주된 성능 평가지표는 Fbeta를 활용할 예정이고 보조수단으로 Recall을 활용하겠습니다.
![initial](https://user-images.githubusercontent.com/88478829/169639782-9fe799b4-6ce9-4154-b17f-45db8db74187.png)
![initial](https://user-images.githubusercontent.com/88478829/169639780-bbf5b2bc-3f8d-4ae0-96d3-a0a4ff30d460.png)


### Model Select
모델링 선택은 Fbata score를 사용했습니다.  
lightgbm 모델은 다양한 모델처럼 좋은 성능을 보이지만 빠르다는 장점이 있습니다.  
따라서, lightgbm을 최종 모델로 선정하여 하이퍼 파라미터를 조정하였습니다. 
Name|#Params|GridsearchCV Fbeta|Validaton Fbeta
---|---|---|---|
RandomForest|max_depth, min_samples_leaf, min_samples_split|0.9999|1.0|
XGboost|learning_rate, gamma, max_depth|0.9999|1.0|
LightGBM|learning_rate|0.9999|1.0|
SVM|C, gamma, kernel|0.9999|1.0|
  
  

### Model Fine Tuning
fbeta=2로 고정한 뒤, learing_rate, max_depth를 조정해가며   
최고의 재현율과 fbeta값이 나오는 모델을 선정하는 단계입니다.
```python
lgb_score_ = []
params = []
lgb_params = {'learning_rate' : np.linspace(0.01, 0.1, 10)}
scoring = {'recall_score': make_scorer(recall_score),
          'fbeta_score': make_scorer(fbeta_score, beta=2)}
for lr in lgb_params['learning_rate']:
  for md in [md for md in range(1, 10)]:
    params.append([lr, md])
    lgb_model = lgb.LGBMClassifier(objective='binary', learning_rate=lr, n_estimators=100, subsample=0.75, 
                                colsample_bytree=0.8, tree_method='gpu_hist', random_state=CFG['SEED'],
                                max_depth=md)
    lgb_score = cross_validate(lgb_model, X_train_full, y_train_full, scoring=scoring)
    lgb_score_.append(lgb_score)
```
   
### Model Test
테스트 데이터셋 적용결과 최적의 모델 및 Fbeta,Recall의 그래프, 혼동행렬 결과입니다.
```python
lgb_model = lgb.LGBMClassifier(objective='binary', learning_rate=0.08, n_estimators=100, subsample=0.75, 
                            colsample_bytree=0.8, tree_method='gpu_hist', random_state=CFG['SEED'],
                            max_depth=6)
                            ...
>>최적의 fbeta 성능: 0.9909867691394472
```
![initial](https://user-images.githubusercontent.com/88478829/169640702-82431313-ce0d-467c-8438-a29a09e01e59.png)
![initial](https://user-images.githubusercontent.com/88478829/169640738-1acf4866-a0f1-4675-ae86-68fa04972186.png)
