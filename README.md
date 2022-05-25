# IISE_DataMining

## Anomaly Detection NSL-KDD

### Team
역할|학번|Git|
---|---|---|---|
Team Leader|16102171 김영서|[yskim569](https://github.com/yskim569)
developer|17101934 고세윤|[17101934ksy](https://github.com/17101934ksy/IISE_DataMining)
developer|20102007 김수연|[ehfapddl](https://github.com/ehfapddl)

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
Name|install|Version|
---|---|---|
python||3.7.13
numpy|!pip install numpy|1.21.6
pandas|!pip install pandas|1.3.5
matplotlib|!pip install matplotlib|3.2.2
seaborn|!pip install seaborn|0.11.2
requests|!pip install requests|2.23.0
PIL.image|!pip install image|7.1.2
sklearn|!pip install scikit-learn|1.0.2
xgboost|!pip install xgboost|0.90
lightgbm|!pip install lightgbm|2.2.3

### Package Info
NSL_Model.ipynb: 전체 데이터마이닝 과정 모듈   
main.py: 데이터마이닝 모델 재사용을 위한 모듈  
modular.py: 모델링에 사용한 외부 모듈 

```pathon
IISE_DataMining/
    __init__.py
    main.py
    modular.py
    
    data/
    save_model/
    func/
        __init__.py
        processing.py
    loader/
        __init__.py
        loader.py
    project_ipynb/
        __init__.py
        NSL_Model.ipynb
```

### Fix Randomseed
```python
CFG = {
    'PATH': '/content/drive/MyDrive/Colab Notebooks/데이터마이닝/project',
    'SEED': 41}
def set_env(path, seed):
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  os.chdir(path)
set_env(CFG['PATH'],CFG['SEED'])
```
### Model Object
네트워크 통신의 범위는 컴퓨터를 넘어서 스마트폰과 같은 스마트 기기, 가정에서 사용하는 스마트 가전, 그리고 iot 등으로 확장되었습니다.  
이렇게 범위가 심하게 증가함에 따라서 통신의 과정이 다양화되고 복잡해졌습니다.  
네트워크 공격은 더욱 고도화되고 지능화되는 등, 개인 정보 유출이나 서버 해킹 등의 위험성이 높아졌습니다.  
저희는 이상탐지를 활용해 데이터 분석을 통한 네트워크 데이터 이상탐지에 대해서 연구하여  
네트워크 침입을 탐지할 수 있는 이상탐지 모델을 만들어 보았습니다.

<img src="https://user-images.githubusercontent.com/88478829/169640650-c726ffbe-1494-430d-9da6-81ba06fbdfd7.png" width="80%" height="300"/>  

### Model Metrics
주제 특성에 맞도록 주된 성능 평가지표는 Fbeta를 활용할 예정이고 보조수단으로 Recall을 활용하겠습니다.  
sklearn에서는 혼동행렬 계산 시 label=1을 양성으로 설정하므로 pos_label=0으로 설정하였습니다.
<img src="https://user-images.githubusercontent.com/88478829/169787624-4f3cdc7e-68e4-43ae-ae8e-45ac4203dd05.png" width="80%" height="300"/>
<img src="https://user-images.githubusercontent.com/88478829/169639782-9fe799b4-6ce9-4154-b17f-45db8db74187.png" width="40%" height="300" float="left"/> <img src="https://i.stack.imgur.com/swW0x.png" width="40%" height="300" float="right"/>

### Model Select
모델링 선택은 Fbata score를 사용했습니다.  
lightgbm 모델은 다양한 모델처럼 좋은 성능을 보이지만 빠르다는 장점이 있습니다.  
따라서, lightgbm을 최종 모델로 선정하여 하이퍼 파라미터를 조정하였습니다. 
Name|#Params|GridsearchCV Fbeta|Validaton Fbeta
---|---|---|---|
RandomForest|max_depth, min_samples_leaf, min_samples_split|0.9945|0.9952|
XGboost|learning_rate, gamma, max_depth|0.9999|1.0|
LightGBM|learning_rate|0.9999|1.0|
SVM|C, gamma, kernel|0.9973|1.0|
  
  

### Model Fine Tuning
fbeta=2로 고정한 뒤, learing_rate, max_depth를 조정해가며   
최고의 재현율과 fbeta값이 나오는 모델을 선정하는 단계입니다.
최고의 Fbeta_Score의 하이퍼 파라미터는 Learinng_rate = 0.08, Max_depth= 6입니다
```python
lgb_score_ = []
params = []
lgb_params = {'learning_rate' : np.linspace(0.01, 0.1, 10)}
scoring = {'recall_score': make_scorer(recall_score, pos_label=0),
          'fbeta_score': make_scorer(fbeta_score, beta=2, pos_label=0)}
for lr in lgb_params['learning_rate']:
  for md in [md for md in range(1, 10)]:
    params.append([lr, md])
    lgb_model = lgb.LGBMClassifier(objective='binary', learning_rate=lr, n_estimators=100, subsample=0.75, 
                                colsample_bytree=0.8, tree_method='gpu_hist', random_state=CFG['SEED'],
                                max_depth=md)
    lgb_score = cross_validate(lgb_model, X_train_full, y_train_full, scoring=scoring)
    lgb_score_.append(lgb_score)
```

<img src="https://user-images.githubusercontent.com/88478829/169807488-11d8a8d2-120b-42b6-b510-64a42e40d087.png" width="40%" height="300" float="left"/> <img src="https://user-images.githubusercontent.com/88478829/169815456-c801cc35-1cdf-4b90-8ba4-9e3b617df2c7.png" width="40%" height="300" float="right"/>

  
### Model Test
테스트 데이터셋 결과 FN이 144가 나왔습니다.  

<img src="https://user-images.githubusercontent.com/88478829/169789897-0a1b3dcd-e945-46a5-8d00-5a65289c1997.png" width="40%" height="300" float="left"/>      

  
### Model PostProcessing
Fbeta는 Precision을 어느 정도 반영한다는 점에서 한계가 존재합니다.   
과적합을 줄이는 방안으로 learning_rate 및 max_depth를 줄여서 과적합을 줄이는 과정을 진행하였습니다.   
또한, threshold를 조정하여 FP은 증가하지만 FN을 줄일 수 있는 지점을 구하였습니다.        
  
그 지점은 임계점이 0.22507250725072508입니다.   
최적의 모델을 후처리한 후, Fbeta, recall의 그래프, 혼동행렬 결과입니다.  
(왼쪽 사진: Validation set 결과, 오른쪽 사진 후처리 이후 Test set 결과)  
혼동행렬 결과 FN이 144 -> 35로 감소한 것을 확인할 수 있습니다.  
FP는 0 -> 233으로 증가했지만, 보안이라는 특수성을 고려하면 성능이 향상되었다고 판단할 수 있습니다.

```python
lgb_model = lgb.LGBMClassifier(objective='binary', learning_rate=0.01, n_estimators=100, subsample=0.75, 
                            colsample_bytree=0.8, tree_method='gpu_hist', random_state=CFG['SEED'],
                            max_depth=1)
                            ...

threshold = confusion_matrix(y_val_pred, y_val)
thr_ = np.linspace(0.5, 0, 10000)
idx = -1
for i, thr in enumerate(thr_):
  y_val_pred = np.where(y_val_prob[:, 0] >= thr, 0, 1)
  if np.sum(confusion_matrix(y_val, y_val_pred) == threshold) != 4:
    idx = i
    break
    
print(f'새로운 threshold: {thr_[idx]}')
>> 0.22507250725072508
```

<img src="https://user-images.githubusercontent.com/88478829/169812435-d518db86-dffb-49e0-82de-adc1bc3efa00.png" width="40%" height="300" float="left"/> <img src="https://user-images.githubusercontent.com/88478829/169815147-08c297a3-fae6-4600-9281-787673847945.png" width="40%" height="300" float="right"/>
<img src="https://user-images.githubusercontent.com/88478829/169793676-b99b9969-0048-4c7d-b66e-7ef793e056ef.png" width="80%" height="300"/>

### Model Deploy
```python

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
    ```
    패키지 다운로드 후 main.py를 실행하면 y_test의 혼동행렬을 확인할 수 있습니다.
