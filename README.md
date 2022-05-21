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

### Environment
```python
python 3.7.13
numpy
pandas
sklearn
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

### Model Select
Name|#Params|GridsearchCV Fbeta|Validaton Fbeta
---|---|---|---|
RandomForest|max_depth, min_samples_leaf, min_samples_split|0.9999|1.0|
XGboost|learning_rate, gamma, max_depth|0.9999|1.0|
LightGBM|learning_rate|0.9999|1.0|
SVM|C, gamma, kernel|0.9999|1.0|

lightgbm 모델은 XGboost과 비교하면 비슷한 성능을 보이지만 빠르다는 장점이 있습니다.
따라서, lightgbm을 최종 모델로 선정하여 하이퍼 파라미터를 조정하였습니다.
