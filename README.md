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
RandomForest|max_depth, min_samples_leaf, min_samples_split|0.99|1.0|
홍길동|97점|78점|93점|
이순신|89점|93점|97점|
