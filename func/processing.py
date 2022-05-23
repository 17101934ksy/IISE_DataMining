import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Pretrained model Processing
def train_preprocessing(df, drop_columns=['attack', 'level'], label='attack_flag', pretrained=True):
  """
  df: 데이터프레임
  drop_columns: 원핫인코딩에 적용하지 않거나, 제거할 object columns
  label: 데이터프레임에서 분리할 정답 레이블
  pretrained: 사전에 학습한 모델의 전처리를 활용하려면 True, False라면 새롭게 X_Train 범주형 자료 전처리
  return: 원핫인코딩이 적용된 데이터프레임, 정답레이블, service dict, flag dict, 원핫인코더
  """

  df = df.drop(drop_columns, axis=1)
  label_ = df.pop('attack_flag')

  df['protocol_type'] = df['protocol_type'].apply(lambda x: 1 if x == 'tcp' \
                                                else 2 if x == 'udp' \
                                                else 3 if x == 'icmp' \
                                                else 4)
  
  if pretrained:
    service = {'IRC': 50, 'X11': 47, 'Z39_50': 65, 'aol': 43, 'auth': 11, 'bgp': 22, 'courier': 20, 'csnet_ns': 45,
               'ctf': 19, 'daytime': 31, 'discard': 57, 'domain': 7, 'domain_u': 51, 'echo': 64, 'eco_i': 28, 'ecr_i': 4,
               'efs': 27, 'exec': 3, 'finger': 33, 'ftp': 52, 'ftp_data': 56, 'gopher': 38, 'harvest': 17, 'hostnames': 41,
               'http': 53, 'http_2784': 67, 'http_443': 36, 'http_8001': 8, 'imap4': 13, 'iso_tsap': 25, 'klogin': 5, 'kshell': 44,
               'ldap': 63, 'link': 15, 'login': 46, 'mtp': 54, 'name': 39, 'netbios_dgm': 55, 'netbios_ns': 0, 'netbios_ssn': 24, 'netstat': 59,
               'nnsp': 60, 'nntp': 34, 'ntp_u': 49, 'other': 29, 'pm_dump': 26, 'pop_2': 2, 'pop_3': 35, 'printer': 16, 'private': 9, 'red_i': 14,
               'remote_job': 58, 'rje': 66, 'shell': 21, 'smtp': 18, 'sql_net': 42, 'ssh': 48, 'sunrpc': 6, 'supdup': 30, 'systat': 69, 'telnet': 1, 'tftp_u': 37,
               'tim_i': 32, 'time': 68, 'urh_i': 23, 'urp_i': 10, 'uucp': 61, 'uucp_path': 40, 'vmnet': 12, 'whois': 62}

    flag = {'OTH': 4, 'REJ': 7, 'RSTO': 0, 'RSTOS0': 10, 'RSTR': 2, 'S0': 5, 'S1': 3, 'S2': 6, 'S3': 1, 'SF': 9, 'SH': 8}

  else:
    service = {}
    for idx, i in enumerate(set(df['service'])):
      service[i] = idx
    flag = {}
    for idx, i in enumerate(set(df['flag'])):
      flag[i] = idx

  df['service'] = df['service'].apply(lambda x: service[x] if x in service.keys() else -1)
  df['flag'] = df['flag'].apply(lambda x: flag[x] if x in flag.keys() else -1)

  for c in ['protocol_type', 'attack_type', 'flag', 'service']:
    df[c] = df[c].astype('object')
  df_temp = df[['protocol_type', 'attack_type', 'flag', 'service']]
  
  oh_encoder = OneHotEncoder()
  oh_encoder.fit(df_temp)
  df_temp = oh_encoder.transform(df_temp).toarray()

  df = df.drop(['protocol_type', 'attack_type', 'flag', 'service'], axis=1).to_numpy()
  df = np.concatenate((df, df_temp), axis=1)

  return df, label_.to_numpy(), service, flag, oh_encoder


#########################################################################################################

def test_preprocessing(df, service, flag, oh_encoder, drop_columns=['attack', 'level'], label='attack_flag'):
  """
  df: 데이터프레임
  service: X_train에 해당하는 serivce 레이블을 train에 그대로 적용하기 위한 고유 샘플 딕셔너리
  flag: X_train에 해당하는 flag 레이블을 train에 그대로 적용하기 위한 고유 샘플 딕셔너리
  on_encoder: X_train에 적용된 onehot인코더
  drop_columns: 원핫인코딩에 적용하지 않거나, 제거할 object columns
  label: 데이터프레임에서 분리할 정답 레이블
  return: 원핫인코딩이 적용된 데이터프레임, 정답레이블  
  """

  df = df.drop(drop_columns, axis=1)
  label_ = df.pop('attack_flag')

  df['protocol_type'] = df['protocol_type'].apply(lambda x: 1 if x == 'tcp' \
                                                else 2 if x == 'udp' \
                                                else 3 if x == 'icmp' \
                                                else 4)
    
  df['flag'] = df['flag'].apply(lambda x: flag[x] if x in flag.keys() else -1)
  df['service'] = df['service'].apply(lambda x: service[x] if x in service.keys() else -1)

  for c in ['protocol_type', 'attack_type', 'flag', 'service']:
    df[c] = df[c].astype('object')
  df_temp = df[['protocol_type', 'attack_type', 'flag', 'service']]
  
  df_temp = oh_encoder.transform(df_temp).toarray()

  df = df.drop(['protocol_type', 'attack_type', 'flag', 'service'], axis=1).to_numpy()
  df = np.concatenate((df, df_temp), axis=1)

  return df, label_.to_numpy()



#########################################################################################################

def lgb_predict_threshold(lgb_model, X_test, threshod=0.22507250725072508):
  y_test_prob = lgb_model.predict_proba(X_test)
  y_test_pred = np.where(y_test_prob[:, 0] > threshod, 0, 1)

  return y_test_pred