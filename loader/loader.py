import pandas as pd

class DataLoader():

  def __init__(self, path):

    self.columns = ['duration' ,'protocol_type' ,'service' ,'flag' ,'src_bytes' ,'dst_bytes','land',
                    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
                    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                    'dst_host_srv_rerror_rate', 'attack', 'level']

    self.attack = {
                  'dos_attacks': ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],                 
                  'probe_attacks': ['ipsweep','mscan','nmap','portsweep','saint','satan'],
                  'privilege_attacks': ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm'],
                  'access_attacks': ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail', \
                                  'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop'],
                  'attack_labels': ['Normal','DoS','Probe','Privilege','Access']
                  }

    self.path = path

    # 데이터 처리
    self.data = pd.DataFrame(data=pd.read_csv(self.path))
    self.data.columns=self.columns

    # 공격 타입 판단
    self.data['attack_type'] = self.data['attack'].apply(lambda x: 1 if x in self.attack['dos_attacks'] \
                                                         else 2 if x in self.attack['probe_attacks'] \
                                                         else 3 if x in self.attack['privilege_attacks'] \
                                                         else 4 if x in self.attack['access_attacks'] \
                                                         else 0)

    # 공격 유무 판단
    self.data['attack_flag'] = self.data['attack'].apply(lambda x: 1 if x == 'normal' else 0)

  def __return__(self):
    return self.data  
