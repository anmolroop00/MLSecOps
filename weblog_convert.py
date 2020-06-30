import re
def extract(filename):
    with open(filename) as f:
        log = f.read()
        regexp2 = r"(?P<ip>.*?) (?P<remote_log_name>.*?) (?P<userid>.*?) \[(?P<date>.*?) (?P<timezone>.*?)\] \"(?P<request_method>.*?) (?P<path>.*?) (?P<request_version>.*?)\" (?P<status>.*?) (?P<length>.*?) \"(?P<referrer>.*?)\" \"(?P<user_agent>.*?)\""
        ips_list = re.findall(regexp2, log)
        return ips_list
logs = extract('/var/log/httpd/access_log')
import numpy as np
log_arr = np.array(logs)
ip=log_arr[:,0]
import pandas as pd
dataset = pd.DataFrame({'IP': log_arr[:, 0], 'A': log_arr[:, 1],'B':log_arr[:, 2],'Date&Time':log_arr[:, 3],'TZ':log_arr[:, 4],'C':log_arr[:, 5],'Site':log_arr[:, 6],'Protocol':log_arr[:, 7],'Status':log_arr[:, 8],'Length':log_arr[:, 9]})
print(dataset.head())
dataset.to_csv('weblog.csv')
