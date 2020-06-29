#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset=pd.read_csv('weblog.csv')
dataset.head()


# In[3]:


import numpy as np
dataset=dataset[['IP','Staus']]
dataset.head()


# In[4]:


ip=dataset.IP.unique().tolist()
dataset['Code']=np.zeros(len(dataset))
for i in range(len(dataset)):
    dataset.Code[i]=ip.index(dataset.IP[i])
  
dataset.head()


# In[5]:


data=dataset[["Code","Staus"]]
df=data.copy()
df['Code'] = df['Code'].astype(int)
print(df.dtypes)
log=df.values
log


# In[6]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(log)
pd.DataFrame(data_scaled).describe()
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[12]:


plt.figure(figsize=(15,6))
plt.scatter(dataset['IP'],dataset["Staus"],marker='*')
plt.xlabel('IP')
plt.ylabel('Status code')
plt.show()


# In[9]:


blocked=[]
for i in range(len(dataset)):
    if dataset['Staus'][i]==401 or dataset['Staus'][i]==403:
        blocked.append(dataset['IP'][i])
print(blocked)
with open('blocked.html', 'w+') as filehandle:
    for listitem in blocked:
        filehandle.write('=>%s\t\t' % listitem)


# In[ ]:




