#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
dataset=pd.read_csv("/root/weblog.csv")
dataset.head()
dataset=dataset[['IP','Status']]
dataset.head()
ip=dataset.IP.unique().tolist()
dataset['Code']=np.zeros(len(dataset))
for i in range(len(dataset)):
    dataset.Code[i]=ip.index(dataset.IP[i])
data=dataset[["Code","Status"]]
df=data.copy()
df['Code'] = df['Code'].astype(int)
print(df.dtypes)
log=df.values
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(log)
pd.DataFrame(data_scaled).describe()
SSE = []
for cluster in range(1,15):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
frame = pd.DataFrame({'Cluster':range(1,15), 'SSE':SSE})
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
plt.figure(figsize=(12,6))
plt.scatter(dataset['IP'],dataset["Status"],marker='*')
plt.xlabel('IP')
plt.ylabel('Status code')
plt.show()
blocked=[]
for i in range(len(dataset)):
    if dataset['Status'][i]==404 or dataset['Status'][i]==403:
        blocked.append(dataset['IP'][i])
print(blocked)
with open('blocked.txt', 'w+') as filehandle:
    for listitem in blocked:
        filehandle.write('%s\n' % listitem)

