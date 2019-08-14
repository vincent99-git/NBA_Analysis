#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture as GMM
#import mplcursors


# In[145]:


df=pd.read_csv("2018-2019_Overall_NBA_Stats_by_Player.csv")
df=df.sample(frac=1)


# In[146]:


print(np.corrcoef(df.Age.array, df.PTS.array))


# In[147]:


df = df[df.MP > 96]

df["TRB/MP"]=df["TRB"]/df["MP"]
df["AST/MP"]=df["AST"]/df["MP"]
df["STL/MP"]=df["STL"]/df["MP"]
df["BLK/MP"]=df["BLK"]/df["MP"]
df["PTS/MP"]=df["PTS"]/df["MP"]
df["FGA/MP"]=df["FGA"]/df["MP"]
df["3PA/MP"]=df["3PA"]/df["MP"]
df["2PA/MP"]=df["2PA"]/df["MP"]
df["FTA/MP"]=df["FTA"]/df["MP"]
df["PF/MP"]=df["PF"]/df["MP"]

df.head()


# In[148]:


fig, ax = plt.subplots()

x_var="AST/MP"
y_var="TRB/MP"
colors = {'SG':'blue', 'PF':'red', 'PG':'green', 'C':'purple', 'SF':'orange', 'PF-SF':'black', 'SF-SG':'black', 'SG-PF':'black', 'C-PF':'black', 'SG-SF':'black', 'PF-C':'black'}
ax.scatter(df[x_var], df[y_var], c=df['Pos'].apply(lambda x: colors[x]), s = 10)

# set a title and labels
ax.set_title('NBA Dataset')
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)

#mplcursors.cursor(hover=True)


# In[149]:


#dfn = df[["3PA/MP","3P%","2PA/MP","2P%","FT%","TRB/MP","AST/MP","STL/MP","BLK/MP","PF/MP"]]
dfn = df[["AST/MP","TRB/MP"]]

print(dfn.corr())
plt.matshow(dfn.corr())
plt.show()

dfn.isnull().values.any()


# In[150]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(dfn)
print(kmeans.cluster_centers_)
d0=dfn[y_kmeans == 0]
d1=dfn[y_kmeans == 1]
d2=dfn[y_kmeans == 2]
d3=dfn[y_kmeans == 3]
d4=dfn[y_kmeans == 4]


# In[151]:


#Visualising the clusters
plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')
plt.scatter(d4[x_var], d4[y_var], s = 10, c = 'orange', label = 'D4')

#plt.scatter(df['sepal length'], df['sepal width'], c=df['class'].apply(lambda x: colors[x]))

#Plotting the centroids of the clusters
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

#plt.legend()


# In[152]:


#df["Kmeans_Class"]= y_kmeans
#print(df[["Pos","Kmeans_Class"]])


# In[153]:


spect = SpectralClustering(n_clusters = 4, affinity='nearest_neighbors')
y_spect = spect.fit_predict(dfn)
d0=dfn[y_spect == 0]
d1=dfn[y_spect == 1]
d2=dfn[y_spect == 2]
d3=dfn[y_spect == 3]
d4=dfn[y_spect == 4]


# In[154]:


#Visualising the clusters
plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')
plt.scatter(d4[x_var], d4[y_var], s = 10, c = 'orange', label = 'D4')


# In[155]:


cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_cluster = cluster.fit_predict(dfn)
d0=dfn[y_cluster == 0]
d1=dfn[y_cluster == 1]
d2=dfn[y_cluster == 2]
d3=dfn[y_cluster == 3]
d4=dfn[y_cluster == 4]


# In[156]:


#Visualising the clusters
plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')
plt.scatter(d4[x_var], d4[y_var], s = 10, c = 'orange', label = 'D4')


# In[157]:


neighb = KNeighborsClassifier(n_neighbors=4)
y_neighb = cluster.fit_predict(dfn, df["Pos"])
d0=dfn[y_neighb == 0]
d1=dfn[y_neighb == 1]
d2=dfn[y_neighb == 2]
d3=dfn[y_neighb == 3]
d4=dfn[y_neighb == 4]


# In[158]:


#Visualising the clusters
plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')
plt.scatter(d4[x_var], d4[y_var], s = 10, c = 'orange', label = 'D4')


# In[159]:


#df["KNN_Class"]= y_neighb
#df_compare=df[["Pos","KNN_Class"]]
#df_compare.sort_values(by="Pos")
#df_compare.loc[df['Pos'] == "SG"]


# In[171]:


gmm = GMM(n_components=4)
y_gmm = gmm.fit_predict(dfn)
d0=dfn[y_gmm == 0]
d1=dfn[y_gmm == 1]
d2=dfn[y_gmm == 2]
d3=dfn[y_gmm == 3]
d4=dfn[y_gmm == 4]

#Visualising the clusters
plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')
plt.scatter(d4[x_var], d4[y_var], s = 10, c = 'orange', label = 'D4')


# In[173]:


df["GMM_Class"]= y_neighb

probs = gmm.predict_proba(dfn)
print(probs[:5].round(3))

size = 10 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(df[x_var], df[y_var], c=df['Pos'].apply(lambda x: colors[x]), s = size)


# In[174]:


plt.hist(df.loc[df['Pos'] == "PG"].GMM_Class)
plt.show()
plt.hist(df.loc[df['Pos'] == "SG"].GMM_Class)
plt.show()
plt.hist(df.loc[df['Pos'] == "SF"].GMM_Class)
plt.show()
plt.hist(df.loc[df['Pos'] == "PF"].GMM_Class)
plt.show()
plt.hist(df.loc[df['Pos'] == "C"].GMM_Class)
plt.show()


# In[175]:


df.to_csv("output.csv")


# In[164]:


norm_age = df["Age"].sort_values()-18
plt.hist(norm_age)


# In[165]:


gamma = stats.gamma
a, loc, scale = 3, 0, 2
size = 20000
y = gamma.rvs(a, loc, scale, size=size)

x = np.linspace(0, y.max(), 100)
# fit
param = gamma.fit(norm_age, floc=0)
pdf_fitted = gamma.pdf(x, *param)
plt.plot(x, pdf_fitted, color='r')

# plot the histogram
plt.hist(norm_age, normed=True, bins=30)

plt.show()


# In[166]:


print(param)
print(gamma.mean(*param))


# In[167]:


df.sort_values(by="AST", ascending=False).head(10)


# In[ ]:





# In[ ]:




