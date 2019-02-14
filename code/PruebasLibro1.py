import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
X=df.loc[:,2:].values
y=df.loc[:,1].values
"""le=LabelEncoder()
y=le.fit_transform(y)
le.transform(['M','B'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,stratify=y,random_state=1)
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
pipe_lr.fit(X_train,y_train)
y_pred=pipe_lr.predict(X_test)
print(pipe_lr.score(X_test,y_test))
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#X,y=make_moons(n_samples=400,noise=0.05,random_state=0)
db=DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
y_db=db.fit_predict(X)
km=KMeans(n_clusters=2,random_state=0)
y_km=km.fit_predict(X)

plt.scatter(X[:,0],X[:,1])
plt.show()

plt.scatter(X[y_db==0,0],X[y_db==0,1],c='lightblue',edgecolor='black',marker='o',s=40,label='cluster 1')
plt.scatter(X[y_db==1,0],X[y_db==1,1],c='red',edgecolor='black',marker='o',s=40,label='cluster 2')
plt.legend()
plt.show()

plt.scatter(X[y_km==0,0],X[y_km==0,1],c='lightblue',edgecolor='black',marker='o',s=40,label='cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],c='red',edgecolor='black',marker='o',s=40,label='cluster 2')
plt.legend()
plt.show()
