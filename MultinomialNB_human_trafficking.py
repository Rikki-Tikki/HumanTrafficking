
"""
Multi-nomial Naive Bayes

"""
import pandas as pd
df = pd.read_csv("df.csv")
#df.head(5)
#%%
df.columns

y = df['RecruiterRelationship']
X = df[['CountryOfExploitation','typeOfExploitConcatenated']]
#y.head(5)
#X.head(5)

#%%
from sklearn import preprocessing
for column in X.columns:
    if X[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        X[column] = le.fit_transform(X[column])

#%%
from sklearn.naive_bayes import MultinomialNB as MNB
clf = MNB(alpha=1.0,class_prior=None, fit_prior=True)
clf.fit(X,y)

#%%
print(y.dtypes)

#%%

#print(X)
# import numpy as np
#Xpredict = pd.DataFrame(np.array([0,25,0]).reshape(1,3), columns = ['CountryOfExploitation','citizenship','typeOfExploitConcatenated'])
y1 = clf.predict(X)

from sklearn import metrics
metrics.accuracy_score(y,y1)

print(y.value_counts())
#%%
