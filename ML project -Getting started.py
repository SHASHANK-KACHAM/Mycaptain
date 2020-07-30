
# coding: utf-8

# In[5]:


import sys
print("Python :{}".format(sys.version))
import scipy
print("Scipy :{}".format(scipy.__version__))
import numpy 
print("Numpy :{}".format(numpy.__version__))
import matplotlib
print("Matplotlib:{}".format(matplotlib.__version__))
import pandas
print("Pandas :{}".format(pandas.__version__))
import sklearn
print("Sklearn :{}".format(sklearn.__version__))


# In[11]:


import pandas as pd
from  pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier 


# In[13]:


# loading data 
url= "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset= pd.read_csv(url,names=names)


# In[14]:


#dimensions of the dataset
print(dataset.shape)


# In[15]:


#take a peek at the data
dataset.head(20)


# In[16]:


#statistical summary 
dataset.describe()


# In[17]:


#class distribution 
dataset.groupby('class').size()


# In[24]:


dataset['class'].unique()


# In[25]:


# univariate plots -box and whisker plots 
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()


# In[26]:


#histogram of the variable 
dataset.hist()
plt.show()


# In[27]:


#multivariate plots 
scatter_matrix(dataset)
plt.show()


# In[39]:


#creating a validation dataset 
#splitting dataset
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2,random_state=1)


# In[40]:


#Logistic Regression
#LinearDiscriminant analysis
#KNN
#Classification and regression Trees
#Gaussian Naive bayes
#Support Vector machine

#building models
models =[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))


# In[42]:


#evaluate the created models 
results=[]
names=[]
for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s : %f (%f)' % (name,cv_results.mean(),cv_results.std()))


# In[43]:


#compare our models 
plt.boxplot(results,labels=names)
plt.title('Algorithm Comparison')
plt.show()


# In[44]:


#make predictions on svm
model=SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_val)


# In[47]:


# evaluate our predictions 
print(accuracy_score(Y_val,predictions))
print(confusion_matrix(Y_val,predictions))
print(classification_report(Y_val,predictions))

