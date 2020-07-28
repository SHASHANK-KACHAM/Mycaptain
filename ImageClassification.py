
# In[13]:


#importing dependencies 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


train=pd.read_csv("F:/Datasets/train.csv")
test=pd.read_csv("F:/Datasets/test.csv")
print ("Training dataset has %i observations and %i variables" %(train.shape[0], train.shape[1]))
print ("Testing dataset has %i observations and %i variables" %(test.shape[0], test.shape[1]))


# In[16]:


train.head()


# In[29]:


#extracting data from the dataset and viewing them up close 
a=train.iloc[3,1:].values
#reshaping the extracted data into the reasonable size
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[17]:


#Seperate the target and independant variables
df_x=train.iloc[:,1:]
df_y=train.iloc[:,0]


# In[31]:


#creating test and train size /batches
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.head())


# In[19]:


rf=RandomForestClassifier(n_estimators=100)


# In[20]:


rf.fit(X_train,y_train)


# In[8]:


pred=rf.predict(X_test)
print ("Classification Report")
print(classification_report(y_test, pred))
print ("Confusion Report")
print(confusion_matrix(y_test, pred))


# In[9]:


pred


# In[11]:


#check predicted accuracy
s=y_test.values

#calculate number of correctly predicted values 
count=0
for i in range(len(pred)):
    if(pred[i]==s[i]):
        count=count+1
print(count)


# In[32]:


#total values of that to prediction cdose was run on
len(pred)


# In[33]:


#accuracy value
count/len(pred)

