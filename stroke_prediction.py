#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# # Importing Dataset

# In[2]:


dataset=pd.read_csv(r'G:\Study Material\Projects\Health Care Stroke Dataset\healthcare-dataset-stroke-data.csv')
dataset


#  Creating copy of Dataset

# In[3]:


df=dataset.copy()
df                            # We'll be working with df in whole notebook


# In[4]:


df.isnull().sum()                       # Checking Null values for each column


# In[5]:


df.columns


# In[6]:


for i in df.columns:
    print(f"Unique values For {i} =>",df[i].unique())
    print("\n")


# # Dealing with Missing data

# In[7]:


df['bmi'].fillna(df['bmi'].mean(), inplace=True)
df


# In[8]:


df.isnull().sum()


# # Exploratory Data Analysis

# **Super Biased Dataset**

# In[9]:


df['stroke']=df['stroke'].replace(to_replace=1,value="Yes")
df['stroke']=df['stroke'].replace(to_replace=0,value="No")


# In[10]:


fig=px.bar(df,x='gender',color='stroke',barmode='group')
fig.show()


# In[11]:


fig=px.bar(df,y='hypertension',x='stroke',color='gender',barmode='group')
fig.show()


# In[12]:


fig=px.bar(df,y='heart_disease',x='stroke',color='gender',barmode='group')
fig.show()


# In[13]:


fig=px.bar(df,y='stroke',x='ever_married',color='gender',barmode='group')
fig.show()


# In[14]:


fig=px.bar(df,x='work_type',color='gender',barmode='group')
fig.show()


# In[15]:


px.histogram(df,x='bmi',color='stroke')


# In[16]:


px.histogram(df,x='avg_glucose_level',color='stroke')


# In[17]:


stroke_list=df['stroke'].value_counts()
stroke_list


# In[18]:


fig=px.pie(values=stroke_list.values,names=stroke_list.index)
fig.show()


# In[19]:


work_list=df['work_type'].value_counts()
work_list

fig=px.pie(values=work_list.values,names=work_list.index)
fig.show()


# In[20]:


work_list=df['smoking_status'].value_counts()
work_list

fig=px.pie(values=work_list.values,names=work_list.index)
fig.show()


# In[21]:


px.bar(df,x='smoking_status',color='stroke',barmode='group')


# # Checkin For Outliers

# In[22]:


def remove_outliers(data):
    arr=[]
    #print(max(list(data)))
    q1=np.percentile(data,25)
    q3=np.percentile(data,75)
    iqr=q3-q1
    mi=q1-(1.5*iqr)
    ma=q3+(1.5*iqr)
    #print(mi,ma)
    for i in list(data):
        if i<mi:
            i=mi
            arr.append(i)
        elif i>ma:
            i=ma
            arr.append(i)
        else:
            arr.append(i)
    #print(max(arr))
    return arr


# In[23]:


px.box(df,y='age',x='gender')


# In[24]:


fig=px.box(df,y='bmi')
fig.show()


# In[25]:


outlier=pd.DataFrame()


# In[26]:


q1=np.percentile(df['bmi'],25)
q3=np.percentile(df['bmi'],75)
iqr=q3-q1
mi=q1-(1.5*iqr)
ma=q3+(1.5*iqr)


# In[ ]:





# In[27]:


print(mi,ma)


# In[28]:


outlier=df[(df['bmi']<mi) | (df['bmi']>ma)]


# In[29]:


outlier


# In[30]:


df['bmi']=remove_outliers(df['bmi'])


# In[31]:


fig=px.box(df,y='bmi')
fig.show()


# In[32]:


fig=px.box(df,y='age',x='stroke')
fig.show()


# In[33]:


smoked=df[df['stroke']=='Yes']
smoked


# In[34]:


smoked[smoked['age']==1.32]


# In[35]:


smoked[smoked['age']==14]


# As there are only two outliers in dataset with age==1.32 and 14 who are having stroke I am removing these rows

# In[36]:


df = df.drop(index=[132,245]).reset_index()
df


# In[37]:


fig=px.box(df,y='avg_glucose_level',color='stroke',points='all')
fig.show()


# A blood sugar level less than 140 mg/dL (7.8 mmol/L) is normal. A reading of more than 200 mg/dL (11.1 mmol/L) after two hours indicates diabetes. A reading between 140 and 199 mg/dL (7.8 mmol/L and 11.0 mmol/L) indicates prediabetes.

# In[38]:


q1=np.percentile(df['avg_glucose_level'],25)
q3=np.percentile(df['avg_glucose_level'],75)
iqr=q3-q1
mi=q1-(1.5*iqr)
ma=q3+(1.5*iqr)


# In[39]:


print(mi,ma)


# In[40]:


outlier=df[(df['avg_glucose_level']<mi) | (df['avg_glucose_level']>ma)]
outlier


# 628 rows are there with outliers under avg_glucose_level so we cannot drop these rows directly.
# These many rows make 12.5% of our dataset

# In[41]:


df['avg_glucose_level']=remove_outliers(df['avg_glucose_level'])
df


# # Dealing with Categorical Variable

# **Working With Gender Column**

# In[42]:


df['gender'].unique()


# In[43]:


df['gender'].value_counts()


# In[44]:


df['gender']=df['gender'].replace(to_replace='Other',value='Male')
df['stroke']=df['stroke'].replace(to_replace='Yes',value=1)
df['stroke']=df['stroke'].replace(to_replace='No',value=0)


# In[45]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['ever_married']=le.fit_transform(df['ever_married'])
df['Residence_type']=le.fit_transform(df['Residence_type'])


# In[46]:


df=df.drop(['id','index'],axis=1)
df                                                


# In[47]:


stroked=df[df['stroke']==1]
not_stroked=df[df['stroke']==0]


# In[48]:


print(stroked.shape,not_stroked.shape)


# In[49]:


df['smoking_status'].value_counts()


# In[50]:


df['smoking_status'].mode()[0]


# In[51]:


df['smoking_status']=df['smoking_status'].replace(to_replace='never smoked',value=-1)
df['smoking_status']=df['smoking_status'].replace(to_replace='Unknown',value=0)
df['smoking_status']=df['smoking_status'].replace(to_replace='formerly smoked',value=1)
df['smoking_status']=df['smoking_status'].replace(to_replace='smokes',value=2)


# In[52]:


df['smoking_status'].value_counts()


# In[53]:


df


# In[54]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[5])],remainder='passthrough')
df=ct.fit_transform(df)
df=pd.DataFrame(df)
df


# In[55]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# # Splitting Dataset 

# In[56]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[57]:


print(x_train.shape,y_train.shape)


# # Balancing Dataset using SMOTE

# In[58]:


#!pip install imbalanced-learn


# In[59]:


from imblearn.over_sampling import SMOTE
smt=SMOTE(random_state=0)


# In[60]:


x_train_smote,y_train_smote=SMOTE().fit_resample(x_train,y_train)


# In[61]:


print(x_train_smote.shape,y_train_smote.shape)


# In[62]:


from collections import Counter
print("Before SMOTE", Counter(y_train))
print("After SMOTE", Counter(y_train_smote))


# # Machine Learning ALgorithms

# 
# 
# 1.   Logistic Regression
# 
# 

# In[63]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train_smote,y_train_smote)


# In[64]:


lr_pred=lr.predict(x_test)
lr_pred


# In[65]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm_lr=confusion_matrix(y_test,lr_pred)
cm_lr


# In[66]:


acc_lr=accuracy_score(y_test,lr_pred)
acc_lr


# 
# 
# 2.   Decision Tree Classifier
# 
# 

# In[67]:


from sklearn.tree import DecisionTreeClassifier
dsc=DecisionTreeClassifier()
dsc.fit(x_train_smote,y_train_smote)


# In[68]:


dsc_pred=dsc.predict(x_test)
dsc_pred


# In[69]:


cm_dsc=confusion_matrix(y_test,dsc_pred)
cm_dsc


# In[70]:


acc_dsc=accuracy_score(y_test,dsc_pred)
acc_dsc


# 
# 
# 3.   Random Forest Classifier
# 
# 
# 

# In[71]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=400)
rfc.fit(x_train_smote,y_train_smote)


# In[72]:


rfc_pred=rfc.predict(x_test)
rfc_pred


# In[73]:


cm_rfc=confusion_matrix(y_test,rfc_pred)
cm_rfc


# In[74]:


acc_rfc=accuracy_score(y_test,rfc_pred)
acc_rfc


# 
# 
# 4.   XGBoost
# 
# 
# 

# In[75]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train_smote,y_train_smote)


# In[76]:


xgb_pred=xgb.predict(x_test)


# In[77]:


cm_xgb=confusion_matrix(y_true=y_test,y_pred=xgb_pred)
cm_xgb


# In[78]:


acc_xgb=accuracy_score(y_test,xgb_pred)
acc_xgb


# # Feature Scaling
# 

# In[79]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_smote=sc.fit_transform(x_train_smote)
x_test=sc.transform(x_test)


# # Neural Networks

# In[ ]:





# In[80]:


from keras.models import Input,Model
from keras.layers import Dense


# In[81]:


i=Input(shape=[14])

layer1=Dense(units=10,activation='relu')(i)
layer2=Dense(units=10,activation='relu')(layer1)
out=Dense(units=1,activation='sigmoid')(layer2)

ann=Model(inputs=i,outputs=out)


# In[82]:


ann.summary()


# In[83]:


from keras.utils.vis_utils import plot_model


# In[84]:


plot_model(ann, to_file='ann_model_plot.png', show_shapes=True, show_layer_names=True)


# # Compiling the Model

# In[85]:


ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# # Training modle

# In[86]:


ann.fit(x_train_smote,y_train_smote,batch_size=32,epochs=15)


# In[87]:


y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)
y_pred


# In[88]:


cm=confusion_matrix(y_test,y_pred)
cm


# In[89]:


score=accuracy_score(y_test,y_pred)
score


# # Results

# In[90]:


print("Logistic Regression Accuracy=>{:.2%}".format(acc_lr))
print("Decision Tree Classifier Accuracy=>{:.2%}".format(acc_dsc))
print("Random Forest Classifeir Accuracy=>{:.2%}".format(acc_rfc))
print("XGB Classifeir Accuracy=>{:.2%}".format(acc_xgb))
print("Neural Network Accuracy=>{:.2%}".format(score))


# Front-End Fireworks




