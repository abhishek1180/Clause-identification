#!/usr/bin/env python
# coding: utf-8
Data Set Information:

This dataset describes a set of 102 molecules of which 39 are judged by human experts to be musks and the remaining 63 molecules are judged to be non-musks. The goal is to learn to predict whether new molecules will be musks or non-musks. However, the 166 features that describe these molecules depend upon the exact shape, or conformation, of the molecule. Because bonds can rotate, a single molecule can adopt many different shapes. To generate this data set, all the low-energy conformations of the molecules were generated to produce 6,598 conformations. Then, a feature vector was extracted that describes each conformation.

This many-to-one relationship between feature vectors and molecules is called the "multiple instance problem". When learning a classifier for this data, the classifier should classify a molecule as "musk" if ANY of its conformations is classified as a musk. A molecule should be classified as "non-musk" if NONE of its conformations is classified as a musk.Attribute Information:

molecule_name: Symbolic name of each molecule. Musks have names such as MUSK-188. Non-musks have names such as NON-MUSK-jp13.
conformation_name: Symbolic name of each conformation. These have the format MOL_ISO+CONF, where MOL is the molecule number, ISO is the stereoisomer number (usually 1), and CONF is the conformation number.
f1 through f162: These are "distance features" along rays (see paper cited above). The distances are measured in hundredths of Angstroms. The distances may be negative or positive, since they are actually measured relative to an origin placed along each ray. The origin was defined by a "consensus musk" surface that is no longer used. Hence, any experiments with the data should treat these feature values as lying on an arbitrary continuous scale. In particular, the algorithm should not make any use of the zero point or the sign of each feature value.
f163: This is the distance of the oxygen atom in the molecule to a designated point in 3-space. This is also called OXY-DIS.
f164: OXY-X: X-displacement from the designated point.
f165: OXY-Y: Y-displacement from the designated point.
f166: OXY-Z: Z-displacement from the designated point.
class: 0 => non-musk, 1 => musk

Please note that the molecule_name and conformation_name attributes should not be used to predict the class.
# In[1]:


# Import library
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


# Create data frame and replace missing values with NaN
missing_values = ["n/a", "na", "--"]
data=pd.read_csv('./Task.csv',na_values = missing_values)
data.head(2)


# In[3]:


# Shape of dataframe
print('Data_shape: ',data.shape)
#Count all NaN in a DataFrame
print('Total_Null_values: ',data.isnull().sum().sum())
# column index of dataframe
print('Column_index: ', data.columns)
#Total unique value of columns molecule_name and conformation_name
print(' Total_molecule_name:', len(data['molecule_name'].unique()),'\n','Total_conformation_name: ', len(data['conformation_name'].unique()))


# In[4]:


# Unique values of columns molecule_name and conformation_name
data['molecule_name'].unique()


# In[5]:


# Grouping each unique value in column molecule_name and then count in each group
data.groupby("molecule_name")["conformation_name"].count()


# In[6]:


# group of first values from each group
grouping_data=data.groupby('molecule_name')
grouping_data.first()


# In[7]:


# group of first molecule in column molecule_name
grouping_data.get_group('MUSK-212')


# In[8]:


grouping_data.head(1)


# In[9]:


data.describe()


# In[10]:


# removing 'ID','molecule_name','conformation_name','class' from columns
columns_list=list(data.columns)
columns_list= [name for name in columns_list if name not in ('ID','molecule_name','conformation_name','class')]


# In[11]:


# plotting normal distribution of each features of data
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

fig, ax1 = plt.subplots(ncols=1, figsize=(6, 5))

ax1.set_title('Before Scaling')
for name in columns_list:
    sns.kdeplot(data[name], ax=ax1)
    
plt.show()


# In[12]:


# creating features_data for splitting into training and testing dataset
data_removed_label=data.drop(['class'],axis=1)
# creating label_data for splitting into training and testing dataset
data_label=data[['class']]

data_removed_column=data.drop(['ID','molecule_name','conformation_name'],axis=1)
# In[13]:


# Splitting into training and testing dataset
from sklearn.model_selection import train_test_split
data_train,data_val,data_train_label,data_val_label=train_test_split(data_removed_label,data_label,test_size=0.2,random_state=1)


# In[14]:


# Removing Unnecessary features from training and testing dataset which we may use later.
data_train_removed_column=data_train.drop(['ID','molecule_name','conformation_name'],axis=1)
data_val_removed_column=data_val.drop(['ID','molecule_name','conformation_name'],axis=1)


# In[15]:


# Preprocessing the training and testing dataset with standard Scaler
from sklearn.preprocessing import StandardScaler
stds=StandardScaler()
scaled_data_train_array=stds.fit_transform(data_train_removed_column)
scaled_data_val_array=stds.transform(data_val_removed_column)


# In[16]:


# Convert scaled training and testing arrays into dataframe 
scaled_data_val=pd.DataFrame(scaled_data_val_array,columns=columns_list)
scaled_data_train=pd.DataFrame(scaled_data_train_array,columns=columns_list)


# In[17]:


#  plotting normal distribution of each features of training_data before scaling and after scaling
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
for name in columns_list:
    sns.kdeplot(data_train_removed_column[name], ax=ax1)

ax2.set_title('After Scaling')
for name in columns_list:
    sns.kdeplot(scaled_data_train[name], ax=ax2)

plt.show()


# ## MLP model

# In[18]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,make_scorer


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# create the model
new_model=Sequential()
n_cols=scaled_data_train.shape[1]
new_model.add(Dense(100,activation='tanh',input_shape=(n_cols,)))
new_model.add(Dropout(0.5))
new_model.add(Dense(100,activation='tanh'))
new_model.add(Dropout(0.5))
new_model.add(Dense(100,activation='tanh'))
new_model.add(Dropout(0.5))
new_model.add(Dense(1,activation='sigmoid'))
# comile the mdoel
new_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

from livelossplot.tf_keras import PlotLossesCallback
new_model.fit(scaled_data_train,
              data_train_label,
              validation_data=(scaled_data_val,
              data_val_label),
              epochs=100,
              callbacks=[PlotLossesCallback()],
              verbose=0)


# In[20]:


# predict validation dataset
predict_data_val_label=new_model.predict(scaled_data_val,verbose=0)


# In[21]:


# converting predicted validation dataset into dataframe
predict_data_val_frame=pd.DataFrame(predict_data_val_label,columns=['class'])


# In[22]:


## Predicted Validation dataset labels
# we set threshold value to 0.5 to classify classes in validation dataset.
print('Predicted_value -','\n','class 1:',len(predict_data_val_frame[predict_data_val_frame['class']>=0.5]),'\n','class 0:',len(predict_data_val_frame[predict_data_val_frame['class']<0.5]))


# In[23]:


# Actual Validation dataset labels
print('Actual_value -','\n','class 1:',len(data_val_label[data_val_label['class']==1]),'\n','class 0:',len(data_val_label[data_val_label['class']==0]))


# In[24]:


## Creating predicted Validation dataset label column in only class 0 and 1.
# Set 0 if value<0.5 and 1 if >=0.5 
predict_data_val_frame['class'] = np.where((predict_data_val_frame['class']<0.5),0,predict_data_val_frame['class'])
predict_data_val_frame['class'] = np.where((predict_data_val_frame['class']>=0.5),1,predict_data_val_frame['class'])


# In[25]:


# F1 score, precision, recall
from sklearn.metrics import classification_report
print(classification_report(data_val_label, predict_data_val_frame))

