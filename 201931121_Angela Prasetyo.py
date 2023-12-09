#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('/Users/ANGELA//Data/Data.csv')
data


# In[3]:


from sklearn.preprocessing import LabelEncoder


# In[4]:


encoder = LabelEncoder()


# In[5]:


data['Status']=encoder.fit_transform(data['Status'])
data


# In[6]:


data['Hadir_Training']=encoder.fit_transform(data['Hadir_Training'])
data


# In[7]:


data['Durasi_Waktu_Pembelajaran']=encoder.fit_transform(data['Durasi_Waktu_Pembelajaran'])
data


# In[8]:


print("Mapping Durasi Waktu Pembelajaran")
for i, Durasi_Waktu_Pembelajaran in enumerate (encoder.classes_):
    print(Durasi_Waktu_Pembelajaran , "=",i)


# In[9]:


data['Pembelajaran_Materi']=encoder.fit_transform(data['Pembelajaran_Materi'])
data


# In[10]:


data1=data[['Status','Hadir_Training','Durasi_Waktu_Pembelajaran','Pembelajaran_Materi']]
data1


# In[11]:


cdf=data1.fillna(data1.mean())
cdf.head(5)


# In[12]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[13]:


viz = data[['Status']]
viz. hist()
plt.show()


# In[14]:


viz = data[['Hadir_Training']]
viz. hist()
plt.show()


# In[15]:


viz = data[['Durasi_Waktu_Pembelajaran']]
viz. hist()
plt.show()


# In[16]:


viz = data[['Pembelajaran_Materi']]
viz. hist()
plt.show()


# In[17]:


cdf.describe()


# In[18]:


from sklearn import linear_model
import statsmodels.api as sm


# In[19]:


regr = linear_model.LinearRegression()
X = data[['Hadir_Training','Pembelajaran_Materi','Durasi_Waktu_Pembelajaran']]
y = data['Status']
regr.fit (X, y)


# In[20]:


print ('Coefficients:', regr.coef_)
print ('Intercept:', regr.intercept_)


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


test_x = np.asanyarray(data[['Hadir_Training',
                             'Pembelajaran_Materi','Durasi_Waktu_Pembelajaran']])
test_y = np.asanyarray(data[['Status']])
regr.fit(test_x, test_y)


# In[23]:


train_x = np.asanyarray(data[['Hadir_Training',
                             'Pembelajaran_Materi','Durasi_Waktu_Pembelajaran']])
train_y = np.asanyarray(data[['Status']])
regr.fit(train_x, train_y)


# In[46]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math

test_y_ = regr.predict(test_x)


# In[47]:


print("Mean absolute error (MAE): %.2f" % mean_absolute_error(test_y_,test_y))
print("Mean Squared error (MSE): %.2f" % mean_squared_error(test_y_,test_y))
print("Roots Mean Squared error (RMSE)mae,: %.2f" % math.sqrt(mean_squared_error(test_y_,test_y)))


# In[26]:


plt.scatter(data.Hadir_Training, data.Status, color='blue')
plt.scatter(data.Pembelajaran_Materi, data.Status , color='yellow')
plt.scatter(data.Durasi_Waktu_Pembelajaran, data.Status, color='red')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0],color='red')
plt.xlabel("Hadir_Training,Pembelajaran_Materi,Durasi_Waktu_Pembelajaran")
plt.ylabel("Status")
plt.show()


# In[27]:


data = pd.read_csv('/Users/ANGELA/Data/Data.csv')
data


# In[28]:


pip install geopandas


# In[29]:


pip install pandas


# In[30]:


import pandas as pd
import geopandas


# In[31]:


data['coordinates']=data[['Longtitude',"Latitude"]].values.tolist()
data.head()


# In[32]:


from shapely.geometry import Point
data['coordinates'] = data['coordinates'].apply(Point)
data.head()


# In[33]:


type(data)


# In[34]:


stations = geopandas.GeoDataFrame(data, geometry='coordinates')
type(stations)


# In[35]:


data['COUNTER'] =1     

lulus = data[data['Status'] == 'Lulus'].groupby(['Provinsi', 'Latitude', 'Longtitude', 'Status']).size().reset_index(name='LulusCount')
tidaklulus = data[data['Status'] == 'Tidak Lulus'].groupby(['Provinsi', 'Latitude', 'Longtitude', 'Status']).size().reset_index(name='TidakLulusCount')


# In[36]:


stations.plot()


# In[37]:


import folium


# In[38]:


osm_map = folium.Map(location=[stations.Latitude.mean(), stations.Longtitude.mean()], zoom_start=7)
osm_map


# In[39]:


pip install geojson


# In[40]:


from geojson import Point


# In[41]:


point = folium.features.GeoJson(stations.to_json())


# In[42]:


osm_map.add_child(point)
osm_map


# In[43]:


data.head()


# In[44]:


for data in range(len(lulus)):
   folium.Marker(
      location=[lulus.iloc[data]['Latitude'], lulus.iloc[data]['Longtitude']],
      popup=('Provinsi:'+'<br>'+ str(lulus.iloc[data]['Provinsi'])+'<br>'+'Status:'+'<br>'+ str(lulus.iloc[data]['Status']) +'<br>' + 'Jumlah:'+'<br>'+ str(lulus.iloc[data]['LulusCount']))
   ).add_to(osm_map)
osm_map


# In[45]:


for data in range(len(tidaklulus)):
   folium.Marker(
      location=[tidaklulus.iloc[data]['Latitude'], tidaklulus.iloc[data]['Longtitude']],
      popup=('Provinsi:'+'<br>'+ str(tidaklulus.iloc[data]['Provinsi'])+'<br>'+'Status:'+'<br>'+ str(tidaklulus.iloc[data]['Status']) +'<br>' + 'Jumlah:'+'<br>'+ str(tidaklulus.iloc[data]['TidakLulusCount']))
   ).add_to(osm_map)
osm_map


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




