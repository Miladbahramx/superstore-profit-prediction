#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# In[2]:


data = pd.read_csv(r'D:/Python/DATA/New/samplesuperstore.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data['Category'].unique()


# In[9]:


data['Category'].value_counts()


# In[10]:


data['Sub-Category'].value_counts()


# In[11]:


data['Region'].value_counts()


# In[12]:


data['Country/Region'].value_counts()


# In[13]:


data['City'].value_counts()


# In[14]:


data['Sales'].sum()


# In[15]:


data['Profit'].sum()


# In[16]:


data['Sales'].mean()


# In[17]:


data.groupby('Category')['Sales'].sum().sort_values(ascending=False)


# In[18]:


data.groupby('Category')['Profit'].sum().sort_values(ascending=False)


# In[19]:


sales_by_cat = data.groupby('Category')['Sales'].sum().sort_values()
sales_by_cat.plot(kind='barh')
plt.title('Sales by Category')
plt.xlabel('Sales')
plt.ylabel('Category')
plt.show()


# In[20]:


profit_by_cat = data.groupby('Category')['Profit'].sum().sort_values()
profit_by_cat.plot(kind='barh')
plt.title('Sales by Profit')
plt.xlabel('Profit')
plt.ylabel('Category')
plt.show()


# In[21]:


data.groupby("Region")["Sales"].sum().sort_values().plot(kind="barh")
plt.title("Sales by Region")
plt.show()


# In[22]:


top_products = (
    data.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_products.plot(kind="bar")

plt.title("Top 10 Products")
plt.show()


# In[23]:


top_profit = (
    data.groupby("Product Name")["Profit"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

top_profit.plot(kind="bar")

plt.title("Top 10 Profitable Products")
plt.show()


# In[24]:


product_profit = (
    data.groupby("Product Name")["Profit"]
    .sum()
    .sort_values()
)
product_profit.head(10)


# In[25]:


plt.figure(figsize=(10,5))
sns.histplot(data['Sales'],bins=50)
plt.title('Sales Distribution')
plt.show()


# In[26]:


plt.figure(figsize=(10,5))
sns.histplot(data['Profit'],bins=50)
plt.title('Profit Distribution')
plt.show()


# In[27]:


plt.figure(figsize=(10,5))
sns.boxplot(x=data["Sales"])
plt.show()


# In[28]:


plt.figure(figsize=(10,5))
sns.boxplot(x=data["Profit"])
plt.show()


# In[29]:


corr = data[
    ['Sales','Profit','Quantity','Discount']
].corr()

corr


# In[30]:


plt.figure(figsize=(8,6))

sns.heatmap(
    corr , 
    annot=True ,
    cmap='coolwarm'    
)

plt.show()


# In[31]:


plt.figure(figsize=(10,6))

sns.scatterplot(
    data=data,
    x="Discount",
    y="Profit"
)

plt.show()


# In[32]:


x = data[['Sales','Quantity','Discount']]
y = data['Profit']


# In[33]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)


# In[34]:


model = LinearRegression()
model.fit(x_train , y_train)


# In[35]:


y_pred = model.predict(x_test)


# In[36]:


print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))


# In[37]:


data['Order Date'] = pd.to_datetime(data['Order Date'])


# In[38]:


data['month'] = data['Order Date'].dt.month


# In[39]:


data['year'] = data['Order Date'].dt.year


# In[40]:


data_encoded = pd.get_dummies(data , columns=['Category','Region'] , drop_first=True)


# In[41]:


features = ['Sales','Quantity','Discount','month','year']


# In[42]:


features += [col 
             for col in data_encoded.columns
             if col.startswith('Category_')
             or col.startswith('Region_')
            ]


# In[43]:


x = data_encoded[features]
y = data_encoded['Profit']


# In[44]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)


# In[45]:


model = LinearRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)


# In[46]:


print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))


# In[47]:


rf = RandomForestRegressor(n_estimators=100 , random_state=42)


# In[48]:


rf.fit(x_train , y_train)


# In[49]:


y_pred_rf = rf.predict(x_test)


# In[50]:


print(r2_score(y_test,y_pred_rf))
print(mean_absolute_error(y_test,y_pred_rf))


# In[51]:


importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf.feature_importances_})

importance = importance.sort_values(
    by='Importance',
    ascending=False)

print(importance)


# In[52]:


plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature')

plt.show()


# In[53]:


train_score = rf.score(x_train, y_train)
test_score = rf.score(x_test, y_test)

print('Train:', train_score)
print('Test :', test_score)


# In[54]:


scores = cross_val_score(
    rf,
    x,
    y,
    cv=5,
    scoring='r2')

print(scores)
print(scores.mean())


# In[ ]:




