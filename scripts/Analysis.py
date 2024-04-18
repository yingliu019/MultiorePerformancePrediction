#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Analysis can be run locally and then migrate to crunchy
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.metrics import root_mean_squared_error
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
from xgboost import plot_importance


# In[57]:


#!pip list --format=freeze > requirements.txt


# # Load Training Data

# In[2]:


directory = r'C:\Users\yingl\OneDrive\Desktop\MultiorePerformancePrediction\MultiorePerformancePrediction\data\training_data'

def collect_training_data(directory = ''):
    lst = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                lst.append(pd.read_csv(os.path.join(root, file)))
    
    df_raw = pd.concat(lst)
    print(f'length starts: {len(df_raw)}')
    for col in ['branch-instructions', 'branch-misses', 'cache-misses', 'cache-references', 'cpu-cycles', 'instructions', 
                'stalled-cycles-frontend', 'L1-icache-load-misses', 'LLC-load-misses', 'LLC-loads', 'LLC-stores', 
                'L1-dcache-prefetch-misses', 'L1-dcache-prefetches', 'L1-icache-loads',
                'branch-load-misses', 'branch-loads', 'dTLB-load-misses', 'dTLB-loads', 'iTLB-load-misses', 'iTLB-loads']:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    #df_raw.to_csv('training_data_all.csv', index=False)
    #df_raw[numeric_feature_lst].corr().to_csv('corr.csv')
    return df_raw

df_raw = collect_training_data(directory)

# def f(row):
#     if any([row['run_id'].startswith(i) for i in ['bfs', 'lavaMD', 'kmeans', 'myocyte']]):
#         return 'rodinia'
#     else:
#         return 'parsec'

# # df_raw['benchmark'] = df_raw.apply(f, axis=1)

df_raw['runtime_serial'] = df_raw['speed_up'] * df_raw['compute_time']
df_raw['IPC'] = df_raw['instructions'] / df_raw['cpu-cycles']
df_raw['IPS'] = df_raw['instructions'] / df_raw['compute_time']


# In[3]:


print(set(df_raw['hostname']))


# ## Anomaly Detection

# In[4]:


sns.displot(df_raw[['speed_up', 'benchmark']], x="speed_up", hue="benchmark")
plt.show()


# In[5]:


df_raw = df_raw.loc[df_raw['speed_up'] <= 100]
sns.displot(df_raw[['speed_up', 'benchmark']], x="speed_up", hue="benchmark")
plt.show()


# In[6]:


plt.figure(figsize=(10, 6))
sns.set(style='whitegrid') 
sns.boxplot(x="threads",
                y="speed_up",
                data=df_raw[['speed_up', 'threads', 'benchmark']])


# In[53]:


fig, axes = plt.subplots(3, 4, figsize=(10, 8))
p_list = sorted(list(set(df_raw['program'])))
for i, ax in enumerate(axes.flatten()):
    p = p_list[i]
    df_slice = df_raw[['threads', 'speed_up', 'program']].loc[df_raw['program'] == p].groupby(['program', 'threads']).mean().reset_index()
    ax.plot(df_slice['threads'], df_slice['speed_up'], linewidth=1.0)
    ax.set_title(p, size=10)

fig.supxlabel('threads')
fig.supylabel('speed up')
plt.show()


# In[8]:


fig, axes = plt.subplots(3, 3, figsize=(10, 8))
h_list = sorted(list(set(df_raw['hostname'])))
for i, ax in enumerate(axes.flatten()):
    h = h_list[i]
    df_slice = df_raw[['threads', 'speed_up', 'hostname']].loc[df_raw['hostname'] == h].groupby(['hostname', 'threads']).mean().reset_index()
    ax.plot(df_slice['threads'], df_slice['speed_up'], linewidth=1.0)
    ax.set_title(h, size=10)

fig.supxlabel('threads')
fig.supylabel('speed up')
plt.show()


# ## Missing Data

# In[9]:


class_feature_lst = []
numeric_feature_lst = []

for col in list(df_raw): 
    null_pct = df_raw[col].isnull().sum() / len(df_raw)
    if null_pct <= 0.2:
        #print(col, null_pct, df_raw.dtypes[col], df_raw.iloc[0])
        if df_raw.dtypes[col] == 'object':
            class_feature_lst.append(col)
        else:
            numeric_feature_lst.append(col)

numeric_feature_lst.remove('speed_up')
numeric_feature_lst.remove('compute_time')
print(numeric_feature_lst)
print(class_feature_lst)


# # Feature Examine

# In[10]:


# take out null
df = df_raw[numeric_feature_lst + ['speed_up']]
sns.heatmap(df.corr())
# for col in list(df):
#     df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
print(f'length after dropping: {len(df)}')

# return df
# df = preprocess_training(df_raw)


# ## Numerical Feature (p value)

# In[11]:


X = df[numeric_feature_lst]
y = df['speed_up']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# ## Categorical Feature (chi test)

# In[12]:


for to_test in class_feature_lst:
    # Create a DataFrame
    #to_test = 'Architecture'
    df_cat_test = df_raw[['speed_up', to_test]]
    
    # Create a contingency table
    table = pd.crosstab(df_cat_test[to_test], df_cat_test['speed_up'])
    
    # Perform the chi-square test
    chi2, p, dof, expected = chi2_contingency(table)
    
    # Interpret the results
    if p < 0.1:
        print('Y --', to_test, p)
    else:
        print('N --', to_test, p)


# ## PCA for Virtulization

# In[13]:


df_pca = df_raw[numeric_feature_lst + ['benchmark']]
df_pca.dropna(inplace=True)

d = {}
d['data'] = df_pca[numeric_feature_lst]
d['benchmark'] = df_pca['benchmark']


# In[14]:


pca = PCA(n_components=2)
components = pca.fit_transform(d['data'])

fig = px.scatter(components, x=0, y=1, color=d['benchmark'])
fig.show()


# # Preprocess data

# ## Train Test Split

# In[15]:


df_tmp = df_raw[numeric_feature_lst + ['speed_up', 'hostname', 'program']]
df_tmp.dropna(inplace=True)

df_train = df_tmp.loc[df_tmp['hostname'] != 'crunchy6']
df_test = df_tmp.loc[df_tmp['hostname'] == 'crunchy6']

print(f'df_train length {len(df_train)}, df_test length {len(df_test)}')

X_train = df_train[numeric_feature_lst]
y_train = df_train['speed_up']
X_test = df_test[numeric_feature_lst]
y_test = df_test['speed_up']
df_result = df_test[['speed_up', 'threads', 'program']]


# In[16]:


X_train_random = X_train.copy()
X_test_random = X_test.copy()
X_train_random["RANDOM"] = np.random.RandomState(42).randn(X_train.shape[0])
X_test_random["RANDOM"] = np.random.RandomState(42).randn(X_test.shape[0])


# In[17]:


numeric_selected_feature_lst = ['CPU(s)','cpu-cycles','host_memused','dTLB-loads','L1-dcache-loads','instructions','branch-loads',
'Stepping','CPU MHz','LLC-stores','LLC-load-misses','CPU family','BogoMIPS','branch-load-misses','context-switches','cache-misses',
'cpu-migrations','iTLB-loads','msr/mperf/','branch-instructions','task-clock','L1-dcache-prefetch-misses',
#'compute_time',
'msr/tsc/','threads','cpu-clock','branch-misses','stalled-cycles-frontend','minor-faults','L1-icache-load-misses','dTLB-load-misses',
'iTLB-load-misses','page-faults','L1-dcache-load-misses','LLC-loads','IPC','cache-references','runtime_serial','IPS',]

df_tmp_2 = df_raw[numeric_selected_feature_lst + ['speed_up', 'hostname']]
df_tmp_2.dropna(inplace=True)

df_train_limited = df_tmp_2.loc[df_tmp_2['hostname'] != 'crunchy6']
df_test_limited = df_tmp_2.loc[df_tmp_2['hostname'] == 'crunchy6']

print(f'df_train length {len(df_train)}, df_test length {len(df_test)}')

X_train_limited = df_train_limited[numeric_selected_feature_lst]
y_train_limited = df_train_limited['speed_up']
X_test_limited = df_test_limited[numeric_selected_feature_lst]
y_test_limited = df_test_limited['speed_up']


# ## Normalization

# In[18]:


scaler = StandardScaler().fit(X_train) 
X_train_std = scaler.transform(X_train) 
X_test_std = scaler.transform(X_test) 


# # Prediction

# In[19]:


df_result


# ## 1) Linear Regression

# ### 1.1) Linear Regression with selected features

# In[20]:


# def predict_regression(df):
#     X = df[numeric_feature_lst]
#     y = df['speed_up']

#     # Split the DataFrame into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     # Create a scikit-learn model
#     model = LinearRegression()

#     # Fit the model to the training data
#     model.fit(X_train, y_train)

#     # Make predictions on the testing data
#     y_pred = model.predict(X_test) # need to cap it >= 0

#     # Evaluate the model's performance
#     print('model.score', model.score(X_test, y_test))
#     # print('RMSE', root_mean_squared_error(y_test, y_pred))
#     print('MSE', mean_squared_error(y_test, y_pred))
#     print('MAE', mean_absolute_error(y_test, y_pred))

# predict_regression(df)


# In[22]:


def perform_linear_and_ridge_regression(X_train, X_test, y_train, y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 
    lin_reg_parameters = { 'fit_intercept': [True, False] }
    lin_reg = GridSearchCV(LinearRegression(), lin_reg_parameters, cv=5)
    lin_reg.fit(X=X_train, y=y_train)
    y_pred = lin_reg.predict(X_test)
    # print('model.score', model.score(X_test, y_test))
    # print('RMSE', root_mean_squared_error(y_test, y_pred))
    print('MSE', mean_squared_error(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))

# X_std = transformer.transform(X)
# X_std = pd.DataFrame(X_std, columns=X.columns)
perform_linear_and_ridge_regression(X_train, X_test, y_train, y_test)
perform_linear_and_ridge_regression(X_train_std, X_test_std, y_train, y_test)
# perform_linear_and_ridge_regression(X=X, Y=y)


# ### 1.2) Linear regression w/ L1 regularization

# In[23]:


def perform_lr_l1(X_train, X_test, y_train, y_test):
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    # print('RMSE', root_mean_squared_error(y_test, y_pred))
    print('MSE', mean_squared_error(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))

perform_lr_l1(X_train, X_test, y_train, y_test)
perform_lr_l1(X_train_std, X_test_std, y_train, y_test)


# In[24]:


# reg = LassoCV(cv=5, random_state=0).fit(X_train_std, y_train)
# y_pred = reg.predict(X_test_std)
# print('MSRE', mean_squared_error(y_test, y_pred))
# print('MAE', mean_absolute_error(y_test, y_pred))


# ### 1.3)* Linear regression w/ L2 regularization

# In[25]:


def perform_lr_l2(X_train, X_test, y_train, y_test):
    ridgecv = linear_model.RidgeCV(alphas=[0.001, 0.01, 0.1, 0.5, 1, 10], cv=5)
    ridgecv.fit(X_train, y_train) # not converging is not std
    y_pred = ridgecv.predict(X_test)
    # print('RMSE', root_mean_squared_error(y_test, y_pred))
    print('MSE', mean_squared_error(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))
    df_result['lr_l2'] = y_pred

perform_lr_l2(X_train, X_test, y_train, y_test)
# perform_lr_l2(X_train_std, X_test_std, y_train, y_test)


# ## 2)* KNN 

# In[26]:


def perform_knn(X_train, X_test, y_train, y_test):
    #create new a knn model
    knn2 = KNeighborsRegressor()
    #create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 25)}
    #use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
    #fit model to data
    knn_gscv.fit(X_train, y_train)
    rf = KNeighborsRegressor(n_neighbors = knn_gscv.best_params_['n_neighbors'])
    
    y_pred = knn_gscv.predict(X_test)
    # print('RMSE', root_mean_squared_error(y_test, y_pred))
    print('MSE', mean_squared_error(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))
    df_result['knn'] = y_pred

perform_knn(X_train, X_test, y_train, y_test)
# perform_knn(X_train_std, X_test_std, y_train, y_test)


# ### w/ Limited Feature

# In[27]:


perform_knn(X_train_limited, X_test_limited, y_train, y_test)


# ## 3)* Random Forest - Bagging

# In[28]:


def perform_rf(X_train, X_test, y_train, y_test):
    param_grid = {
        'max_depth': [5,10,20,30],
        'max_features' : [5,10,20,30],
        'n_estimators': [20,50]}
    
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0), 
                               param_grid=param_grid,
                               cv=KFold(n_splits=5, shuffle=True, random_state=1))
    
    grid_search.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators = grid_search.best_params_['n_estimators'],
                               max_features = grid_search.best_params_['max_features'],
                               max_depth = grid_search.best_params_['max_depth'],
                               random_state = 0)
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print('MSRE', mean_squared_error(y_test, y_pred))
    print('MAE', mean_absolute_error(y_test, y_pred))
    df_result['random_forest'] = y_pred
    
    plt.figure(figsize=(20, 16))
    global_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    global_importances.sort_values(ascending=True, inplace=True)
    global_importances.plot.barh(color='green')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance - Built-in Method")

perform_rf(X_train, X_test, y_train, y_test)
# perform_knn(X_train_std, X_test_std, y_train, y_test)


# ### RF: Feature Importance w/ Random

# In[30]:


rf_random = RandomForestRegressor(n_estimators=100, random_state=42)
rf_random.fit(X_train_random, y_train)

global_importances_random = pd.Series(rf_random.feature_importances_, index=X_train_random.columns)
global_importances_random.sort_values(ascending=True, inplace=True)

plt.figure(figsize=(20, 16))
global_importances_random.plot.barh(color='green')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Global Feature Importance - Built-in Method")


# In[31]:


global_importances_random


# ## 4) Boosting

# ### 4.1) Gradient Boosting

# In[32]:


gb_reg = GradientBoostingRegressor(random_state=0)
gb_reg.fit(X_train, y_train)
y_pred = gb_reg.predict(X_test)
# print('RMSE', root_mean_squared_error(y_test, y_pred))
print('MSE', mean_squared_error(y_test, y_pred))
print('MAE', mean_absolute_error(y_test, y_pred))
df_result['gradient_boost'] = y_pred


# #### Feature Importance

# In[33]:


# print(reg.feature_importances_)
plt.figure(figsize=(20, 16))
global_importances = pd.Series(gb_reg.feature_importances_, index=X_train.columns)
global_importances.sort_values(ascending=True, inplace=True)
global_importances.plot.barh(color='green')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Global Feature Importance - Built-in Method")


# ### 4.2)* XgBoosting
# 
# https://www.kaggle.com/code/carlmcbrideellis/an-introduction-to-xgboost-regression

# In[34]:


xg_regressor=xgb.XGBRegressor(eval_metric='rmsle')
param_grid = {"max_depth":    [4, 5, 6],
              "n_estimators": [500, 600, 700],
              "learning_rate": [0.01, 0.015]}

# try out every combination of the above values
search = GridSearchCV(xg_regressor, param_grid, cv=5).fit(X_train, y_train)

print("The best hyperparameters are ",search.best_params_)

xg_regressor=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],
                           eval_metric='rmse') # mae

xg_regressor.fit(X_train, y_train)
predictions = xg_regressor.predict(X_test)

# print('RMSE', root_mean_squared_error(y_test, predictions))
print('MSE', mean_squared_error(y_test, predictions))
print('MAE', mean_absolute_error(y_test, predictions))
df_result['xgboost'] = predictions


# #### Feature Importance

# In[52]:


plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xg_regressor, max_num_features=25, ax=ax)
plt.show()


# #### Feature Importance w/ Random

# In[54]:


xg_regressor_rn =xgb.XGBRegressor(eval_metric='rmsle')
param_grid = {"max_depth":    [4, 5, 6],
              "n_estimators": [500, 600, 700],
              "learning_rate": [0.01, 0.015]}

# try out every combination of the above values
search = GridSearchCV(xg_regressor_rn, param_grid, cv=5).fit(X_train, y_train)

print("The best hyperparameters are ",search.best_params_)

xg_regressor_rn=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],
                           eval_metric='rmsle')

xg_regressor_rn.fit(X_train_random, y_train)
predictions = xg_regressor_rn.predict(X_test_random)

# print('RMSE', root_mean_squared_error(y_test, predictions))
print('MSE', mean_squared_error(y_test, predictions))
print('MAE', mean_absolute_error(y_test, predictions))

plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(20,16))
plot_importance(xg_regressor_rn, max_num_features=100, ax=ax)
plt.show()


# ## (optional) Neural Network

# In[37]:


clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# print('RMSE', root_mean_squared_error(y_test, predictions))
print('MSE', mean_squared_error(y_test, predictions))
print('MAE', mean_absolute_error(y_test, predictions))


# ## Summary

# In[42]:


print('-------------- all')
df_result = df_result[df_result['threads'] != 1]
print('MSE', mean_squared_error(df_result['speed_up'], df_result['lr_l2']))
print('MSE', mean_squared_error(df_result['speed_up'], df_result['knn']))
print('MSE', mean_squared_error(df_result['speed_up'], df_result['random_forest']))
print('MSE', mean_squared_error(df_result['speed_up'], df_result['gradient_boost']))
print('MSE', mean_squared_error(df_result['speed_up'], df_result['xgboost']))

for t in sorted(set(df_result['threads'])):
    print('--------------', t)
    df_result_sub = df_result[df_result['threads'] == t]
    print('MSE', mean_squared_error(df_result_sub['speed_up'], df_result_sub['lr_l2']))
    print('MSE', mean_squared_error(df_result_sub['speed_up'], df_result_sub['knn']))
    print('MSE', mean_squared_error(df_result_sub['speed_up'], df_result_sub['random_forest']))
    print('MSE', mean_squared_error(df_result_sub['speed_up'], df_result_sub['gradient_boost']))
    print('MSE', mean_squared_error(df_result_sub['speed_up'], df_result_sub['xgboost']))


# In[48]:


print('-------------- all')
print('MAE', mean_absolute_error(df_result['speed_up'], df_result['lr_l2']))
print('MAE', mean_absolute_error(df_result['speed_up'], df_result['knn']))
print('MAE', mean_absolute_error(df_result['speed_up'], df_result['random_forest']))
print('MAE', mean_absolute_error(df_result['speed_up'], df_result['gradient_boost']))
print('MAE', mean_absolute_error(df_result['speed_up'], df_result['xgboost']))

for t in sorted(set(df_result['threads'])):
    print('--------------', t)
    df_result_sub = df_result[df_result['threads'] == t]
    print('MAE', mean_absolute_error(df_result_sub['speed_up'], df_result_sub['lr_l2']))
    print('MAE', mean_absolute_error(df_result_sub['speed_up'], df_result_sub['knn']))
    print('MAE', mean_absolute_error(df_result_sub['speed_up'], df_result_sub['random_forest']))
    print('MAE', mean_absolute_error(df_result_sub['speed_up'], df_result_sub['gradient_boost']))
    print('MAE', mean_absolute_error(df_result_sub['speed_up'], df_result_sub['xgboost']))


# In[ ]:




