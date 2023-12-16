#!/usr/bin/env python
# coding: utf-8

# In[73]:


import warnings

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report, ConfusionMatrixDisplay,precision_score,recall_score, f1_score,roc_auc_score,roc_curve, balanced_accuracy_score 
import six
import sys
sys.modules['sklearn.externals.six'] = six
import joblib
sys.modules['sklearn.externals.joblib'] = joblib


# In[170]:


# Load the dataset
file_path = 'Downloads/stroke_data.csv' 
data = pd.read_csv(file_path)
data.head(10)


# In[171]:


data.info()


# In[172]:


data_vis = data.copy()
unique_values = {}
for col in data_vis.columns:
    unique_values[col] = data_vis[col].value_counts().shape[0]

pd.DataFrame(unique_values, index=['unique value count']).transpose()


# In[173]:


# All data columns except for color
feature_cols = [x for x in data_vis.columns if x not in 'Stroke']
plt.figure(figsize=(25,35))
# loop for subplots
for i in range(len(feature_cols)):
    plt.subplot(8,5,i+1)
    plt.title(feature_cols[i])
    plt.xticks(rotation=90)
    plt.hist(data_vis[feature_cols[i]],color = "deepskyblue")
    
plt.tight_layout()


# In[175]:


data = data.drop(data[data.age < 0].index)
data_vis = data_vis.drop(data_vis[data_vis.age < 0].index)
len(data. index)


# In[176]:


# Traget values frequency
plt.figure(figsize=(8,6))
labels = ['Stroke', 'No-Stroke']
sizes = [data_vis['stroke'].value_counts()[1],data_vis['stroke'].value_counts()[0]]
colors = ['crimson', 'deepskyblue']
explode = (0.01,0.01)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, autopct='%.1f%%', colors=colors, data = data_vis);


# In[178]:


data.drop('stroke', axis=1).corrwith(data.stroke).plot(kind='bar', grid=True, figsize=(10, 6), title="Correlation with Stroke",color="deepskyblue");


# In[179]:


sns.set(rc = {'figure.figsize':(12,12)})
sns.heatmap(data.corr(),vmin=-1, vmax=1, annot = True, fmt='.1g',cmap= 'coolwarm')


# In[181]:


#scale the data before pairplot
data_pairplot = data.copy()
float_columns = [x for x in data.columns if x in ['bmi','age','avg_glucose_level']]

sc = StandardScaler()
data_pairplot[float_columns] = sc.fit_transform(data_pairplot[float_columns])
data_pairplot.head(4)


# In[ ]:





# In[183]:


data.info()
data["sex"].fillna(0, inplace = True) #fill 0 instead NA in the gender column
data = data.drop(data[data.age < 0].index) #drop rows with negative age


# In[184]:


data_skew = data[['age','avg_glucose_level','bmi']]
skew = pd.DataFrame(data_skew.skew())
skew.columns = ['skew']
skew['too_skewed'] = skew['skew'] > .75
skew


# In[131]:


qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')
data[['bmo']] = qt.fit_transform(data[['bmi']])
data[['avg_glucose_level']] = qt.fit_transform(data[['avg_glucose_level']])


# In[185]:


sc = StandardScaler()
data[['bmi']] = sc.fit_transform(data[['bmi']])
data[['age']] = sc.fit_transform(data[['age']])
data[['avg_glucose_level']] = sc.fit_transform(data[['avg_glucose_level']])
data.head()


# In[193]:


(data[['bmi','age','avg_glucose_level']]).describe()


# In[194]:


y = (data['stroke']).astype(int)
X = data.loc[:, data.columns != 'stroke']  # everything except "stroke"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[195]:


print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


# In[196]:


from sklearn.model_selection import GridSearchCV
# defining parameter range
param_grid = {'n_neighbors': [1,3,5,7,9,11,13,15,17,19],  #odd numbers because there are 2 classes in target coulmn
              'weights': ['distance', 'uniform']}  
gridKNN = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
gridKNN.fit(X_train, y_train)


# In[197]:


print(gridKNN.best_params_)


# In[198]:


y_pred_test = gridKNN.predict(X_test)
y_pred_train = gridKNN.predict(X_train)


# In[199]:


print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))


# In[200]:


def train_evaluate_model(y_test):
    #fit the model instance 
    predictions = y_pred_test # calculate predictions

    #compute metrics for evaluation
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)

    #create a dataframe to visualize the results
    eval_df = pd.DataFrame([[accuracy, f1, precision, recall, balanced_accuracy, auc]], columns=['accuracy', 'f1_score', 'precision', 'recall', 'balanced_accuracy', 'auc'])
    return eval_df


# In[201]:


results = train_evaluate_model(y_test)
results.index = ['K Nearest Neighbors - Method 1']
results.style.background_gradient(cmap = sns.color_palette("blend:green,red", as_cmap=True))


# ## Naive Bayes

# In[202]:


cv_N = 10
nb = {'gaussian': GaussianNB(),
      'bernoulli': BernoulliNB()
      }
scores = {}
for key, model in nb.items():
    s = cross_val_score(model, X_train, y_train, cv=cv_N, n_jobs=cv_N, scoring='accuracy')
    scores[key] = np.mean(s)
scores


# In[203]:


# fitting the model
GNB = GaussianNB()
GNB.fit(X_train, y_train)


# In[204]:


y_pred_test = GNB.predict(X_test)
y_pred_train = GNB.predict(X_train)
print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))


# In[205]:


resultsNB = train_evaluate_model(y_test)
resultsNB.index = ['Naive Bayes - Method 1']
results = results.append(resultsNB)
results.style.background_gradient(cmap = sns.color_palette("blend:red,green", as_cmap=True))


# DECISION TREE CLASSIFER

# In[206]:


dt = DecisionTreeClassifier(random_state=42)
dt = dt.fit(X_train, y_train)


# In[207]:


#dt = DecisionTreeClassifier(random_state=42)
dt = dt.fit(X_train, y_train)

# defining parameter range
param_grid = {'max_depth':range(1, dt.tree_.max_depth+1, 2),
              'max_features': range(1, len(dt.feature_importances_)+1)}  
gridDT = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, n_jobs=-1)
  
# fitting the model for grid search
gridDT.fit(X_train, y_train)


# In[208]:


print(gridDT.best_params_)


# In[209]:


y_pred_test = gridDT.predict(X_test)
y_pred_train = gridDT.predict(X_train)


# In[210]:


print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))


# In[211]:


resultsDT = train_evaluate_model(y_test)
resultsDT.index = ['Decision Trees - Method 1']
results = results.append(resultsDT)
results.style.background_gradient(cmap = sns.color_palette("blend:red,green", as_cmap=True))


# Random Forrest Classifer

# In[212]:


RF = RandomForestClassifier(oob_score=True, 
                            random_state=42, 
                            warm_start=True,
                            n_jobs=-1)

# defining parameter range
param_grid = {'n_estimators':[15, 20, 30, 40, 50, 100, 150, 200, 300, 400]
              }  
gridRF = GridSearchCV(RF, param_grid)
  
# fitting the model for grid search
gridRF.fit(X_train, y_train)


# In[213]:


y_pred_test = gridRF.predict(X_test)
y_pred_train = gridRF.predict(X_train)


# In[214]:


print(accuracy_score(y_train, y_pred_train))
print(accuracy_score(y_test, y_pred_test))


# In[215]:


resultsRF = train_evaluate_model(y_test)
resultsRF.index = ['Random Forest - Method 1']
results = results.append(resultsRF)
results.style.background_gradient(cmap = sns.color_palette("blend:red,green", as_cmap=True))


# In[ ]:




