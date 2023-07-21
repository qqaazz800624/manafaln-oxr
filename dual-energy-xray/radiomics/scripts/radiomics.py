#%%

import pandas as pd
from scipy.stats import ttest_ind
import xgboost as xgb
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

#%%

data_path = '/home/u/qqaazz800624/manafaln-oxr/cardiovascular-calcification/radiomics/data/radiomics_simplified.csv'
cac_path = '/home/u/qqaazz800624/manafaln-oxr/cardiovascular-calcification/radiomics/data/CAC_scores.csv'
data_all_path = '/home/u/qqaazz800624/manafaln-oxr/cardiovascular-calcification/radiomics/data/df_all.csv'

df = pd.read_csv(data_path)
df_cac = pd.read_csv(cac_path)
df_all = pd.read_csv(data_all_path)

#%%

df.shape

#%%

df_all.loc[df_all['cac_score'] > 400, 'Y'] = 1
df_all.loc[df_all['cac_score'] <= 400, 'Y'] = 0

#%%

X, y = df_all.drop(columns=['Unnamed: 0','uid','cac_score','Y']), df_all['Y']

X_tv, X_test, y_tv, y_test = train_test_split(X, y, 
                                              test_size=.2, 
                                              random_state=123,
                                              stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, 
                                                  test_size=.25, 
                                                  random_state=12,
                                                  stratify=y_tv)

#%%

# XGBoost
model_xgb = XGBClassifier(n_estimators=1000)
model_xgb.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              early_stopping_rounds=30)


#%%

def print_result(model, X, y):
  pred = model.predict(X)
  prob = model.predict_proba(X)[:, 1]
  auc = roc_auc_score(y, prob)
  print(confusion_matrix(y, pred))
  print(f'AUC = {auc:.3f}')

# Print XGBoost result
print('Training')
print_result(model_xgb, X_train, y_train)
print('------')

print('Validation')
print_result(model_xgb, X_val, y_val)
print('------')

print('Testing')
print_result(model_xgb, X_test, y_test)
print('------')


#%%

ax = xgb.plot_importance(model_xgb, max_num_features=5)
plt.rcParams['figure.figsize'] = [5,5]

fig = ax.get_figure()

#%%

a = df_all['image_front_hard_original_glszm_SizeZoneNonUniformityNormalized']
b = df_all['image_front_soft_original_glszm_SizeZoneNonUniformityNormalized']

t, p = ttest_ind(a, b, equal_var=False)
print(t)
print(p)

#%%

a = df_all['image_front_hard_original_ngtdm_Contrast']
b = df_all['image_front_combined_original_ngtdm_Contrast']

t, p = ttest_ind(a, b, equal_var=False)
print(t)
print(p)


#%%

a = df_all['image_front_hard_original_glrlm_RunEntropy']
b = df_all['image_front_soft_original_glrlm_RunEntropy']

t, p = ttest_ind(a, b, equal_var=False)
print(t)
print(p)


#%%






#%%




#%%




#%%





#%%




#%%




#%%