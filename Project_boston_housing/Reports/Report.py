#!/usr/bin/env python
# coding: utf-8

# # Project "Boston housing"
# Stanin Vladislav

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, pearsonr


# ### Data installation

# In[2]:


data  = pd.read_csv('../data/BostonHousing.csv')
data


# ### little EDA

# In[3]:


data.info()


# In[4]:


data.isna().sum()


# **No problems with data**

# #### Distribution of target variable (_medv_):

# In[5]:


sns.histplot(data[['medv']], stat='density', color='blue', legend=False);
plt.xlabel('medv');


# **Looks like normal**

# In[6]:


predictors = data.iloc[:,:-1]
medv = data.iloc[:,-1]


# ## Standartization

# In[7]:


means = predictors.mean(axis=0)
stds = predictors.std(axis=0)

scaled_preds = (predictors - means) / stds

scaled_preds.head()


# ## First linear model

# In[8]:


X = sm.add_constant(scaled_preds)
model_scaled = sm.OLS(medv, X)
results_scaled = model_scaled.fit()

print(results_scaled.summary())


# **The _age_ and _indus_ predictors not significantly predicts the _medv_**  
# _Checking model without these predictors:_

# In[9]:


model_new = sm.OLS(medv, X.drop(columns=["indus", "age"]))
results_new = model_new.fit()

print(results_new.summary())


# **Model did not changed significantly arfter removing those predictors.  
# Nevertheless, F-statistics and R-adj were increased (and p-val decreased).**  
# In the rest of report i will name these paramters as **Statistics** (If they become better, I will say that Statistics _increased_)

# In[10]:


X_new = X.drop(columns=["indus", "age"])


# ## Checking model

# ### Linear relationship

# In[11]:


fig = sns.pairplot(pd.concat([scaled_preds.drop(columns=["indus", "age"]), medv], axis=1), kind="reg", corner=True);
fig.savefig("pairplot.png");


# **Most of predictors have linear relationship with _medv_, but _lstat_ which have looking like exponental relationship**  
# Let's check:

# In[12]:


sns.scatterplot(y=predictors.lstat, 
                x=medv);


# In[13]:


print(f"medv and lstat correlated with Pearson\'s r value {pearsonr(y=predictors.lstat, x=medv).statistic} and pvalue {pearsonr(y=predictors.lstat, x=medv).pvalue}")


# **Checking relationship between _lstat_ and _-ln(medv)_**

# In[14]:


sns.scatterplot(y=predictors.lstat, 
                x=-np.log(medv+1));


# In[15]:


print(f'ln(medv) and lstat correlated with Pearson\'s r value {pearsonr(y=predictors.lstat, x=np.log(medv+1)).statistic.round(4)} and pvalue {pearsonr(y=predictors.lstat, x=np.log(medv+1)).pvalue}')


# **Since Pearson's r not changed significantly, we can leave it**  
# **However,  
# Additional check of model without _lstat_ predictor:**

# In[16]:


model_wo_lstat = sm.OLS(medv, X_new.drop(columns=["lstat"]))
results_wo_lstat = model_wo_lstat.fit()

print(results_wo_lstat.summary())


# **Statistics of model decresed, therefore we leave _lstat_ predictor in model**

# ### Checking if distribution of residuals is normal

# In[17]:


prediction = results_new.get_prediction(X_new)
medv_predicted = prediction.predicted_mean


# In[18]:


residuals = medv - medv_predicted


# In[19]:


sns.histplot(residuals, color='lightblue', binwidth=1);
plt.xlabel('Residuals');


# **Seems like normal...**

# In[20]:


sm.qqplot(residuals, line='s');


# **Still lloks like normal, with some deviations (richest houses)**  
# **but let's check these deviations:**

# ### Checking devations
# (Calculating Cook's distances)

# In[21]:


influence = results_new.get_influence()
cooks = influence.cooks_distance
n_deviations = (cooks[1] < 0.05).sum() 
print(f'There are {n_deviations} significant deviations')


# **There are no deviations!**

# ### Checking homoscedacity

# In[22]:


sns.scatterplot(x = medv, y = residuals);
plt.hlines(0,0, 51, color = 'red');


# **Only some very expensive houses deviate. Nevertheless, we can consider that the model has homoscedasticity**

# ### Checking VIF

# In[23]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[24]:


def vif(preds):
    vif_data = pd.DataFrame()
    vif_data["Predictor"] = preds.columns
    vif_data["VIF"] = [variance_inflation_factor(preds.values, i) for i in range(len(preds.columns))]
    return vif_data


# In[25]:


vif(X_new)


# **There are several predictors that is linked with others**

# In[26]:


X_new_2 = X_new.drop(columns=["tax"])
vif(X_new_2)


# **_It looks much better_**  
# VIF of _rad_ reduced significantly and it means thar _tax_ and _rad_ was linked

# #### Checking new model without _tax_

# In[27]:


model_new_2 = sm.OLS(medv, X_new_2)
results_new_2 = model_new_2.fit()

print(results_new_2.summary())


# **Statistics increased!**  
# **Now model looks pretty!**

# ### The largest modulo value of coefficient belongs to _lstat_ predictor

# #### _rm_ vs raw _medv_

# In[28]:


sns.scatterplot(x = medv, y = predictors.lstat);


# #### _lstat_ vs predicted _medv_

# In[29]:


sns.scatterplot(x = medv_predicted, y = predictors.lstat);
plt.xlabel('medv_predicted');


# ## OPTIONAL

# **Let's see parameters of suburbs that have _medv_ variable > mean + 1 sd** - It will be named _data_medv_high_  
# **And suburbs that have _medv_ variable < mean + 1 sd** - It will be named _data_medv_low_

# In[30]:


mean_medv = medv.mean()
std_medv = medv.std()


# In[31]:


treschold = mean_medv + std_medv


# In[32]:


data_medv_high = data.loc[data.medv > treschold]
data_medv_low = data.loc[data.medv <= treschold]


# ### Looking through correlation between medv and all predictors:

# In[33]:


(pd.concat([medv, X_new_2.drop(columns=['const'])], axis=1)).corr().iloc[:,0]


# **I will check predictors with high correclatioon values**

# **One-sided Mann-Whitney U test for all predictors vs _medv_:**

# In[34]:


for col in data_medv_high.columns:
    print(col, '- in cheap suburbs greater')
    print(f"p-value - {mannwhitneyu(data_medv_low[col], data_medv_high[col], alternative='greater').pvalue}")
    print(col, '- in expensive suburbs greater')
    print(f"p-value - {mannwhitneyu(data_medv_low[col], data_medv_high[col], alternative='less').pvalue}")
    print()


# **I'm very picky and suggest that predictors _b, rad, dis, age and chas_ not significantly affects the cost of house**

# ### Testing _lstat_

# **Also, according to summary of last model, the most important parameter for higher cost is lower status of the population (percent)**

# In[35]:


data_medv_low.lstat.describe()


# In[36]:


data_medv_high.lstat.describe()


# In[37]:


b1 = sns.histplot(data_medv_low.lstat, kde=True,
                  alpha=0.6, binwidth=1);
b2 = sns.histplot(data_medv_high.lstat, kde=True,
                  alpha=0.6, binwidth=1);
plt.xlabel('Lower status of the population (percent)');
plt.legend(['Cheap', 'Expensive']);


# In[38]:


pval_mann_lstat = mannwhitneyu(data_medv_high.lstat, data_medv_low.lstat, alternative='less').pvalue
print('Expensive suburbs have greater value of lower status of the population (percent) then between cheap suburbs\n'
      f'with pvalue of Mann-Whitney U test: {pval_mann_lstat}')


# ### Testing _rm_

# In[39]:


data_medv_low.rm.describe()


# In[40]:


data_medv_high.rm.describe()


# In[41]:


b1 = sns.histplot(data_medv_low.rm, kde=True,
                  alpha=0.6, binwidth=0.5);
b2 = sns.histplot(data_medv_high.rm, kde=True,
                  alpha=0.6, binwidth=0.5);
plt.legend(['Cheap', 'Expensive']);
plt.xlabel('Average number of rooms per dwelling.');


# In[42]:


pval_mann_rm = mannwhitneyu(data_medv_low.dis, data_medv_high.rm, alternative='less').pvalue
print('Expensive suburbs have greater average number of rooms per dwelling than cheap ones', 
      f'with one sided Mann-Whitney U test p-value: {pval_mann_rm}')


# ### Testing _ptratio_

# In[43]:


data_medv_low.ptratio.describe()


# In[44]:


data_medv_high.ptratio.describe()


# In[45]:


sns.histplot(data_medv_low.ptratio, kde=True, alpha=0.6, binwidth=0.5);
sns.histplot(data_medv_high.ptratio, kde=True, alpha=0.6, binwidth=0.5);
plt.legend(['Cheap', 'Expensive']);
plt.xlabel('Pupil-teacher ratio by town');
# plt.ylim(0, 60);


# In[46]:


pval_mann_ptratio = mannwhitneyu(data_medv_low.ptratio, data_medv_high.ptratio, alternative='greater').pvalue
print('Expensive suburbs have less Pupil-teacher ratio by town than cheap ones', 
      f'with one sided Mann-Whitney U test p-value: {pval_mann_ptratio}')


# **After testing three most significant predictors I can say that higher cost of suburbs significantly depends on** 
# 1) lower status of population
# 2) number of rooms
# 3) pupil-teacher ratio  
# 
# The first parameter is interesting because the population with "low status" simply cannot buy expensive houses. So the correlation is high.   
# The second parameter is clear - big house = big price.  
# The third is the dependence on education. The small number of students in one class is characteristic of private schools, which usually educate the children of wealthy parents.

# ### Testing _nox_

# In[47]:


data_medv_low.nox.describe()


# In[48]:


data_medv_high.nox.describe()


# In[49]:


b1 = sns.histplot(data_medv_low.nox, kde=True,
                  alpha=0.6, binwidth=0.01);
b2 = sns.histplot(data_medv_high.nox, kde=True,
                  alpha=0.6, binwidth=0.01);
plt.legend(['Cheap', 'Expensive']);
plt.xlabel('Nitrogen oxides concentration (parts per 10 million).');
# plt.ylim(0,15);


# In[50]:


pval_mann_nox = mannwhitneyu(data_medv_low.nox, data_medv_high.nox, alternative='greater').pvalue
print('Expensive suburbs have less Nitrogen oxides concentration (parts per 10 million) than cheap ones', 
      f'with one sided Mann-Whitney U test p-value: {pval_mann_nox}')


# **The same shape of distribution, but different mean value**  
# Looks like it is two groups among all suburbs: with high nitrogen oxides concentration and lowe (two maximum peaks on the distribution)

# ### Testing _crim_

# In[51]:


data_medv_low.crim.describe()


# In[52]:


data_medv_high.crim.describe()


# In[53]:


b1 = sns.histplot(data_medv_low.crim, kde=True,
                  alpha=0.6, binwidth=3);
b2 = sns.histplot(data_medv_high.crim, kde=True,
                  alpha=0.6, binwidth=3);
plt.legend(['Cheap', 'Expensive']);
plt.xlabel('Per capita crime rate by town.');
plt.ylim(0, 150)


# In[54]:


pval_mann_crim = mannwhitneyu(data_medv_low.crim, data_medv_high.crim, alternative='greater').pvalue
print('Expensive suburbs have less per capita crime rate by town than cheap ones', 
      f'with one sided Mann-Whitney U test p-value: {pval_mann_crim}')


# **Cheap districs in some occausions have very high criminality rate (up to 89). Expensive ones have maximum 9 _crim_**

# ### Testing _zn_

# In[55]:


data_medv_low.zn.describe()


# In[56]:


data_medv_high.zn.describe()


# In[57]:


b1 = sns.histplot(data_medv_low.zn, kde=True,
                  alpha=0.6, binwidth=5);
b2 = sns.histplot(data_medv_high.zn, kde=True,
                  alpha=0.6, binwidth=5);
plt.legend(['Cheap', 'Expensive']);
plt.xlabel('Proportion of residential land zoned for lots over 25,000 sq.ft.');
plt.ylim(0, 60)


# In[58]:


pval_mann_zn = mannwhitneyu(data_medv_low.zn, data_medv_high.zn, alternative='less').pvalue
print('Expensive suburbs have greater proportion of residential land zoned for lots over 25,000 sq.ft. than cheap ones', 
      f'with one sided Mann-Whitney U test p-value: {pval_mann_zn}')


# #### Scatter

# In[59]:


sns.scatterplot(x=medv, y=scaled_preds.zn);


# **But there are a lot of zero values**  
# Yes expensive houses have less zero _zn_, but it looks like there are simply less expensive houses.  
# However correlation exists: a lot of such zones => less houses in suburbs and more lands per house => higher cost

# ### Testing indus

# In[60]:


data_medv_low.indus.describe()


# In[61]:


data_medv_high.indus.describe()


# In[62]:


b1 = sns.histplot(data_medv_low.indus, kde=True,
                  alpha=0.6, binwidth=1);
b2 = sns.histplot(data_medv_high.indus, kde=True,
                  alpha=0.6, binwidth=1);
plt.legend(['Cheap', 'Expensive']);
plt.xlabel('Proportion of non-retail business acres per town');
plt.ylim(0, 50)


# In[63]:


pval_mann_indus = mannwhitneyu(data_medv_low.zn, data_medv_high.zn, alternative='less').pvalue
print('Expensive suburbs have greater proportion of non-retail business acres per town than cheap ones', 
      f'with one sided Mann-Whitney U test p-value: {pval_mann_indus}')


# **The plot looks like expensive houses should have a lower _indus_ value, but the Mann-Whitney test shows us the opposite information. I think, that simply number of cheap houses is greater and it affected the results.**  
# Also, it looks like there two grups of suburbs: with high and low values of non-retail business acres.  
# But some correlation exists!

# ### Testing _tax_

# In[64]:


data_medv_low.tax.describe()


# In[65]:


data_medv_high.tax.describe()


# In[66]:


b1 = sns.histplot(data_medv_low.tax, kde=True,
                  alpha=0.6, binwidth=50);
b2 = sns.histplot(data_medv_high.tax, kde=True,
                  alpha=0.6, binwidth=50);
plt.legend(['Cheap', 'Expensive']);
plt.xlabel('full-value property-tax rate per \$10,000.');


# **The data don't look any different! + there are two peaks that indicate the existence of two groups depending on tax value** 

# ### CONCLUSION

# **In my opinion, the most important aspects to consider when choosing an area to build a house are**
# 
# - Lower status of the population
# But dependency there may be inverse relationship - houses in suburbs are expensive and people with lower status cannot buy such houses
# - Number of rooms  
# It is very significant parameter
# - Pupil per teacher  
# Most of rich people want their children to study in private schools with high level of personal education 
# 
# **_More comlicated parameters:_**
# - Criminality  
# Criminality in rich subrubs is less on average, but cause of this distribution is because there are several "deviations" in cheap subrubs where criminality rates are extrimely high
# - Nitrogen oxides concentrationLevel of nitrogen dioxide and non-retail business acres per town  
# I think this parameters depend on location of various factories and industrial compnies. Such districts are more ecologically friendly. There are no noises, smells and, in fact, people with lower status.
# - Proportion of residential land zoned for lots over 25,000 sq.ft  
# Expensive subrubs have more lands per house, but it looks like there is small correlation. May be there some cheap subrubs that have small population with empty zones wuthout houses.
# - Full-value property-tax rate per \$10,000.  
# This parameter measures cost of public services. But correlation is not very significant. May be it is bacuse some cheap subrubs have big taxes and people live in cheap houses.

# ## Best House ever:

# 1) With big nuber of rooms
# 2) With good educational insitutions nearby
# 3) No industrial complexes nearby
# 4) Big teritory 
# 5) Good public services in town
# 6) No criminality in suburb
