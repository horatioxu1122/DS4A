#!/usr/bin/env python
# coding: utf-8

# # Auxiliary code script

# In[21]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
from scipy import stats

# In[22]:


### Load relevant packages
# import pandas                  as pd

# import numpy                   as np
# import matplotlib.pyplot       as plt
# import seaborn                 as sns
# import statsmodels.formula.api as sm
# import chart_studio.plotly     as py
# https://community.plot.ly/t/solved-update-to-plotly-4-0-0-broke-application/26526/2
# import os

#%matplotlib inline
# plt.style.use('ggplot')


# In[23]:


Data = pd.read_csv("data/company_dataset.csv")


# In[24]:


sns.boxplot(x="male_female", y="pay_yearly", data=Data)
# Data.boxplot(grid= False, column = ['pay'], by = ['gender'])
plt.title("Pay vs Gender")


# In[25]:


## A simple t test for difference of means
t2, p2 = stats.ttest_ind(
    Data.loc[Data["male_female"] == "M", "pay_yearly"],
    Data.loc[Data["male_female"] == "F", "pay_yearly"],
)
print("t = " + str(t2))
print("p = " + str(p2))


# In[26]:


# Scatterplot of pay vs age
plt.scatter(Data["age_years"], Data["pay_yearly"], color="orange")
plt.title("Pay vs. Age", fontsize=20, verticalalignment="bottom")
plt.xlabel("Age")
plt.ylabel("Pay")


# In[27]:


Data.groupby("male_female")["pay_yearly"].describe().round(2).T.to_html().replace(
    "\n", ""
)


# In[28]:


# Guess the correlation

plt.figure(figsize=(12, 10))
rho = [0.999, -0.999, 0.5, -0.7, 0.001, -0.3]
cor_list = []
np.random.seed(10)
for i, r in enumerate(rho):
    plt.subplot(2, 3, i + 1)
    mean, cov = [4, 6], [(1, r), (r, 1)]
    x, y = np.random.multivariate_normal(mean, cov, 150).T
    ax = sns.scatterplot(x=x, y=y, color="g")
    cor_list.append(np.corrcoef(x, y)[0, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Guess the Correlation")


# In[29]:


# Guess the correlation (answers)

plt.figure(figsize=(12, 10))
rho = [0.999, -0.999, 0.5, -0.7, 0.001, -0.3]
cor_list = []
np.random.seed(10)
for i, r in enumerate(rho):
    plt.subplot(2, 3, i + 1)
    mean, cov = [4, 6], [(1, r), (r, 1)]
    x, y = np.random.multivariate_normal(mean, cov, 150).T
    ax = sns.scatterplot(x=x, y=y, color="g")
    cor_list.append(np.corrcoef(x, y)[0, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Correlation: ~" + str(r))


# In[30]:


# Correlation matrix
corr_mat = Data[
    ["pay_yearly", "age_years", "seniority_years", "performance_score"]
].corr()
corr_mat
sns.heatmap(corr_mat, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True)
plt.title("Correlation Matrix")


# In[31]:


# Line of best fit
sns.lmplot(
    x="age_years",
    y="pay_yearly",
    ci=None,
    data=Data,
    line_kws={"color": "blue"},
    scatter_kws={"color": "orange"},
)
plt.title("Pay vs. Age", fontsize=20, verticalalignment="bottom")
plt.xlabel("Age")
plt.ylabel("Pay")


# In[32]:


model1 = "pay_yearly~age_years"
lm1 = sm.ols(formula=model1, data=Data).fit()
print(lm1.summary())


# In[33]:


# R squared is Pearson's r squared
corr_age_pay = np.corrcoef(Data["pay_yearly"], Data["age_years"])[0, 1]
r2 = corr_age_pay ** 2
print("Rho is " + str(round(corr_age_pay, 4)) + " and R2 is " + str(round(r2, 4)))


# In[35]:


# Model with age and gender
model2 = "pay_yearly~age_years + male_female"
lm2 = sm.ols(formula=model2, data=Data).fit()
print(lm2.summary())


# In[36]:


# Boxplot pay vs. education
sns.boxplot(x="education", y="pay_yearly", data=Data)
plt.title("Pay vs. Education", fontsize=20, verticalalignment="bottom")


# In[37]:


# Model 3
model3 = "pay_yearly~age_years + male_female + education"
lm3 = sm.ols(formula=model3, data=Data).fit()
print(lm3.summary())


# In[39]:


# Last model
model4 = "pay_yearly~job_title + age_years + performance_score + education + seniority_years + male_female"
lm4 = sm.ols(formula=model4, data=Data).fit()
print(lm4.summary())


# In[44]:


pd.crosstab(Data["male_female"], Data["job_title"])


# In[73]:


ax = (
    pd.crosstab(Data["male_female"], Data["seniority_years"], normalize="index")
    .loc["M"]
    .plot(kind="bar", color="grey", legend=False)
)
ax.set_title("Distribution of gender vs. seniority (MALE)")
ax.set_xlabel("Seniority")
ax.set_ylabel("% of people")

plt.show()

ax = (
    pd.crosstab(Data["male_female"], Data["seniority_years"], normalize="index")
    .loc["F"]
    .plot(kind="bar", color="grey", legend=False)
)
ax.set_title("Distribution of gender vs. seniority (FEMALE)")
ax.set_xlabel("Seniority")
ax.set_ylabel("% of people")
plt.show()


# In[45]:


sns.countplot(x="seniority_years", hue="male_female", data=Data)
plt.title(
    "Seniority distribution for men vs. women", fontsize=20, verticalalignment="bottom"
)


# In[74]:


pl = sns.countplot(x="job_title", hue="male_female", data=Data)
pl.set_xticklabels(pl.get_xticklabels(), rotation=40, ha="right")
plt.title(
    " Distribution of men vs. women across various roles",
    fontsize=20,
    verticalalignment="bottom",
)


# In[75]:


Data.groupby("male_female").count()


# In[76]:


Data.count()
