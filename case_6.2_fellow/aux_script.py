#!/usr/bin/env python
# coding: utf-8

# # How should we price homes in Seattle?


import matplotlib.pyplot as plt

# Load relevant packages
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.graphics.gofplots import qqplot as qqplot

sns.set_style("white")
plt.style.use("ggplot")


# In[2]:


houses = pd.read_csv("kc_house_data.csv")


# ## Exercise 1

# In[4]:


houses["price"].describe().to_frame().to_html().replace("\n", "")


# In[5]:


## histogram plot of price
# sns.distplot(houses['price'],fit=stats.laplace, kde=False)
sns.distplot(houses["price"], fit=stats.norm, kde=False)
sns.set(color_codes=True)
plt.xticks(rotation=90)
plt.title("Histogram of prices")


# In[6]:


## Explanation of QQ plots - Original distribution
percentile_ticks = []
for i in range(1, 101, 1):
    percentile_ticks.append(i / 100)
percentile_ticks

np.random.seed(1880)
theor_data = np.random.normal(loc=145, scale=19, size=10000)
theor_data = pd.Series(theor_data)
percentiles_theor_data = theor_data.quantile(percentile_ticks)
plt.figure(figsize=(15, 4), dpi=100)
theor_data.plot.kde()
plt.vlines(percentiles_theor_data, ymin=0, ymax=0.01, color="orange")
plt.title("A normal distribution and its percentiles")
plt.plot()


# In[7]:


## Explanation of QQ plots - A sample from the original distribution
percentile_ticks = []
for i in range(1, 101, 1):
    percentile_ticks.append(i / 100)
percentile_ticks

np.random.seed(1880)
sample_data = np.random.normal(loc=145, scale=19, size=100)
sample_data = pd.Series(sample_data)
percentiles_sample_data = sample_data.quantile(percentile_ticks)
plt.figure(figsize=(15, 4), dpi=100)
sample_data.plot.kde(linewidth=0)  # This one doesn't plot the curve
plt.vlines(percentiles_sample_data, ymin=0, ymax=0.01, color="orange")
plt.title("A sample from the same distribution and its percentiles")
plt.plot()


# In[8]:


## Explanation of QQ plots - QQ plot of original vs. sample
plt.figure(figsize=(4, 4), dpi=90)
plt.scatter(percentiles_theor_data, percentiles_sample_data, s=3)
plt.xlabel("Percentiles of the original distribution")
plt.ylabel("Percentiles of the 100-observations sample")
plt.title(
    "Quantile vs. quantile scatterplot\n for the original distribution and the sample"
)


# In[9]:


## Explanation of QQ plots - A sample from ANOTHER NORMAL distribution
percentile_ticks = []
for i in range(1, 101, 1):
    percentile_ticks.append(i / 100)
percentile_ticks

np.random.seed(179)
sample_data_other = np.random.normal(loc=1485, scale=190, size=100)
sample_data_other = pd.Series(sample_data_other)
percentiles_sample_data_other = sample_data_other.quantile(percentile_ticks)
plt.figure(figsize=(15, 4), dpi=100)
sample_data_other.plot.kde()
plt.vlines(percentiles_sample_data_other, ymin=0, ymax=0.001, color="orange")
plt.title("A sample from another normal distribution and its percentiles")
plt.plot()


# In[10]:


## Explanation of QQ plots - QQ plot of original vs. sample from another normal distribution
plt.figure(figsize=(4, 4), dpi=90)
plt.scatter(percentiles_theor_data, percentiles_sample_data_other, s=3)
plt.xlabel("Percentiles of the original distribution")
plt.ylabel("Percentiles of the 100-observations sample")
plt.title(
    "Quantile vs. quantile scatterplot\n for the original distribution and a sample from\n another normal distribution"
)


# In[11]:


## QQ plot of price
stats.probplot(x=houses["price"], dist="norm", plot=plt)
plt.title("QQ Plot for Prices")
plt.show()


# ## Exercise 2

# In[12]:


## linear relation between sqft_living and price
sns.lmplot(x="sqft_living", y="price", data=houses, line_kws={"color": "red"}, aspect=2)
plt.title("Price vs. Sqft_living")


# ## Exercise 3

# In[13]:


## QQ plot of price
plt.subplot(2, 1, 1)
stats.probplot(np.log(houses["price"]), dist="norm", plot=plt)
plt.title("QQ Plot for Log Prices")
plt.show()
plt.subplot(2, 1, 2)
sns.distplot(np.log(houses["price"]), fit=stats.norm, kde=False)
sns.set(color_codes=True)
plt.xticks(rotation=90)
plt.title("Histogram of Log prices")


# In[14]:


np.log(houses["price"]).describe().to_frame().to_html().replace("\n", "")


# ### Building a linear model with transformed variables

# In[15]:


mod1 = smf.ols(formula="np.log(price) ~ np.log(sqft_living)", data=houses).fit()
print(mod1.summary())


# In[16]:


## linear relation between sqft_living and price
houses2 = houses.copy()
houses2["sqft_living_log"] = np.log(houses["sqft_living"])
houses2["price_log"] = np.log(houses["price"])
sns.lmplot(
    x="sqft_living_log",
    y="price_log",
    data=houses2,
    line_kws={"color": "red"},
    aspect=2,
)
plt.title("Price vs. Sqft_living (log-log)")


# In[17]:


mod2 = smf.ols(formula="np.log(price) ~ sqft_living", data=houses).fit()
print(mod2.summary())


# In[18]:


## linear relation between sqft_living and price
sns.lmplot(
    x="sqft_living", y="price_log", data=houses2, line_kws={"color": "red"}, aspect=2
)
plt.title("Log of Price vs. Sqft_living")


# ## Exercise 4

# In[19]:


houses["log_price"] = np.log(houses["price"])
houses["log_sqft_living"] = np.log(houses["sqft_living"])

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

sns.regplot(
    x="log_sqft_living",
    y="log_price",
    data=houses,
    line_kws={"color": "red"},
    scatter_kws={"s": 5},
    ax=axes[0],
)
axes[0].set_title("Log-Price vs. Log_Sqft_living")

sns.regplot(
    x="sqft_living",
    y="log_price",
    data=houses,
    line_kws={"color": "red"},
    scatter_kws={"s": 5},
    ax=axes[1],
)
axes[1].set_title("Log-Price vs. Sqft_living")


# ### Box-Cox transformation

# In[20]:


price, fitted_lambda = stats.boxcox(houses["price"])
round(fitted_lambda, 2)


# ## Exercise 5

# In[21]:


mod3 = smf.ols(
    formula="np.log(price) ~ np.log(sqft_living)+ np.log(sqft_lot) +bedrooms + floors + bathrooms +C(waterfront) + condition  + C(view) + grade + yr_built + lat + long ",
    data=houses,
).fit()
print(mod3.summary())


# ## Exercise 6

# In[22]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.scatter(
    houses["lat"], np.log(houses["price"]), c="b", alpha=0.6, edgecolors="none", s=10
)
plt.xlabel("Latitude")
plt.ylabel("Log Price")
ax.set_title("Log Price against latitude")
plt.show()


# ## Exercise 7

# In[23]:


## lat square effect
mod4 = smf.ols(
    formula="np.log(price) ~ np.log(sqft_living)+ np.log(sqft_lot) +bedrooms + floors + bathrooms+ condition + C(waterfront) + C(view) + grade + yr_built + lat + I(lat**2) + long ",
    data=houses,
).fit()
print(mod4.summary())


# In[24]:


print(mod3.aic)


# In[25]:


print(mod4.aic)


# ## Exercise 8

# In[26]:


sns.lmplot(
    x="lat",
    y="log_price",
    data=houses,
    hue="waterfront",
    height=8,
    scatter_kws={"s": 1.5},
)
plt.title("Log-price vs. Latitude")


# In[27]:


lm = sns.lmplot(
    x="lat",
    y="log_price",
    data=houses,
    hue="view",
    height=8,
    scatter_kws={"s": 2},
    sharex=False,
    sharey=False,
    ci=None,
)
axes = lm.axes
axes[0, 0].set_ylim(11, 17)
axes[0, 0].set_xlim(47.1, 47.8)


# Modeling interaction effects

# In[28]:


formula = "np.log(price) ~ lat*C(waterfront)"
mod5_1 = smf.ols(formula=formula, data=houses).fit()
print(mod5_1.summary())


# ## Exercise 9

# In[29]:


formula = "np.log(price) ~ lat*C(view)"
mod5_2 = smf.ols(formula=formula, data=houses).fit()
print(mod5_2.summary())


# ### Incorporating interaction effects into a linear model

# In[30]:


houses["renovated"] = houses["yr_renovated"] > 0

formula = (
    "np.log(price) ~ np.log(sqft_living)+ np.log(sqft_lot) + bedrooms + floors + bathrooms "
    "+ C(waterfront) + condition  + C(view) + grade + yr_built + lat + I(lat**2) "
    "+ long + C(zipcode)+ C(renovated)"
)
mod6 = smf.ols(formula=formula, data=houses).fit()
print(mod6.summary())


# In[31]:


## effect of a waterfront view different for houses that were recently renovated
formula = (
    "np.log(price) ~ np.log(sqft_living)+ np.log(sqft_lot) + bedrooms + floors + bathrooms "
    "+ C(waterfront) + condition  + C(view) + grade + yr_built + lat + I(lat**2) "
    "+ long + C(zipcode)+ C(renovated) + C(waterfront)*C(renovated)"
)
mod6a = smf.ols(formula=formula, data=houses).fit()
print(mod6a.summary())


# In[32]:


formula = (
    "np.log(price) ~ np.log(sqft_living)*C(renovated) + np.log(sqft_lot) + bedrooms + floors + bathrooms "
    " + condition + C(view) + grade + yr_built + lat*C(waterfront) + I(lat**2) "
    "+ long + C(zipcode)"
)
mod7 = smf.ols(formula=formula, data=houses).fit()
print(mod7.summary())


# In[33]:


print("The AIC of mod6 is ", mod6.aic)
print("The AIC of mod7 is ", mod7.aic)


# In[34]:


formula = (
    "np.log(price) ~ np.log(sqft_living)*C(waterfront) + np.log(sqft_living)*C(renovated) + np.log(sqft_lot)"
    "+ bedrooms + floors + bathrooms "
    "+ C(waterfront) + condition + C(view) + grade + yr_built + lat + I(lat**2) + long + C(zipcode)"
)
mod8 = smf.ols(formula=formula, data=houses).fit()
print(mod8.summary())


# In[35]:


print("The AIC of mod7 is ", mod7.aic)
print("The AIC of mod8 is ", mod8.aic)


# In[36]:


mod7.rsquared


# ### Exercise 10

# In[38]:


formula = (
    "np.log(price) ~ np.log(sqft_living)*C(renovated) + np.log(sqft_lot) + bedrooms + floors + bathrooms "
    " + condition + C(view) + grade + yr_built + lat*C(waterfront) + I(lat**2) "
    "+ long + C(zipcode) + I(yr_built**2)"
)

mod9 = smf.ols(formula=formula, data=houses).fit()
print(mod9.summary())


# In[39]:


print(mod9.aic)


# In[41]:


# built dummy variable to separate houses with a basement and houses with no basement
houses["has_basement"] = (houses["sqft_basement"] > 0) * 1.0


# In[42]:


# estimate a model with an interaction between the longitude coordinate
# and the presence of a basement.
formula = (
    "np.log(price) ~ np.log(sqft_living)*C(renovated) + np.log(sqft_lot) + bedrooms + floors + bathrooms "
    " + condition + C(view) + grade + yr_built + lat*C(waterfront) + I(lat**2) "
    "+ long + C(zipcode) + has_basement * long"
)
mod10 = smf.ols(formula=formula, data=houses).fit()
print(mod10.summary())


# In[43]:


# the r-squared results
r7 = mod7.rsquared
r9 = mod9.rsquared
r10 = mod10.rsquared

# the aic results
aic7 = mod7.aic
aic9 = mod9.aic
aic10 = mod10.aic

print("-------------- R Squared results --------------")
print("Model 7 -", r7)
print("Model 9 -", r9)
print("Model 10 -", r10)
print("\n----------------- AIC results -----------------")
print("AIC 7 -", aic7)
print("AIC 9 -", aic9)
print("AIC 10 -", aic10)


# In[38]:


formula = (
    "np.log(price) ~ np.log(sqft_living)+ np.log(sqft_lot) + bedrooms + floors + bathrooms "
    "+ C(waterfront) + condition  + C(view) + grade + yr_built +  + lat + I(lat**2) "
    "+ long + C(zipcode)+ C(renovated) + I(yr_built**2)"
)
mod6b = smf.ols(formula=formula, data=houses).fit()
print(mod6b.summary())


# In[40]:


# built dummy variable to separate houses with a basement and houses with no basement
houses["has_basement"] = (houses["sqft_basement"] > 0) * 1.0

formula = (
    "np.log(price) ~ np.log(sqft_living)+ np.log(sqft_lot) + bedrooms + floors + bathrooms "
    "+ C(waterfront) + condition  + C(view) + grade + yr_built +  + lat + I(lat**2) "
    "+ long + C(zipcode)+ C(renovated) + C(has_basement)*long"
)
mod6c = smf.ols(formula=formula, data=houses).fit()
print(mod6c.summary())
