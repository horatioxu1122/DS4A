#!/usr/bin/env python
# coding: utf-8

# In[75]:


import math

import matplotlib.pyplot as plt
import numpy as np

# Importing libraries
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import nct

# In[14]:


# Documentation
# https://www.fia.fs.fed.us/library/database-documentation/current/ver80/FIADB%20User%20Guide%20P2_8-0.pdf#page=166
# The natural_reserve_a.csv file is the CA_TREE.csv file from https://apps.fs.usda.gov/fia/datamart/CSV/datamart_csv.html
# But only has the CN and the HT columns
reserve_a = pd.read_csv("data/natural_reserve_a.csv")
reserve_a.head()


# ### Population distribution

# In[15]:


sns.displot(x=reserve_a["HT"], kde=True, height=5, aspect=3, color="green")
plt.title("Distribution of tree height (in feet)")
plt.axvline(reserve_a["HT"].mean(), color="#8a450c")
plt.annotate(
    "μ", (reserve_a["HT"].mean() + 2, 10000), color="#8a450c", fontsize="x-large"
)


# ### Population mean distribution

# In[16]:


sns.displot(x=[reserve_a["HT"].mean()], height=5, aspect=3, bins=500, color="green")
plt.title("Distribution of the population mean μ")


# ### Comparing p-values
#
# #### Sample A and sample B

# In[17]:


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10.5, 6.5)
df, nc = 199, 0
mu = reserve_a["HT"].mean()
S = reserve_a["HT"].std()
n = 200
se = S / math.sqrt(n)
x = np.linspace(
    nct.ppf(0.0000001, df, nc, loc=mu, scale=se),
    nct.ppf(0.9999999, df, nc, loc=mu, scale=se),
    n,
)
ax.plot(x, nct.pdf(x, df, nc, loc=mu, scale=se), label="nct pdf", color="blue")

# Plot the values
plt.axvline(mu, color="#8a450c")
plt.annotate("μ", (mu + 0.2, 0.04), color="#8a450c", fontsize="x-large")
plt.axvline(60, color="purple")
plt.annotate("A", (60.2, 0.04), color="purple", fontsize="x-large")
plt.axvline(62, color="grey")
plt.annotate("B", (62.2, 0.04), color="grey", fontsize="x-large")

plt.title("Sampling distribution of the sample mean")


# #### Sample B and sample C

# In[18]:


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10.5, 6.5)
df, nc = 199, 0
mu = reserve_a["HT"].mean()
S = reserve_a["HT"].std()
n = 200
se = S / math.sqrt(n)
x = np.linspace(
    nct.ppf(0.0000001, df, nc, loc=mu, scale=se),
    nct.ppf(0.9999999, df, nc, loc=mu, scale=se),
    n,
)
ax.plot(x, nct.pdf(x, df, nc, loc=mu, scale=se), label="nct pdf", color="blue")

# Plot the values
plt.axvline(mu, color="#8a450c")
plt.annotate("μ", (mu + 0.2, 0.04), color="#8a450c", fontsize="x-large")
plt.axvline(62, color="grey")
plt.annotate("B", (62.2, 0.04), color="grey", fontsize="x-large")
plt.axvline(69, color="#eb3471")
plt.annotate("C", (69.2, 0.04), color="#eb3471", fontsize="x-large")

plt.title("Sampling distribution of the sample mean")


# #### $\alpha$ and $p$-values
#
#

# In[79]:


# Rejecting the null hypothesis
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10.5, 6.5)
df, nc = 199, 0
mu = reserve_a["HT"].mean()
S = reserve_a["HT"].std()
n = 200
se = S / math.sqrt(n)
x = np.linspace(
    nct.ppf(0.0000001, df, nc, loc=mu, scale=se),
    nct.ppf(0.9999999, df, nc, loc=mu, scale=se),
    n,
)
ax.plot(x, nct.pdf(x, df, nc, loc=mu, scale=se), label="nct pdf", color="blue")
ax.set_title(
    "Rejecting the null hypothesis\nThe value is beyond the critical value, so it is considered extreme enough"
)

x_alpha = np.delete(x, np.where(x < 60))
ax.fill_between(
    x_alpha, nct.pdf(x_alpha, df, nc, loc=mu, scale=se), color=[(1, 0.27, 0.5, 0.5)]
)

x_pval = np.delete(x, np.where(x < 62))
ax.fill_between(
    x_pval, nct.pdf(x_pval, df, nc, loc=mu, scale=se), color=[(1, 1, 0, 0.5)]
)
ax.annotate(text="α", xy=(60.6, 0.04), size=20)
ax.annotate(text="p", xy=(62.6, 0.008), size=20)
ax.annotate(text="p < α", xy=(66, 0.08), size=20)


# In[74]:


# Failing to reject the null hypothesis
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10.5, 6.5)
df, nc = 199, 0
mu = reserve_a["HT"].mean()
S = reserve_a["HT"].std()
n = 200
se = S / math.sqrt(n)
x = np.linspace(
    nct.ppf(0.0000001, df, nc, loc=mu, scale=se),
    nct.ppf(0.9999999, df, nc, loc=mu, scale=se),
    n,
)
ax.plot(x, nct.pdf(x, df, nc, loc=mu, scale=se), label="nct pdf", color="blue")
ax.set_title("Failing to reject the null hypothesis\nThe value is not extreme enough")

x_pval = np.delete(x, np.where(x < 60))
ax.fill_between(
    x_pval, nct.pdf(x_pval, df, nc, loc=mu, scale=se), color=[(1, 1, 0, 0.5)]
)

x_alpha = np.delete(x, np.where(x < 62))
ax.fill_between(
    x_alpha, nct.pdf(x_alpha, df, nc, loc=mu, scale=se), color=[(1, 0.27, 0.5, 0.5)]
)


ax.annotate(text="α", xy=(62.6, 0.008), size=20)
ax.annotate(text="p", xy=(60.6, 0.04), size=20)
ax.annotate(text="p > α", xy=(66, 0.08), size=20)


# #### $p$-values to the left

# In[87]:


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10.5, 6.5)
df, nc = 199, 0
mu = reserve_a["HT"].mean()
S = reserve_a["HT"].std()
n = 200
se = S / math.sqrt(n)
x = np.linspace(
    nct.ppf(0.0000001, df, nc, loc=mu, scale=se),
    nct.ppf(0.9999999, df, nc, loc=mu, scale=se),
    n,
)
ax.plot(x, nct.pdf(x, df, nc, loc=mu, scale=se), label="nct pdf", color="blue")
ax.set_title(
    "A left-tailed test\nThe more to the left the value, the more extreme it is"
)

x_pval = np.delete(x, np.where(x > 54))
ax.fill_between(
    x_pval, nct.pdf(x_pval, df, nc, loc=mu, scale=se), color=[(1, 1, 0, 0.5)]
)

ax.annotate(text="p", xy=(52.9, 0.02), size=20)


# #### Two-tailed test

# In[118]:


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10.5, 6.5)
df, nc = 199, 0
mu = reserve_a["HT"].mean()
S = reserve_a["HT"].std()
n = 200
se = S / math.sqrt(n)
x = np.linspace(
    nct.ppf(0.0000001, df, nc, loc=mu, scale=se),
    nct.ppf(0.9999999, df, nc, loc=mu, scale=se),
    n,
)
ax[0].plot(x, nct.pdf(x, df, nc, loc=mu, scale=se), label="nct pdf", color="blue")
ax[0].set_title("A two-tailed test\nThe furthest from the mean, the more extreme")

x_pval_left = np.delete(x, np.where(x > (mu - 3.5)))
ax[0].fill_between(
    x_pval_left, nct.pdf(x_pval_left, df, nc, loc=mu, scale=se), color=[(1, 1, 0, 0.5)]
)

x_pval_right = np.delete(x, np.where(x < (mu + 3.5)))
ax[0].fill_between(
    x_pval_right,
    nct.pdf(x_pval_right, df, nc, loc=mu, scale=se),
    color=[(1, 1, 0, 0.5)],
)

ax[0].annotate(text="p/2", xy=(52.1, 0.01), size=20)
ax[0].annotate(text="p/2", xy=(60.9, 0.01), size=20)

# Second plot
ax[1].plot(x, nct.pdf(x, df, nc, loc=mu, scale=se), label="nct pdf", color="blue")
ax[1].set_title("A two-tailed test\nThe furthest from the mean, the more extreme")

x_alpha_left = np.delete(x, np.where(x > (mu - 3.0)))
ax[1].fill_between(
    x_alpha_left,
    nct.pdf(x_alpha_left, df, nc, loc=mu, scale=se),
    color=[(1, 0.27, 0.5, 0.5)],
)

x_alpha_right = np.delete(x, np.where(x < (mu + 3.0)))
ax[1].fill_between(
    x_alpha_right,
    nct.pdf(x_alpha_right, df, nc, loc=mu, scale=se),
    color=[(1, 0.27, 0.5, 0.5)],
)

ax[1].annotate(text="α/2", xy=(52.1, 0.01), size=20)
ax[1].annotate(text="α/2", xy=(60.9, 0.01), size=20)

plt.plot()


# ## Data analysis

# In[2]:


import pandas as pd

# In[42]:


# The CA_TREE.csv file can be retrieved from https://apps.fs.usda.gov/fia/datamart/CSV/datamart_csv.html

california = pd.read_csv(
    "data/CA_TREE.csv", usecols=["CN", "INVYR", "HT", "SPCD", "SITREE"]
)
california = pd.merge(
    california, pd.read_csv("data/species.csv"), on="SPCD", how="left"
)
california = california[california["INVYR"] == 2010]
california = california.dropna(subset=["SITREE"])
california = california.drop(columns=["INVYR", "SPCD"])
california.head()


# In[60]:


species_to_test = california["scientific_name"].value_counts() > 150
species_to_test = species_to_test[species_to_test == True].index
species_to_test


# In[79]:


forest = california[california["scientific_name"].isin(species_to_test)]
forest = forest.groupby("scientific_name").sample(100, random_state=1991, replace=False)
forest


# In[80]:


forest.to_csv("data/forest_survey.csv")


# In[65]:


forest.sample(5).to_html().replace("\n", "")


# In[70]:


california_means = (
    california[california["scientific_name"].isin(species_to_test)]
    .groupby("scientific_name")["SITREE"]
    .mean()
)
california_means.to_frame().to_csv("data/california_means.csv")


# In[72]:


california_means


# In[73]:


california_means.loc["Sequoia sempervirens"]


# In[94]:


results = []
for scientific_name in species_to_test:
    mu_california = california_means.loc[scientific_name]
    filtered = forest[forest["scientific_name"] == scientific_name]
    # We subtract 3 feet from the samples to create variability and make some differences significant
    p = stats.ttest_1samp(filtered["SITREE"] - 3, mu_california, nan_policy="omit")[1]
    results.append([scientific_name, p])

results = pd.DataFrame(results)
results.columns = ["scientific_name", "p-value"]
results = results.set_index("scientific_name")
results = results.join(
    forest[["scientific_name", "common_name"]]
    .drop_duplicates()
    .set_index("scientific_name")
)
results = results[["common_name", "p-value"]]
results


# In[95]:


results.to_html().replace("\n", "")


# In[100]:


results[results["p-value"] <= 0.05].to_html().replace("\n", "")


# In[102]:


forest.groupby("scientific_name")["SITREE"].mean()


# In[114]:


comparison_means = pd.merge(
    california_means.to_frame(),
    forest.groupby("scientific_name")["SITREE"].mean().to_frame(),
    right_index=True,
    left_index=True,
)

comparison_means.columns = ["sample mean (forest)", "population mean (California)"]
comparison_means["sample mean (forest)"] = round(
    comparison_means["sample mean (forest)"], 2
)
comparison_means = comparison_means[
    comparison_means.index.isin(results[results["p-value"] <= 0.05].index)
]
comparison_means.to_html().replace("\n", "")


# In[115]:


# Using Bonferroni correction
results[results["p-value"] <= (0.05 / 18)].to_html().replace("\n", "")
