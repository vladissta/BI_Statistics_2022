#!/usr/bin/env python
# coding: utf-8

# # PROJECT 1
# Stanin Vladislav

# ### Importing requirements

# In[1]:


import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# ### Function that rquires path, file extension and separator to parse all files in path and concate them all in one dataframe
# For example, situation can happen, when files are with .txt file extension and separator in them is ','

# In[2]:


def concat_files(path, separator, file_extension):
    all_files = glob.glob(f'{path}/*.{file_extension}')
    df = pd.concat([pd.read_table(file, sep=separator) for file in all_files])
    return df


# #### Concating data about athletes

# In[3]:


df = concat_files('../athlete_events', ',', 'csv')


# In[4]:


df.head(3)


# ## Checking the Data

# #### Checking types of variables

# In[5]:


df.info(show_counts=True)


# #### Checking common statistics of variables

# In[6]:


df.describe(include = 'all')


# In[7]:


for i in ['Sex', 'Team', 'Games', 'Season', 'City', 'Sport', 'Medal']:
    print(df[i].unique())


# _There are some deviations that are already visible_

# #### Checking NULLs

# In[8]:


df.isna().sum()


# _Obviously some sportsmen did not have medals. Also we consider that some athletes did not measured their Height and Weight and did not wanted to say their age (maybe)_

# #### Deleting sportsmen without name and sport, beacuse sportsmen could not be without sport 

# In[9]:


df = df.drop(df.loc[df.Name.isnull()].index[0], axis=0)
df = df.drop(df.loc[df['Sport'].isnull()].index)


# In[10]:


df.isna().sum()


# #### Checking Event and Season and repair missed data 

# In[11]:


df.loc[df['Event'].isnull()]


# In[12]:


df.loc[df['Event'].isnull(), "Event"] = 'Football Men\'s Football'
df.loc[df['Event'].isnull(), "Sport"] = 'Football'


# In[13]:


df.loc[df.Games == '2004 Summe', "Event"] = '2004 Summer'


# In[14]:


df.loc[df.Games == '2000 Su', "Event"] = '2000 Summer'


# #### Checking Sex

# In[15]:


df['Sex'].value_counts().sort_index()


# In[16]:


df.loc[df.Sex == 'G']


# _2 athletes turned out to be with gender "G" - with all my respect for minorities, I can't leave it this way, because in the years in which these athletes performed, gender-non-decided people did not perform (especially from Russia and Czechoslovakia). So i changed the sex according to hi sports and names_

# In[17]:


df.loc[df['Sex'] == 'G', 'Sex'] = 'M'


# #### Checking Age

# In[18]:


df['Age'].value_counts().sort_index()


# In[19]:


df.loc[df.Age == 240]


# _Too old... We have to change this data to NULL_

# In[20]:


df.loc[df.Age == 240, 'Age'] = None


# #### Checking Height

# In[21]:


df['Height'].value_counts().sort_index()


# In[22]:


df.loc[df.Height == 340]


# _It is to high!_

# In[23]:


df.loc[df.Height == 340, 'Height'] = None


# #### Checking Weight

# In[24]:


df['Weight'].value_counts().sort_index()


# _It is already okay_

# ## EDA

# ### The age of the youngest athletes of both sexes at the 1992 Olympics.

# In[25]:


df.loc[df.Year == 1992].groupby('Sex').Age.min().to_frame()


# ### The average value and standard deviation of the Height variable for athletes of each sex.

# In[26]:


df.groupby('Sex').agg(Mean_value = ('Height', 'mean'),
                      Standard_deviation = ('Height', 'std'))


# ### The average value and standard deviation of the Height variable for female tennis players at the 2000 Olympics.

# In[27]:


df.loc[(df.Sex == 'F') & 
       (df.Sport == 'Tennis') & 
       (df.Year == 2000)].agg(Mean_value = ('Height', 'mean'), 
                              Standard_deviation = ('Height', 'std')).round(1)


# ### Heaviest athlete's sport in at the 2006 Olympics

# In[28]:


df.loc[
    (df.Weight == df.loc[df.Year == 2006].Weight.max()) &
    (df.Year == 2006)].Sport


# ### Number of gold medals which were received by women from 1980 to 2010

# In[29]:


df.loc[(df.Sex == "F") & (df.Medal == "Gold") & (df.Year.isin(range(1980,2011)))].Medal.count()


# ### Number of times has John Aalberg participated in the Olympic Games in different years

# In[30]:


diff_years = df.loc[df.Name == "John Aalberg"].Year.unique().size
all_times = df.loc[df.Name == "John Aalberg"].shape[0]
num_of_games = df.loc[df.Name == "John Aalberg"].Games.unique().size
print(f'Different years he participated: {diff_years}. Participated at different competitions: {all_times} times.')


# ### The least and most represented (by number of participants) age groups of athletes at the 2008 Olympics. 
# ##### Age groups: [15-25), [25-35), [35-45), [45-55].

# In[31]:


categories = pd.cut(df.loc[df.Year == 2008].Age,bins=(15,25,35,45,55), right= False)

ages_df = categories.value_counts().agg(['idxmax', 'idxmin']).reset_index().\
replace('idxmax', 'Most represented').replace('idxmin', 'Least represented').set_index('index')
ages_df.index.names = ['Most or least']
ages_df


# ### How much has the number of sports at the 2002 Olympics more compared to the 1994 Olympic Games

# In[32]:


df.loc[df.Year == 2002].Sport.unique().size - df.loc[df.Year == 1994].Sport.unique().size


# ### The top 3 countries for each type of medals for the Winter and Summer Olympics

# In[33]:


medal_top_df = df.groupby(['Season', "Medal", "NOC"]).NOC.count().sort_values(ascending=False).\
groupby(level=["Medal", "Season"]).head(3).sort_index(level=[0,1]).\
reindex(['Gold', 'Silver', "Bronze"], level=1).to_frame().rename(columns = {'NOC': 'Count'})
medal_top_df


# ###  Height_z_scores variable with the values of the Height variable after its standardization

# In[34]:


Height_z_scores = stats.zscore(df.Height.dropna())
Height_z_scores


# ### Height_min_max_scaled variable with the values of the Height variable after applying min-max normalization to it.
# ##### Optional

# In[35]:


Height = df.Height.dropna()
Height_min_max_scaled = (Height - Height.min())/(Height.max() - Height.min())
Height_min_max_scaled


# ### Compared the height, weight and age of men and women who participated in the Winter Olympic Games. 
# ##### The results designed to use them for the article.

# In[36]:


new_df = df.loc[df.Season == 'Winter',['Sex','Height','Weight', 'Age']].groupby('Sex', as_index=False).agg(['min','mean', 'max', 'std']).round(2).transpose()
new_df = new_df.rename(columns = {'F':'Female', 'M':'Male'}, 
                       index = {'min':'Minimum value', 'max':'Maximum value', 'std':'Standard deviation', 'mean':'Average value'})
new_df.rename_axis()
new_df.rename_axis(["Characteristic", 'Statistics'], axis='index', inplace=True)
new_df.rename_axis("Sex of athlete:", axis="columns", inplace=True)


# In[37]:


s1 = new_df.style.format(formatter={'Female': "{:.2f}", 'Male': "{:.2f}"})
s1 = s1.set_table_styles([{'selector': 'th', 'props': 'text-align: center;'},
                    {'selector': 'th', 'props': 'text-align: center;'},
                    {'selector': '', 'props': 'border: 1px solid #000066;'},
                    {'selector': 'td', 'props': 'border: 1px solid #000066;'},
                    {'selector': 'th', 'props': 'border: 1px solid #000066;'},
                     {'selector': 'caption','props': 'caption-side: bottom; font-size:1.25em;'}],overwrite=False, axis=1)

s1.set_caption("Table 1. Height, Weight and Age of Male and Female athletes on winter olympics.")

for l0 in ['Height', 'Weight', 'Age']:
    s1 = s1.set_table_styles({(l0, 'Standard deviation'): [{'selector': '', 'props': 'border-bottom: 2px solid black;'}], 
                        (l0, 'Minimum value'): [{'selector': '.level0', 'props': 'border-bottom: 2px solid black;'}],
                        (l0, 'Minimum value'): [{'selector': '.level0', 'props': 'border: 2px solid black;'}]},
                        overwrite=False, axis=1)

s1


# ##### Making tables for article

# In[38]:


print(s1.to_latex(), file = open('latex_table_with_style.txt', 'w'))
print(s1.to_latex(), file = open('latex_table.txt', 'w'))
print(new_df.to_markdown(), file = open('markdown_table.txt', 'w'))


# ### Let's compare Medal and Team variables

# _Making top Teams with the most number of medals of all time_

# In[39]:


from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, levene, bartlett
import statsmodels.api as sm


# In[40]:


medals = df.loc[df.Medal.notna()].groupby('NOC').Medal.count().reset_index().sort_values(by='Medal', ascending=False)
num_medals = df.loc[df.Medal == "Gold"].groupby('NOC').Medal.count().reset_index().\
sort_values(by='Medal', ascending=False).rename(columns = {'Medal': 'Number of medals'})
num_medals.head(10)


# #### _Normalizing to number of unique events these countries participated_

# In[41]:


num_sport = df.groupby('NOC').Sport.nunique()
num_sport_medals = num_medals.merge(num_sport, on='NOC').rename(columns={'Sport':'Number of sports participated'})
num_sport_medals


# In[42]:


num_sport_medals['Medal per sport'] = num_sport_medals['Number of medals']/num_sport_medals['Number of sports participated']


# In[43]:


num_sport_medals = num_sport_medals.sort_values(by = 'Number of medals', ascending=False)
num_sport_medals


# ##### Checking if we can make linear model

# In[44]:


x=num_sport_medals['Number of medals']
y=num_sport_medals['Number of sports participated']

x = sm.add_constant(x)


# _Deleting deviations_

# In[45]:


model = sm.OLS(y, x).fit()


# In[46]:


cooks = model.get_influence().cooks_distance


# In[47]:


df_wo_devs = num_sport_medals[cooks[1] > 0.05]


# In[48]:


levene(df_wo_devs['Number of medals'],
df_wo_devs['Number of sports participated']).pvalue


# In[49]:


bartlett(df_wo_devs['Number of medals'],
df_wo_devs['Number of sports participated']).pvalue


# _Homoscedacity condition have not been approved_

# In[50]:


sns.scatterplot(x=df_wo_devs['Number of medals'], y=df_wo_devs['Number of sports participated']);


# In[51]:


pearsonr(df_wo_devs['Number of medals'],
df_wo_devs['Number of sports participated']).statistic


# ##### Let's see the plots

# In[74]:


fig, ax = plt.subplots(1, 3, figsize = (14, 4))
sns.barplot(data=num_sport_medals, x='NOC', y='Medal per sport', ax = ax[0]);
ax[0].set(xticklabels=[]);
ax[0].tick_params(axis='y', which='major', labelsize=7)
sns.barplot(data=num_sport_medals, x='NOC', y='Number of medals', ax = ax[1]);
ax[1].set(xticklabels=[]);
ax[1].tick_params(axis='y', which='major', labelsize=7)
sns.barplot(data=num_sport_medals, x='NOC', y='Number of sports participated', ax = ax[2]);
ax[2].set(xticklabels=[]);
ax[2].tick_params(axis='y', which='major', labelsize=7)


# We could not use correlation test, therefore we made 3 plots. All NOCs on plots sorted by Number of medals (you can see it on 2nd plot).  
# 
# As seen on 1st plot, it is a little bit similar to 2nd. However some NOCs have numbers of medals significantly greater then  number of sports they participated -> These NOCs are professionals in their sports. (It is peaks on 1st plot and gaps on 3rd plot)  
# 
# Low values of Medal per sport variable (on 1st plot) belong to some NOCs with high number of sports they participated (3rd plot). These NOCs tries a lot of different sports, but still have small number of medals.  
# 
# In common we can conclude that number of sports in which NOC participate could affect the number of medals they have. But there are some deviations, where some NOCs are professionals in their small amount of sports and NOCs that could not find they successufull sport  

# ### Some additional hypothesis

# #### Are the differences in height in women and men significant?

# In[53]:


f_height = df.loc[df.Sex == 'F'].Height.dropna()


# In[54]:


m_height = df.loc[df.Sex == 'M'].Height.dropna()


# _Checking distribution_

# In[55]:


fig, axes = plt.subplots(1,2,figsize = (8,3))

sns.histplot(data=f_height, ax=axes[0], binwidth=1);
sns.histplot(data=m_height, ax=axes[1], binwidth=1);

axes[0].set_title('Female');
axes[1].set_title('Male');

axes[0].tick_params(axis='both', which='major', labelsize=8);
axes[1].tick_params(axis='both', which='major', labelsize=8);


# _Checking common statistics_

# In[56]:


f_height.describe()


# In[57]:


m_height.describe() 


# **There are a lot of values and distribution seems to be normal, we use a t-test**

# In[58]:


t = ttest_ind(m_height, f_height)
t.pvalue


# In[59]:


mann = mannwhitneyu(m_height, f_height)
mann.pvalue


# **pvalue is very small in both tests -> Significantly different**

# #### Let's check for weight and age sticking to the pipeline described above

# In[60]:


m_weight = df.loc[df.Sex == 'M'].Weight.dropna()
f_weight = df.loc[df.Sex == 'F'].Weight.dropna()
m_age = df.loc[df.Sex == 'M'].Age.dropna()
f_age = df.loc[df.Sex == 'F'].Age.dropna()


# **_Weight_**

# In[61]:


fig, axes = plt.subplots(1,2,figsize = (8,3))

sns.histplot(data=f_weight, ax=axes[0], binwidth=2);
sns.histplot(data=m_weight, ax=axes[1], binwidth=2);

axes[0].set_title('Female');
axes[1].set_title('Male');

axes[0].tick_params(axis='both', which='major', labelsize=8);
axes[1].tick_params(axis='both', which='major', labelsize=8);


# In[62]:


m_weight.describe()


# In[63]:


f_weight.describe()


# In[64]:


t = ttest_ind(m_weight, f_weight)
t.pvalue


# **_Age_**

# In[65]:


fig, axes = plt.subplots(1,2,figsize = (8,3))

sns.histplot(data=f_age, ax=axes[0], binwidth=1);
sns.histplot(data=m_age, ax=axes[1], binwidth=1);

axes[0].set_title('Female');
axes[1].set_title('Male');

axes[0].tick_params(axis='both', which='major', labelsize=8);
axes[1].tick_params(axis='both', which='major', labelsize=8);


# In[66]:


m_age.describe()


# In[67]:


f_age.describe()


# In[68]:


t = ttest_ind(m_age, f_age)
t.pvalue


# **Both the average weight and average age of athletes of different sexes are significantly different**  
# _Both distributions seems normal and number of values was huge_

# #### Is the average number of medals in Women and Men significant?

# In[69]:


m_medals = df.loc[(df.Sex == 'M') & (df.Medal.notnull())].groupby('Name').Medal.count()
f_medals = df.loc[(df.Sex == 'F') & (df.Medal.notnull())].groupby('Name').Medal.count()


# In[70]:


f_medals.value_counts()


# _Distribution is not normal, but number of values is still big_

# In[71]:


m_medals.describe()


# In[72]:


f_medals.describe()


# _It is better to use another test since distribution significantly not normal_

# In[73]:


mann = mannwhitneyu(m_medals, f_medals)
mann.pvalue


# **Differences still significant.**

# _I am not sexist. It were the most common tests that i vave in my mind with comparison of 2 groups_
