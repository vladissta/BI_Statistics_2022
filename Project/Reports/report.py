#!/usr/bin/env python
# coding: utf-8

# # PROJECT 1
# Stanin Vladislav

# ### Importing requirements

# In[1]:


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


# _Obviously some sportsmen did not have medals. Also we consider that some athletes did not measured their Height and Weight and did not wanted to say their age (maybe). And nobody took such statistics especially in the several first games_

# #### Deleting sportsmen without name and sport, beacuse sportsmen could not be without sport and name

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


# _2 athletes turned out to be with gender "G" - with all my respect for minorities, I can't leave it this way, because in the years in which these athletes performed, gender-non-decided people did not perform (especially from Russia and Czechoslovakia). So i changed the sex according to their sports and names_

# In[17]:


df.loc[df['Sex'] == 'G', 'Sex'] = 'M'


# #### Checking Age

# In[18]:


df['Age'].value_counts().sort_index()


# In[19]:


df.loc[df.Age == 240]


# _Too old... We have to change this data to 24, beacuse she have some another data in dataset - he was 24 in 1912_

# In[20]:


df.loc[(df.Name == "Flicien Jules mile Courbet") & (df.Year == 1912)].Age


# In[21]:


df.loc[df.Age == 240, 'Age'] = 24


# #### Checking Height

# In[22]:


df['Height'].value_counts().sort_index()


# In[23]:


df.loc[df.Height == 340]


# _It is to high! But let's take a look to another data about his height_

# In[24]:


df.loc[df.Name == 'Kirsty Leigh Coventry (-Seward)'].Height.value_counts()


# She is definetely 176!

# In[25]:


df.loc[df.Height == 340, 'Height'] = 176


# #### Checking Weight

# In[26]:


df['Weight'].value_counts().sort_index()


# _It is already okay_

# ## EDA

# ### The age of the youngest athletes of both sexes at the 1992 Olympics.

# In[27]:


df.loc[df.Year == 1992].groupby('Sex').Age.min().to_frame()


# ### The average value and standard deviation of the Height variable for athletes of each sex.

# In[28]:


df.groupby('Sex').agg(Mean_value = ('Height', 'mean'),
                      Standard_deviation = ('Height', 'std'))


# ### The average value and standard deviation of the Height variable for female tennis players at the 2000 Olympics.

# In[29]:


df.loc[(df.Sex == 'F') & 
       (df.Sport == 'Tennis') & 
       (df.Year == 2000)].agg(Mean_value = ('Height', 'mean'), 
                              Standard_deviation = ('Height', 'std')).round(1)


# ### Heaviest athlete's sport in at the 2006 Olympics

# In[30]:


print(f"Sport of heviest athlete: {(df.loc[(df.Weight == df.loc[df.Year == 2006].Weight.max()) & (df.Year == 2006)].Sport).reset_index().Sport[0]}")


# ### Number of gold medals which were received by women from 1980 to 2010

# In[31]:


df.loc[(df.Sex == "F") & (df.Medal == "Gold") & (df.Year.isin(range(1980,2011)))].Medal.count()


# ### Number of times has John Aalberg participated in the Olympic Games in different years

# In[32]:


diff_years = df.loc[df.Name == "John Aalberg"].Year.unique().size
all_times = df.loc[df.Name == "John Aalberg"].shape[0]
num_of_games = df.loc[df.Name == "John Aalberg"].Games.unique().size
print(f'Different years he participated: {diff_years}. Participated at different competitions: {all_times} times.')


# ### The least and most represented (by number of participants) age groups of athletes at the 2008 Olympics. 
# ##### Age groups: [15-25), [25-35), [35-45), [45-55].

# In[33]:


categories = pd.cut(df.loc[df.Year == 2008].Age,bins=(15,25,35,45,55), right= False)

ages_df = categories.value_counts().agg(['idxmax', 'idxmin']).reset_index().\
replace('idxmax', 'Most represented').replace('idxmin', 'Least represented').set_index('index')
ages_df.index.names = ['Most or least']
ages_df


# ### How much has the number of sports at the 2002 Olympics more compared to the 1994 Olympic Games

# In[34]:


print(f" In 2002 Olympics there were {df.loc[df.Year == 2002].Sport.unique().size - df.loc[df.Year == 1994].Sport.unique().size} more sports then in 1994 Olympics")


# ### The top 3 countries for each type of medals for the Winter and Summer Olympics

# In[35]:


medal_top_df = df.groupby(['Season', "Medal", "NOC"]).NOC.count().sort_values(ascending=False).\
groupby(level=["Medal", "Season"]).head(3).reindex(['Gold', 'Silver', "Bronze"], level=1).to_frame().\
rename(columns = {'NOC': 'Count'})
medal_top_df


# ###  Height_z_scores variable with the values of the Height variable after its standardization

# In[76]:


df['Height_z_scores'] = (df.Height - df.Height.mean()) / df.Height.std()
df.Height_z_scores


# ### Height_min_max_scaled variable with the values of the Height variable after applying min-max normalization to it.
# ##### Optional

# In[77]:


df['Height_min_max_scaled'] = (df.Height - df.Height.min()) / (df.Height.max() - df.Height.min())
df['Height_min_max_scaled']


# ### Compared the height, weight and age of men and women who participated in the Winter Olympic Games. 
# ##### The results designed to use them for the article.

# _As we have huge data (big amount of values), t-test could be applied_

# In[78]:


t_height = stats.ttest_ind(df.loc[df.Sex == 'F', 'Height'].dropna(),
df.loc[df.Sex == 'M', 'Height'].dropna())
t_weight = stats.ttest_ind(df.loc[df.Sex == 'F', 'Weight'].dropna(),
df.loc[df.Sex == 'M', 'Weight'].dropna())
t_age = stats.ttest_ind(df.loc[df.Sex == 'F', 'Age'].dropna(),
df.loc[df.Sex == 'M', 'Age'].dropna())


# In[79]:


new_df = df.loc[df.Season == 'Winter',['Sex','Height','Weight', 'Age']].groupby('Sex', as_index=False).agg(['min','mean', 'max', 'std', "count"]).round(2).transpose()
new_df = new_df.rename(columns = {'F':'Female', 'M':'Male'}, 
                       index = {'min':'Minimum value', 
                                'max':'Maximum value',
                                'std':'Standard deviation',
                                'mean':'Average value',
                                'count': "Total values"})
new_df.rename_axis()
new_df.rename_axis(["Characteristic", 'Statistics'], axis='index', inplace=True)
new_df.rename_axis("Sex of athlete:", axis="columns", inplace=True)

new_df['T-test'] = ''
new_df.loc[('Height', 'Average value'), 'T-test'] = f' Statistic = {t_height[0].round(2)}'
new_df.loc[('Weight', 'Average value'), 'T-test'] = f'Statistic = {t_weight[0].round(2)}'
new_df.loc[('Age', 'Average value'), 'T-test'] = f'Statistic = {t_age[0].round(2)}'

new_df.loc[('Height', 'Maximum value'), 'T-test'] = f'p-value = {t_height[1].round(2)}'
new_df.loc[('Weight', 'Maximum value'), 'T-test'] = f'p-value = {t_weight[1].round(2)}'
new_df.loc[('Age', 'Maximum value'), 'T-test'] = f'p-valuec = {t_age[1].round(2)}'


# In[80]:


s1 = new_df.style.format(formatter={'Female': "{:.1f}", 'Male': "{:.1f}", 'T-test p-value': "{:.5f}"})
s1 = s1.set_table_styles([{'selector': 'th', 'props': 'text-align: center;'},
                    {'selector': 'th', 'props': 'text-align: center;'},
                    {'selector': '', 'props': 'border: 1px solid #000066;'},
                     {'selector': 'caption','props': 'caption-side: bottom; font-size:1.25em;'}],overwrite=False, axis=1)

s1.set_caption("Table 1. Height, Weight and Age of Male and Female athletes on winter olympics.")

for l0 in ['Height', 'Weight', 'Age']:
    s1 = s1.set_table_styles({(l0, 'Total values'): [{'selector': '', 'props': 'border-bottom: 2px solid black;'}], 
                        (l0, 'Minimum value'): [{'selector': '.level0', 'props': 'border-bottom: 2px solid black;'}],
                        (l0, 'Minimum value'): [{'selector': '.level0', 'props': 'border: 2px solid black;'}]},
                        overwrite=False, axis=1)

s1


# ##### Making tables for article

# In[81]:


print(s1.to_latex(), file = open('../Tables for article/latex_table_with_style.txt', 'w'))
print(s1.to_latex(), file = open('../Tables for article/latex_table.txt', 'w'))
print(new_df.to_markdown(), file = open('../Tables for article/markdown_table.txt', 'w'))


# ### Let's compare Medal and Team variables

# _Making top Teams with the most number of medals of all time_

# In[82]:


medals = df.loc[df.Medal.notna()].groupby('Team').Medal.count().reset_index().sort_values(by='Medal', ascending=False)
num_medals = df.loc[df.Medal == "Gold"].groupby('Team').Medal.count().reset_index().\
sort_values(by='Medal', ascending=False).rename(columns = {'Medal': 'Number of medals'})
num_medals.head(10)


# We already can assume that some Teams have more medals then others! But it is difficult to make some correlation. Let's assume that Team and Medal variables connected via Sports variable (in particular - the number of kinds of sports Team is participated in)

# #### How many kinds of sports Teams is participated in?

# In[83]:


num_sport = df.groupby('Team').Sport.nunique()
num_sport_medals = num_medals.merge(num_sport, on='Team').rename(columns={'Sport':'Number of sports participated'})
num_sport_medals


# In[84]:


sns.scatterplot(x=num_sport_medals['Number of medals'], y=num_sport_medals['Number of sports participated']);


# _It seemas like it is better to use spearman test (nonlinear, non-homoscedastic etc.)_

# In[85]:


print(f"Spearman's r is {stats.spearmanr(num_sport_medals['Number of medals'], num_sport_medals['Number of sports participated']).correlation}")


# ##### Let's see some additional plots !

# In[86]:


fig, ax = plt.subplots(1, 2, figsize = (14, 4))
sns.barplot(data=num_sport_medals, x='Team', y='Number of medals', ax = ax[0]);
ax[0].set(xticklabels=[]);
ax[0].tick_params(axis='y', which='major', labelsize=7)
sns.barplot(data=num_sport_medals, x='Team', y='Number of sports participated', ax = ax[1]);
ax[1].set(xticklabels=[]);
ax[1].tick_params(axis='y', which='major', labelsize=7)


# All Teams on plots sorted by Number of medals (you can see it on 1st plot).  
# 
# Some Teams have numbers of medals significantly greater then  number of sports they participated -> These Teams are professionals in their sports or it is particular sport team!. (It is gaps on 2nd plot)  
# 
# Some Teams with high number of sports they participated (2nd plot - high values) have low number of medals. These Teams tries a lot of different sports, but still have small number of medals.  
# 
# In common we can conclude that number of sports in which Team participate affects the number of medals they have. But there are some deviations, where some Teams are professionals in their small amount of sports (or one kind of sport) and Teams that could not find they successufull sport
# 
# That is why correlation is not equal to 1!
# 
# **So Team and Medal variables is connected. Particular Teams partcicpate in many kind of sports or even in only one and then recieve many or few medals!**

# ## Some additional hypothesis

# ### Is the average number of medals in Women and Men significant?

# In[87]:


m_medals = df.loc[(df.Sex == 'M') & (df.Medal.notnull())].groupby('Name').Medal.count()
f_medals = df.loc[(df.Sex == 'F') & (df.Medal.notnull())].groupby('Name').Medal.count()


# In[88]:


f_medals.value_counts()


# _Distribution is not normal, but number of values is still big_

# In[89]:


m_medals.describe()


# In[90]:


f_medals.describe()


# _It is better to use another test (Mann-Whitney) since distribution significantly not normal_

# In[51]:


mann = stats.mannwhitneyu(m_medals, f_medals)
print(f'p-value is {mann.pvalue}')


# _**Differences still significant.**_

# ### Is weights of swimmers is significantly differ from footballer's?

# In[52]:


footbalers_weights = df.loc[(df.Sport == 'Football')].Weight.dropna()
swimmers_weights = df.loc[(df.Sport == 'Swimming')].Weight.dropna()


# In[53]:


footbalers_weights.describe()


# In[54]:


swimmers_weights.describe()


# _There are almost no differences yet!_

# In[55]:


print(f"p-value = {stats.ttest_ind(swimmers_weights, footbalers_weights).pvalue}")


# _They have the same weight! Is that mean that footballers would not drown..?_

# ### Is it true that over time more Teams have become involved in games?

# In[56]:


num_of_nocs = df.loc[df.Season == 'Summer'].groupby('Year').Team.nunique().reset_index()


# In[57]:


sns.scatterplot(data=num_of_nocs, x = 'Year', y="Team");


# In[58]:


print(f" Test of homoscedacity (p-value): {stats.levene(num_of_nocs.Year, num_of_nocs.Team).pvalue}")


# It is better to use Spearman's test

# In[59]:


print(f"Spearman's r is {stats.spearmanr(num_of_nocs.Year, num_of_nocs.Team).correlation}")


# _More teams each year!_
