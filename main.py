# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# %%
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_data.describe(include='all')


# %%
size_train = len(train_data)
dataset = pd.concat([train_data, test_data]).reset_index(drop=True)


# %%
dataset.isna().sum()


# %%
dataset.sample(20)

# %% [markdown]
# # EDA

# %%
EDA = False

# %% [markdown]
# ### Age

# %%
if EDA:
    ax = sns.kdeplot(train_data.loc[train_data['Survived']==True, 'Age'], shade=True, color='r')
    ax = sns.kdeplot(train_data.loc[train_data['Survived']==False, 'Age'], shade=True, color='b', ax=ax)
    ax_legend = ax.legend(['Survived', 'Not Survived'])


# %%
if EDA:
    g = sns.FacetGrid(data=train_data, col='Survived')
    g = g.map(sns.distplot, 'Age')


# %%
if EDA:
    features = ['SibSp', 'Pclass', 'Sex', 'Parch', 'Embarked']
    for f in features:
        sns.catplot(x=f, y='Age', data=dataset, kind='box')
    sns.heatmap(dataset[features + ['Age']].corr(), annot=True)


# %%
if EDA:
    sns.catplot(x='Pclass', y='Age', hue='Sex', data=dataset, kind='bar')

# %% [markdown]
# ### Fare

# %%
if EDA:
    fig = plt.figure(figsize=[12, 4])
    axes = fig.subplots(1, 2)
    ax = sns.distplot(train_data['Fare'], ax=axes[0])
    ax = sns.distplot(train_data['Fare'].map(lambda x: np.log(x) if x > 0 else -10), ax=axes[1])

# %% [markdown]
# ### Pclass

# %%
if EDA:
    g = sns.catplot(x='Pclass', y='Survived', hue='Sex', data=train_data, kind='bar')

# %% [markdown]
# ### Embarked

# %%
if EDA:
    g = sns.catplot(x='Embarked', y='Survived', hue='Sex', data=train_data, kind='bar')


# %%
if EDA:
    sns.catplot(x='Pclass', col='Embarked', data=train_data, kind='count')

# %% [markdown]
# ### Family

# %%
if EDA:
    sns.catplot(x='SibSp', y='Survived', data=train_data, kind='bar')


# %%
if EDA:
    sns.catplot(x='Parch', y='Survived', data=train_data, kind='bar')

# %% [markdown]
# ### Sex

# %%
if EDA:
    sns.catplot(x='Sex', y='Survived', data=train_data, kind='bar')

# %% [markdown]
# ## Data processing 

# %%
dataset.isna().sum()

# %% [markdown]
# ### Age, Fare, Embarked

# %%
dataset['Embarked'].fillna('S', inplace=True)


# %%
dataset[dataset['Fare'].isnull()]


# %%
dataset['Fare'].fillna(dataset.loc[(dataset['Pclass'] == 3) & (dataset['Embarked'] == 'S'), 'Fare'].median(), inplace=True)


# %%
dataset['Age'] = dataset.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))


# %%
dataset.isnull().sum()


# %%
dataset['AgeBand'] = pd.qcut(dataset['Age'], 10)
if EDA:
    sns.countplot(x='AgeBand', hue='Survived', data=dataset)


# %%
dataset['FareBand'] = pd.qcut(dataset['Fare'], 13)
if EDA:
    sns.countplot(x='FareBand', hue='Survived', data=dataset)

# %% [markdown]
# ### Cabin

# %%
dataset['Cabin'] = dataset['Cabin'].str.extract(r'^(\S)', expand=False).fillna('M')


# %%
if EDA:
    sns.catplot(x='Cabin', y='Survived', data=dataset.loc[:size_train], kind='bar')


# %%
dataset['Cabin'].replace(['A', 'B', 'C', 'T'], 'ABC', inplace=True)
dataset['Cabin'].replace(['D', 'E'], 'DE', inplace=True)
dataset['Cabin'].replace(['F', 'G'], 'FG', inplace=True)

# %% [markdown]
# ### Name

# %%
dataset.loc[dataset['Name'].str.contains('\('), 'Name'].sample(20)


# %%
dataset['Surname'] = dataset['Name'].str.extract(r'^([^,]+),', expand=False)
dataset['Title'] = dataset['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
dataset['Married'] = (dataset['Title'] == 'Mrs').astype(np.int)
dataset['Title'].replace(['Ms', 'Mrs', 'Mlle', 'Countess', 'Lady', 'Dona', 'Mme'], 'Miss', inplace=True)
dataset['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Rev', 'Don', 'Sir'], 'Noble', inplace=True)


# %%
if EDA:
    sns.countplot(x='Title', hue='Survived', data=dataset[:size_train])


# %%
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
if EDA:
    sns.countplot(x='FamilySize', hue='Survived', data=dataset)


# %%
dataset.loc[dataset['FamilySize'] == 1, 'FamilyType'] = 'Alone'
dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] < 5), 'FamilyType'] = 'Small'
dataset.loc[(dataset['FamilySize'] >= 5) & (dataset['FamilySize'] < 7), 'FamilyType'] = 'Medium'
dataset['FamilyType'].fillna('Large', inplace=True)


# %%
if EDA:
    sns.countplot(x='FamilyType', hue='Survived', data=dataset)


# %%
surname_fam_rates = dataset[:size_train].groupby(['Surname', 'FamilySize'])['Survived'].median().to_dict()
surname_fam_rates = {k: v for k, v in surname_fam_rates.items() if k[1] > 1}
surname_rates = dataset[:size_train].groupby('Surname')['Survived', 'FamilySize'].median().to_dict('index')
surname_rates = {k: v['Survived'] for k, v in surname_rates.items() if v['FamilySize'] > 1}

surname_train = set(dataset.iloc[:size_train]['Surname'].tolist())
surname_test = set(dataset.iloc[size_train:]['Surname'].tolist())
surname_set = surname_train.intersection(surname_test)


# %%
average_mean = dataset['Survived'].mean()
def get_surname_rate(sur, fams):
    if sur not in surname_set:
        return average_mean, 0
    elif (sur, fams) not in surname_fam_rates.keys():
        if sur in surname_rates.keys():
            return surname_rates[sur], 1
        else:
            return average_mean, 0
    else:
        return surname_fam_rates[(sur, fams)], 1

surname_df = dataset[['Surname', 'FamilySize']].apply(lambda x: get_surname_rate(x['Surname'], x['FamilySize']), axis=1, result_type='expand').rename({0: 'SurnameSurvived', 1: 'SurnameSurvivedValid'}, axis=1)

dataset = pd.concat((dataset, surname_df), axis=1)

# %% [markdown]
# ### Ticket

# %%
dataset['TicketFreq'] = dataset.groupby('Ticket')['Ticket'].transform('count')
ticket_survived = dataset[:size_train].groupby('Ticket')['Survived'].median().to_dict()
ticket_count = dataset[:size_train].groupby('Ticket')['TicketFreq'].median().to_dict()
ticket_train = set(dataset.iloc[:size_train]['Ticket'].tolist())
ticket_test = set(dataset.iloc[size_train:]['Ticket'].tolist())
ticket_set = ticket_train.intersection(ticket_test)


# %%
dataset['TicketSurvived'] = dataset['Ticket'].transform(lambda x: ticket_survived[x] if x in ticket_set and ticket_count[x] > 1 else average_mean)
dataset['TicketSurvivedValid'] = dataset['Ticket'].transform(lambda x: 1 if x in ticket_set and ticket_count[x] > 1 else 0)


# %%
dataset['SurvivalRate'] = (dataset['TicketSurvived'] + dataset['SurnameSurvived']) / 2
dataset['SurvivalRateValid'] = (dataset['TicketSurvivedValid']  + dataset['SurnameSurvivedValid']) / 2
dataset.drop(['TicketSurvived', 'SurnameSurvived', 'TicketSurvivedValid', 'SurnameSurvivedValid'], axis=1, inplace=True)

# %% [markdown]
# ### Finalize

# %%
dataset.drop(['Name', 'Ticket', 'Surname', 'FamilySize', 'Age', 'Fare'], axis='columns', inplace=True)


# %%
dataset.head(10)


# %%
from sklearn.preprocessing import LabelEncoder, StandardScaler

dataset = pd.get_dummies(dataset, columns=['Pclass'])
dataset = pd.get_dummies(dataset, columns=['FamilyType'])
dataset = pd.get_dummies(dataset, columns=['Cabin'])
dataset = pd.get_dummies(dataset, columns=['Embarked'])
dataset = pd.get_dummies(dataset, columns=['Title'])
dataset['Sex'] = LabelEncoder().fit_transform(dataset['Sex'])
dataset['AgeBand'] = LabelEncoder().fit_transform(dataset['AgeBand'])
dataset['FareBand'] = LabelEncoder().fit_transform(dataset['FareBand'])

# %% [markdown]
# ## Start training

# %%
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score


# %%
y = dataset.iloc[:size_train]['Survived']
X = dataset.iloc[:size_train].drop(columns=['Survived', 'PassengerId'], axis=1)
X_test = dataset.iloc[size_train:].drop(columns=['Survived', 'PassengerId'], axis=1)


# %%
model = RandomForestClassifier( criterion='gini', 
                                n_estimators=1100,
                                max_depth=5,
                                min_samples_split=4,
                                min_samples_leaf=5,
                                max_features='auto',
                                oob_score=True,
                                random_state=0,
                                n_jobs=-1,
                                verbose=1)
model.fit(X, y)
y_pred = model.predict(X_test).astype(np.int)
results = pd.DataFrame({'PassengerId': dataset.iloc[size_train:]['PassengerId'], 'Survived': y_pred})
results.to_csv('data/submission_0807_rf.csv', index=False)

# %% [markdown]
# ## Finish

# %%
y_test = pd.read_csv('data/test_label.csv')['Survived']
score = accuracy_score(y_test, y_pred)
print(score)


# %%



