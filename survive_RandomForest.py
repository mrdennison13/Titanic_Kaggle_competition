import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

import seaborn as sns

#read in the data
df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")


#first, check which features there are
df_train.columns


#See what has NANs etc.
df_train.info()
df_test.info()

df_train.isnull().sum()
df_test.isnull().sum()


#make a list of those columns (maybe we need it later)
na_cols_train=df_train.columns[df_train.isnull().any()]
na_cols_test=df_test.columns[df_test.isnull().any()]

#and lets replace them:
#1) embarked (in training set)
df_train[df_train['Embarked'].isnull()]
#two women, paying the same fare and in the same first-class cabin
#show the average fare for each embarkation by class
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df_train)
plt.show()
#since they paid 80 a pop, this matches most closely with 'C'
df_train["Embarked"] = df_train["Embarked"].fillna('C')

#2) Fare in the test set
df_test[df_test['Fare'].isnull()]
#Third class chap, embarked at S, male, no family
#replacing it with the mean, rather than the median, probably makes most sense
median_fare=df_test[(df_test['Pclass'] == 3) & (df_test['Embarked'] == 'S') & (df_test['Sex'] == 'male') & (df_test['SibSp'] + df_test['Parch'] == 0)]['Fare'].median()
df_test["Fare"] = df_test["Fare"].fillna(median_fare)


#could try the modal group instead for embarcation
#embarked: change to modal group for the class and gender of the passenger
#n_group = df_train[(df_train['Pclass'] == 1) & (df_train['Sex'] == 1)]['Embarked'].mode()[0]
#df_train['Embarked'].fillna(n_group, inplace=True)



#the others have a lot more missing, so lets ignore them for now


#Now see what makes sense to use for features

#check the correlations between features (useful for continuous ones, e.g. age and fare)
df_train.corr()
#fare looks good

#cross tabulate the features take only a few values (sex, class, embarked)
pd.crosstab(df_train.Sex, df_train.Survived, margins=True)
pd.crosstab(df_train.Pclass, df_train.Survived, margins=True)
pd.crosstab(df_train.Embarked, df_train.Survived, margins=True)
#all these look pretty useful (although I have no idea why the point of embarkation would matter: the cabin location they got, perhaps?


#make a box-plot to check age and fare
sns.boxplot(x="Survived", y="Age", data=df_train)
sns.boxplot(x="Survived", y="Fare", data=df_train)
#fare looks good to use, age does not (considering how many are missing, best to ignore it)

#now lets check the family size, SibSp and Parch
sns.factorplot(x="Parch", y="Survived", data=df_train);
sns.factorplot(x="SibSp", y="Survived", data=df_train);
#More survive if SibSp and Parch are small, but dip again on 0

#Plot the total family size (Parch + SibSp), so make a family size variable (add the number of siblings to the number of parents)
df_train['Fam_size'] = df_train['SibSp'] + df_train['Parch'] 
df_test['Fam_size']  = df_test['SibSp']  + df_test['Parch']
sns.factorplot(x="Fam_size", y="Survived", data=df_train);

#Fam_size between 1 and 3 is clearly the most likely to survive
#discritize it (1 = singleton, 0 = 1-3 family members, 2 = 4+ family members)
df_train['Fam_type']  = 0
df_train.loc[(df_train['Fam_size'] < 1), 'Fam_type'] = 1
df_train.loc[(df_train['Fam_size'] > 3), 'Fam_type'] = 2

df_test['Fam_type']  = 0
df_test.loc[(df_test['Fam_size'] < 1), 'Fam_type'] = 1
df_test.loc[(df_test['Fam_size'] > 3), 'Fam_type'] = 2

sns.factorplot(x="Fam_type", y="Survived", data=df_train);


#as features we have
#Pclass
#Sex
#Fare
#Embarked
#Family type
features = ['Pclass','Sex','Fare', 'Embarked','Fam_type']


#change the genders: Male = 0, Female = 1
df_train = df_train.replace(['male','female'], [0,1])
df_test = df_test.replace(['male','female'], [0,1])

#change the embarcation: C=0, S=1, Q=2
df_train = df_train.replace(['C','S','Q'], [0,1,2])
df_test = df_test.replace(['C','S','Q'], [0,1,2])

df_train.corr()['Survived']





#make them into something that can be read by the random forest function
train_features = df_train[features].values.reshape(-1,5)
train_label    = df_train.Survived.reshape(-1,1)


#split it into a training and a test set (60% to train)
X_train, X_test1, y_train, y_test1 = train_test_split(train_features, train_label, test_size=0.4)
#and into a validation and test set (50/50)
X_test, X_val, y_test, y_val = train_test_split(X_test1, y_test1, test_size=0.5)

test_features = df_test[features].values.reshape(-1,5)



#feature scaling for the fare (not needed for random forest)

#from sklearn import preprocessing
#std_scale = preprocessing.StandardScaler().fit(df_train['Fare'])
#df_train['Fare']  = std_scale.transform(df_train['Fare'])#
#df_test['Fare']  = std_scale.transform(df_test['Fare'])

#or manually for the fare
#mu_train    = df_train['Fare'].mean()
#range_train = (df_train['Fare'].max()-df_train['Fare'].min())

#df_train['Fare'] -= mu_train            #subtract the mean
#df_train['Fare'] /= range_train           #divide by the range

#df_test['Fare'] -= mu_train  
#df_test['Fare'] /= range_train




#we will use the random forest classifer to train it, starting with the inbuilt defaults
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


#rf = RandomForestClassifier(min_samples_leaf=3, n_estimators=500, max_depth=241, max_features=3, oob_score = True)


#now find the accuracy on the test set
1.0 - float(np.absolute(y_test - rf.predict(X_test).reshape((y_test.shape[0],1))).sum())/float(y_test.shape[0])


#and show the important features
importance = rf.feature_importances_
importance = pd.DataFrame(importance, index=df_train[features].columns, columns=["Importance"])


#now lets try to find the opitmal parameters for the Random Forest classifer

#Function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


            

#make the data readible
y_train = train_label.reshape(train_label.shape[0],)
X_train = train_features



#test the max_features first
param_grid = {
    "max_depth": [1, 501],
    "max_features": range(1,6),
    "min_samples_leaf": [1, 1000],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

#use 250 estimators
rf = RandomForestClassifier(n_estimators=250)
grid_search = GridSearchCV(rf, param_grid=param_grid)

start = time()

grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_)

#regardless of the other params, it looks like 3/4 is the best.
#also, bootstrap and criterion do not matter much


#we can probably ignore criterion and bootstrap, and set max_features to 3
param_grid = {
    "max_depth": range(1, 501, 50),
    "min_samples_leaf": range(1, 101, 10),
}

#use 250 estimators
rf = RandomForestClassifier(n_estimators=250,max_features=3)
grid_search = GridSearchCV(rf, param_grid=param_grid)

start = time()

grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_)


#high max depth is good, so lets set it to 501, and check the min_samples_leaf
param_grid = {"min_samples_leaf": range(1, 51, 1)}

#use 250 estimators
rf = RandomForestClassifier(n_estimators=250,max_features=3,max_depth=501)
grid_search = GridSearchCV(rf, param_grid=param_grid)

start = time()

grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_)



#looks like 2,3,4 are good values, so go with 3
rf = RandomForestClassifier(n_estimators=250,max_features=3,max_depth=501,min_samples_leaf=3)

#split it into a training and a test set (60% to train)
X_train, X_test, y_train, y_test = train_test_split(train_features, train_label, test_size=0.4)

rf.fit(X_train, y_train)
#now find the accuracy on the test set
1.0 - float(np.absolute(y_test - rf.predict(X_test).reshape((y_test.shape[0],1))).sum())/float(y_test.shape[0])

#and output the data set to submit
test_features = df_test[features].values.reshape(-1,5)
df_test_out = df_test
df_test_out['Survived'] = rf.predict(test_features)
df_test_out[['PassengerId', 'Survived']].to_csv('prediction_Dennison_RF_features5.csv',  index = False)




#We can also cross-validate it

rf = RandomForestClassifier(min_samples_split=2, n_estimators=250,max_features=3,max_depth=501,min_samples_leaf=3)
kf = KFold(df_train.shape[0], n_folds=5, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

predictions = cross_validation.cross_val_predict(rf, df_train[features],df_train["Survived"],cv=kf)
predictions = pd.Series(predictions)
scores = cross_val_score(rf, df_train[features], df_train["Survived"],scoring='f1', cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


#This gives am accuracy lower than above (about 0.73), still not sure as to why





#And this is it for random forest.

# I find that on the test set I get somewhere between 0.74 and 0.78, and it varies each time
# (it is a method that makes use of random sampling, and we have a small-ish data set)
