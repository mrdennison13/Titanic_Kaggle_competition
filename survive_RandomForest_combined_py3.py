import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectFromModel

from sklearn import preprocessing

import seaborn as sns


#read in the data
df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")

labels_survived = df_train.Survived

#make a combined data set: more data to replace e.g. ages
data = df_train.append(df_test, ignore_index=True)
data.drop('Survived', 1, inplace=True)


#replace NANs which are simple to do (i.e. only 1 or 2) 
data["Embarked"] = data["Embarked"].fillna('C')

median_fare=data[(data['Pclass'] == 3) & (data['Embarked'] == 'S') & (data['Sex'] == 'male') & (data['SibSp'] + data['Parch'] == 0)]['Fare'].median()
data["Fare"] = data["Fare"].fillna(median_fare)


#Make multi-value features into separate features
def make_features(df, col):
    dummies = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummies],axis=1)
    return df


data = make_features(data,'Embarked')
data = make_features(data,'Pclass')



def make_fam_type(df, cuts):
    """Add up the family size and set it into three different types"""
    df['Fam_size']  = df['SibSp']  + df['Parch']
    df['Fam_type']  = 'Small_family'
    df.loc[(df['Fam_size'] < cuts[0]), 'Fam_type'] = 'Singleton'
    df.loc[(df['Fam_size'] > cuts[1]), 'Fam_type'] = 'Large_family'
    return df



cutss = (1, 3)

data = make_fam_type(data, cutss)
data = make_features(data,'Fam_type')



def make_str_ints(df, col):
    """Turn features which have string entries into integer entries"""
    labels = df[col].value_counts().index.tolist()
    df[col] = df[col].replace(labels, range(0,len(labels)))
    return df
    

data = make_str_ints(data, 'Sex')


#How about name length
def make_name_len(df, bins):
    df['Name_len'] = df.Name.str.len()
    labels = range(0,len(bins)-1)
    df['Name_cat'] = pd.cut(df['Name_len'], bins, labels=labels)
    return df


data.Name.str.len().max()
data.Name.str.len().min()
#12 min, 82 max

#bin it
bins = (10, 26, 55, 82)
data =  make_name_len(data, bins)



def make_titles(df):
    """Extract the title of the passenger and make it into features"""
    #lots of Mr, Mrs, Master etc. Less of the more obscure titles. Let's make the following features
    # Mr
    # Master
    # Mrs      (also Mme)
    # Miss     (also Ms, Mlle)
    # Military (Col, Major, Capt)
    # Rev 
    # Dr 
    # Aristocracy Female (Countess, Lady, Dona)
    # Aristocracy Male   (Don, Jonkheer, Sir)
    df['Title'] = df.Name.str.split(", ").str[1]
    df['Title'] = df.Title.str.split(".").str[0]
    #remove the 'the '
    df.Title = df.Title.map(lambda x: x.lstrip('the '))
    df['Mr'] = 0
    df.ix[df['Title']=='Mr', 'Mr'] = 1
    df['Master'] = 0
    df.ix[df['Title']=='Master', 'Master'] = 1
    df['Mrs'] = 0
    df.ix[(df['Title']=='Mrs') | (df['Title']=='Mme'), 'Mrs'] = 1
    df['Miss'] = 0
    df.ix[(df['Title']=='Miss') | (df['Title']=='Ms') | (df['Title']=='Mlle'), 'Miss'] = 1
    df['Military'] = 0
    df.ix[(df['Title']=='Col') | (df['Title']=='Major') | (df['Title']=='Capt'), 'Military'] = 1
    df['Rev'] = 0
    df.ix[df['Title']=='Rev', 'Rev'] = 1
    df['Dr'] = 0
    df.ix[df['Title']=='Dr', 'Dr'] = 1
    df['Arist_F'] = 0
    df.ix[(df['Title']=='Lady') | (df['Title']=='Countess') | (df['Title']=='Dona'), 'Arist_F'] = 1
    df['Arist_M'] = 0
    df.ix[(df['Title']=='Sir') | (df['Title']=='Jonkheer') | (df['Title']=='Don'), 'Arist_M'] = 1
    return df


data = make_titles(data)

data.Title.value_counts()



#now lets try to replace the missing ages, to make that a feature
#first, lets try using the random forest to learn on the data we have with ages, and then replace them accordingly
def calc_age(df, rf, features):
    """Use the random forest regressor to predict the ages of the passengers"""
    df_train = df.loc[df['Age'].notnull()]
    df_test  = df.loc[df['Age'].isnull()]
    train_features = df_train[features].values.reshape(-1,len(features))
    train_label    = df_train.Age.reshape(-1,1)
    test_features =  df_test[features].values.reshape(-1,len(features))
    #split it into a training and a test set (60% to train)
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_label, test_size=0.3)
    rf.fit(X_train, y_train)
    #now find the accuracy on the test set
    diff = (np.absolute(y_test - rf.predict(X_test).reshape((y_test.shape[0],1))))
    print('mean:   ', np.mean(diff))
    print('median: ', np.median(diff))
    print('std:    ', np.std(diff))
    age_predict = rf.predict(test_features)
    df.loc[ (df.Age.isnull()), 'Age_guess' ] = age_predict
    df.loc[ (df.Age.notnull()), 'Age_guess' ] = df_train['Age']
    return df


data = make_str_ints(data, 'Title')
data = make_str_ints(data, 'Fam_type')



features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Sex', 'Title']
rf = RandomForestRegressor(n_estimators=1500,max_features=3, max_depth=1500, min_samples_leaf=5)
data=calc_age(data, rf, features)


def make_age_bins(df, rf, bins, features, col):
    df_train = df.loc[df[col].notnull()]
    df_test  = df.loc[df[col].isnull()]
    labels = range(0,len(bins)-1)
    df_train['Age_cat'] = pd.cut(df_train[col], bins, labels=labels)
    df_test['Age_cat'] =  pd.cut(df_test[col], bins, labels=labels)
    train_features = df_train[features].values.reshape(-1,len(features))
    train_label    = df_train['Age_cat'].astype('int8').reshape(-1,1)
    test_features = df_test[features].values.reshape(-1,len(features))
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_label, test_size=0.3)
    rf.fit(X_train, y_train)
    #now find the accuracy on the test set
    print(float(np.absolute(y_test.reshape(y_test.shape[0],) - np.rint(rf.predict(X_test)).astype(np.int64)).sum())/float(y_test.shape[0]))
    print(np.count_nonzero((np.rint(rf.predict(X_test)).astype(np.int64)-y_test.reshape(y_test.shape[0],)))/float(y_test.shape[0]))
    #and generate data for missing ones
    age_class_predict = np.rint(rf.predict(test_features)).astype(np.int64)
    df.loc[ (df.Age.isnull()), 'Age_cat' ] = age_class_predict
    df.loc[ (df.Age.notnull()), 'Age_cat' ] = df_train['Age_cat']
    df['Young']  = 0
    df.loc[(df['Age_cat'] == 0), 'Young']  = 1
    df['Mid-life']  = 0
    df.loc[(df['Age_cat'] == 1), 'Mid-life']  = 1
    df['Old']  = 0
    df.loc[(df['Age_cat'] == 2), 'Old']  = 1
    return df


rf = RandomForestRegressor(min_samples_split=2, n_estimators=250,max_features=2, max_depth=501, min_samples_leaf=5)

#passenger class, the number of siblings, parents and the fare seem to be quite good indicators
features = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Fam_type', 'Fam_size', 'Mr','Mrs','Master','Miss','Military', 'Dr', 'Rev', 'Arist_F', 'Arist_M']

#young children are most likely to survive, then it drops of sharply
#lets take kids (0-18) adults (18-55) and oldies (55+) 
bins = [0, 18,55, 81]

data =  make_age_bins(data, rf, bins, features, 'Age')



def make_cabin_features(df, features, rf):
    df['N_Cabins']=df.Cabin.str.count(' ')+1.0
    df_train = df.loc[df['Cabin'].notnull()]
    df_test = df.loc[df['Cabin'].isnull()]
    train_features = df_train[features].values.reshape(-1,len(features))
    train_label    = df_train['N_Cabins'].reshape(-1,1)
    test_features = df_test[features].values.reshape(-1,len(features))
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_label, test_size=0.2)
    rf.fit(X_train, y_train)
    #now find the accuracy on the test set
    print(float(np.absolute(y_test.reshape(y_test.shape[0],) - np.rint(rf.predict(X_test))).sum())/float(y_test.shape[0]))
    print(np.count_nonzero((np.rint(rf.predict(X_test))-y_test.reshape(y_test.shape[0],)))/float(y_test.shape[0]))
    age_class_predict = np.rint(rf.predict(test_features))
    df.loc[ (df.Cabin.isnull()), 'N_Cabins' ] = age_class_predict
    df.loc[ (df.Cabin.notnull()), 'N_Cabins' ] = df_train['N_Cabins']
    df['Cabin_letter'] = df.Cabin.astype(str).str[0]
    level = df.Cabin_letter.unique()
    for n in level:
        df[n]  = 0
        df.loc[(df['Cabin_letter'] == n), n] = 1
    return df



data = make_cabin_features(data, features, rf)


#Drop the features we do not need

data.drop('Age', 1, inplace=True)
data.drop('Embarked', 1, inplace=True)
data.drop('Cabin', 1, inplace=True)
data.drop('Name', 1, inplace=True)
data.drop('PassengerId', 1, inplace=True)
data.drop('Ticket', 1, inplace=True)
data.drop('Title', 1, inplace=True)
data.drop('Age_cat', 1, inplace=True)
data.drop('Cabin_letter', 1, inplace=True)


#all the remaining columns are features
features = data.columns.tolist()

#scale the features (so that age and fare go from -1 to 1 etc.)
std_scale = preprocessing.StandardScaler().fit(data[features])
data[features]  = std_scale.transform(data[features])#


#and now we separate it back out to our training and test data
train_data = data[features][0:891]
train_data['Survived'] = labels_survived
test_data = data[features][891:]


#now to train it
def train_model(df, df_test, rf, features, label, split=0.3):
    """function to learn from our data, print an estimate of the accuracy and predictions outputted"""
    #make data into something that can be read by the random forest function
    train_features = df[features].values.reshape(-1,len(features))
    train_label    = df[label].values.reshape(-1,)
    #split it into a training and a test set (split% to test)
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_label, test_size=split)
    #we will use the random forest classifer to train it
    rf.fit(X_train, y_train)
    #now find the accuracy on the test set
    print('Percent accuracy: ', 100.0*(1.0 - float(np.absolute(y_test - rf.predict(X_test)).sum())/float(y_test.shape[0])))
    #now predict from the test set
    return rf.predict(df_test[features])


rf = RandomForestClassifier(n_estimators=1500, max_features=4,max_depth=501, min_samples_leaf=3, random_state=2)
results = train_model(train_data, test_data, rf, features, 'Survived', 0.2)



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


            

def search_params(df, gs, report, features, label, param_grid):
    #make the data readible
    X_train = df[features].values.reshape(-1,len(features))
    y_train = df[label].values.reshape(-1,)
    start = time()
    gs.fit(X_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(gs.cv_results_['params'])))
    report(gs.cv_results_)



    
# params to test
param_grid = {
    "max_features": [4,16,31],
    "min_samples_leaf": [2, 3, 4]
}

rf = RandomForestClassifier(n_estimators=1500, max_depth=501, random_state=1)
gs = GridSearchCV(rf, param_grid=param_grid,scoring='accuracy')

search_params(train_data, gs, report, features, 'Survived', param_grid)

rf = RandomForestClassifier(n_estimators=1500, max_features=16,max_depth=501, min_samples_leaf=4)
res = train_model(train_data, test_data, rf, features, 'Survived', 0.2)


#random forest is random (shock horror), so lets run it 100 times and average the results
n_loop = 100
res_sum = res
for i in range(n_loop-1):
    res_sum += train_model(train_data, test_data, rf, features, 'Survived', 0.2)



res_avg = np.rint(res_sum/float(n_loop)).astype(int)




#and show the important features
importance = rf.feature_importances_
importance = pd.DataFrame(importance, index=train_data[features].columns, columns=["Importance"])
importance.sort_values(by=['Importance'], ascending=False, inplace=True)
print(importance)

#and select the important ones to make a reduced data set
model = SelectFromModel(rf, prefit=True)
train_data_reduced = pd.DataFrame()
train_data_reduced = pd.DataFrame(model.transform(train_data[features]))
train_data_reduced['Survived'] = train_data['Survived']
test_data_reduced = pd.DataFrame()
test_data_reduced = pd.DataFrame(model.transform(test_data[features]))

features_reduced = range(train_data_reduced.shape[1]-1)


#search for the best parameters again
param_grid = {
    "max_features": [2,3,4,8],
    "min_samples_leaf": [2, 3, 4]
}
rf = RandomForestClassifier(n_estimators=1500, max_depth=501, random_state=1)
gs = GridSearchCV(rf, param_grid=param_grid,scoring='accuracy')

search_params(train_data_reduced, gs, report, features_reduced, 'Survived', param_grid)


rf = RandomForestClassifier(n_estimators=1500, max_features=8,max_depth=501, min_samples_leaf=3, random_state=2)
res_reduced = train_model(train_data_reduced, test_data_reduced, rf, features_reduced, 'Survived', 0.2)

n_loop = 100
res_reduced_sum = res_reduced
for i in range(n_loop-1):
    res_reduced_sum += train_model(train_data_reduced, test_data_reduced, rf, features_reduced, 'Survived', 0.2)
    print('loops done: ', i+2, '    ', 100.0*float(i+2)/float(n_loop),'%')


res_reduced_avg = np.rint(res_reduced_sum/float(n_loop)).astype(int)

#this actually turns out to be worse, so we output the normal result



def output_result(res, out_file):
    """output the results on the test set as a csv"""
    df_out = pd.DataFrame()
    df_out['PassengerId'] = range(892,892+len(results))
    df_out['Survived'] = res
    df_out[['PassengerId', 'Survived']].to_csv(out_file,  index = False)



out_file = 'prediction_Dennison_RF.csv'
output_result(res_avg, out_file)




#We can also cross-validate it
kf = KFold(df_train.shape[0], n_folds=5, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

def cross_val(df, kf, cv, rf, features, label):
    """Cross validate the training model"""
    predictions = cross_validation.cross_val_predict(rf, df[features],df[label],cv=kf)
    predictions = pd.Series(predictions)
    scores = cross_val_score(rf, df[features], df[label],scoring='f1', cv=kf)
    # Take the mean of the scores (because we have one for each fold)
    print(scores.mean())



cross_val(train_data, kf, cv, rf, features, 'Survived')
#This gives am accuracy lower than above (about 0.73), still not sure as to why
