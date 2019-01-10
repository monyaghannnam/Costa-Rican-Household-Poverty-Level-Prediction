
import pandas as pd
import numpy as np
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")



# since each column has numeric and string value  and i need just to convert string
mapping = {"yes": 1, "no": 0}

# Apply same operation to both train and test
for df in [train, test]:
    # Fill in the values with the correct mapping
    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)
    
from sklearn.preprocessing import LabelEncoder
train['idhogar'] = LabelEncoder().fit_transform(train['idhogar'])
test['idhogar'] = LabelEncoder().fit_transform(test['idhogar'])



#households where the family members do not all have the same target
unique_values = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
different_households = unique_values[unique_values != True]
train[train['idhogar'] == different_households.index[64]][['idhogar', 'parentesco1', 'Target']]


for each_household in different_households.index:
    
    #find the correct label
    true_target = int(train[(train['idhogar'] == each_household) & (train['parentesco1'] == 1.0)]['Target'])
    
    #assign the correct label for each member
    train.loc[train['idhogar'] == each_household, 'Target'] = true_target

unique_values = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
different_households = unique_values[unique_values != True]


#check if any column has missing values

train.columns[train.isna().any()]
# 5 column has missing values 1.v2a1  2.v18q1  3.rez_esc 4.meaneduc 5.SQBmeaned
#1.v2a1 - Monthly rent payment.
#2.v18q1 - number of tablets household owns.
#3.rez_esc - Years behind in school.
#4.meaneduc average years of education for adults (18+).
#5.education of adults (>=18) in the household

#handling missing value and replace it eith 0

train['v2a1']=train['v2a1'].fillna(0.0)
train['v18q1']=train['v18q1'].fillna(0.0)
train['rez_esc']=train['rez_esc'].fillna(0.0)
train['meaneduc']=train['meaneduc'].fillna(0.0)
train['SQBmeaned']=train['SQBmeaned'].fillna(0.0)

test['v2a1']=test['v2a1'].fillna(0.0)
test['v18q1']=test['v18q1'].fillna(0.0)
test['rez_esc']=test['rez_esc'].fillna(0.0)
test['meaneduc']=test['meaneduc'].fillna(0.0)
test['SQBmeaned']=test['SQBmeaned'].fillna(0.0)

################################################################
#feature selection and cross validation

###############################################################

# Splitting data into dependent and independent variable
# X is the independent variables matrix
x_train = train.iloc[:,1:-1]

# y is the dependent variable vector
y_train = train.iloc[:,142]

# Droping useless columns 
x_train.drop(['idhogar'], axis = 1, inplace = True)

x_test=test.iloc[:,1:142]
x_test.drop(['idhogar'], axis = 1, inplace = True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)

# run RandomForest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)


# run XGBoost model
#from xgboost import XGBClassifier
#classifier = XGBClassifier()
#classifier.fit(x_train, y_train)

#run LightGBM model
#from lightgbm import LGBMClassifier
#classifier = LGBMClassifier()
#classifier.fit(x_train, y_train)

#predictig the test set result
y_pred=classifier.predict(x_test)

###############confusion matrix####################
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train1 = sc_x.fit_transform(x_train1)
x_test1 = sc_x.transform(x_test1)



#Predecting the test set resutls
y_pred1 = classifier.predict(x_test1)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred1)

#accuracy
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 
accuracy(cm)
##################################################

#create asubmission file

y_id=test['Id']
sbmt=pd.DataFrame(({'Id':y_id, 'Target':y_pred}))
sbmt.to_csv('submission.csv',index=False)

#score
#RandomForest
#https://www.kaggle.com/monyaghannam/monya1
