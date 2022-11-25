import pandas
import numpy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

train = pandas.read_csv("train.csv")
print(train)

#Récupération des colonnes pour compter les valeurs 
for column in train.columns:
    print(train[column].value_counts())
    print('Number of unique values: ',train[column].nunique())
    print('Number of null values: ',train[column].isna().sum())

# Selon l'age des personnes 

# Selon s'ils sont en cryptoSleep
# Selon leurs positions dans la cabine 