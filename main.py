import pandas
import numpy
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

train = pandas.read_csv("train.csv")
test = pandas.read_csv("train.csv")
# print(train)

# Rechercher si des valeurs sont incorrectes   
columns = list(train.columns)

def newList(name) -> str:
 return f"{name} : "  + str(len(train.isna().query(expr=f"{name} == True")))
 
for column in columns:
 print(newList(column))

 # Age moyenne
mean_age = round(train['Age'].mean())

# de modifier de façon importante les données
# inplace permet de modifier de façon permanente la colonne
print(train['Age'].head(10))
train.Age.fillna(mean_age, inplace=True)
test.Age.fillna(mean_age, inplace=True)
print(train['Age'].head(10))

#Mettre false à 0 et true à 1
train['Transported'].replace(False, 0, inplace=True)
test['Transported'].replace(False, 0, inplace=True)
train['Transported'].replace(True, 1, inplace=True)
test['Transported'].replace(True, 1, inplace=True)
#Mettre false à 0 et true à 1
train['CryoSleep'].replace(False, 0, inplace=True)
test['CryoSleep'].replace(False, 0, inplace=True)
train['CryoSleep'].replace(True, 1, inplace=True)
test['CryoSleep'].replace(True, 1, inplace=True)
#Mettre false à 0 et true à 1
train['VIP'].replace(False, 0, inplace=True)
test['VIP'].replace(False, 0, inplace=True)
train['VIP'].replace(True, 1, inplace=True)
test['VIP'].replace(True, 1, inplace=True)
#Mettre false à 0 et true à 1
# cabin = train['Cabin'].str.split('/')
# print(cabin)
# print(train['Transported'])


#Supprimer les colonnes qui ne seront pas utilisées
# contenant des valeurs textuelles ou ne semblant n'avoir pas d'intérêt pour l'analyse axis 1=column, 0=line
train.drop(['PassengerId','HomePlanet','Destination', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=True)
test.drop(['PassengerId','HomePlanet','Destination', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name'], axis=1, inplace=True)
# print(train.head())

#Nb_personnes_transportées des passagers transportés ou non
nb_total = len(train)
transported = len(train.query(expr="Transported == 1"))
no_transported = len(train.query(expr="Transported == 0"))
print(f"transporté : {transported}")
print(f"No_transporté : {no_transported}")

print(train.head())
print(type(train))
# Nombre de personne qui on fait de la cryosleep ou non
cryosleep = len(train.query(expr= "CryoSleep == 1.0"))
not_cryosleep = len(train.query(expr= "CryoSleep == 0.0"))
print(f"cryosleep : {cryosleep}")
print(f"not_cryosleep : {not_cryosleep}")

train = train.fillna(0.0).astype(int)
test = test.fillna(0.0).astype(int)
print(train)
#Regression Lineaire
vectors = ['Age', 'CryoSleep', 'VIP']
dependent = ['Transported']

x = train[vectors]
y = train[dependent]


x_test = test[vectors]

#Instancie LinearRegression
model = LinearRegression().fit(x, y)
# # Avec .fit(), vous calculez les valeurs optimales de l'entrée et la sortie existantes, x et y, comme arguments.
model.predict(x_test)
# #Étant donné un intervalle, les valeurs en dehors de l'intervalle sont écrêtées sur les bords de l'intervalle. Par exemple, si un intervalle de [0, 1] est spécifié, les valeurs inférieures à 0 deviennent 0 et les valeurs supérieures à 1 deviennent 1.
results = numpy.round(numpy.clip(model.predict(x_test),0, 1))

# # Regression linéaire
# train_score = model.score(x, y.ravel())
# print("score linear", train_score)
# #On ajoute la colonne results aux données de tests
test['Transported'] = results
print(test.head(100)) 

rf_model = RandomForestClassifier(n_estimators=50)
rf_model.fit(x, y)
predict = rf_model.predict(x)
results = list(predict)
print("rf_model", results)
test['Transported'] = results
print(test.head(100))
train_score = rf_model.score(x, y)
print("score forest", train_score)