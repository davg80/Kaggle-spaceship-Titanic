import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
import copy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
#diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# print(diabetes_X.shape)
# print('----------------------------------------------------------------')
# print(diabetes_y.shape)
# print('----------------------------------------------------------------')
# exit()

DATASET_LOCATION = './train.csv'
originDataset: pandas.DataFrame = pandas.read_csv(DATASET_LOCATION)
dataset = copy.deepcopy(originDataset)

# remove null values
# print(dataset.isnull().sum())
dataset.dropna(axis=0, inplace=True)

# get deck
deck: pandas.DataFrame = dataset[dataset['Cabin'].str.endswith('P')]
# get side
side: pandas.DataFrame = dataset[dataset['Cabin'].str.endswith('S')]
figure0: plt.figure = plt.figure(0)

# save pie chart to file
plt.pie([len(deck), len(side)])
plt.legend(['Deck', 'Side'])
plt.savefig('deck_and_side.png')
plt.close()
figure1: plt.figure = plt.figure(1)

cryoYY = deck.query(expr="Transported == True and CryoSleep == True")
cryoNN = deck.query(expr="Transported == False and CryoSleep == False")
cryoNY = deck.query(expr="Transported == False and CryoSleep == True")
cryoYN = deck.query(expr="Transported == True and CryoSleep == False")

for i, frame in enumerate([cryoYY, cryoNN, cryoYN, cryoNY]):
    plt.hist(frame['CryoSleep'], alpha=0.6, lw=3)

#plt.xlim(0, 1)
#plt.ylim(0, 10)
plt.legend(["Transported == True and CryoSleep == True", "Transported == False and CryoSleep == False",
            "Transported == False and CryoSleep == True", "Transported == True and CryoSleep == False"])
plt.savefig('transported_cryosleep.png')
plt.close()

# train data
dataset.drop('HomePlanet', inplace=True, axis=1)
dataset.drop('Cabin', inplace=True, axis=1)
dataset.drop('Destination', inplace=True, axis=1)
dataset.drop('Name', inplace=True, axis=1)
figure2: plt.figure = plt.figure(2)
reg: LinearRegression = LinearRegression()
X: np.ndarray = np.array(dataset)
y: np.ndarray = np.array(dataset['Age'].values.tolist())
X.transpose()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model: LinearRegression = reg.fit(X, y)
pred: any = reg.predict(X)

plt.scatter(x_test[:, 0], y_test, color="blue", linewidth=3)
#plt.plot(x_test[:, 0], y_test, color="blue", linewidth=3)
#plt.savefig('train_data.png')
plt.show()
#plt.close()
