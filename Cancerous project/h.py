import pandas as pd

dataset = pd.read_csv('data.csv')
print(dataset)

#split the X & Y variable
X = dataset.iloc[:, [0,2,3,4]].values
Y = dataset.iloc[:,1].values

#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train = pd.get_dummies(0,1)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting RF to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

#Predicting the Test set results
Y_pred = classifier.predict(X_test)

Y_pred = classifier.predict([[842302,17.99,10.38,122.8]])
print('Prediction is: ')
print(Y_pred)


#m= malegnant
#b= benign
