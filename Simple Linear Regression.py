from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

dataset = pd.read_csv('score.csv')
dataset

## input for this data is hours and the output for this data is scores
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1:]

# Splitting the data into test and Train using the library train,test,split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 1/3)

print('x_train data: ','\n',x_train)
print('\n')
print('x_test data: ','\n',x_test)
print('\n')
print('y_train data: ','\n',y_train)
print('\n')
print('y_test data: ','\n',y_test)
print('\n')

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

plt.scatter(x_train,y_train,color = 'green')
plt.plot(x_train.values,pd.DataFrame(x_pred)[0].values)
plt.show()