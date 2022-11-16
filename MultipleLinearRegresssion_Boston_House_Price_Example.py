import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('boston.csv')
df

df.describe()

df.info()

df.isnull().sum()

X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS']]
Y = df[['MEDV']]

# Splitting the data into test and Train using the librarie train,test,split
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=2)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)
print(regressor.score(x_train,
                      y_train))  ## gives how much we can predict the data
print(regressor.score(x_test,
                      y_test))  ## while test the data it gives the accuracy

plt.plot(x_train.values, pd.DataFrame(x_pred)[0].values)
