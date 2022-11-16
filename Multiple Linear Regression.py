df = pd.read_csv('bengaluru_house_prices.csv')
df.head()

df = df.dropna()

df = df[['size', 'total_sqft', 'bath', 'balcony', 'price']]
df.dropna()
X = df[['balcony', 'bath']]
Y = df[['price']]

# Splitting the data into test and Train usinglibraryrarie train,test,split
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
