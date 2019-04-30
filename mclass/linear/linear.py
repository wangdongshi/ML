# House price forecast problem
# Use linear regression model

import pandas as pd
from io import StringIO
from sklearn import linear_model
import matplotlib.pyplot as plt

csv_data = "square_feet,price\n150,6450\n200,7450\n250,8450\n300,9450\n350,11450\n400,15450\n600,18450\n"

df = pd.read_csv(StringIO(csv_data))
print(df)
regr = linear_model.LinearRegression()

regr.fit(df['square_feet'].values.reshape(-1,1), df['price'])
a, b = regr.coef_, regr.intercept_
area = [[238.5]]
result = regr.predict(area)[0]

# print result
print()
print('a = %.3f' % a)
print('b = %.3f' % b)
print('area = 238.5, price = %f' %result)

# draw
plt.scatter(df['square_feet'], df['price'], color='blue')
plt.plot(df['square_feet'], regr.predict(df['square_feet'].values.reshape(-1,1)), color='red', linewidth=4)
plt.show()

