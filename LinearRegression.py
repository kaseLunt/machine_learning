# load the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the dataset
df = pd.read_csv('winequality-white.csv')

# subset the data containing all columns except target
x = df[[col for col in df.columns if col != 'quality']]

# subset the data containing the labels
y = df[['quality']]

# splits the data into 75 % training and 25 % test sets. Set random_state = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# initializes and run the Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)


# calculate R2 and Mean Square Error(MSE)
r2,mse= r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)

print(r2,mse)