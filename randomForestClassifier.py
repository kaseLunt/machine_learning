# loads the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# load the car dataset
df = pd.read_csv('car_evaluation.csv')

# check for null values
print(df.isnull().sum())

# Encode the data
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

x = df.iloc[:, :-1]

y = df.iloc[:, -1]


x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.25, random_state=0)

rfc = RandomForestClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)

confusion_matrix = confusion_matrix(y_test, pred)
print(confusion_matrix)