import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import csv

forest = RandomForestRegressor(100)

data = pd.read_csv("internship_train.csv")

x = data.drop('target', axis=1)
y = data['target']

forest.fit(x, y)

data_test = pd.read_csv("internship_hidden_test.csv")
Y_predict = forest.predict(data_test)
Y_predict = pd.Series(Y_predict, name="target")
result = pd.merge(data_test, Y_predict)
with open("train_result.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(result)


