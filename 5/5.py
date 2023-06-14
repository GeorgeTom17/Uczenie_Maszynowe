import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare", #cechy
]


def preprocess(data):
    data["Sex"] = data["Sex"].apply(
        lambda x: 0 if x in ["male"] else 1)
    data = data.applymap(np.nan_to_num)  # przygotowanie danych
    return data


dataset_filename = "titanic.tsv"

data = pd.read_csv(dataset_filename, header=0, sep="\t")
columns = data.columns[1:]
data = data[FEATURES + ["Survived"]]
data = preprocess(data)

data_train, data_test = train_test_split(data, test_size=0.2) #podział

y_train = pd.DataFrame(data_train["Survived"])

x_train = pd.DataFrame(data_train[FEATURES])
model = LogisticRegression()
model.fit(x_train, np.ravel(y_train))

y_expected = pd.DataFrame(data_test["Survived"])
x_test = pd.DataFrame(data_test[FEATURES])
y_predicted = model.predict(x_test)


################################## ewaluacja

precision, recall, fscore, support = precision_recall_fscore_support(
    y_expected, y_predicted, average="micro"
)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-score: {fscore}")

score = model.score(x_test, y_expected)

print(f"Model score: {score}")