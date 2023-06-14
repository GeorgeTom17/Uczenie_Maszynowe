import numpy as np
import pandas as pd

alldata = pd.read_csv(
    "titanic.tsv",
    header=0,
    sep="\t",
    usecols=[
        "Survived",
        "PassengerId",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked"
    ],
)

alldata["Embarked"] = alldata["Embarked"].apply(
    lambda x: 0 if x in [np.nan] else x
)
alldata["Sex"] = alldata["Sex"].apply(
    lambda x: 0 if x in ["male"] else 1
)
alldata = pd.get_dummies(alldata, columns=["Embarked"])
alldata["Cabin"] = alldata["Cabin"].apply(
    lambda x: 0 if x in [np.nan] else x.translate({ord(i): None for i in 'ABCDEFGHIJK'})
)
alldata["Ticket"] = alldata["Ticket"].apply(
    lambda x: 0 if x in ["LINE"] else x.translate({ord(i): None for i in '[ABCDEFGHIJKLMNOPQRSTUVWXYZ/.]'})
)
print("Liczba rekordów przed usunięciem NaN:", len(alldata))

alldata = alldata.dropna()

print("Liczba rekordów po usunięciu NaN:", len(alldata)) #w celu pozbycia się NaN age, ponieważ nie można tego zamienić na 0
print(alldata)
