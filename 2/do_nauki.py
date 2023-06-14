import csv
import matplotlib as plt


with open("data2.csv", encoding="utf-8") as data_file:
    reader = csv.reader(data_file)
    for row in reader:
        print(row)