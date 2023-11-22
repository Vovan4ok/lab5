from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np


def task1(data_set):
    print("Statistical information:")
    print(data_set.describe(), "\n")
    print("Class distribution by ", data_set.groupby('Type').size(), "\n")
    #df = pd.DataFrame(data_set)
    #missing_values = df.isna()
    #print("DataFrame з пропущеними значеннями")
    #print(df)
    #print("\nКількість непорожніх значень у кожному стовпці:")
    #print(df.info())
    #print("\nЧи пропущено значення:")
    #print(missing_values)


def task2(data_set):
    color_wheel = {
        "1": "red",
        "2": "blue",
        "3": "green",
        "5": "grey",
        "6": "yellow",
        "7": "purple"
    }
    colors = data_set["Type"].map(lambda x: color_wheel.get(x))
    scatter_matrix(data_set, c=colors)
    plt.show()


def get_train_and_test_data(data_set):
    heart_size = data_set.iloc[:, :-1].values
    heart_class = data_set.iloc[:, 9].values
    train_size, test_size, train_class, test_class = train_test_split(heart_size, heart_class, test_size=0.2)
    return train_size, test_size, train_class, test_class


def get_new_size(train_size, test_size):
    scaler = StandardScaler()
    scaler.fit(train_size)
    train_size = scaler.transform(train_size)
    test_size = scaler.transform(test_size)
    return train_size, test_size


def task3_5(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set)
    classifier = RandomForestClassifier()
    classifier.fit(train_size, train_class)
    test_predict = classifier.predict(test_size)
    score_forest = classifier.score(test_size, test_class)
    print("Correct predictions proportion = ", score_forest, "\n")
    print(classification_report(test_class, test_predict))
    cm = confusion_matrix(test_class, test_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()


def task6(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set)
    train_size, test_size = get_new_size(train_size, test_size)
    classifier = RandomForestClassifier()
    classifier.fit(train_size, train_class)
    test_predict = classifier.predict(test_size)
    score_forest = classifier.score(test_size, test_class)
    print("Correct predictions proportion = ", score_forest, "\n")
    print(classification_report(test_class, test_predict))
    cm = confusion_matrix(test_class, test_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()


def task7(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set)
    train_size, test_size = get_new_size(train_size, test_size)
    classifier = RandomForestClassifier(n_estimators=250, max_depth=10, n_jobs=-1, min_samples_leaf=2, bootstrap=False)
    classifier.fit(train_size, train_class)
    test_predict = classifier.predict(test_size)
    score_forest = classifier.score(test_size, test_class)
    print("Correct predictions proportion = ", score_forest, "\n")
    #print(classification_report(test_class, test_predict))
    #cm = confusion_matrix(test_class, test_predict)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    #disp.plot()
    #plt.show()


def get_type_indexes(test_class, class_name):
    indexes = np.where(test_class == class_name)
    indexes = indexes[0]
    return indexes[:3]


def get_objects(test_size, test_class):
    unique_classes = np.unique(test_class)
    indexes = []
    for i in range(len(unique_classes)):
        indexes.append(get_type_indexes(test_class, unique_classes[i]))
    elements = []
    for i in range(len(indexes)):
        elements.append(test_size[indexes[i]])
    return elements


def euclide_distance(glass1, glass2):
    epsilon = 0
    for i in range(len(glass1)):
        epsilon += pow((glass1[i] - glass2[i]), 2)
    return sqrt(epsilon)


def find_closest(element, train_size):
    min_distance = 10000000000
    min_index = 0
    for i in range(len(train_size)):
        current_distance = euclide_distance(element, train_size[i])
        if current_distance < min_distance:
            min_distance = current_distance
            min_index = i
    return train_size[min_index]


def all_true(arr):
    flag = True
    for i in range(len(arr)):
        #print(arr[i])
        if arr[i] == False:
            flag = False
    return flag


def get_type(size, types, element):
    for i in range(len(size)):
        is_element = element == size[i]
        if all_true(is_element):
            return types[i]


def task10(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set)
    elements = get_objects(test_size, test_class)
    for i in range(len(elements)):
        help = elements[i]
        for j in range(len(help)):
            closest_element = find_closest(help[j], train_size)
            print("the neighbor to an element " + str(help[j]) + " is " + str(closest_element))
            type1 = get_type(test_size, test_class, help[j])
            type2 = get_type(train_size, train_class, closest_element)
            print("Their types " + type1 + " -- " + type2)


def main():
    data_set = pd.read_csv("Glass.csv")
    data_set["Type"] = [str(i) for i in data_set["Type"]]
    #task1(data_set)
    #task2(data_set)
    #task3_5(data_set)
    #task6(data_set)
    #task7(data_set)
    task10(data_set)


main()
