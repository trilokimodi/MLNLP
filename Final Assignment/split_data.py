import pandas
import math
import numpy as np
import os
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt

def split_data(trainDataSize, filePath):
    whole_data = pandas.read_csv(filePath)
    index_0 = whole_data[whole_data['is_humor'] == 0].index
    index_1 = whole_data[whole_data['is_humor'] == 1].index
    ratio_0 = len(index_0) / len(whole_data)
    train_0_freq = math.floor(ratio_0 * trainDataSize)
    train_1_freq = trainDataSize - train_0_freq
    np.random.seed(7)
    train_0_index = np.random.choice(index_0, size=train_0_freq, replace=False)
    np.random.seed(7)
    train_1_index = np.random.choice(index_1, size=train_1_freq, replace=False)
    test_data = whole_data.copy()
    train_index = np.concatenate((train_0_index, train_1_index))
    test_data.drop(train_index, inplace=True)
    train_data = whole_data.copy()
    test_index = [i for i in range(len(whole_data)) if i not in train_index]
    train_data.drop(test_index, inplace=True)
    return train_data, test_data

def tokenize(document):
    doc = word_tokenize(document)
    tokenizeDoc = ' '.join([str(token.lower()) for token in doc])
    return tokenizeDoc

def tokenize_docs(train_data, test_data):
    train_data["text"] = train_data["text"].apply(tokenize)
    test_data["text"] = test_data["text"].apply(tokenize)
    return train_data, test_data

def write_csv(train_data, test_data, trainDataName, testDataName):
    filePath = os.getcwd() + trainDataName
    train_data.to_csv(filePath, index=False)
    filePath = os.getcwd() + testDataName
    test_data.to_csv(filePath, index=False)

def regression_data(train_data, test_data): # Data where is_humor == 1
    train_index_0 = train_data[train_data['is_humor'] == 0].index
    test_index_0 = test_data[test_data['is_humor'] == 0].index
    train_data.drop(train_index_0, inplace=True)
    test_data.drop(test_index_0, inplace=True)
    return train_data, test_data

def maxLength(train_data, test_data):
    xList = train_data.text.tolist()
    xList = xList + test_data.text.tolist()
    lenDocs = list()
    for idoc in range(len(xList)):
        lenDocs.append(len(xList[idoc]))
    lenDocs = np.array(lenDocs)
    print(np.max(lenDocs))
    print(np.mean(lenDocs))

def class_rater(train_data, test_data):
    rating = train_data.humor_rating.tolist()
    rating = rating + test_data.humor_rating.tolist()
    rating_class = np.zeros(len(rating), int)
    for i in range(len(rating)):
        if 0 <= rating[i] < 1:
            rating_class[i] = 1
        elif 1 <= rating[i] < 2:
            rating_class[i] = 2
        elif 2 <= rating[i] < 3:
            rating_class[i] = 3
        elif 3 <= rating[i] < 4:
            rating_class[i] = 4
        elif 4 <= rating[i] < 5:
            rating_class[i] = 5
    plt.hist([rating_class[i] for i in rating_class if i > 0], range=(1,5))
    plt.savefig(os.getcwd() + "\\rating_class.png")



filePath = os.getcwd() + "\\whole.csv"
train, test = split_data(7000, filePath)
train, test = tokenize_docs(train, test)
trainDataName = "\\train.csv"
testDataName = "\\test.csv"
write_csv(train, test, trainDataName, testDataName)
maxLength(train, test)
class_rater(train, test)
train_reg, test_reg = regression_data(train, test)
write_csv(train_reg, test_reg, "\\train_reg.csv", "\\test_reg.csv")
