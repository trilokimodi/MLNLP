import spacy
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict

def most_common_words(filePath, encoding, numOfMostCommonWords, returnFrequency):
    wordCounter = Counter()
    fileHandler = open(filePath, encoding=encoding)
    with fileHandler:
        for line in fileHandler:
            tokens = line.lower().split()
            wordCounter.update(tokens)

    fileHandler.close()
    if returnFrequency == 'N':
        print(Counter(wordCounter).most_common(numOfMostCommonWords))
        wordCounter.clear()
    elif returnFrequency == 'Y':
        returnWordCounter = Counter(wordCounter).most_common(numOfMostCommonWords)
        wordCounter.clear()
        return returnWordCounter

def most_common_word_following(filePath, encoding, followingWhichWord, numOfMostCommonWords):
    wordPairCounter = defaultdict(Counter)
    fileHandler = open(filePath, encoding=encoding)
    with fileHandler:
        for line in fileHandler:
            tokens = line.lower().split()
            for t1, t2 in zip(tokens, tokens[1:]):
                wordPairCounter[t1][t2] += 1
    
    print(Counter(wordPairCounter[followingWhichWord]).most_common(numOfMostCommonWords))
    wordPairCounter.clear()
    fileHandler.close()


filePath = list()
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt")
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\europarl.txt")
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\wikipedia.txt")

fileEncoding = list()
fileEncoding.append("ISO-8859-1")
fileEncoding.append("UTF-8")
fileEncoding.append("UTF-8")

fileDetails = dict(zip(filePath, fileEncoding))

numOfMostCommonWords = 10
wordFrequency = list()
# most_common_words(filePath[0], fileEncoding[0], numOfMostCommonWords, 'N')
# most_common_word_following(filePath[0], fileEncoding[0], 'red', numOfMostCommonWords)
wordFrequency.append(most_common_words(filePath[0], fileEncoding[0], 100, 'Y'))
y = [wordFrequency[x][1] for x in range(len(wordFrequency))]
plt.plot([wordFrequency[x][1] for x in range(len(wordFrequency))])
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Rank-frequency plot")
plt.show()

plt.loglog([wordFrequency[x][1] for x in range(len(wordFrequency))])
plt.xlabel("Log(Rank)")
plt.ylabel("Log(Frequency)")
plt.title("Logged rank-frequency plot")
plt.show()

# Comparing corpora
commonWordsBooks = [wordFrequency[x][0] for x in range(len(wordFrequency))]
wordFrequency = most_common_words(filePath[1], fileEncoding[1], 100, 'Y')
