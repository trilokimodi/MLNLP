import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import numpy as np

class Corpus:

    def __init__(self, filePath, encoding):
        self.wordCounter = Counter()
        fileHandler = open(filePath, encoding=encoding)
        with fileHandler:
            for line in fileHandler:
                tokens = line.lower().split()
                self.wordCounter.update(tokens)
        fileHandler.close()

    def most_common_word(self, numOfMostCommonWords, needToStore):
        if needToStore == 'Y':
            self.corpusMostCommonWords = Counter(self.wordCounter).most_common(numOfMostCommonWords)
        elif needToStore == 'N':
            print(Counter(self.wordCounter).most_common(numOfMostCommonWords))

    def most_common_word_following(self, filePath, encoding, followingWhichWord, numOfMostCommonWords):
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

books = Corpus(filePath[0], fileEncoding[0])
euroParl = Corpus(filePath[1], fileEncoding[1])
wikipedia = Corpus(filePath[2], fileEncoding[2])

books.most_common_word(numOfMostCommonWords, 'N')
euroParl.most_common_word(numOfMostCommonWords, 'N')
wikipedia.most_common_word(numOfMostCommonWords, 'N')

books.most_common_word_following(filePath[0], fileEncoding[0], 'red', numOfMostCommonWords)
euroParl.most_common_word_following(filePath[1], fileEncoding[1], 'red', numOfMostCommonWords)
wikipedia.most_common_word_following(filePath[2], fileEncoding[2], 'red', numOfMostCommonWords)

numOfMostCommonWords = 100

books.most_common_word(numOfMostCommonWords, 'Y')
euroParl.most_common_word(numOfMostCommonWords, 'Y')
wikipedia.most_common_word(numOfMostCommonWords, 'Y')

plotFrame = np.zeros((4, numOfMostCommonWords))
plotFrame[0] = np.arange(1, 101, 1)
for iWord in range(numOfMostCommonWords):
    plotFrame[1][iWord] = books.corpusMostCommonWords[iWord][1]
    plotFrame[2][iWord] = euroParl.corpusMostCommonWords[iWord][1]
    plotFrame[3][iWord] = wikipedia.corpusMostCommonWords[iWord][1]

plt.plot(plotFrame[0], plotFrame[1], label="Books")
plt.plot(plotFrame[0], plotFrame[2], label="EuroParl")
plt.plot(plotFrame[0], plotFrame[3], label="Wikipedia")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.legend()
plt.title("Rank-frequency plot")
plt.show()

plt.loglog(plotFrame[0], plotFrame[1], label="Books")
plt.loglog(plotFrame[0], plotFrame[2], label="EuroParl")
plt.loglog(plotFrame[0], plotFrame[3], label="Wikipedia")
plt.xlabel("Log(Rank)")
plt.ylabel("Log(Frequency)")
plt.title("Logged rank-frequency plot")
plt.legend()
plt.show()

# Comparing corpora

numOfMostCommonWords = 500
books.most_common_word(numOfMostCommonWords, 'Y')
euroParl.most_common_word(numOfMostCommonWords, 'Y')
wikipedia.most_common_word(numOfMostCommonWords, 'Y')

wordsFrame = list()
for iFrame in range(3):
    wordsFrame.append(list())

for iWord in range(numOfMostCommonWords):
    wordsFrame[0].append(books.corpusMostCommonWords[iWord][0])
    wordsFrame[1].append(euroParl.corpusMostCommonWords[iWord][0])
    wordsFrame[2].append(wikipedia.corpusMostCommonWords[iWord][0])

# Comparing books and europarl
# typicalWords = set(wordsFrame[0]) & set(wordsFrame[1])  # Not necessarily in decreasing order and hence enumeration is done.

typicalWordsFrame = list()
for iWord in range(numOfMostCommonWords):
    for jWord in range(numOfMostCommonWords):
        if wordsFrame[0][iWord] == wordsFrame[1][jWord]:
            iWordFreq = books.corpusMostCommonWords[iWord][1]
            jWordFreq = euroParl.corpusMostCommonWords[iWord][1]
            typicalWordsFrame.append((wordsFrame[0][iWord], iWordFreq, jWordFreq, iWordFreq + jWordFreq))

typicalWordsFrame = sorted(typicalWordsFrame, key=lambda tup: tup[3], reverse=True)

plotTypicalWords = np.zeros((3, len(typicalWordsFrame)))
plotTypicalWords[0] = np.arange(1, len(typicalWordsFrame) + 1, 1)
for iWord in range(len(typicalWordsFrame)):
    plotTypicalWords[1][iWord] = typicalWordsFrame[iWord][1]
    plotTypicalWords[2][iWord] = typicalWordsFrame[iWord][2]

plt.plot(plotTypicalWords[0], plotTypicalWords[1], label="Books")
plt.plot(plotTypicalWords[0], plotTypicalWords[2], label="EuroParl")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.legend()
plt.title("Rank-frequency plot for typical words")
plt.show()

# plotting top 10 words

plt.plot([typicalWordsFrame[iWord][0] for iWord in range(10)], [typicalWordsFrame[iWord][3] / 2 for iWord in range(10)])
plt.xlabel("Word")
plt.ylabel("Average frequency")
plt.legend()
plt.title("Word vs average frequency plot for top 10 typical words")
plt.show()

# Processing text for Machine Learning

maxVocabSize = 1000
numOfMostCommonWords = maxVocabSize - 2  # 2 for padding and out-of-vocabulary
books.most_common_word(numOfMostCommonWords, 'Y')
euroParl.most_common_word(numOfMostCommonWords, 'Y')
wikipedia.most_common_word(numOfMostCommonWords, 'Y')


class IntegerVocabulary:
    def __init__(self, mostCommonWords, maxVocabSize):
        self.integerVocab = list()
        wordCode = 1
        for iWord in range(maxVocabSize - 2):
            self.integerVocab.append((mostCommonWords[iWord][0], wordCode))
            wordCode += 1
        self.integerVocab.append(("Padding", wordCode))
        self.integerVocab.append(("OTHER", wordCode + 1))

    def build_vocabulary(self, filePath, encoding, maxBatchSize):
        codedWords = [iWord[0] for iWord in self.integerVocab]
        maxLines = 0
        fileHandler = open(filePath, encoding=encoding)
        with fileHandler:
            for line in fileHandler:
                maxLines += 1
        fileHandler.close()
        self.batches = list()
        lineCount = 0
        iBatch = 0
        lineIteration = 0
        tokens = list()
        for iList in range(maxBatchSize):
            tokens.append(list())
        fileHandler = open(filePath, encoding=encoding)
        with fileHandler:
            for line in fileHandler:
                tokens[lineCount] = line.lower().split().copy()
                if lineCount == maxBatchSize - 1 or lineIteration == maxLines - 1:
                    lineCount = 0
                    maxLength = max([len(tokens[iList]) for iList in range(maxBatchSize)])
                    if lineIteration == maxLines - 1:
                        maxBatchSize = lineCount
                    self.batches.append(np.zeros((maxBatchSize, maxLength)))
                    iList = 0
                    while iList < maxBatchSize:
                        for iWord in range(maxLength):
                            if iWord < len(tokens[iList]):
                                if tokens[iList][iWord] in codedWords:
                                    self.batches[iBatch][iList][iWord] = codedWords.index(tokens[iList][iWord]) + 1
                                else:
                                    self.batches[iBatch][iList][iWord] = self.integerVocab[len(self.integerVocab) - 1][1]
                            else:
                                self.batches[iBatch][iList][iWord] = self.integerVocab[len(self.integerVocab) - 2][1]
                        iList += 1
                    iBatch += 1
                lineCount += 1
                lineIteration += 1

        fileHandler.close()

booksIV = IntegerVocabulary(books.corpusMostCommonWords, maxVocabSize)
euroParlIV = IntegerVocabulary(euroParl.corpusMostCommonWords, maxVocabSize)
wikipediaIV = IntegerVocabulary(wikipedia.corpusMostCommonWords, maxVocabSize)

booksIV.build_vocabulary(filePath[0], fileEncoding[0], 8)
euroParlIV.build_vocabulary(filePath[1], fileEncoding[1], 8)
wikipediaIV.build_vocabulary(filePath[2], fileEncoding[2], 8)
