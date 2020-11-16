import spacy
from collections import Counter
from collections import defaultdict
from pyinflect import getAllInflections


filePath = list()
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt")
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\europarl.txt")
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\wikipedia.txt")

fileEncoding = list()
fileEncoding.append("ISO-8859-1")
fileEncoding.append("UTF-8")
fileEncoding.append("UTF-8")

# Trying out NLP toolkit

nlp = spacy.load('en_core_web_sm')
doc = nlp("Save the planet, says David. David lives in London and works for Planet Earth")


# html = spacy.displacy.render(doc, style='dep', page=True)
# spacy.displacy.serve(doc, style="dep")

# result = nlp("Save the planet, says David. David lives in London and works for Planet Earth")
# spacy.displacy.render(result, style = 'dep', jupyter = True)


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

    def frequentNouns(self):
        self.nounCounter = Counter()
        for iWord in self.corpusMostCommonWords:
            result = nlp(iWord[0])
            for token in result:
                if token.pos_ == "PROPN":
                    self.nounCounter[token.text] = iWord[1]

    def frequentLocations(self):
        self.locCounter = Counter()
        for iWord in self.corpusMostCommonWords:
            doc = nlp(iWord[0])
            for entity in doc.ents:
                if entity.label_ == "GPE":
                    self.locCounter[entity.text] = iWord[1]

    def frequentDrinks(self, filePath, encoding, drinkInflectionSet):
        self.drinkCounter = Counter()
        fileHandler = open(filePath, encoding=encoding)
        with fileHandler:
            for line in fileHandler:
                flagDrink = 0
                iDrink = 0
                while flagDrink == 0 and iDrink < len(drinkInflectionSet):
                    if drinkInflectionSet[iDrink] in line:
                        flagDrink = 1
                        result = nlp(line)
                        for token in result:
                            if token.pos_ == "PROPN":
                                self.drinkCounter[token.text] += 1
                    iDrink += 1
        fileHandler.close()


    def most_common_noun_loc_drink(self, numOfMostCommonWords, problem):
        if problem == 'Noun':
            print(Counter(self.nounCounter).most_common(numOfMostCommonWords))
        elif problem == 'Loc':
            print(Counter(self.locCounter).most_common(numOfMostCommonWords))
        elif problem == 'Drink':
            print(Counter(self.drinkCounter).most_common(numOfMostCommonWords))

filePath = list()
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt")
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\europarl.txt")
filePath.append("D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\wikipedia.txt")

fileEncoding = list()
fileEncoding.append("ISO-8859-1")
fileEncoding.append("UTF-8")
fileEncoding.append("UTF-8")

books = Corpus(filePath[0], fileEncoding[0])
euroParl = Corpus(filePath[1], fileEncoding[1])
wikipedia = Corpus(filePath[2], fileEncoding[2])

numOfMostCommonWords = 1000
books.most_common_word(numOfMostCommonWords, 'Y')
euroParl.most_common_word(numOfMostCommonWords, 'Y')
wikipedia.most_common_word(numOfMostCommonWords, 'Y')

drinkInflections = getAllInflections('drink')
drinkInflectionsList = list(drinkInflections.values())
drinkInflectionsSet = list()
for iList in range(len(drinkInflectionsList)):
    for jList in range(len(drinkInflectionsList[iList])):
        drinkInflectionsSet.append(drinkInflectionsList[iList][jList])
drinkInflectionsSet = list(set(drinkInflectionsSet))

books.frequentNouns()
books.most_common_noun_loc_drink(10, 'Noun')
wikipedia.frequentLocations()
wikipedia.most_common_noun_loc_drink(10, 'Loc')
euroParl.frequentDrinks(filePath[1], fileEncoding[1], drinkInflectionsSet)
euroParl.most_common_noun_loc_drink(50, 'Drink')


# result = nlp("Save the planet, says Attenborough. David lives is from Isleworth and works for Planet Earth")
# spacy.displacy.render(result, style = 'dep', jupyter = True)
# for token in result:
#   print(token.text, "\t|\t", token.pos_, "\t|\t",token.lemma_, "\t|\t",token.head, "\t|\t",token.dep_, "\t|\t")
#
# for entity in result.ents:
#   print(entity.text,"\t|\t",entity.label_,"\t|\t")