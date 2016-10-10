#!/usr/bin/python

# count P(c), the frequency of each class
# for each w in/ not in  vocabulary , count P(w|c)
# for a new document doc, P(doc|c) = PIE(P(wi'|c)), wi' represent each word in document
# classify:calculate P(doc|c) * P(c) and compare each ci


from numpy import *
import datetime
import time


# use regular expression & change all letter into lowercase
def TextParse(text):
    import re
    # test for TexParse:
    # r'\W*': 100 runtimes, around 40% are at the error rate of 0.1
    # r'\[^a-z1-9]*': 100runtimes, <10% are at the error rate of 0.1
    TokenList = re.split(r'\W*', text)
    WordList = []
    for token in TokenList:
        if len(token) > 2:
            WordList.append(token.lower())

    return WordList


#CreateVocabulary
def CreateVocabulary(Doclist):
    Vocab = []
    for doc in Doclist:
        for word in doc:
            if (word not in Vocab):
                Vocab.append(word)
    return Vocab


def CreateBOWVec(vocabulary, doc):
    vector = zeros(len(vocabulary))
    for word in doc:
        if word in vocabulary:
            vector[vocabulary.index(word)] += 1
    #    else:
    #        print "The word %s is not in vocabulary" %word

    return vector



# get all the parameter needed to make a predction:
# 1. the two trained vector v[p(wi| c = 0)] & v[p(wi | c = 1)]
# 2. the p(c = 0) & p(c = 1)
def TrainNaiveBayes(TrainingMatrix, DocClass):

    ClassNum = int(max(DocClass)) + 1
    DocNum = len(DocClass)
    WordsNum = len(TrainingMatrix[0])
    Total = 0;


    # calculate the probability of each class P(ci)
    # P(ci) = (cnt(words in ci) + 1) / (cnt(total words) + cnt(class))
    # Use Laplace Smooth here

    Freq = ones(ClassNum)
    for i in range(DocNum):
        ci = DocClass[i]
        Freq[ci] += sum(TrainingMatrix[i])
    ProbCi = Freq / sum(Freq)

    # calculate the prior probability P(wj|ci)
    # in order to avoid 0
    # P(wj|ci) = (1 + cnt(wj in ci)) / (cnt(words in ci) + |V|)
    # |V| = the number of elements in training Vocabulary
    # Practically speaking, use the |V'| = the number of elements in Vocabulary performs better
    V = len(TrainingMatrix[0]);
    h = len(TrainingMatrix); w = len(TrainingMatrix[0]);
    for j in range(w):
        Flag = 0;
        for i in range(h):
            if TrainingMatrix[i][j] > 0:
                Flag = 1;
                break;
        if (Flag == 0):
            V -= 1;
    #print V


    ClassVector = []
    for ci in range(ClassNum):
        ClassVector.append((ones(WordsNum)))

    DocIndex = 0
    for DocVec in TrainingMatrix:
        ci = int(DocClass[DocIndex])
        ClassVector[ci] += DocVec
        DocIndex += 1

    ProbWC = []
    sub = len(ClassVector[0])
    for ci in range(ClassNum):
        ProbWC.append(ClassVector[ci]/(sum(ClassVector[ci]) - sub + V))




    #print >> fo, line
    #print >> fo, "The Probability Distribution of Class Ci:"
    #print >> fo, ProbCi
    #print >> fo, line

    #print >>fo, "The Probability Matrix P(wj|ci):"
    #print >>fo, ProbWC
    #print >>fo, line

    return log(ProbCi), log(ProbWC)



def NaiveBayesClassifier(pCi, pWC, vector):
    #p(c|w) = p(w|c) * p(c)/p(w)
    #compare sum(log(p(wj|ci))) + log(p(ci))
    maximum = 0
    prediction = -1
    ClassNum = len(pCi)
    for i in range(ClassNum):
        temp = sum(vector * pWC[i]) + pCi[i]
        if ((prediction == -1) or (temp > maximum)):
            maximum = temp;
            prediction = i;
    return prediction

def SpamTest():

    # get the document
    DocList = []; ClassList = []; IndexList = []; Emails = []
    ind = 0;
    for i in range(25):
        # add ham documents
        doc = open("../email/ham/%d.txt" % (i+1)).read()
        Emails.append(doc)
        doc = TextParse(doc)
        DocList.append(doc)
        ClassList.append(1)
        IndexList.append(ind)
        ind += 1

        # add spam documents
        doc = open("../email/spam/%d.txt" % (i+1)).read()
        Emails.append(doc)
        doc = TextParse(doc)
        DocList.append(doc)
        ClassList.append(0)
        IndexList.append(ind)
        ind += 1

    # Create VocabularyBag
    # Vocabulary = ['Chinese', 'love', 'great'.....] the set of words
    # VocabCount = [1, 2, 5, 4] the total count of words
    # VocabClasss[i][j]: the number of word wi in class cj


    Vocabulary = CreateVocabulary(DocList)
    # seperate the documents as training sets and test sets
    import random
    debug = 0;

    TestIndex = []; TestDoc = []; TestClass = [];
    for i in range(10):
        if (debug == 1):
            x = i
        else:
            x = random.randint(0,49)
            while x in TestIndex:
                x =random.randint(0,49)
        TestIndex.append(x)
        TestDoc.append(DocList[x])
        TestClass.append(ClassList[x])

    # create training set
    TrainIndex = []; TrainDoc = []; TrainClass = [];
    for i in IndexList:
        if i not in TestIndex:
            TrainIndex.append(i)
            TrainDoc.append(DocList[i])
            TrainClass.append(ClassList[i])

    # create training matrix
    TrainingMatrix = []
    for doc in TrainDoc:
        vec = CreateBOWVec(Vocabulary, doc)
        TrainingMatrix.append(vec)


    #Use NaiveBayes to train the data
    pCi, pWC = TrainNaiveBayes(TrainingMatrix, TrainClass)

    # use testset to evaluate the NaiveBayes method
    ErrorNum = 0; Ham2SpamNum = 0
    i = 0
    for doc in TestDoc:
        vec = CreateBOWVec(Vocabulary, doc)
        prediction = NaiveBayesClassifier(pCi, pWC, vec)

        if (prediction != TestClass[i]):
            ErrorNum += 1
            if (ErrorNum == 1):
                print >> fo, "Classification Error:"
                print >> fo, line

            print >> fo, "No. %d sample, class: %d, prediction: %d" % (i, TestClass[i], prediction)
            print >> fo, "Email Content:\n"
            print >> fo, Emails[TestIndex[i]]
            print >> fo, line
            if (prediction == 0):
                 Ham2SpamNum += 1

        i += 1

    print >> fo, "Total test samples:", len(TestDoc)
    print >> fo, "error classification number:", ErrorNum
    print >> fo, "The error rate is:", ErrorNum * 1.0 / len(TestDoc)

    return ErrorNum, Ham2SpamNum


def main():
    ans = 0; ham2spam = 0;runtimes = 100;
    for i in range(runtimes):
        error, seriouserror = SpamTest()
        ans += error
        ham2spam += seriouserror

    print >>fo, line
    print >>fo, "Run SpamTest for %d times\nTotal number of error classification: %d,\nTotal number of test samples: %d\namong which there are %d ham emails have been classified into spam emails." % (runtimes, ans, runtimes * 10, ham2spam)

    print  "Run SpamTest for %d times\nTotal number of error classification: %d,\nTotal number of test samples: %d\namong which there are %d ham emails have been classified into spam emails." % (runtimes, ans, runtimes * 10, ham2spam)



starttime = datetime.datetime.now()
start = time.clock()


fo = open("email.out", "w")
# a line
line = ""
for i in range(66):
    line += '-'

main()


end = time.clock()
endtime = datetime.datetime.now()

print >>fo, line
print >>fo, "RunningTime: %s sec" % (end-start)
print >>fo, "WaitingTime: %s"  % (endtime - starttime)
print "RunningTime: %s sec" % (end-start)
print "WaitingTime: %s sec"  % (endtime - starttime)
