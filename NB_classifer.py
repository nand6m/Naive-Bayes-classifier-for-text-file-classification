import os
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from collections import Counter
import random
import math
stop = set(stopwords.words('english'))
import operator

#sents =  word_tokenize(text)
class PreProcessing(object):
    def __init__(self,TrainingPath,TestPath,NumOfSets):
        self.TrainingPath=TrainingPath
        self.TestPath=TestPath
        self.AllClassList=os.listdir(TrainingPath)
        Rand_num = random.sample(range(20), NumOfSets)
        self.class_list = [self.AllClassList[a] for a in range(20) if a in Rand_num]
        self.TrainingDataSet={}
        self.TestDataSet = {}
        self.NumOfFiles=0
        self.vocabulary=[]
        self.NumOfDataInEachClass={}
        self.CondProb={}
        self.NumOfClassTerm={}

    def read_dataset(self,class_path,class_name):
        path = os.path.join(class_path,class_name)
        DataSet=[]
        self.NumOfFiles+=len(os.listdir(path))
        self.NumOfDataInEachClass[class_name]=len(os.listdir(path))
        for filename in os.listdir(path):
            f = open(os.path.join(path, filename), 'r')
            x = f.readlines()
            LinesNumber=0
            for i in range(len(x)):
                if x[i].find('Lines')==0 :
                    LinesNumber=i
            #    if LinesNumber!=0:
            #        x[i]=x[i].strip('\n')
            #        x[i] = x[i].strip('\t')
            #x[:] = [item for item in x if item != '']
            DataSet.extend(x[(LinesNumber+1):])
        return DataSet

    def read_OneFile(self,file_path,file_name):

        f = open(file_path + '\\' + file_name, 'r')
        x = f.readlines()
        LinesNumber=0
        for i in range(len(x)):
            if x[i].find('Lines')==0 :
                LinesNumber=i
        #    if LinesNumber!=0:
        #        x[i]=x[i].strip('\n')
        #        x[i] = x[i].strip('\t')
        #x[:] = [item for item in x if item != '']
        return x[(LinesNumber+1):]

    def FilterData(self,DataSet):
        filtered_words = []
        # ======================== word tokenize and delete stop words  ===================================
        for i in DataSet:
            filtered_words.extend([i for i in word_tokenize(i.lower()) if i not in stop and not i.isdigit() and not i.isdecimal()
                   and not i.isnumeric() and i.isalpha()])
        # ========== start: Dict type of counting and sorting  ====================================
        return filtered_words
        # ========== end: Dict type of counting and sorting  ===================================

    def calc_score(self,Testmatch):
        score=0
        for v in Testmatch:
            if (self.vocabulary.__contains__(v)):
                score += math.log2(self.CondProb[c + "_" + v])
        return score
## ===================================================================================================
## ===================================================================================================
TrainingPath='20news-bydate-train'
TestPath='20news-bydate-test'
#TrainingPath='sample.train'
#TestPath='sample.test'

NB=PreProcessing(TrainingPath,TestPath,5)
#NB.class_list = ['comp.graphics', 'rec.autos', 'sci.electronics', 'sci.space', 'talk.politics.misc']

print("These datasets are selected:",NB.class_list)
print("-------------------Training starts--------------------------------")
#choose a class for training

NB.CondProb={}
priorineachclass = list()
for c in NB.class_list:
    DataSet=NB.read_dataset(NB.TrainingPath,c)
    NB.vocabulary.extend(NB.FilterData(DataSet))
    NB.TrainingDataSet[c]=Counter(NB.vocabulary)

for c in NB.class_list:
    priorineachclass.append(NB.NumOfDataInEachClass[c] / NB.NumOfFiles)
    temp = 0
    for v in NB.vocabulary:
        temp += NB.TrainingDataSet[c][v]
    NB.NumOfClassTerm[c]=temp
    for v in NB.vocabulary:
        NB.CondProb[c + '_' + v] = (NB.TrainingDataSet[c][v] + 1) / (NB.NumOfClassTerm[c] + len(NB.vocabulary))


#========== end : Array type of counting and sorting ====================================
print("-------------------Training done --------------------------------")

print("-------------------Testing starts--------------------------------")

score = {}
acc = {}
for c in NB.class_list: acc[c]=0

for TestSet in NB.class_list:
    path = os.path.join(NB.TestPath, TestSet)
    OneClassTestData = []
    for i in os.listdir(path):       # loop for reading all test data of one class for test
        OneClassTestData.append(NB.FilterData(NB.read_OneFile(path,i)))
    for i in range(len(OneClassTestData)): # loop for finding the class of each file
        # filtering to only use the available word
        Testmatch=list()
        for t in OneClassTestData[i]:
            if (NB.vocabulary.__contains__(t)):
                Testmatch.append(t)

        predictedclass = ""
        flag = 0

        for c in NB.class_list:
            countofclass = 0
            score[c] = math.log2(priorineachclass[countofclass])
            countofclass += 1
            score[c] += NB.calc_score(Testmatch)
            #for v in Testmatch:
            #    if (NB.vocabulary.__contains__(v)):
            #        score[c] += calc_score(Testmatch) += math.log2(NB.CondProb[c + "_" + v])
            if (flag == 0):
                maxscore = score[c]
                predictedclass = c
                flag = 1
            if (score[c] > maxscore):
                maxscore = score[c]
                predictedclass = c
        if (predictedclass == TestSet):
            acc[TestSet] += 1
        print("The class of", TestSet, i, "is:", predictedclass)

print("-------------------Testing Done--------------------------------")
avg_accuracy=0
for j in NB.class_list:
    path = os.path.join(NB.TestPath, j)
    accuracy = (acc[j] / len(os.listdir(path))) * 100
    avg_accuracy+=accuracy
    print("Accuracy obtained on the testing of ",j, 'dataset is : ', accuracy)
print("The average accuracy obtained on these",len(NB.class_list), " test data is : ", avg_accuracy/len(NB.class_list))


