#------------------------------------------------------------------------------------------------
# FILE NAME:      testing.py
# DESCRIPTION:    uses python3 to test a program to detect spam vs ham
#                 using Multinomial Naive Bayes
# USAGE:          python3 testing.py
#                 
# notes:          Program uses multinomial Naive Bayes to calculate the probability of spam          
#                 Split the csv file half for training and the other half for testing
#
# MODIFICATION HISTORY
# Author               Date           version
#-------------------  ------------    -------------------------------------------------------
# Annette McDonough   2021-10-22      1.0 first version 
# Annette McDonough   2021-10-24      1.1 reading in csv file and spliting it 
# Annette McDonough   2021-10-25      1.2 working on dictionary
# Annette McDonough   2021-10-31      1.3 copied over from training for testing
# Annette McDonough   2021-11-02      1.4 not in correct format and not what Prateek wanted
# Annette McDonough   2021-11-02      1.5 trying had to go back to traing and fix output format 
#                                         extract values from dataframe
# Annette McDonough   2021-11-03      1.6 back to rewritting 90% of test program
# Annette McDonough   2021-11-04      1.7 input files working well
# Annette McDonough   2021-11-04      1.8 accuracy is very low need to figure out why || array isn't working
# Annette McDonough   2021-11-05      1.9 accuracy is working now
# Annette McDonough   2021-11-06      2.0 adding argparse to finish up code
#-----------------------------------------------------------------------------------------------
import pandas as pd
import csv
import re
import argparse

#------------------------------------------------------------------------------------------
# FUNCTION:     processTestCsv()
# DESCRIPTION:  processes csv file
# notes:        parses through the csv file to calculate spam vs ham
#-----------------------------------------------------------------------------------------
def processTestCsv(filename1):

    smsSpamT = pd.read_csv(filename1, sep=",,,", engine='python',
             error_bad_lines=False, quoting=csv.QUOTE_NONE,header=None)

    smsSubT = smsSpamT.drop(smsSpamT.index[:1])
    col1 = []
    col2 = []

    for i in smsSubT[0]:
        k = i.split(",", 1)
        col1.append(k[0])
        col2.append(k[1])

    data1 = []
    data1.append(col1)
    data1.append(col2)

    newDf2 = pd.DataFrame(data1).transpose()
    newDf2.columns=['Label', 'SMS']
    return newDf2

#------------------------------------------------------------------------------------------
# FUNCTION:     classifyTest(mess)
# DESCRIPTION:  distinguishes between spam and ham
# notes:        
#-----------------------------------------------------------------------------------------
def classifyTest(mess):

    mess = re.sub('\W', ' ', mess)
    mess = mess.lower().split()

    pSpamMess = 0
    pHamMess = 0
   # calculate probability for each mess and compare
    for word in mess:
        if word in paramSpam:
            pSpamMess += paramSpam[word]

        if word in paramHam:
            pHamMess += paramHam[word]
    
    # make probability guess
    if pHamMess > pSpamMess:
        return 'ham'
    elif pHamMess < pSpamMess:
        return 'spam'
    else:
        return 'ham'
        


#------------------------------------------------------------------------------------------
# FUNCTION:     main()
# DESCRIPTION:  programs driver
# notes:        uses data to train for spam
#-----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('-i',metavar='N',  nargs='+',help='run testing inputfile')
parser.add_argument('-is',metavar='N', nargs='+',dest='ipfile',help='run input os.csv')
#parser.add_argument('-iS',metavar='N',nargs='+',help='run input os.csv')
parser.add_argument('-ih',metavar='N', nargs='+',help='run input oh.csv')
parser.add_argument('-o',metavar='N',nargs='+',help='run output o.csv')

args = parser.parse_args()
filename1 = args.i[0]
filename2 = args.ipfile[-1]
filename3 = args.ih[-1]
filename4 = args.o[-1]


smsSpam = processTestCsv(filename1)

# randomize the data
data = smsSpam.sample(frac=1, random_state=1)

# calculate len of data
testingL = round(len(data))

# split data into 2 like slides show. I still have half the orig data
# for testing
testSet = data[:testingL].reset_index(drop=True)

print('testing lines:', testSet.shape)

print('spam and ham test percentages:\n', testSet['Label'].value_counts(normalize=True))

wordSpam = []
wordHam = []
instanceHam = []
instanceSpam = []
vocab = []
# open spam file 
with open(filename2, mode='r') as csvFileS:
    csvReader = csv.reader(csvFileS)
    nSpam = next(csvReader)
    nSpam = nSpam[0]
    print("countSpam:", nSpam)

    for row in csvReader:
        wordSpam.append(row[0])
        instanceSpam.append(row[1])
    # read in oh.csv file
with open(filename3, mode='r') as csvFileH:
    csvReader2 = csv.reader(csvFileH)
    nHam = next(csvReader2)
    nHam = nHam[0]
    print("countHam:", nHam)

    for row in csvReader2:
        wordHam.append(row[0])
        instanceHam.append(row[1])
# append both arrays to get vocab 
vocab.extend(wordSpam)
vocab.extend(wordHam)
# get rid of duplicates
vocab = list(set(vocab))
# calculate total number of words
totalNum = int(nSpam) + int(nHam)

print("total number of words:", totalNum)

# calculate probability pHam and pSpam
pHam = int(nSpam) / totalNum
pSpam = int(nHam) / totalNum

# print number of unique vocab words
print("\nunique vocabulary:", len(vocab))
# print number of spam and ham words
print("Number of spam words:", nSpam)
print("Number of ham words:", nHam)

# number of vocabulary words
nVocab = len(vocab)

# Laplace smoothing alpha = add 1
alpha = 1

# create dictionary
paramSpam = {}
paramHam = {}

# we have two parallel arrays we need to tap into
# wordHam[] instancesHam[] 
# wordSpam[] instanceSpam[]
count = 0
# calculate parameters
for word in wordSpam:
    # need to figure out array extraction
    nWordSpam = instanceSpam[count]
    # Naive Bayes calculation
    pWordSpam = (int(nWordSpam) + alpha) / (int(nSpam) + alpha*nVocab)
    paramSpam[word] = pWordSpam
    count +=1

count = 0
for word in wordHam:
    # need value from wordHam
    nWordHam = instanceHam[count]
    # Naive Bayes calculations
    pWordHam = (int(nWordHam) + alpha) /(int(nHam) + alpha*nVocab)
    paramHam[word] = pWordHam
    count +=1
    


# test program
testSet['predicted'] = testSet['SMS'].apply(classifyTest)
print(testSet.head())

correct = 0
total = testSet.shape[0]

for row in testSet.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
# print out calculations
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)

testSet = testSet[['predicted', 'SMS']]
testSet.to_csv(filename4)
print(testSet.head())
