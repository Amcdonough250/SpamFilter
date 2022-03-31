#------------------------------------------------------------------------------------------------
# FILE NAME:      training.py
# DESCRIPTION:    uses python3 to train a program to detect spam vs ham
#                 using Multinomial Naive Bayes
# USAGE:          python3 training.py
#                 
# notes:          Program uses multinomial Naive Bayes to calculate the probability of spam          
#                 Split the csv file half for training and the other half for testing
#
# MODIFICATION HISTORY
# Author               Date           version
#-------------------  ------------    -------------------------------------------------------
# Annette McDonough   2021-10-22      1.0 first version 
# Annette McDonough   2021-10-24      1.1 reading in csv file and spliting it 
# Annette McDonough   2021-10-26      1.2 working on dictionary
# Annette McDonough   2021-11-02      1.3 need to rework what needs to be output to file
# Annette McDonough   2021-11-03      1.4 writing to file properly now
#-----------------------------------------------------------------------------------------------
import pandas as pd
import csv
import re
import argparse

#------------------------------------------------------------------------------------------
# FUNCTION:     processCsv()
# DESCRIPTION:  processes csv file 
# notes:        parses through the csv file to calculate spam vs ham
#-----------------------------------------------------------------------------------------
def processCsv(filename1):

    
    smsSpam = pd.read_csv(filename1, sep=",,,", engine='python',
            error_bad_lines=False, quoting=csv.QUOTE_NONE,header=None)

    smsSub = smsSpam.drop(smsSpam.index[:1])
    col1 = []
    col2 = []

    for i in smsSub[0]:
        k = i.split(",", 1)
        col1.append(k[0])
        col2.append(k[1])

    data1 = []
    data1.append(col1)
    data1.append(col2)

    newDf = pd.DataFrame(data1).transpose()
    newDf.columns=['Label', 'SMS']
    return newDf

#------------------------------------------------------------------------------------------
# FUNCTION:     classify(mess)
# DESCRIPTION:  distinguishes between spam and ham
# notes:        
#-----------------------------------------------------------------------------------------
def classify(mess):

    mess = re.sub('\W', ' ', mess)
    mess = mess.lower().split()

    pSpamMess = pSpam
    pHamMess = pHam

    for word in mess:
        if word in paramSpam:
            pSpamMess *= paramSpam[word]

        if word in paramHam:
            pHamMess *= paramHam[word]

    print('P(Spam|message):', pSpamMess)
    print('P(Ham|message):', pHamMess)

    if pHamMess > pSpamMess:
        print('Label: Ham')
    elif pHamMess < pSpamMess:
        print('Label: Spam')
    else:
        print('Need a human to classify equal probability')


#------------------------------------------------------------------------------------------
# FUNCTION:     main()
# DESCRIPTION:  programs driver
# notes:        uses data to train for spam
#-----------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='process files')
parser.add_argument('-i',metavar='N',nargs='+',help='file processor')
parser.add_argument('-os',metavar='N',nargs='+', help='output spam file')
parser.add_argument('-oh',metavar='N',nargs='+',help='output ham file')

args = parser.parse_args()
filename1 = args.i[0]
filename2 = args.os[-1]
filename3 = args.oh[-1]

smsSpam1 = processCsv(filename1)
smsSpam1.head()


# randomize the data
dataR = smsSpam1.sample(frac=1, random_state=1)

# calculate split len of data
trainingTestI = round(len(dataR))

# split data into 2 like slides show. I still have half the orig data
# for testing
trainSet = dataR[:trainingTestI].reset_index(drop=True)

print("number of lines in file and number of cols:\n", trainSet.shape)
print("spam and ham percentages:\n",trainSet['Label'].value_counts(normalize=True))

# remove punctuation
trainSet['SMS'] = trainSet['SMS'].str.replace('\W', ' ')
# replace uppercase to lowercase
trainSet['SMS'] = trainSet['SMS'].str.lower()

# split string in SMS catagorey
trainSet['SMS'] = trainSet['SMS'].str.split()

# initialize array with vocab to create a dictionary
vocab = []
for sms in trainSet['SMS']:
    for word in sms:
        vocab.append(word)

vocab = list(set(vocab))

# print number of unique vocab words
print("\nunique vocabulary:", len(vocab))

# create dictionary to use and itterate through 
smsCount = {uniqueWord: [0] * len(trainSet['SMS']) for uniqueWord in vocab}

for index, sms in enumerate(trainSet['SMS']):
    for word in sms:
        smsCount[word][index] += 1

# take dictionary and put it into new dataframe
wordCount = pd.DataFrame(smsCount)
wordCount.head()

trainSetClean = pd.concat([trainSet, wordCount], axis=1)
trainSetClean.head()

# seperate spam and ham
spamMess = trainSetClean[trainSetClean['Label'] == 'spam']
hamMess = trainSetClean[trainSetClean['Label'] == 'ham']

# Naive Bayes calculations P(Spam) and P(Ham)
pSpam = len(spamMess) / len(trainSetClean)
pHam = len(hamMess) / len(trainSetClean)

# NSpam
nWordsPerSpam = spamMess['SMS'].apply(len)
nSpam = nWordsPerSpam.sum()
print("Number of spam words:", nSpam)

#NHam
nWordsPerHam = hamMess['SMS'].apply(len)
nHam = nWordsPerHam.sum()
print("Number of ham words:", nHam)

# drop first column of dataframe
testDropS = spamMess
testDropS = testDropS.drop(["Label", "SMS"], axis = 1)
testArrayWordSpam = []
testArraySumSpam = []
countSpam = 0
# sum each instance
for word in vocab:
    testSumSpam = spamMess[word].sum()
    if testSumSpam != 0:
        testArrayWordSpam.append(word)
        testArraySumSpam.append(testSumSpam)
        countSpam += 1
# write to os.csv file
with open(filename2, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([nSpam])
    for x in range(countSpam):
        writer.writerow([testArrayWordSpam[x], testArraySumSpam[x]])

# drop first column of dataframe
testDropH = hamMess
testDropH = testDropH.drop(["Label", "SMS"], axis = 1)
testArrayWordHam = []
testArraySumHam = []
countHam = 0
# sum each instances
for word in vocab:
    testSumHam = hamMess[word].sum()
    if testSumHam != 0:
        testArrayWordHam.append(word)
        testArraySumHam.append(testSumHam)
        countHam += 1
# write to oh.csv file
with open(filename3, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([nHam])
    for x in range(countHam):
        writer.writerow([testArrayWordHam[x], testArraySumHam[x]])


# nVocab
nVocab = len(vocab)

# Laplace smoothing alpha = add 1
alpha = 1

# create parameters
paramSpam = {uniqueWord:0 for uniqueWord in vocab}
paramHam = {uniqueWord:0 for uniqueWord in vocab}

# calculate parameters
for word in vocab:
    # (Nw|spam + alpha) / (Nspam + (alpha * Nvocab))
    nWordSpam = spamMess[word].sum()
    pWordSpam = (nWordSpam + alpha) / (nSpam + alpha*nVocab)
    paramSpam[word] = pWordSpam

    # (Nw|ham + alpha) / (Nham + (alpha * Nvocab))
    nWordHam = hamMess[word].sum()
    pWordHam = (nWordHam + alpha) / (nHam + alpha*nVocab)
    paramHam[word] = pWordHam


print("The message: 'SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info' is classified as: ")
classify('SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info')

print("The message: 'Sorry, I'll call later' is classified as: ")
classify("Sorry, I'll call later")


