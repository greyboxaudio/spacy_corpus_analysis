import datetime
import spacy
import csv
import pandas as pd
import numpy as np

tStart = datetime.datetime.now()
print(tStart.time())

searchItem1 = "kupfrig"
searchItem2 = "kupfern"
csvInput1 = "webkorpus_kupfrig_dwds_export_2023-12-21_16_49_28.csv"
csvInput2 = "webkorpus_kupfern_dwds_export_2023-12-21_17_32_11.csv"
headsItem1 = np.zeros(shape=(5000),dtype=np.uint64)
headsItem2 = np.zeros(shape=(5000),dtype=np.uint64)

nlp = spacy.load("de_dep_news_trf")
pd.set_option('display.max_colwidth', None)

with open('output/output.csv', 'w', newline='', encoding="utf-8") as csvfile:
    dataWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["token","token.tag_","token.head.lemma_","token.head.tag_","source"]
    dataWriter.writerow(header)
    df = pd.read_csv("input/" + csvInput1, encoding="utf-8")
    rows=df.shape[0]
    for i in range(0, rows, 1):
        s = df.iat[i,5]
        doc = nlp(s)
        for token in doc:
            if (token.lemma_ == searchItem1):
                row = [token.lemma_,token.tag_,token.head.lemma_,token.head.tag_,s]
                dataWriter.writerow(row)
                headsItem1[i] = token.head.lemma
    df = pd.read_csv("input/"+csvInput2, encoding="utf-8")
    rows=df.shape[0]
    for i in range(0, rows, 1):
        s = df.iat[i,5]
        doc = nlp(s)
        for token in doc:
            if (token.lemma_ == searchItem2):
                row = [token.lemma_,token.tag_,token.head.lemma_,token.head.tag_,s]
                dataWriter.writerow(row)
                headsItem2[i] = token.head.lemma
print(headsItem2)
headsItem1Sorted = np.unique(headsItem1,False,False,False)
print(headsItem1Sorted)
for x in np.nditer(headsItem1Sorted):
    whereHeads = np.where(headsItem2 == x)
    if whereHeads != 0:
        print(whereHeads)

tEnd = datetime.datetime.now()
tDuration = tEnd - tStart
print(tDuration)
