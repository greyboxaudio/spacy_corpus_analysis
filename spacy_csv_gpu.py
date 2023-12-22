searchItem1 = "kupfrig"
searchItem2 = "kupfern"
csvInput1 = "webkorpus_kupfrig_dwds_export_2023-12-21_16_49_28.csv"
csvInput2 = "webkorpus_kupfern_dwds_export_2023-12-21_17_32_11.csv"
import spacy
spacy.prefer_gpu()
nlp = spacy.load("de_dep_news_trf")
import csv
import pandas as pd
pd.set_option('display.max_colwidth', None)
import cupy as cp
headsItem1 = cp.zeros(shape=(5000),dtype=cp.uint64)
headsItem2 = cp.zeros(shape=(5000),dtype=cp.uint64)
with open('output.csv', 'w', newline='', encoding="utf-8") as csvfile:
    dataWriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ["token","token.tag_","token.head.lemma_","token.head.tag_","source"]
    dataWriter.writerow(header)
    df = pd.read_csv(csvInput1, encoding="utf-8")
    rows=df.shape[0]
    for i in range(0, rows, 1):
        s = df.iat[i,5]
        doc = nlp(s)
        for token in doc:
            if (token.lemma_ == searchItem1):
                row = [token.lemma_,token.tag_,token.head.lemma_,token.head.tag_,s]
                dataWriter.writerow(row)
                headsItem1[i] = token.head.lemma
    df = pd.read_csv(csvInput2, encoding="utf-8")
    rows=df.shape[0]
    for i in range(0, rows, 1):
        s = df.iat[i,5]
        doc = nlp(s)
        for token in doc:
            if (token.lemma_ == searchItem2):
                row = [token.lemma_,token.tag_,token.head.lemma_,token.head.tag_,s]
                dataWriter.writerow(row)
                headsItem2[i] = token.head.lemma
headsItem1Sorted = cp.unique(headsItem1,True,True,True)
headsItem2Sorted = cp.unique(headsItem2,False,False,True)
print(headsItem1Sorted)
print(headsItem2Sorted)
