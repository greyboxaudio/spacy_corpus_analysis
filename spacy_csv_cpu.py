#compare two csv tables with corpus data and find matching POS
#import all the things
import datetime
import spacy
import csv
import pandas as pd
import numpy as np
import pickle

#print current time for benchmarking purposes
tStart = datetime.datetime.now()
print(tStart.time(),"program start")

#let spacy use an available GPU to speed up the process
spacy.prefer_gpu()
#pd.set_option("display.max_colwidth", None)

#modifiable parameters
searchItem1 = "silbrig"
searchItem2 = "silbern"
maxEntries = 5000
csvInput1 = "input/dwds_silbrig_export_2023-12-29_16 37 56.csv"
csvInput2 = "input/dwds_silbern_export_2023-12-29_16 37 06.csv"
nlp = spacy.load("de_dep_news_trf")

#create empty arrays
headsItem1 = np.zeros(shape=(maxEntries),dtype=np.uint64)
headsItem2 = np.zeros(shape=(maxEntries),dtype=np.uint64)

#print current time for benchmarking purposes
print(datetime.datetime.now()-tStart,"tokenize first input")

#open a new csv to write data to
with open("output/"+searchItem1+"_"+searchItem2+"_raw.csv", "w", newline="", encoding="utf-8") as csvfile:
    dataWriter = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
    #write column headers
    header = ["token","token.tag_","token.head.lemma_","token.head.lemma","match","token.head.tag_","source"]
    dataWriter.writerow(header)
    #load input into a pandas dataframe
    df = pd.read_csv(csvInput1, encoding="utf-8")
    #get total number of rows in dataframe
    rows=df.shape[0]
    #iterate over each row
    for i in range(0, rows, 1):
        #get data from the sixth column, which contains the text data
        s = df.iat[i,5]
        #iterate over each token of the text data
        for token in nlp(s):
            if (token.lemma_ == searchItem1):
                row = [token.lemma_,token.tag_,token.head.lemma_,token.head.lemma,0,token.head.tag_,s]
                dataWriter.writerow(row)
                #add token.lemma to an array for later analysis
                headsItem1[i] = token.head.lemma
    #print elapsed time for benchmarking purposes
    print(datetime.datetime.now()-tStart,"tokenize second input")
    #load input into a pandas dataframe
    df = pd.read_csv(csvInput2, encoding="utf-8")
    #get total number of rows in dataframe
    rows=df.shape[0]
    #iterate over each row
    for i in range(0, rows, 1):
        #get data from the sixth column, which contains the text data
        s = df.iat[i,5]
        #iterate over each token of the text data
        for token in nlp(s):
            if (token.lemma_ == searchItem2): 
                row = [token.lemma_,token.tag_,token.head.lemma_,token.head.lemma,0,token.head.tag_,s]
                dataWriter.writerow(row)
                #add token.lemma to an array for later analysis
                headsItem2[i] = token.head.lemma

#print elapsed time for benchmarking purposes
print(datetime.datetime.now()-tStart,"sort and match data")

#speed up further development by storing/recalling the tokenized data with pickle
buffer = open("output/"+searchItem1+"_"+searchItem2+"_buffer.pkl", "wb")
pickle.dump(headsItem1, buffer)
pickle.dump(headsItem2, buffer)

#read previously written .csv to a new dataframe
df = pd.read_csv("output/"+searchItem1+"_"+searchItem2+"_raw.csv", encoding="utf-8")
#find unique values of the first array and sort them in ascending order
headsItem1Sorted = np.unique(headsItem1,False,False,False)
#iterate over sorted array
for x in np.nditer(headsItem1Sorted):
    #check the unsorted second array for matching values
    whereHeads = np.where(headsItem2 == x)
    #if return tuple is not empty and x is greater than 0
    if (any(map(len,whereHeads)) == True) and (x > 0):
        #select all rows that match the currently selected value
        rows = df.loc[df["token.head.lemma"] == x]
        #iterate over each row
        for i in range(0,rows.index.shape[0],1):
            #flip the "match" cell in the dataframe from 0 to 1
            df.iat[rows.index[i],4] = 1

#clean up and sort data
df = df[df.match == 1]
df = df.sort_values(["token.head.lemma_","token"])
df = df.drop(labels="token.head.lemma", axis=1)
df = df.drop(labels="match", axis=1)
#write dataframe to new .csv
df.to_csv("output/"+searchItem1+"_"+searchItem2+"_processed.csv", encoding="utf-8")

#print elapsed time for benchmarking purposes
print(datetime.datetime.now()-tStart,"program complete")
