#Stock sentiment Analysis
#Top 25 headlines of a specific company based on which it has a label
#Label 0 says it will have a -ve impact ie Stock price is not going to increase
#Label 1 says it will have a +ve impact ie Stock price is going to increase

import pandas as pd
df = pd.read_csv('/content/Combined_News_DJIA.csv' , encoding="ISO-8859-1")

train=df[df['Date'] < '20150101']
test=df[df['Date'] > '20141231']
#In time series dataset older data set can be put under train
#and latest dataset can be put under test

#Removing Punctuations
data=train.iloc[:,2:27] #will select all rows but exclude
data.replace("[^a-zA-Z]"," " ,regex=True, inplace=True) #Apart from alphabets replace every other character with space
#Renaming column names for ease of access
list1=[i for i in range (25)]
new_Index=[str(i) for i in list1]
data.columns=new_Index

#Converting headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)

#Join all the sentences
' '.join(str(x) for x in data.iloc[1,0:25])

#Combine sentences of all the records
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
headlines[0]

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer #Converting sentences into vectors
from sklearn.ensemble import RandomForestClassifier

#Implement bag of words
countvector=CountVectorizer(ngram_range=(2,2)) #ngram is used to combine two words
traindataset=countvector.fit_transform(headlines)

#Traindataset is the sparse matrix(0,1)


#implement RandomForest Clssifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

#Predict for the test Dataset
test_transform=[]
for row in range(0,len(test.index)):
  test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=countvector.transform(test_transform)
prediction=randomclassifier.predict(test_dataset)

#import library to check accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
matrix=confusion_matrix(test['Label'],prediction)
print(matrix)
score=accuracy_score(test['Label'],prediction)
print(score)
report=classification_report(test['Label'],prediction)
print(report)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

tfidfvector=TfidfVectorizer(ngram_range=(2,2))
traindataset=tfidfvector.fit_transform(headlines)
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

#Predict for the test Dataset
test_transform=[]

for row in range(0,len(test.index)):
  test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=tfidfvector.transform(test_transform)
prediction=randomclassifier.predict(test_dataset)

matrix=confusion_matrix(test['Label'],prediction)
print(matrix)
score=accuracy_score(test['Label'],prediction)
print(score)
report=classification_report(test['Label'],prediction)
print(report)



from sklearn.naive_bayes import MultinomialNB
naive=MultinomialNB()

naive.fit(traindataset,train['Label'])
test_transform=[]

for row in range(0,len(test.index)):
  test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=tfidfvector.transform(test_transform)
prediction=naive.predict(test_dataset)

matrix=confusion_matrix(test['Label'],prediction)
print(matrix)
score=accuracy_score(test['Label'],prediction)
print(score)
report=classification_report(test['Label'],prediction)
print(report)