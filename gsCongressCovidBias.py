#!/usr/bin/env python3

### Current tasks: 
### 	[DONE] modify to output a cleaner data file
### 	[DONE] modify stopwords -- remove parties, add states
###     Add stemming

import glob
import pandas as pd
import string
import numpy as np
from nltk.corpus import stopwords
import math
import csv
import re
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV #not in grid_search any more maybe?
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from nltk.stem.porter import PorterStemmer
import us


congressSpeech = "covidUtterances.tsv"
outfile = "fullPredictions.tsv"
#article = "articles/mediaText/article0.html"
#articleList = glob.glob("allScrape/article99*.html")
#articleList = glob.glob("articles/mediaText/article*.html")
#articleList = glob.glob("allScrape/article*.html")
articleList = glob.glob("allStoryText/article*.html")
#print(articleList)

#my_stop = ['democrat', 'democratic', 'republican', 'democrats', 'republicans']
my_stop = [state.name.lower() for state in us.states.STATES] #list of all US states should be stop words; congressional uses of state names are collinear with the congressperson
stop = my_stop + stopwords.words('english') + ['carolina', 'dakota', 'hampshire', 'island', 'jersey', 'mexico', 'new', 'north', 'rhode', 'south', 'west', 'york'] #these are spit out by the tokenizer
#stop = stopwords.words('english')

def wordclean(w):

    w = w.replace('.', '')
    w = w.replace(',','')
    w = w.replace(';','')
    w = w.replace(')','')
    w = w.replace('(','')
    w = w.replace("'s","")
    w = w.replace("s'","")
    #w = w.replace("'","")
    w = w.lower()

    return w

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

inDF = pd.read_csv(congressSpeech, sep='\t', header=0)
porter_stemmer = PorterStemmer()

#cleanup routine
inDF['speech'] = inDF['speech'].apply(lambda x: str(x)) #stringify everything
numInIt = re.compile('\d')
inDF['speech']= inDF['speech'].apply(lambda x: ' '.join(x for x in x.split() if not numInIt.search(x))) #eliminate numbers and words that contain numbers

inDF['speech'] = inDF['speech'].apply(lambda x: x.lower()) #lowercase everything
inDF['speech'] = inDF['speech'].apply(lambda x: x.lstrip()) #eliminate leading whitespace
inDF['speech'] = inDF['speech'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation)) #elim punctuation
#print(f"I'm about to get stemming, {inDF['speech']}")
#inDF['stemmed_speech'] = inDF['speech'].apply(stem_sentences)
#print(f"I've got a stemmed set of speeches now, {inDF['stemmed_speech']}")

#try letting scikit handle this
#inDF['speech'] = inDF['speech'].apply(lambda x:' '.join(x for x in x.split() if not x in stop)) #eliminate stop and custom stop words

inDF = inDF.dropna()
inDF = inDF.reset_index(drop=True)
minusNA = inDF.shape
print(inDF.head(3))
#print(inDF)

y = inDF['is_republican']
X = inDF['speech']
#X = inDF['stemmed_speech']
#print(y)

estimators = [("tf_idf", TfidfVectorizer(ngram_range=(1,3), stop_words=stop)), ("ridge", linear_model.Ridge())] 
#estimators = [("tf_idf", TfidfVectorizer(ngram_range=(1,3), stop_words=stop)), ("lasso", linear_model.Lasso())] 
model = Pipeline(estimators)


#print(f"X is {X}")
#print("===========================")
#print(f"y is {y}")

model = model.fit(X, y)
#if we do any hyper parameter tuning it can happen in here
finalModel = model

print(f"For the model I now have {finalModel}")
tf_idf_model = finalModel.named_steps["tf_idf"]
ridge_model = finalModel.named_steps["ridge"]
print(tf_idf_model)
print(ridge_model)
coefficients = pd.DataFrame({"names":tf_idf_model.get_feature_names(), "coef":ridge_model.coef_})
print("Best")
print(coefficients.sort_values("coef", ascending=False).head(50))
print("Worst")
print(coefficients.sort_values("coef", ascending=False).tail(50))


## now we bring in the article text and clean it up, then predict

outFH = open(outfile, 'w') #rewrites outfile fresh
outFH.write("articleID\tprediction\n")

for article in articleList: 
	with open(article) as f: 
		s_article = " ".join([l.rstrip() for l in f]) 
		p = re.compile(r'<.*?>') #tag killing regex
		articleClean = re.sub(p, '', s_article) #drops the tags

##Same cleanup routine as above
		articleClean = articleClean.lower() #lowercase everything
		articleClean = articleClean.lstrip() #eliminate leading whitespace 
		#articleClean = re.sub(r'[^\w^\s]', '', articleClean)
		#print(f'After cleanup, our article is {articleClean}')
		#articleClean = [porter_stemmer.stem(word) for word in articleClean]
		#print(f'After cleanup and stemming, our article is {articleClean}')
		prediction = finalModel.predict([articleClean]) 
		#print(f"Article is {article} and prediction is {prediction}")
		#print(f"Article is {article} and prediction is {prediction}")
		justID = article.split('/')[1]
		justID = justID.split('.')[0]
		justID = justID.lstrip('article')
		outFH.write(f"{justID}\t{prediction[0]}\n")


#for i in range(len(prediction)):
	#print(f"Title is {titleList[i]} and prediction is {classPreds[i]}")
#	print(f"Article is {article} and prediction is {prediction}")

