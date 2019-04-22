# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 13:33:31 2019

@author: Yshah
"""

import numpy as np
import pandas as pd

import pip

!pip install twitter

from twitter import Twitter
from twitter import OAuth

from pandas.io.json import json_normalize
#Access code for twitter API
apikey='ZCOq2I9HhSwlLthu7ZokZkO3R'
apisecretkey='AZRY5lMmcbkMYPz95rrMSjQkZjFJLvUFUNKOYdMQWWJB16FdU9' 
accesstoken='581601951-EZulBjxCjasHQq7VLVGgaUBN13f0ZnR4eX98fEaE'
accesstokensecret='XVvCnDJtOgqScUdgzmo80fsbBBsvslTzJ8yzcfoBTCZIF'

#Authorization of Access code for twitter API
oauth = OAuth ( accesstoken , accesstokensecret, apikey, apisecretkey )
api = Twitter ( auth=oauth )

#Trends Available 
tloc = api.trends.available()
print(tloc)
#Normalize the collected data
dfloc=json_normalize(tloc)
print(dfloc.country.value_counts())
#Tracking countires with the starting word New
dfNew=dfloc[dfloc['name'].str.contains('New')]
dfNew[['name','woeid']]
ny=dfNew.loc[dfNew.name=='New York','woeid']
#gives you data structure of ny
type(ny)
#Data type and number of values
ny.values
ny.values[0]

ny_trend= api.trends.place(_id=ny.values[0])
dfny=json_normalize(ny_trend)

dfny.trend
type(dfny.trends)
dfny.trends.shape
dftrends=json_normalize(dfny.trends.values[0])

#User Tweets about Donald Trup

tjson=api.statuses.user_timeline(screen_name="realDonaldTrump",tweet_mode='extended',count = 200)
dftrump=json_normalize(tjson)
dftrump.shape

dftrump['id']

mid = dftrump['id'].min()
mid=mid-1
tjson2=api.statuses.user_timeline(screen_name="realDonaldTrump",tweet_mode='extended',count = 200,max_id = mid)
dftrump2=json_normalize(tjson2)

mid_l=dftrump2['id'].max()

df = pd.DataFrame()
mid=0

for i in range(34):
    if i==0:
        tjson=api.statuses.user_timeline(screen_name="realDonaldTrump",tweet_mode='extended',count = 200)
    else:
        tjson=api.statuses.user_timeline(screen_name="realDonaldTrump",tweet_mode='extended',count = 200,max_id = mid)
    if len(tjson)>0:
        dftrump=json_normalize(tjson)
        mid=dftrump['id'].min()
        mid=mid-1
        #df = df.append(df,ignore_index=True)
        df = pd.concat([df, dftrump], ignore_index=True)

#NLP
#Textblob
import pip

#!pip install textblob
#!python -m textblob.download_corpora

from textblob import TextBlob

import numpy as np
import pandas as pd

tx =df.loc[0,'full_text']
blob = TextBlob (tx)
blob.tags
blob.sentences[0].words
blob.noun_phrases
blob.ngrams(3)
blob.correct( )
blob.words[3].spellcheck( )
blob.detect_language( )
blob. translate (to= 'ar' )


verbs=[ ]
for word, tag in blob.tags:
    if tag == 'VB ' :
        verbs.append(word.lemmatize( ))
nouns = [ ]
for word,tag in blob.tags:
    if tag == 'NN' :
        nouns.append(word.lemmatize( ) )
nounsp = [ ]
for word,tag in blob.tags:
    if tag == 'NNP' :
        nounsp.append(word.lemmatize( ) )
blob.sentiment
blob.sentiment.polarity
blob.sentiment.subjectivity

polarity =[ ]
subj =[ ]
for t in df.full_text:
    tx=TextBlob( t )
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)
polsubj=pd.DataFrame({'polarity': polarity,'subjectivity ':subj})

polsubj.plot(title='Polarity and Subjectivity' )
#World Cloud
import pip

!pip install wordcloud
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

stop =stopwords.words('english')

wordcloud = WordCloud().generate(t)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show( )

wordcloud2 = WordCloud(background_color="white",stopwords=stop).generate(t)
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

# Display the generated image:
tx2=df.full_text.str.cat(sep=' ')

wordcloud3 = WordCloud(stopwords=stop).generate(tx2)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.show()

stop.append('RT')
stop.append('co')
stop.append('https')
stop.append('amp')

wordcloud4 = WordCloud(background_color="white",stopwords=stop,max_words=1000).generate(tx2)
plt.imshow(wordcloud4, interpolation='bilinear')
plt.axis("off")
plt.show()
