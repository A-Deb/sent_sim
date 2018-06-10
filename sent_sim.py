import pandas as pd
from gensim import corpora, models, similarities

df = pd.read_csv('hackmageddon.csv')
print(len(df))

documents = []
for i in range(0,len(df)):
	documents.append(df.Description[i])

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

dictionary = corpora.Dictionary(texts)
dictionary.save('hackmageddon.dict')

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('hackmageddon.mm', corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

query = "malicious email attack"
vec_bow = dictionary.doc2bow(query.lower().split())
vec_lsi = lsi[vec_bow]
print(vec_lsi)
index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
df2=pd.DataFrame(sims, columns=['attack_id','sim_score'])
for i in range(0,len(df2)):
    df2.text[i]=df.Description[df2.attack_id[i]]

from gensim.models import Word2Vec
model_hack = Word2Vec(sentences=texts, size=100, window=5, min_count=2, workers=4, sg=0)
model_hack.wv.most_similar('researchers')


from gensim.models import FastText
model_hack = FastText(texts, size=100, window=5, min_count=5, workers=4,sg=1)
model_hack.wv.most_similar('researchers')



from elasticsearch import Elasticsearch
es = Elasticsearch(['http://cloudweb01.isi.edu/es/'], 
    http_auth=('effect', 'c@use!23'), port=80)
print(es.info())

#search for blogs/news since certain timeframe. 
results = es.search(index="effect/socialmedia",scroll = '1d', size = 20000,body = {

  "query": {
     "range" : {
        "datePublished" : {
            "gte": "07/01/2016",
            "format": "MM/dd/yyyy"
        }
    }
  }
})

total=results['hits']['total']
print(total)

#get first scroll
temp = results['hits']['hits']
sid = results['_scroll_id']
temp = list(map(lambda x: x['_source'],temp))
df1 = pd.DataFrame(temp)
df1 = df1[['datePublished', 'text','uri']]
print(len(df1))

#establish the output dataframe
bigdata = pd.concat([df1], ignore_index=True)
print(len(bigdata))

#Scroll through the rest of the data appending to the output file; 200K tweets
for i in range(1,10):
    
 results = es.scroll(scroll_id = sid, scroll = '1d')
 temp = results['hits']['hits']
 temp = list(map(lambda x: x['_source'],temp))
 df1 = pd.DataFrame(temp)
 df1 = df1[['datePublished', 'text','uri']]
 ss=len(df1)
 print(ss)

 bigdata = pd.concat([bigdata,df1], ignore_index=True)
 print(len(bigdata))

 #save to .csv 200K tweets = 40MB
bigdata.to_csv('/Users/ashokdeb/Desktop/sent_sim/sent_sim/tweets.csv',index=False)








