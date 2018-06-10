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
