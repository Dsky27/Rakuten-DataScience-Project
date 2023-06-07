import pickle
from math import *
from collections import Counter

 
vocabFile=open('saveFiles/vocabulary.txt','rb')
vocab=pickle.load(vocabFile)
vocabFile.close()

indexInvFile=open('saveFiles/indexInversed.txt','rb')
indexInversed=pickle.load(indexInvFile)
indexInvFile.close()


def tf(term,productId,indInv):
    return indInv[term][productId]

# pondération logarithmique d'un terme
def tfLog(term,productId, indInv):
    tf = tf(term,productId, indInv)
    if tf > 0:
        return 1 +log(tf)
    else:
        return 0
    

def statText(text):
    counter= Counter()
    for term in text:
        counter.update([term])
    stats={}
    stats["freqMax"] = counter.most_common(1)[0][1]
    stats["uniqueTerms"] = len(counter.items())
    tf_moy = sum(counter.values())
    stats["freqMoy"] = tf_moy/len(counter.items())
    return stats


def statsCollection(collection):
    stats={}
    l,r=collection.shape
    stats["nb_docs"]=l
    for index in range(l):
        stats[collection['productid'][index]] = statText(collection['text'][index])
    return stats

def tfNormalise(term,productId,indInv,stats_collection):
        tf = tf(term,productId,indInv)
        tfNormalise = 0.5 + 0.5 * (tf /stats_collection[productId]["freq_max"])
        return tfNormalise

def tfLogNormalise(term,productId, indInv,stats_collection):
        tf = tf(term,productId, indInv)
        tfLogNormalise = (1 +log(tf))/(1 + log(stats_collection[productId]["freq_moy"]))
        return tfLogNormalise
    

def idf(term,indInv,nbDoc):
    return log(nbDoc/len(indInv[term].keys()))