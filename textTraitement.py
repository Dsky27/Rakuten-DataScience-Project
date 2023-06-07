from lib2to3.pgen2.tokenize import tokenize
import re
from nltk import word_tokenize,pos_tag
from nltk.corpus import stopwords
import pandas as pd
from collections import OrderedDict
import pickle

data=pd.read_csv('Xtest.csv')


def clean_str(string):
            
    string = re.sub(r"Ã©", "é", string)
    string = re.sub(r"Ã¨", "è", string)
    string = re.sub(r"NÂ°", "n°", string)
    
    return string

def filtrage(text):

    text=clean_str(str(text))
    textTok=pos_tag(word_tokenize(str(text)))
    trueText=[]
    stopwordsList=stopwords.words('english')+stopwords.words('french')+stopwords.words('german')
    falseType=['CD',':','!','/',':','-','(',')','.']
    for tuple in textTok:
        if (tuple[1] not in falseType) and (tuple[0].lower() not in (stopwordsList or list(map(chr, range(97, 123))))):
            trueText.append(tuple[0].lower())
    return trueText


def tokenizeDf(df,type=2):
    """
    Prend en entrée une dataframe avec un type à renseigner par défault à 2 
    fonction renvoyant la dataframe avec les mots tokenised à la place du texte
    le type 1 renvoit une colonne pour la desigantion et l'autre pour la description
    le type 2 renvoit une seule colonne de designation + decription
    """
    l,r=df.shape
    if type == 1 :   
        columnNames=['designation','description','productid','imageid']
        newdf=pd.DataFrame(columns=columnNames)
        for i in range(l):
            designation=df['designation'][i]
            description=df['description'][i]
            desiTokenise=filtrage(str(designation))
            descrTokenise=filtrage(str(description))
            newdf.loc[i]=[desiTokenise,descrTokenise,df['productid'][i],df['imageid'][i]]

    elif type == 2:
        columnNames=['text','productid','imageid']
        newdf=pd.DataFrame(columns=columnNames)
        for i in range(l):
            designation=df['designation'][i]
            description=df['description'][i]
            desiTokenise=filtrage(str(designation))
            descrTokenise=filtrage(str(description))
            newdf.loc[i]=[desiTokenise+descrTokenise,df['productid'][i],df['imageid'][i]]
        
    return newdf


def dfToVocab(df):
    """
    Prend en entrée une dataframe
    renvoit une liste du vocabulaire présent dans tous les textes (sans occurence)
    """
    l,r=df.shape
    vocab=[]
    for i in range(l):
        vocab=list(set(vocab+df['text'][i]))
    return vocab




def build_inverted_index(collection,type_index):
    """
    collection est une dataframe contenant les tokens de chaques produits
    type_index 1 est un index de documents
    type_index 2 est un index de fréquence
    """
    inverted_index=OrderedDict()
    l,r=collection.shape
    if type_index == 1:
        for index in range(l):
            for term in collection['text'][index]:
                if term in inverted_index.keys():
                    if collection['productid'][index] not in inverted_index[term]:
                        inverted_index[term].append(collection['productid'][index])
                else:
                    inverted_index[term]=[collection['productid'][index]]
    elif type_index ==2:
        for index in range(l):
            for term in collection['text'][index]:
                if term in inverted_index.keys():
                    if collection['productid'][index] in inverted_index[term].keys():
                        inverted_index[term][collection['productid'][index]] = inverted_index[term][collection['productid'][index]] + 1
                    else:
                        inverted_index[term][collection['productid'][index]]= 1
                else:
                    inverted_index[term]=OrderedDict()
                    inverted_index[term][collection['productid'][index]]=1
    return inverted_index


def saveVocIndexInv(data):
    dfToken=tokenizeDf(data)
    vocab=dfToVocab(dfToken)
    indexInversed=build_inverted_index(dfToken,2)
    vocabFile=open('saveFiles/vocabulary.txt','wb')
    pickle.dump(vocab,vocabFile)
    vocabFile.close()
    indexFile=open('saveFiles/indexInversed.txt','wb')
    pickle.dump(indexInversed,indexFile)
    indexFile.close()


dataReduce=data.head(5)
saveVocIndexInv(dataReduce)


