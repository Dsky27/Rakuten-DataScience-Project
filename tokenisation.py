


import nltk,csv,numpy 
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords

import pandas as pd
TOKENIZER=[]
df = pd.read_csv('XTrain.csv', engine='python')

L=['a','z','e','r','t','y','u','i','o','p','q','s','d','f','g','h','j','k','l','m','w','x','c','v','b','n']

def token2():
    TOKENIZER=[]
    for i in range(25):
        line = df.iloc[i,1]
        tokens =nltk.word_tokenize(line)
        posData = pos_tag(tokens)
        for i in range(len(posData)):
            M=[]
            if posData[i][1]!= 'CD' and posData[i][0][0]!= ':' and posData[i][0][0]!= '!'and posData[i][0][0]!= '/' and posData[i][0][0]!= ':' and posData[i][0][0]!= '-' and posData[i][0][0]!= '(' and posData[i][0]!= ')' and posData[i][0][0]!= '.' and posData[i][0][0]!= '<' and posData[i][0][0]!= '>' and posData[i][0][0]!= ';' and posData[i][0][0]!= '@': 
                M.append(posData[i][0])
                TOKENIZER.append(M[0].lower())
    

    
    A=[word for word in TOKENIZER if word not in stopwords.words('english') and word not in stopwords.words('french') and word not in stopwords.words('german') and word not in L]
    return A

def decouper(texte):
    # Découpe le texte par morceau de 3 lettres
    decoupage = [texte[i:i+1] for i in range(0, len(texte))]
    # Recrée la chaine avec un espace toutes les 3 lettres
    return " ".join(decoupage)
 



def dictionnaire(L):
    dict={}
    for i in range (len(L)): #on parcourt la liste
        if decouper(L[i])+' </w>' not in dict:
            dict[decouper(L[i])+' </w>']=1
        else : 
            dict[decouper(L[i])+' </w>']=dict[decouper(L[i])+' </w>']+1
    return dict




