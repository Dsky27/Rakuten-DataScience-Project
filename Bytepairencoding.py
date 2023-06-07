# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:23:33 2022

@author: maxen
"""

# assuming we've extracted from our raw text and this is the character
# vocabulary that we've ended up with, along with their frequency

import os
import re
import time
import numpy as np
from operator import itemgetter
from typing import Dict, Tuple, List, Set
import nltk,csv,numpy 
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords

import pandas as pd

TOKENIZER=[]
df = pd.read_csv('Xtrain.csv', engine='python')

W=['a','z','e','r','t','y','u','i','o','p','q','s','d','f','g','h','j','k','l','m','w','x','c','v','b','n']

def token2():
    TOKENIZER=[]
    for i in range(10):
        line = df.iloc[i,1]
        tokens =nltk.word_tokenize(line)
        posData = pos_tag(tokens)
        for i in range(len(posData)):
            M=[]
            if posData[i][1]!= 'CD' and posData[i][0][0]!= ':' and posData[i][0][0]!= '!'and posData[i][0][0]!= '/' and posData[i][0][0]!= ':' and posData[i][0][0]!= '-' and posData[i][0][0]!= '(' and posData[i][0]!= ')' and posData[i][0][0]!= '.' and posData[i][0][0]!= '<' and posData[i][0][0]!= '>' and posData[i][0][0]!= ';' and posData[i][0][0]!= '@': 
                M.append(posData[i][0])
                TOKENIZER.append(M[0].lower())
    

    
    A=[word for word in TOKENIZER if word not in stopwords.words('english') and word not in stopwords.words('french') and word not in stopwords.words('german') and word not in W]
    return A

def decouper(texte):
    # Découpe le texte par morceau de 3 lettres
    decoupage = [texte[i:i+1] for i in range(0, len(texte))]
    # Recrée la chaine avec un espace toutes les 3 lettres
    return " ".join(decoupage)
 



def dictionnaire(P):
    dict={}
    for i in range(len(P)):
        if decouper(P[i])+' </w>' not in dict:
            dict[decouper(P[i])+' </w>']=1
        else : 
            dict[decouper(P[i])+' </w>']=dict[decouper(P[i])+' </w>']+1
    return dict

vocab = dictionnaire(token2())

def get_pair_stats(vocab):
    """Get counts of pairs of consecutive symbols."""

    pairs = {}
    for word, frequency in vocab.items():
        symbols = word.split()
        print(symbols)

        # count occurrences of pairs
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            current_frequency = pairs.get(pair, 0)
            pairs[pair] = current_frequency + frequency
    print (pairs)
    return pairs
pair_stats = get_pair_stats(vocab)
pair_stats

def merge_vocab(best_pair,vocab_in):
    """Step 3. Merge all occurrences of the most frequent pair"""

    vocab_out = {}

    # re.escape
    # ensures the characters of our input pair will be handled as is and
    # not get mistreated as special characters in the regular expression.
    pattern = re.escape(' '.join(best_pair))
    replacement = ''.join(best_pair)

    for word_in in vocab_in:
        # replace most frequent pair in all vocabulary
        word_out = re.sub(pattern, replacement, word_in)
        vocab_out[word_out] = vocab_in[word_in]

    return vocab_out

best_pair = max(pair_stats, key=pair_stats.get)
print(best_pair)

new_vocab = merge_vocab(best_pair, vocab)
new_vocab

vocab = dictionnaire(token2())

# we store the best pair during each iteration for encoding new vocabulary, more on this later
bpe_codes = {}
num_merges =10  # hyperparameter
for i in range(num_merges):
    print('\niteration', i)
    pair_stats = get_pair_stats(vocab)
    if not pair_stats:
        break

    best_pair = max(pair_stats, key=pair_stats.get)
    bpe_codes[best_pair] = i

    print('vocabulary: ', vocab)
    print('best pair:', best_pair)
    vocab = merge_vocab(best_pair, vocab)

print('\nfinal vocabulary: ', vocab)
print('\nbyte pair encoding: ', bpe_codes)

# first convert an input word to the list of character format
original_word = 'parisien'
word = list(original_word)
word.append('</w>')
word


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs



# attempt to find it in the byte pair codes

bpe_codes_pairs = [(pair, bpe_codes[pair]) for pair in get_pairs(word) if pair in bpe_codes]
print(bpe_codes_pairs)
pair_to_merge = min(bpe_codes_pairs, key=itemgetter(1))[0]
pair_to_merge

def create_new_word(word,pair_to_merge):
    first, second = pair_to_merge
    new_word = []
    i = 0
    while i < len(word):
        try:
            j = word.index(first, i)
            new_word.extend(word[i:j])
            i = j
        except ValueError:
            new_word.extend(word[i:])
            break

        if i < len(word) - 1 and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(first)
            i += 1

    return new_word



def encode(original_word,bpe_codes):
    if len(original_word) == 1:
        return original_word

    word = list(original_word)
    word.append('</w>')

    while True:
        pairs = get_pairs(word)
        bpe_codes_pairs = [(pair, bpe_codes[pair]) for pair in pairs if pair in bpe_codes]
        if not bpe_codes_pairs:
            break

        pair_to_merge = min(bpe_codes_pairs, key=itemgetter(1))[0]
        word = create_new_word(word, pair_to_merge)
    return word
original_word = token2()[17]
encode(original_word, bpe_codes)
print(encode(original_word, bpe_codes))

def new_vocab(vocab): 
    res=[]
    for i in range(len(vocab)):
        for j in range(encode(vocab[i])):
            res.append(encode(vocab[i])[j])
    res=list(set(res))
    return res

def vect(doc,voc):
    res=[0]*len(voc)
    for i in range(len(voc)):
        for j in range(len(doc)):
            if doc[j]==voc[i]:
                res[i]=res[i]+1
    return res

