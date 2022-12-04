#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:52:56 2022

@author: tim_riet
"""

import json
import pandas as pd
import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from scipy import spatial

f = open('TVs-all-merged.json')
data = json.load(f) #converting json file to python dictionary

#transform dictionary into dataframe, and seperate duplicates
file = [] 
for i in range(len(list(data.values()))):
    if len(list(data.values())[i]) == 1:
        file.append(list(data.values())[i][0])
    else:
        for k in range(len(list(data.values())[i])):
            file.append(list(data.values())[i][k])
df = pd.DataFrame(file)
df = df.drop(['url', 'featuresMap'], axis = 1)

def clean_transform(data): #works
    shop  = data['shop']
    title = data['title']
    all_words = [""]*30000
    
    for i in range(0, len(title)): #cleaning titles
        title[i] = title[i].lower()
        title[i] = title[i].replace("'"," ")
        title[i] = title[i].replace('/'," ")
        title[i] = title[i].replace('–'," ")
        title[i] = title[i].replace('-'," ")
        title[i] = title[i].replace(':'," ")
        title[i] = title[i].replace('('," ")
        title[i] = title[i].replace('+'," ")
        title[i] = title[i].replace(')'," ")
        title[i] = title[i].replace('['," ")
        title[i] = title[i].replace(']'," ")
        title[i] = title[i].replace('newegg.com'," ")
        title[i] = title[i].replace('thenerds.net'," ")
        title[i] = title[i].replace('best buy'," ")
        title[i] = title[i].replace('.'," ")
        title[i] = title[i].replace(','," ")
        title[i] = title[i].replace('"','inch')
        title[i] = title[i].replace('  '," ")
        title[i] = title[i].replace('class', " ")
        title[i] = title[i].replace(' inch', "inch")
        title[i] = title[i].replace(' hz', 'hz')
        title[i] = title[i].replace('hertz', 'hz')
        
    #filling vector titlewords
    index = 0
    for i in range(len(title)):
        allwords0 = title[i].split()
        for j in range(0,len(allwords0)):
            all_words[j+index] = allwords0[j]
        index = index + len(allwords0)
    
    all_words =  set(all_words) #remove dups
    all_words.remove("") #remove empty indeces
    all_words = list(all_words)
    
    word_count = np.zeros(len(all_words), dtype=np.int64)
    for i in range(len(word_count)):
       for j in range(len(title)):
            if all_words[i] in title[j]:
                word_count[i] += 1
            
    for i in range(len(all_words)):
        if (word_count[i]>0.85*len(title)):
            all_words[i] = ""

    all_words = list(set(all_words))
    all_words.remove("")
    
    #create brand vector 
    all_brands =  list(("philips", "supersonic", "sharp", "samsung", 
               "toshiba", "hisense", "sony", "lg",  "sanyo",
               "coby", "panasonic", "rca", "vizio", "naxa",
               "sansui", "viewsonic", "avue", "insignia",
               "sunbritetv", "magnavox", "jvc", "haier", 
               "optoma", "nec", "proscan", "venturer", 
               "westinghouse", "pyle", "dynex", "magnavox", 
               "sceptre", "tcl", "mitsubishi", "open box", 
               "curtisyoung", "compaq", "hannspree", 
               "upstar", "azend", "seiki", "craig",
               "contex", "affinity", "hiteker", "epson", 
               "elo", "pyle", "hello kitty", "gpx", "sigmac", 
               "venturer", "elite"))

    brand = [0]*len(title)
    for i in range(0,len(all_brands)):
        for j in range(0,len(title)):
            if all_brands[i] in title[j]:
                brand[j] = all_brands[i]
    
    #create size vector
    all_sizes = [0]*100
    for i in range(0, len(all_sizes)):
        all_sizes[i] = str(i)+'inch'

    size = [0]*len(title)
    for i in range(0, len(all_sizes)):
        for j in range(0, len(title)):
            if all_sizes[i] in title[j]:
                size[j] = all_sizes[i]
                
    title = pd.Series(title)
    shop = pd.Series(shop)
    brand = pd.Series(brand)
    size = pd.Series(size)
    transformed =  pd.concat([title, shop, brand, size], axis = 1)
    transformed.set_axis(['title', 'shop', 'brand', 'size'], axis='columns', inplace=True) 
    
    return transformed, all_words

def sig_band_mat(cleandata, all_words, r, b): #works
    title = cleandata["title"]
    binmat = np.zeros((len(all_words),len(title)), dtype=np.int64)
    for i  in range(len(all_words)): #generating binary  matrix
        for j  in range(len(title)):
            if all_words[i] in title[j]:
                binmat[i,j] = 1
                
    
    
    perm = r*b
    count =  0
    sigmat = np.zeros((perm, len(title)), dtype=np.int64)
    for i in range(perm):
        np.random.shuffle(binmat)
        count  = i
        for j in range(len(title)):
            obs = binmat[:,j]
            for k  in  range(len(all_words)):
                if obs[k] == 1:
                    sigmat[count, j]  = k
                    break
    
    z = np.arange(0, perm, r)
    band_D = {}
    for i in range(len((z))-1):
        band_D["band{0}".format(i)]=sigmat[z[i]:z[i+1]][:]
        band_D["band{0}".format(b-1)]=sigmat[z[b-1]:][:]   
    
    bandfin = np.zeros((b, len(title)))
    for i in range(0,b):
        for j in range(0,len(title)):
           merge1 = band_D['band'+str(i)][:,j] #iterates over bands for al obs
           merge2 = [str(int) for int in merge1] #as string
           merge3 = ''.join(merge2) #joins
           bandfin[i,j] = np.sqrt(float(merge3))
    
    return sigmat, bandfin, binmat
  
    
def candidate_matrix(cleandata, bandmat, r, b):

    candidates = np.zeros((len(cleandata), len(cleandata)), dtype = np.int64)    
    
    for i in range(0,len(cleandata)-1):
        for j in range(i+1,len(cleandata)):
            for k in range(0,b):
                if (bandmat[k,i] == bandmat[k,j]):
                    candidates[i,j] = 1
                    candidates[j,i] = 1
                    break
    
    return candidates



def dissimilarity_matrix(cleandata, can_mat, bin_mat, measure):
    
    brand = cleandata["brand"]
    shop  = cleandata["shop"]
    size = cleandata["size"]
    
    dis_mat = can_mat.astype('float64')
    dis_mat[dis_mat==0] = 100000
    
    for i in range(0,len(dis_mat)): 
        dis_mat[i,i] = 0
        
    def jacdis(A,B):
        intersection = np.logical_and(A,B)
        union = np.logical_or(A,B)
        distance = 1 - (intersection.sum() / float(union.sum()))
        return distance

    def cosine(A, B):
        return spatial.distance.cosine(A, B)
    
    #given certain dissimilarity measure, change 1's into distances
    if (measure == 'jacard'):
        for i in range(0,len(dis_mat)):
            for j in range(i+1,len(dis_mat)):
                if dis_mat[i,j]==1 and shop[i] != shop[j] and brand[i]==brand[j] and  size[j]==size[j]:
                    dis_mat[i,j] = jacdis(bin_mat[:,i], bin_mat[:,j])
                    dis_mat[j,i] = jacdis(bin_mat[:,i], bin_mat[:,j])
                else:
                    dis_mat[i,j] = 100000
                    dis_mat[j,i] = 100000
        
    
    elif (measure == 'cosine'):
        for i in range(0,len(dis_mat)):
            for j in range(i+1,len(dis_mat)):
                if dis_mat[i,j]==1 and shop[i] != shop[j] and brand[i]==brand[j] and  size[j]!=size[j]:
                    dis_mat[i,j] = cosine(bin_mat[:,i], bin_mat[:,j])
                    dis_mat[j,i] = cosine(bin_mat[:,i], bin_mat[:,j])
        
        
    return dis_mat


def clustering(dismatrix, threshold, data):
    clustering = AgglomerativeClustering(affinity='precomputed', 
                                         linkage='complete', distance_threshold = threshold, n_clusters=None)
    clustering = clustering.fit(dismatrix)
    labels = clustering.labels_
    pred_dubs = []
    
    for i in range(0, clustering.n_clusters_):
        clusprods = np.where(labels == i)[0]
        if (len(clusprods)>1):
            pred_dubs.extend(list(combinations(clusprods, 2)))
            
    IDs = data['modelID']
    for i in range(len(IDs)):
        IDs[i] = IDs[i].lower()

    real_dubs = []
    
    for modelID in IDs:
        if modelID not in real_dubs:
            dubs = np.where(IDs == modelID)[0]
            if (len(dubs)>1):
                real_dubs.extend(list(combinations(dubs, 2))) #counted double


    set_real = set(real_dubs)
    real_dubs = list(set_real)
    
    return pred_dubs, real_dubs



def Performance(data, r, b, threshold, measure):
    
    #First transform/clean imput "data"
    cleandata = clean_transform(data)[0]
    all_words = clean_transform(data)[1]
    
    #Use results + imputs "r,b" to define binary- and bandmatrix
    bandmat = sig_band_mat(cleandata, all_words, r, b)[1]
    binmat = sig_band_mat(cleandata, all_words, r, b)[2]
    
    #Use the signature- and bandmatrix to make the candidate matrix
    candidate = candidate_matrix(cleandata, bandmat,r,b)
    
    #use the candidate matrix to compute a dissimilirity matrix, 
    #where we use certain dis measure on signature matrix
    dismatrix = dissimilarity_matrix(cleandata, candidate, binmat, measure)
    
    #after clustering, find vector of predicted duplicate pairs
    #and real dublicate pairs
    preds = clustering(dismatrix = dismatrix, threshold=threshold, data = df)[0]
    reals = clustering(dismatrix, threshold, data)[1]

    #now we want to find TP, FP, FN. 
    TP = 0
    FP = 0
    for i in range(0,len(preds)):
        if preds[i] in reals:
            TP += 1
        else:
            FP +=1
        
    FN = len(reals)-TP
    
    #now predict scores
    dups_found = TP
    comp_made = np.count_nonzero(candidate)/2
    total_dubs = len(reals)
    pair_quality = dups_found/comp_made
    pair_completeness = dups_found/total_dubs
    comp_total = (len(data)*(len(data)-1))/2
    
    frac = comp_made/comp_total
    
    
    #Formulas for F1 and F1*
    def F1_star(pq, pc):
        return 2*(pq*pc)/(pq+pc)

    def F1(TP, FP, FN):
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        return 2*(precision*recall)/(precision+recall)

    
    F1_star = F1_star(pair_quality, pair_completeness)
    F1 = F1(TP, FP, FN)
    
    return frac, pair_quality, pair_completeness, F1_star, F1



#%% Bootstrapping to generate 5 test-train samples

from sklearn.utils import resample

def bootstrap(data, seed):
    bootstrap = resample(df, replace=True, n_samples=len(df), random_state=seed)
    train = bootstrap.drop_duplicates()
    test = pd.concat([df, train]).drop_duplicates(keep=False)
    
    return  train, test

train1, test1 =  bootstrap(df, 11)[0].reset_index(drop=True), bootstrap(df, 11)[1].reset_index(drop=True)
train2, test2 =  bootstrap(df, 12)[0].reset_index(drop=True), bootstrap(df, 12)[1].reset_index(drop=True)
train3, test3 =  bootstrap(df, 13)[0].reset_index(drop=True), bootstrap(df, 13)[1].reset_index(drop=True)
train4, test4 =  bootstrap(df, 14)[0].reset_index(drop=True), bootstrap(df, 14)[1].reset_index(drop=True)
train5, test5 =  bootstrap(df, 15)[0].reset_index(drop=True), bootstrap(df, 15)[1].reset_index(drop=True)


#%% Optimizing code
#Optimize our F1 score with respect to:
#1) threshold for clustering 
#2) optimal words vector (which ones to leave out based on frequency?)


def average_score(r, b, threshold, measure):
    F1_1 = Performance(train1, r, b, threshold, measure)[4]
    F1_2 = Performance(train2, r, b, threshold, measure)[4]
    F1_3 = Performance(train3, r, b, threshold, measure)[4]
    F1_4 = Performance(train4, r, b, threshold, measure)[4]
    F1_5 = Performance(train5, r, b, threshold, measure)[4]
    
    return (F1_1+F1_2+F1_3+F1_4+F1_5)/5


#------------------------------------------------------------------------------------------
#Let's see for different r and b


def all_scores(r, b, threshold, measure):
    results = np.zeros((5, 5))
    results[:,0] = Performance(test1, r, b, threshold, measure)
    results[:,1] = Performance(test2, r, b, threshold, measure)
    results[:,2] = Performance(test3, r, b, threshold, measure)
    results[:,3] = Performance(test4, r, b, threshold, measure)
    results[:,4] = Performance(test5, r, b, threshold, measure)
    
    avg_results = results.mean(axis=1)
    return results, avg_results
    
results10_10 = all_scores(10,10,0.5,'jacard')
results8_13 = all_scores(8, 13, 0.5, 'jacard')
results5_20 = all_scores(5, 20, 0.5, 'jacard')
results4_25 = all_scores(4, 25, 0.5, 'jacard')
results3_34 = all_scores(3, 34, 0.5, 'jacard')
results2_50 = all_scores(2, 50, 0.5, 'jacard')
results1_100 = all_scores(1, 100, 0.5, 'jacard')

results7_15 = all_scores(7, 15, 0.5, 'jacard')
results6_16 = all_scores(6, 16, 0.5, 'jacard')

scores = pd.DataFrame(columns=['r10_b10', 'r8_b13', 'r7_b15', 'r6_b16', 'r5_b20',
                               'r4_b25', 'r3_b34', 'r2_b50', 'r1_b100'], 
                      index=['frac_comp', 'PQ', 'PC', 'F1*', 'F1'])

scores['r10_b10'], scores['r8_b13'], scores['r7_b15'], scores['r6_b16'], scores['r5_b20'],scores['r4_b25'], scores['r3_b34'], scores['r2_b50'], scores['r1_b100'] = results10_10[1], results8_13[1],results7_15[1], results6_16[1], results5_20[1], results4_25[1], results3_34[1], results2_50[1], results1_100[1]

#%% Making graphs: PQ, PC, F1* and F1, all vs frac_comp

import matplotlib.pyplot as plt

#Pair quality plot
plt.plot(scores.iloc[0], scores.iloc[1], 'black')
plt.axis([0,0.25, 0,0.025])
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair quality')
plt.show

#Pair completeness plot
plt.plot(scores.iloc[0], scores.iloc[2], 'black')
plt.axis([0,1, 0,0.7])
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair completeness')
plt.show

#F1* plot
plt.plot(scores.iloc[0], scores.iloc[3], 'black')
#plt.axis([0,1, 0,0.5])
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1*-measure')
plt.show

#F1 plot
plt.plot(scores.iloc[0], scores.iloc[4], 'black')
#plt.axis([0,1, 0,0.5])
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1-measure')
plt.show

#%%Saving scores

#scores.to_excel('scores.xlsx', index=False)
scores = pd.read_excel('scores.xlsx')

