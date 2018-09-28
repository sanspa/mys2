#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:56:00 2018

@author: budi
"""

import matplotlib.pyplot as plt
from math import log
import random
from prosestext import *
import pandas as pd
import datetime

class SpamClassifier(object):
    def __init__(self, tweets,labels,method='bow',gram=1):
        self.tweets, self.labels = tweets,labels
        self.method = method
        self.gram = gram
        self.spam_tweets, self.ham_tweets = (labels==1).sum(),(labels==0).sum()
        self.total_tweets = self.spam_tweets + self.ham_tweets
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0.0
        self.sum_tf_idf_ham = 0.0
        self.tf_spam1 = dict()
        self.tf_ham1 = dict()
        self.tf_spam2 = dict()
        self.tf_ham2 = dict()
        self.tf_spam3 = dict()
        self.tf_ham3 = dict()
        self.word_count =0
        self.vocabsize =0
        self.stemfactory = StemmerFactory()
        self.stemmer = self.stemfactory.create_stemmer()
    
    def train(self):
        print('spam: ',self.spam_tweets)
        print('ham: ',self.ham_tweets)
        self.calc_TF_and_IDF()

        if self.method == 'tfidf':
            self.calc_TF_IDF()
        else:
            if self.method == 'bow':
                self.calc_prob()
            else:
                if self.method == 'stbo':
                    self.sbtrain()
            
    def calc_prob(self):
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1)/(self.spam_words + self.word_count)
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1)/ (self.ham_words + self.word_count)
            
        self.prob_spam_tweet = self.spam_tweets/ self.total_tweets
        self.prob_ham_tweet = self.ham_tweets / self.total_tweets

        
    def calc_TF_and_IDF(self):
        noOfMessages = self.tweets.shape[0]  
        count = list()
        for i in range(noOfMessages):
            processed = praproses(self.tweets[i],self.stemmer,gram=self.gram)
            for word in processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1   
        self.word_count = len(count)
        print
    
    
    def sbtrain(self):
        noOfMessages = self.tweets.shape[0]    
        #print(self.train_tweets['text'][0])       
        for i in range(noOfMessages):
            processed = praproses(self.tweets[i],self.stemmer,gram=1)
            #count = list() #           
            for word in processed:
                if self.labels[i]:
                    self.tf_spam1[word] = self.tf_spam1.get(word, 0) + 1
                else:
                    self.tf_ham1[word] = self.tf_ham1.get(word, 0) + 1
            #count = list() #           
            processed2 = praproses(self.tweets[i],self.stemmer,gram=2)
            for word in processed2:
                if self.labels[i]:
                    self.tf_spam2[word] = self.tf_spam2.get(word, 0) + 1
                else:
                    self.tf_ham2[word] = self.tf_ham2.get(word, 0) + 1
                    
            processed3 = praproses(self.tweets[i],self.stemmer,gram=3)
            for word in processed3:
                if self.labels[i]:
                    self.tf_spam3[word] = self.tf_spam3.get(word, 0) + 1
                else:
                    self.tf_ham3[word] = self.tf_ham3.get(word, 0) + 1
        self.vocabsize = len(self.tf_ham1) + len(self.tf_ham2)
    
                         
        
    def calc_TF_IDF(self):     
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) \
              * log((self.spam_tweets + self.ham_tweets)\
              / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
                    
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            #print(self.prob_spam[word])  
        
        #print("SPAM.....")
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_tweets\
              + self.ham_tweets) / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            
            self.sum_tf_idf_ham += self.prob_ham[word]
        
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
        
        self.prob_spam_tweet, self.prob_ham_tweet = self.spam_tweets \
          / self.total_tweets, self.ham_tweets / self.total_tweets
        #print('===HAM====')
        #print('prob_spam = {}'.format(self.prob_spam_tweet))
        #print('prob_ham = {}'.format(self.prob_ham_tweet))

    def classify(self, tweet):
        processed_text = praproses(tweet,self.stemmer,gram=self.gram)
        pSpam, pHam = 0, 0
        for word in processed_text:          
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
                print('kata spam ',word,' = ',self.tf_spam[word])

                if self.method == 'tfidf':
                    pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                else:
                    pSpam -= log(self.spam_words + self.word_count)
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
                print('kata ham ',word,' = ',self.tf_ham[word])

                if self.method == 'tfidf':
                    pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
                else:
                    pHam -= log(self.ham_words + self.word_count)
        pSpam += log(self.prob_spam_tweet)
        pHam += log(self.prob_ham_tweet)
            
        return pSpam >= pHam
    
    def sbclassify(self,stemmedtweet):
        """ Takes a list of strings as argument and returns the log-probability
        of the sentence using your language model. Use whatever data you
        computed in train() here."""
        hamscore = 0.0
        spamscore = 0.0
        words = praproses(stemmedtweet,self.stemmer,gram=2)
        for word in words:       
            wordtoken = word.split()
            tokenprev = wordtoken[0]
            tokennext = wordtoken[1]
            if word in self.tf_ham2:
                bicount = self.tf_ham2[word]
                bi_unicount = self.tf_ham1[tokenprev]
                hamscore += log(bicount)
                hamscore -= log(bi_unicount)
            else:
                if tokennext in self.tf_ham1:
                    unicount = self.tf_ham1[tokennext]
                else:
                    unicount = 0.4
                hamscore += log(0.4)
                hamscore += log(unicount)
                hamscore -= log(self.ham_words + self.vocabsize)
        
            if word in self.tf_spam2:
                bicount2 = self.tf_spam2[word]
                bi_unicount2 = self.tf_spam1[tokenprev]
                spamscore += log(bicount2)
                spamscore -= log(bi_unicount2)
            else:
                if tokennext in self.tf_spam1:
                    unicount2 = self.tf_spam1[tokennext]
                else:
                    unicount2 = 0.4
                spamscore += log(0.4)
                spamscore += log(unicount2)
                spamscore -= log(self.spam_words + self.vocabsize)        
        return spamscore >= hamscore
        
    
    def predict(self, test_data):
        result = dict()
        if self.method == 'stbo':
            for (i, tweet) in enumerate(test_data):
                result[i] = int(self.sbclassify(tweet))
        else:           
            for (i, tweet) in enumerate(test_data):
                result[i] = int(self.classify(tweet))
        return result
    
    def printP(self):
        print('spam_words:',self.spam_words)
        print('ham_words:',self.ham_words)
        print('prior spamm: ',len(self.prob_spam))
        print('prior ham: ',len(self.prob_ham))
        print('word in spam:',len(self.tf_spam))
        print('word in ham: ',len(self.tf_ham))
        print('total word: ',self.word_count)
        print('list keys:',len(list(self.tf_spam.keys())))
        
    

def metrics(labels, predictions,tweets):
    print('spam: ',(labels==1).sum())
    print('ham: ',(labels==0).sum())
    etext=[]
    eptext=[]
    elabel=[]
    true_pos, true_neg, false_pos, false_neg = 0,0,0,0
    for i in range(len(labels)):
        true_pos += int((labels[i] == 1) and (predictions[i] == 1))
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        if (labels[i] == 0 and predictions[i] == 1):
            false_pos += 1
            etext.append(tweets[i])
            eptext.append(prosestext(tweets[i]))
            elabel.append('fp')
        if (labels[i] == 1 and predictions[i] == 0):
            false_neg += 1
            etext.append(tweets[i])
            eptext.append(prosestext(tweets[i]))
            elabel.append('fn')
    edf = pd.DataFrame(list(zip(etext,eptext,elabel)),columns=['text','stemmedtext','label'])
    filename = '/home/budi/false'+datetime.datetime.now().strftime("%H%M%S")+ '.xlsx'
    writer = pd.ExcelWriter(filename,engine='xlsxwriter')
    edf.to_excel(writer,sheet_name='Sheet1')
    writer.save()
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy) 
    print('   ')
    print("True Positiv: ",true_pos)
    print("False Positiv: ",false_pos)
    print("True Negativ: ",true_neg)
    print("False Negativ: ",false_neg)
    
    
