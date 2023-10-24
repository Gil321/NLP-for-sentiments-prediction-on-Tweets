# In[1]:
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:48:22 2022

@author: baihaicen
"""

# In[2]:


#get_ipython().run_line_magic('run', 'setup.ipynb')


# # All the basic preprocessing in one place
# 
# - text cleaning
# - remove stop words
# - lemmatizaion
# - stemming

# In[3]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import re

import string

# In[4]:

#

train = pd.read_csv(f"data/Corona_NLP_train.csv",encoding='latin-1')
train.head()

test = pd.read_csv(f"data/Corona_NLP_test.csv")
test.head()

combi = train.append(test, ignore_index=True)


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 


combi['tweet'] = np.vectorize(remove_pattern)(combi['OriginalTweet'],"@[\w]*")

combi['tweet'] = combi['tweet'].str.lower()

combi['tweet'] = combi['tweet'].str.replace("https*\S+", " ")

combi['tweet'] = combi['tweet'].str.replace("#\S+", " ")

#combi['tweet'] = combi['tweet'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
combi['tweet'] = combi['tweet'].str.replace("[^a-zA-Z#]", " ")

stop_words = stopwords.words("english")
combi['tweet'] = combi['tweet'].apply(lambda x: ' '.join([word for word in x.split(' ') if word not in stop_words]))

##
tokenized_tweet = combi['tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

## stemming
from nltk.stem.porter import *
stemmer = PorterStemmer()

stemmized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
stemmized_tweet.head()

#


#pip install -U sacremoses
from sacremoses import MosesTokenizer, MosesDetokenizer

detokenizer = MosesDetokenizer()

for i in range(len(stemmized_tweet)):
  stemmized_tweet[i] = detokenizer.detokenize(stemmized_tweet[i], return_str=True)

combi['tweet'] = stemmized_tweet


######
#pip install textblob
from textblob import TextBlob

def analize_sentiment(tw):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(tw)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# We create a column with the result of the analysis:
combi['EM'] = np.array([ analize_sentiment(tw) for tw in combi['tweet'] ])
# We display the updated dataframe with the new column:
display(combi.head(10))

#
df_train_s = pd.DataFrame(combi[:41157], columns=["tweet", "EM"])
df_test_s = pd.DataFrame(combi[41157:], columns=["tweet", "EM"])

x_train = df_train_s["tweet"]
y_train = df_train_s["EM"]

x_test = df_test_s["tweet"]
y_test = df_test_s["EM"]

sentiment = df_train_s.groupby("EM").size()
sentiment
#######

# In[5]:

########
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_1_u = TfidfVectorizer(max_df=0.90, min_df=0, max_features=1000, ngram_range=(1,1))
tfidf_1_b = TfidfVectorizer(max_df=0.90, min_df=0, max_features=1000, ngram_range=(1,2))
tfidf_1_t = TfidfVectorizer(max_df=0.90, min_df=0, max_features=1000, ngram_range=(1,3))
tfidf_1_f = TfidfVectorizer(max_df=0.90, min_df=0, max_features=1000, ngram_range=(1,4))

tfidf_3_u = TfidfVectorizer(max_df=0.90, min_df=0, max_features=3000, ngram_range=(1,1))
tfidf_3_b = TfidfVectorizer(max_df=0.90, min_df=0, max_features=3000, ngram_range=(1,2))
tfidf_3_t = TfidfVectorizer(max_df=0.90, min_df=0, max_features=3000, ngram_range=(1,3))
tfidf_3_f = TfidfVectorizer(max_df=0.90, min_df=0, max_features=3000, ngram_range=(1,4))

x_train_1_u = tfidf_1_u.fit_transform(x_train)
x_train_1_b = tfidf_1_b.fit_transform(x_train)
x_train_1_t = tfidf_1_t.fit_transform(x_train)
x_train_1_f = tfidf_1_f.fit_transform(x_train)

x_train_3_u = tfidf_3_u.fit_transform(x_train)
x_train_3_b = tfidf_3_b.fit_transform(x_train)
x_train_3_t = tfidf_3_t.fit_transform(x_train)
x_train_3_f = tfidf_3_f.fit_transform(x_train)
y_train

x_test_1_u = tfidf_1_u.fit_transform(x_test)
x_test_1_b = tfidf_1_b.fit_transform(x_test)
x_test_1_t = tfidf_1_t.fit_transform(x_test)
x_test_1_f = tfidf_1_f.fit_transform(x_test)

x_test_3_u = tfidf_3_u.fit_transform(x_test)
x_test_3_b = tfidf_3_b.fit_transform(x_test)
x_test_3_t = tfidf_3_t.fit_transform(x_test)
x_test_3_f = tfidf_3_f.fit_transform(x_test)
y_test

# In[6]:
#########
#SVM

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

# In[6.1]
#clf = svm.SVC(kernel='linear') #支持 非/线性kernel！！;默认RBF(radial basis function kernel);non-linear核不支持high dimensions和small train set --》overfitting
svm_clf = svm.LinearSVC(penalty='l2', C=1) #仅支持线性kernel，不支持非线性kernel


#clf.fit(x_train_1_u,y_train)
svm_clf.fit(x_train_1_u,y_train)

#y_pred = clf.predict(x_test_1_u)
y_pred = svm_clf.predict(x_test_1_u)


print(metrics.classification_report(y_test,y_pred))
print("Precision Score : ",precision_score(y_test, y_pred, average='micro'))
print("Recall Score : ",recall_score(y_test, y_pred, average='micro'))
print("F1 Score : ",f1_score(y_test, y_pred, average='micro'))
print("Accuracy Score : ",accuracy_score(y_test, y_pred))


def svm_test(x_train_set,x_test_set):
    
    #clf = svm.SVC(kernel='linear', random_state=123) #支持 非/线性kernel！！;默认RBF(radial basis function kernel);non-linear核不支持high dimensions和small train set --》overfitting
    svm_clf = svm.LinearSVC(penalty='l2', C=1,random_state=123) #仅支持线性kernel，不支持非线性kernel

    #clf.fit(x_train_1_u,y_train)
    svm_clf.fit(x_train_set,y_train)

    #y_pred = clf.predict(x_test_1_u)
    y_pred = svm_clf.predict(x_test_set)

    print(metrics.classification_report(y_test,y_pred))
    print("Precision Score : ",precision_score(y_test, y_pred, average='micro'))
    print("Recall Score : ",recall_score(y_test, y_pred, average='micro'))
    print("F1 Score : ",f1_score(y_test, y_pred, average='micro'))
    print("Accuracy Score : ",accuracy_score(y_test, y_pred))

svm_test(x_train_1_u,x_test_1_u)


def svm_test(x_train_set,x_test_set):
    
    #clf = svm.SVC(kernel='linear',random_state=123) #支持 非/线性kernel！！;默认RBF(radial basis function kernel);non-linear核不支持high dimensions和small train set --》overfitting
    svm_clf = svm.LinearSVC(penalty='l2', C=1, random_state=123) #仅支持线性kernel，不支持非线性kernel

    #clf.fit(x_train_1_u,y_train)
    svm_clf.fit(x_train_set,y_train)

    #y_pred = clf.predict(x_test_1_u)
    y_pred = svm_clf.predict(x_test_set)

    return[precision_score(y_test, y_pred, average='micro'),
           recall_score(y_test, y_pred, average='micro'),
           f1_score(y_test, y_pred, average='micro'),
           accuracy_score(y_test, y_pred)]


#
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

svm_clf = svm.LinearSVC(penalty='l2', C=1, random_state=123)
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)


svm_clf.fit(x_train_1_u,y_train)

score1 = cross_val_score(svm_clf, x_test_1_u, y_test, cv=kfold, scoring='accuracy')
print("Accuracy SCore(CV=10): ", score1.mean())

score2 = cross_val_score(svm_clf, x_test_1_u, y_test, cv=kfold, scoring='precision_weighted' )
print("Precision Score(CV=10): ", score2.mean())  

score3 = cross_val_score(svm_clf, x_test_1_u, y_test, cv=kfold, scoring='f1_weighted' )
print("F1 Score(CV=10): ", score3.mean())  

score4 = cross_val_score(svm_clf, x_test_1_u, y_test, cv=kfold, scoring='recall_weighted' )
print("Recall Score(CV=10): ", score4.mean())   

def svm_cv_test(x_train_set,x_test_set):

    svm_clf = svm.LinearSVC(penalty='l2', C=1, random_state=123)
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)

    svm_clf.fit(x_train_set,y_train)

    score1 = cross_val_score(svm_clf, x_test_set, y_test, cv=kfold, scoring='accuracy')
    print("Accuracy SCore(CV=10): ", score1.mean())
    score2 = cross_val_score(svm_clf, x_test_set, y_test, cv=kfold, scoring='precision_weighted' )
    print("Precision Score(CV=10): ", score2.mean())  
    score3 = cross_val_score(svm_clf, x_test_set, y_test, cv=kfold, scoring='f1_weighted' )
    print("F1 Score(CV=10): ", score3.mean())  
    score4 = cross_val_score(svm_clf, x_test_set, y_test, cv=kfold, scoring='recall_weighted' )
    print("Recall Score(CV=10): ", score4.mean())   

svm_cv_test(x_train_1_u,x_test_1_u)

# In[6.2]

def svm_cv_test(x_train_set, x_test_set):

    svm_clf = svm.LinearSVC(penalty='l2', C=1, random_state=123)
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)

    svm_clf.fit(x_train_set, y_train)

    return[cross_val_score(svm_clf, x_test_set, y_test, cv=kfold, scoring='precision_weighted').mean(),
           cross_val_score(svm_clf, x_test_set, y_test, cv=kfold,
                           scoring='recall_weighted').mean(),
           cross_val_score(svm_clf, x_test_set, y_test,
                           cv=kfold, scoring='f1_weighted').mean(),
           cross_val_score(svm_clf, x_test_set, y_test, cv=kfold, scoring='accuracy').mean()]


svm_cv_test(x_train_1_u,x_test_1_u)
    

svm_test_dataframe = pd.DataFrame([svm_test(x_train_1_u, x_test_1_u),
                                   svm_test(x_train_3_u, x_test_3_u),
                                   svm_test(x_train_1_b, x_test_1_b),
                                   svm_test(x_train_3_b, x_test_3_b),
                                   svm_test(x_train_1_t, x_test_1_t),
                                   svm_test(x_train_3_t, x_test_3_t),
                                   svm_test(x_train_1_f, x_test_1_f),
                                   svm_test(x_train_3_f, x_test_3_f)],
                                  columns=['Precision', 'Recall', 'F1', 'Accuracy'],
                                  index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

svm_test_dataframe

svm_cv_dataframe = pd.DataFrame([svm_cv_test(x_train_1_u, x_test_1_u),
                                 svm_cv_test(x_train_3_u, x_test_3_u),
                                 svm_cv_test(x_train_1_b, x_test_1_b),
                                 svm_cv_test(x_train_3_b, x_test_3_b),
                                 svm_cv_test(x_train_1_t, x_test_1_t),
                                 svm_cv_test(x_train_3_t, x_test_3_t),
                                 svm_cv_test(x_train_1_f, x_test_1_f),
                                 svm_cv_test(x_train_3_f, x_test_3_f)],
                                columns=['Precision(CV=10)', 'Recall(CV=10)', 'F1(CV=10)', 'Accuracy(CV=10)'],
                                index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

svm_cv_dataframe

svm_res = pd.concat([svm_test_dataframe, svm_cv_dataframe], axis=1, join='outer')
svm_res.insert(0, 'Distance', [1000, 3000, 1000, 3000, 1000, 3000, 1000, 3000])

svm_res.to_csv('SVM_result.csv',encoding = 'utf8')                         

# In[7]:
#######
#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn  import  metrics

def knn_test(x_train_set,x_test_set):
    
    knn_clf = KNeighborsClassifier() #k值，（默认5),neighbors
    knn_clf.fit(x_train_set,y_train)
    
    y_pred = knn_clf.predict(x_test_set)

    return[precision_score(y_test, y_pred, average='micro'),
           recall_score(y_test, y_pred, average='micro'),
           f1_score(y_test, y_pred, average='micro'),
           accuracy_score(y_test, y_pred)]


def knn_cv_test(x_train_set, x_test_set):

    knn_clf = KNeighborsClassifier() #k值，（默认5),neighbors
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)

    knn_clf.fit(x_train_set, y_train)

    return[cross_val_score(knn_clf, x_test_set, y_test, cv=kfold, scoring='precision_weighted').mean(),
           cross_val_score(knn_clf, x_test_set, y_test, cv=kfold,
                           scoring='recall_weighted').mean(),
           cross_val_score(knn_clf, x_test_set, y_test,
                           cv=kfold, scoring='f1_weighted').mean(),
           cross_val_score(knn_clf, x_test_set, y_test, cv=kfold, scoring='accuracy').mean()]


    
knn_test_dataframe = pd.DataFrame([knn_test(x_train_1_u, x_test_1_u),
                                   knn_test(x_train_3_u, x_test_3_u),
                                   knn_test(x_train_1_b, x_test_1_b),
                                   knn_test(x_train_3_b, x_test_3_b),
                                   knn_test(x_train_1_t, x_test_1_t),
                                   knn_test(x_train_3_t, x_test_3_t),
                                   knn_test(x_train_1_f, x_test_1_f),
                                   knn_test(x_train_3_f, x_test_3_f)],
                                  columns=['Precision', 'Recall', 'F1', 'Accuracy'],
                                  index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

knn_test_dataframe

knn_cv_dataframe = pd.DataFrame([knn_cv_test(x_train_1_u, x_test_1_u),
                                 knn_cv_test(x_train_3_u, x_test_3_u),
                                 knn_cv_test(x_train_1_b, x_test_1_b),
                                 knn_cv_test(x_train_3_b, x_test_3_b),
                                 knn_cv_test(x_train_1_t, x_test_1_t),
                                 knn_cv_test(x_train_3_t, x_test_3_t),
                                 knn_cv_test(x_train_1_f, x_test_1_f),
                                 knn_cv_test(x_train_3_f, x_test_3_f)],
                                columns=['Precision(CV=10)', 'Recall(CV=10)', 'F1(CV=10)', 'Accuracy(CV=10)'],
                                index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

knn_cv_dataframe

knn_res = pd.concat([knn_test_dataframe, knn_cv_dataframe], axis=1, join='outer')
knn_res.insert(0, 'Distance', [1000, 3000, 1000, 3000, 1000, 3000, 1000, 3000])
knn_res

knn_res.to_csv('KNN_result.csv',encoding = 'utf8')                         

# In[8]:
###
#Decision Tree
from sklearn import tree
from sklearn  import  metrics

# In[8.1]

dt_clf = tree.DecisionTreeClassifier(criterion='gini', random_state=123) #Difference between GINI <--> ENTROPY!
dt_clf.fit(x_train_1_u,y_train)

import matplotlib.pyplot as plt
test=[]
for i in range(200):
    clf=tree.DecisionTreeClassifier(max_depth=i+1
                                    ,criterion='gini'
                                    ,random_state=123
                                    ,splitter='random'
                                    
    )
    clf=clf.fit(x_train_1_f,y_train)
    score=clf.score(x_test_1_f,y_test)
    test.append(score)
    
max_index=np.argmax(test)
plt.figure(figsize=(10,8))
plt.plot(range(1,201),test,color='blue',label='max_depth')
plt.grid(alpha=0.3)
plt.legend()
plt.plot(max_index,test[max_index],'ks')
show_max= '['+str(max_index)+'  '+str(test[max_index])+']'
plt.annotate(show_max,xytext=(max_index,test[max_index]),xy=(max_index,test[max_index]))
plt.savefig('Optimization_nodes_Decision_Tree.jpg')
plt.show()

# In[8.2]

def dt_test(x_train_set,x_test_set):
    
    dt_clf = tree.DecisionTreeClassifier(criterion='gini', random_state=123) #Difference between GINI <--> ENTROPY!
    dt_clf.fit(x_train_set,y_train)
    
    y_pred = dt_clf.predict(x_test_set)

    return[precision_score(y_test, y_pred, average='micro'),
           recall_score(y_test, y_pred, average='micro'),
           f1_score(y_test, y_pred, average='micro'),
           accuracy_score(y_test, y_pred)]


def dt_cv_test(x_train_set, x_test_set):

    dt_clf = tree.DecisionTreeClassifier(criterion='gini', random_state=123)
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)

    dt_clf.fit(x_train_set, y_train)

    return[cross_val_score(dt_clf, x_test_set, y_test, cv=kfold, scoring='precision_weighted').mean(),
           cross_val_score(dt_clf, x_test_set, y_test, cv=kfold,
                           scoring='recall_weighted').mean(),
           cross_val_score(dt_clf, x_test_set, y_test,
                           cv=kfold, scoring='f1_weighted').mean(),
           cross_val_score(dt_clf, x_test_set, y_test, cv=kfold, scoring='accuracy').mean()]


    
dt_test_dataframe = pd.DataFrame([dt_test(x_train_1_u, x_test_1_u),
                                   dt_test(x_train_3_u, x_test_3_u),
                                   dt_test(x_train_1_b, x_test_1_b),
                                   dt_test(x_train_3_b, x_test_3_b),
                                   dt_test(x_train_1_t, x_test_1_t),
                                   dt_test(x_train_3_t, x_test_3_t),
                                   dt_test(x_train_1_f, x_test_1_f),
                                   dt_test(x_train_3_f, x_test_3_f)],
                                  columns=['Precision', 'Recall', 'F1', 'Accuracy'],
                                  index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

dt_test_dataframe

dt_cv_dataframe = pd.DataFrame([dt_cv_test(x_train_1_u, x_test_1_u),
                                 dt_cv_test(x_train_3_u, x_test_3_u),
                                 dt_cv_test(x_train_1_b, x_test_1_b),
                                 dt_cv_test(x_train_3_b, x_test_3_b),
                                 dt_cv_test(x_train_1_t, x_test_1_t),
                                 dt_cv_test(x_train_3_t, x_test_3_t),
                                 dt_cv_test(x_train_1_f, x_test_1_f),
                                 dt_cv_test(x_train_3_f, x_test_3_f)],
                                columns=['Precision(CV=10)', 'Recall(CV=10)', 'F1(CV=10)', 'Accuracy(CV=10)'],
                                index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

dt_cv_dataframe

dt_res = pd.concat([dt_test_dataframe, dt_cv_dataframe], axis=1, join='outer')
dt_res.insert(0, 'Distance', [1000, 3000, 1000, 3000, 1000, 3000, 1000, 3000])
dt_res

dt_res.to_csv('DT_result.csv',encoding = 'utf8') 

# In[9]:
##
#Random Forest
from sklearn.ensemble import RandomForestClassifier

# In[9.1]

kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)
score_test = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=123)
    rfc.fit(x_train_1_u, y_train)
    score = cross_val_score(rfc, x_test_1_u, y_test, cv=kfold).mean()
    score_test.append(score)
score_max = max(score_test)

print('max_score：{}'.format(score_max),
      'num_of_trees:{}'.format(score_test.index(score_max)*10+1))
max_index=(np.argmax(score_test))
x = np.arange(1,201,10)
plt.subplot(111)
plt.plot(x, score_test, 'r-')
plt.plot(max_index*10,score_test[max_index],'ks')
show_max= '['+str(max_index*10)+'  '+str(score_test[max_index])+']'
plt.annotate(show_max,xytext=(max_index,score_test[max_index]),xy=(max_index,score_test[max_index]))
plt.savefig('Learning curve_num_trees_Random_Forest.jpg')
plt.show()

# In[9.2]

def rf_test(x_train_set,x_test_set):
    
    rf_clf = RandomForestClassifier(criterion='gini', random_state=123) #Difference between GINI <--> ENTROPY!
    rf_clf.fit(x_train_set,y_train)
    
    y_pred = rf_clf.predict(x_test_set)

    return[precision_score(y_test, y_pred, average='micro'),
           recall_score(y_test, y_pred, average='micro'),
           f1_score(y_test, y_pred, average='micro'),
           accuracy_score(y_test, y_pred)]


def rf_cv_test(x_train_set, x_test_set):

    rf_clf = RandomForestClassifier(criterion='gini', random_state=123)
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)

    rf_clf.fit(x_train_set, y_train)

    return[cross_val_score(rf_clf, x_test_set, y_test, cv=kfold, scoring='precision_weighted').mean(),
           cross_val_score(rf_clf, x_test_set, y_test, cv=kfold,
                           scoring='recall_weighted').mean(),
           cross_val_score(rf_clf, x_test_set, y_test,
                           cv=kfold, scoring='f1_weighted').mean(),
           cross_val_score(rf_clf, x_test_set, y_test, cv=kfold, scoring='accuracy').mean()]


    
rf_test_dataframe = pd.DataFrame([rf_test(x_train_1_u, x_test_1_u),
                                   rf_test(x_train_3_u, x_test_3_u),
                                   rf_test(x_train_1_b, x_test_1_b),
                                   rf_test(x_train_3_b, x_test_3_b),
                                   rf_test(x_train_1_t, x_test_1_t),
                                   rf_test(x_train_3_t, x_test_3_t),
                                   rf_test(x_train_1_f, x_test_1_f),
                                   rf_test(x_train_3_f, x_test_3_f)],
                                  columns=['Precision', 'Recall', 'F1', 'Accuracy'],
                                  index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

rf_test_dataframe

rf_cv_dataframe = pd.DataFrame([rf_cv_test(x_train_1_u, x_test_1_u),
                                 rf_cv_test(x_train_3_u, x_test_3_u),
                                 rf_cv_test(x_train_1_b, x_test_1_b),
                                 rf_cv_test(x_train_3_b, x_test_3_b),
                                 rf_cv_test(x_train_1_t, x_test_1_t),
                                 rf_cv_test(x_train_3_t, x_test_3_t),
                                 rf_cv_test(x_train_1_f, x_test_1_f),
                                 rf_cv_test(x_train_3_f, x_test_3_f)],
                                columns=['Precision(CV=10)', 'Recall(CV=10)', 'F1(CV=10)', 'Accuracy(CV=10)'],
                                index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

rf_cv_dataframe

rf_res = pd.concat([rf_test_dataframe, rf_cv_dataframe], axis=1, join='outer')
rf_res.insert(0, 'Distance', [1000, 3000, 1000, 3000, 1000, 3000, 1000, 3000])
rf_res

rf_res.to_csv('RF_result.csv',encoding = 'utf8') 

# In[10]:
###
#Logistic Regression
from sklearn.linear_model import LogisticRegression

def lr_test(x_train_set,x_test_set):
    
    lr_clf = LogisticRegression()
    lr_clf.fit(x_train_set,y_train)
    
    y_pred = lr_clf.predict(x_test_set)

    return[precision_score(y_test, y_pred, average='micro'),
           recall_score(y_test, y_pred, average='micro'),
           f1_score(y_test, y_pred, average='micro'),
           accuracy_score(y_test, y_pred)]


def lr_cv_test(x_train_set, x_test_set):

    lr_clf = LogisticRegression()
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=123)

    lr_clf.fit(x_train_set, y_train)

    return[cross_val_score(lr_clf, x_test_set, y_test, cv=kfold, scoring='precision_weighted').mean(),
           cross_val_score(lr_clf, x_test_set, y_test, cv=kfold,
                           scoring='recall_weighted').mean(),
           cross_val_score(lr_clf, x_test_set, y_test,
                           cv=kfold, scoring='f1_weighted').mean(),
           cross_val_score(lr_clf, x_test_set, y_test, cv=kfold, scoring='accuracy').mean()]


    
lr_test_dataframe = pd.DataFrame([lr_test(x_train_1_u, x_test_1_u),
                                   lr_test(x_train_3_u, x_test_3_u),
                                   lr_test(x_train_1_b, x_test_1_b),
                                   lr_test(x_train_3_b, x_test_3_b),
                                   lr_test(x_train_1_t, x_test_1_t),
                                   lr_test(x_train_3_t, x_test_3_t),
                                   lr_test(x_train_1_f, x_test_1_f),
                                   lr_test(x_train_3_f, x_test_3_f)],
                                  columns=['Precision', 'Recall', 'F1', 'Accuracy'],
                                  index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

lr_test_dataframe

lr_cv_dataframe = pd.DataFrame([lr_cv_test(x_train_1_u, x_test_1_u),
                                 lr_cv_test(x_train_3_u, x_test_3_u),
                                 lr_cv_test(x_train_1_b, x_test_1_b),
                                 lr_cv_test(x_train_3_b, x_test_3_b),
                                 lr_cv_test(x_train_1_t, x_test_1_t),
                                 lr_cv_test(x_train_3_t, x_test_3_t),
                                 lr_cv_test(x_train_1_f, x_test_1_f),
                                 lr_cv_test(x_train_3_f, x_test_3_f)],
                                columns=['Precision(CV=10)', 'Recall(CV=10)', 'F1(CV=10)', 'Accuracy(CV=10)'],
                                index=['Unigram', '', 'Bigram', '', 'Trigram', '', 'Four-gram', ''])

lr_cv_dataframe

lr_res = pd.concat([lr_test_dataframe, lr_cv_dataframe], axis=1, join='outer')
lr_res.insert(0, 'Distance', [1000, 3000, 1000, 3000, 1000, 3000, 1000, 3000])
lr_res

lr_res.to_csv('LR_result.csv',encoding = 'utf8') 


# In[11]:
###
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['Precision', 'Recall', 'F1-score', 'Accuracy']
dt = list(dt_test_dataframe.max())
lr = list(lr_test_dataframe.max())
knn = list(knn_test_dataframe.max())
rf = list(rf_test_dataframe.max())
svm = list(svm_test_dataframe.max())

dt_cv = list(dt_cv_dataframe.max())
lr_cv = list(lr_cv_dataframe.max())
knn_cv = list(knn_cv_dataframe.max())
rf_cv = list(rf_cv_dataframe.max())
svm_cv = list(svm_cv_dataframe.max())



x = list(range(len(labels)))
total_width, n = 0.5, 4
width = total_width / n


b1 = plt.bar(x, dt, width=width, label='DT', tick_label=labels)
for i in range(len(x)):
    x[i]=x[i]+width
b2 = plt.bar(x, lr, width=width, label='LR')
for i in range(len(x)):
    x[i]=x[i]+width
b3 = plt.bar(x, knn, width=width, label='KNN')
for i in range(len(x)):
    x[i]=x[i]+width
b4 = plt.bar(x, rf, width=width, label='RF')
for i in range(len(x)):
    x[i]=x[i]+width
b5 = plt.bar(x, svm, width=width, label='SVM')

plt.legend(handles=[b1,b2,b3,b4,b5],loc="lower center",ncol=5)
plt.ylabel("Performance metric(%)",color = 'b')
plt.title("The best testing performance")
plt.savefig('The best testing performance.jpg')
plt.show

# In[12]:
x = list(range(len(labels)))
total_width, n = 0.5, 4
width = total_width / n

c1 = plt.bar(x, dt_cv, width=width, label='DT', tick_label=labels)
for i in range(len(x)):
    x[i]=x[i]+width
c2 = plt.bar(x, lr_cv, width=width, label='LR')
for i in range(len(x)):
    x[i]=x[i]+width
c3 = plt.bar(x, knn_cv, width=width, label='KNN')
for i in range(len(x)):
    x[i]=x[i]+width
c4 = plt.bar(x, rf_cv, width=width, label='RF')
for i in range(len(x)):
    x[i]=x[i]+width
c5 = plt.bar(x, svm_cv, width=width, label='SVM')

plt.legend(handles=[c1,c2,c3,c4,c5],loc="lower center",ncol=5)
plt.ylabel("Performance metric(%)",color = 'b')
plt.title("The best cross-validation performance")
plt.savefig('The best cross-validation performance.jpg')
plt.show




















