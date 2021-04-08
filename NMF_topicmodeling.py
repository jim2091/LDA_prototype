# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:00:58 2021

@author: LJB
"""

## lovit의 코드 사용 ##

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from lovit_textmining_dataset.navernews_10days import get_bow

x, idx_to_vocab, vocab_to_idx = get_bow(date='2016-10-20', tokenize='noun')

n_topics = 100
n_docs, n_terms = x.shape
#가져온 x 안에 shape가 있음. 30091개의 문서, 9774개의 단어.

nmf = NMF(n_components=n_topics)
y = nmf.fit_transform(normalize(x, norm='l1')) # shape = (n_docs, n_topics)
# l1 normalization을 하는 이유는 각 문서의 길이가 다를 수 있고, 그 차이를 알 수 없으므로.
components = nmf.components_ # shape = (n_topics, n_terms)


import numpy as np
from sklearn.preprocessing import normalize

##여기 두 함수는 ldavis를 사용하기 위한 단계임
##ldavis의 경우 확률형식으로 정의된 doc_topic, topic_term 벡터들이 필요한데
##지금 component들의 경우 그 정도를 나타내고 있는 벡터이지만 확률의 형식은 아니다(합이 1이 아님)
##따라서 확률형태로 만들어주기 위한 함수를 만든 것

def y_to_doc_topic(y):
    n_topics = y.shape[1]
    base = 1 / n_topics
    doc_topic_prob = normalize(y, norm='l1')
    rowsum = doc_topic_prob.sum(axis=1)
    doc_topic_prob[np.where(rowsum == 0)[0]] = base
    return doc_topic_prob

def components_to_topic_term(components):
    n_terms = components.shape[1]
    base = 1 / n_terms
    topic_term_prob = normalize(components, norm='l1')
    rowsum = topic_term_prob.sum(axis=1)
    topic_term_prob[np.where(rowsum == 0)[0]] = base
    return topic_term_prob

doc_topic_prob = y_to_doc_topic(y)
topic_term_prob = components_to_topic_term(components)

doc_lengths = np.asarray(x.sum(axis=1)).reshape(-1)
term_frequency = np.asarray(x.sum(axis=0)).reshape(-1)

from pyLDAvis import prepare, show

prepared_data = prepare(
    topic_term_prob,
    doc_topic_prob,
    doc_lengths,
    idx_to_vocab,
    term_frequency,
    R = 30 # num of displayed terms
)

show(prepared_data)
