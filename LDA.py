# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:54:39 2021

@author: LJB
"""

###ratsgo 님의 코드 사용 ###

from collections import Counter
import random

documents = [["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
["R", "Python", "statistics", "regression", "probability"],
["machine learning", "regression", "decision trees", "libsvm"],
["Python", "R", "Java", "C++", "Haskell", "programming languages"],
["statistics", "probability", "mathematics", "theory"],
["machine learning", "scikit-learn", "Mahout", "neural networks"],
["neural networks", "deep learning", "Big Data", "artificial intelligence"],
["Hadoop", "Java", "MapReduce", "Big Data"],
["statistics", "R", "statsmodels"],
["C++", "deep learning", "artificial intelligence", "probability"],
["pandas", "R", "Python"],
["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
["libsvm", "regression", "support vector machines"]]

random.seed(0)

# topic 수 지정
K=4

#################변수선언##################

# 각 토픽이 각 문서에 할당되는 횟수
# Counter로 구성된 리스트
# 각 Counter는 각 문서를 의미
document_topic_counts = [Counter() for _ in documents]


# 각 단어가 각 토픽에 할당되는 횟수
# Counter로 구성된 리스트
# 각 Counter는 각 토픽을 의미
topic_word_counts = [Counter() for _ in range(K)]

# 각 토픽에 할당되는 총 단어수
# 숫자로 구성된 리스트
# 각각의 숫자는 각 토픽을 의미함
topic_counts = [0 for _ in range(K)]

# 각 문서에 포함되는 총 단어수
# 숫자로 구성된 리스트
# 각각의 숫자는 각 문서를 의미함
# list(map(함수, 리스트)) 로 쓰면 리스트의 각 원소에 함수를 적용하여 새 리스트에 넣어줌
# 이번에 처음 보는 방법!
document_lengths = list(map(len, documents))

# 단어 종류의 수
distinct_words = set(word for document in documents for word in document)
V = len(distinct_words)

# 총 문서의 수
D = len(documents)

#####################함수 선언########################

# 수식에서의 A(d번째 문서가 k번째 토픽과 맺고 있는 연관성 정도)
def p_topic_given_document(topic, d, alpha=0.1):
    # 문서 d의 모든 단어 가운데 topic에 속하는
    # 단어의 비율 (alpha를 더해 smoothing)
    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))


# 수식에서의 B(d번째 문서의 n번째 단어(w(d,n))가 k번째 토픽과 맺고 있는 연관성 정도)
def p_word_given_topic(word, topic, beta=0.1):
    # topic에 속한 단어 가운데 word의 비율
    # (beta를 더해 smoothing)
    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + V * beta))

# 수식에서의 A*B (d번째 문서 i번째 단어의 토픽 z(d,i)가 j번째에 할당될 확률)
def topic_weight(d, word, k):
    # 문서와 문서의 단어가 주어지면
    # k번째 토픽의 weight를 반환
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)



def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k) for k in range(K)])

def sample_from(weights):
    # i를 weights[i] / sum(weights)
    # 확률로 반환
    total = sum(weights)
    # 0과 total 사이를 균일하게 선택
    rnd = total * random.random()
    # 아래 식을 만족하는 가장 작은 i를 반환
    # weights[0] + ... + weights[i] >= rnd
    for i, w in enumerate(weights):
        # enumerate는 인덱스와 원소를 튜플(,)의 형태로 만들어줌.
        rnd -= w
        if rnd <= 0:
            return i

####################### main #######################

# 각 단어를 임의의 토픽에 랜덤 배정
# w_d,n 모두를 임의 배정한 것.
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]

# 위와 같이 랜덤 초기화한 상태에서
# AB를 구하는 데 필요한 숫자를 세어봄
for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

AB_list=[]
for iter in range(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):
            # 깁스 샘플링 수행을 위해
            # 샘플링 대상 word의 topic정보들을 제외하고 세어봄
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            # 깁스 샘플링 대상 word의 topic정보들을 제외한 
            # 말뭉치 모든 word의 topic 정보를 토대로
            # 샘플링 대상 word의 새로운 topic을 선택
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic

            # 샘플링 대상 word의 새로운 topic을 반영해 
            # 말뭉치 정보 업데이트
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1
    AB_list.append(topic_weight(0, 0, 0))