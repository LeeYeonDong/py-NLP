import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import Counter
import numpy as np
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine
import stanza
from infomap import Infomap
from math import log2
import re
from random import randint, random
import math
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import norm
from scipy.stats import dirichlet
from scipy.stats import pareto
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import random
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import SpectralClustering
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import time
from nltk.util import ngrams
from sklearn.preprocessing import normalize
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, matthews_corrcoef
from scipy.spatial.distance import braycurtis


# 파일 경로 설정
# file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_processed.csv' # 수정
# file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined.csv' # 1000*20
file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined1000.csv'

# CSV 파일 불러오기
df_filtered = pd.read_csv(file_path)
df_filtered = df_filtered.astype(str)
df_filtered.dtypes
df_filtered = df_filtered[['id', 'title', 'keywords', 'year', 'abstract', 'authors']]
df_filtered = df_filtered.dropna(subset=['id', 'title', 'keywords', 'year', 'abstract', 'authors'])
df_filtered['keywords']

# keyword 열에서 가장 많은 키워드 수 계산
nltk.download('punkt')

# GloVe 임베딩 로드 함수
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 단어 벡터 간 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

# GloVe 임베딩 파일 경로
glove_file_path = r'D:\대학원\논문\textrank\rawdata\glove.6B.100d.txt' 
glove_embeddings = load_glove_embeddings(glove_file_path)

time.sleep(3)


# 문장 간 유사도를 계산하는 함수 (GloVe 임베딩 사용)
def sentence_similarity(sentence1, sentence2, embeddings):
    words1 = word_tokenize(sentence1.lower())
    words2 = word_tokenize(sentence2.lower())
    
    vectors1 = [embeddings[word] for word in words1 if word in embeddings]
    vectors2 = [embeddings[word] for word in words2 if word in embeddings]
    
    if not vectors1 or not vectors2:
        return 0  # 벡터가 없는 경우 유사도 0
    
    # 각 단어 벡터의 평균을 문장 벡터로 사용
    sentence_vector1 = np.mean(vectors1, axis=0)
    sentence_vector2 = np.mean(vectors2, axis=0)
    
    return cosine_similarity(sentence_vector1, sentence_vector2)

# Precision, Recall, F1 계산 함수
def calculate_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)
    
    true_positive = len(extracted_set & actual_set)
    false_positive = len(extracted_set - actual_set)
    false_negative = len(actual_set - extracted_set)
    
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

# ROUGE 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

# FM Index, ARI, MCC, Bray-Curtis Dissimilarity를 계산하는 함수
def calculate_additional_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)

    # 이진화된 벡터로 변환 (키워드 존재 여부에 따라 1, 0으로 구분)
    extracted_binary = [1 if word in extracted_set else 0 for word in actual]
    actual_binary = [1] * len(actual)

    if len(extracted_binary) == len(actual_binary):
        fm_index = fowlkes_mallows_score(actual_binary, extracted_binary)
        ari = adjusted_rand_score(actual_binary, extracted_binary)
        mcc = matthews_corrcoef(actual_binary, extracted_binary)
        bray_curtis = braycurtis(extracted_binary, actual_binary)
    else:
        fm_index, ari, mcc, bray_curtis = 0, 0, 0, 1  # 다른 크기의 경우 처리

    return fm_index, ari, mcc, bray_curtis

# DataFrame에 적용하여 모든 메트릭스 계산
def apply_metrics(df):
    # Precision, Recall, F1, ROUGE 계산
    df['metrics'] = df.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split()), axis=1)
    df[['precision', 'recall', 'f1']] = pd.DataFrame(df['metrics'].tolist(), index=df.index)

    df['rouge'] = df.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split()), axis=1)
    df['rouge1'] = df['rouge'].apply(lambda x: x['rouge1'].fmeasure)
    df['rougeL'] = df['rouge'].apply(lambda x: x['rougeL'].fmeasure)

    # FM Index, ARI, MCC, Bray-Curtis Dissimilarity 계산
    df['additional_metrics'] = df.apply(lambda row: calculate_additional_metrics(row['extracted_keywords'], row['keywords'].split()), axis=1)
    df[['FM_Index', 'ARI', 'MCC', 'Bray_Curtis']] = pd.DataFrame(df['additional_metrics'].tolist(), index=df.index)

    return df


#### 1.1 textrank
time.sleep(3)
df_filtered_11 = df_filtered.copy()
df_filtered_11.dtypes

# Textrank 키워드 추출 함수
def textrank_keywords(text, top_n=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences:
        keywords.extend(word_tokenize(sentence.lower()))
        if len(set(keywords)) >= top_n:
            break
    
    return list(set(keywords))[:top_n]

# 각 행의 'keywords'에서 단어 개수를 계산하여 'num_keywords' 열 생성
df_filtered_11['num_keywords'] = df_filtered_11['keywords'].apply(lambda x: len(x.split()))

# 'num_keywords'를 top_n으로 사용하여 'extracted_keywords' 생성
df_filtered_11['extracted_keywords'] = df_filtered_11.apply(
    lambda row: textrank_keywords(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_11[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_11 = apply_metrics(df_filtered_11)

# 최종 결과 출력
df_result11 = df_filtered_11[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 1.2 textrank + term frequency, term postion, word co-occurence
time.sleep(3)
df_filtered_12 = df_filtered.copy()
df_filtered_12.dtypes

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2):  # window size 조절
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank 키워드 추출 함수
def textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    tf = calculate_tf(text)
    
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 제목과 초록에서 단어의 중요도를 반영하여 유사도 행렬을 계산
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # word position importance 반영
                        weight_i = 1 if word_i in word_tokenize(title.lower()) else beta
                        weight_j = 1 if word_j in word_tokenize(title.lower()) else beta
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
                        similarity_matrix[i][j] += similarity * weight_i * weight_j  # 중요도 반영

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences:
        keywords.extend(word_tokenize(sentence.lower()))
        if len(set(keywords)) >= top_n:
            break
    
    return list(set(keywords))[:top_n]

# 각 행의 'keywords'에서 단어 개수를 계산하여 'num_keywords' 열 생성
df_filtered_12['num_keywords'] = df_filtered_12['keywords'].apply(lambda x: len(x.split()))

# 'num_keywords'를 top_n으로 사용하여 'extracted_keywords' 생성
df_filtered_12['extracted_keywords'] = df_filtered_12.apply( lambda row: textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5) if pd.notnull(row['abstract']) else [], axis=1)

# 데이터 프레임 출력 
print(df_filtered_12[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_12 = apply_metrics(df_filtered_12)

# 최종 결과 출력
df_result12 = df_filtered_12[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 1.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting 
time.sleep(3)
df_filtered_13 = df_filtered.copy()

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'likely', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2): # window size 조절
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i+1, min(i+1+window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank 키워드 추출 함수 수정
def textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    
    sentences = [s for s, w in weighted_sentences]
    weights = [w for s, w in weighted_sentences]
    
    words = word_tokenize(text.lower())
    
    # 단어의 TF 값 계산
    tf = calculate_tf(text)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # 유사도 행렬 계산
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # word position importance 반영
                        weight_i = 1 if word_i in word_tokenize(title.lower()) else beta
                        weight_j = 1 if word_j in word_tokenize(title.lower()) else beta
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
            similarity_matrix[i][j] = similarity * ((weights[i] + weights[j]) / 2)  # 가중치 적용
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences[:top_n]:
        keywords.extend(word_tokenize(sentence.lower()))
    
    return list(set(keywords))

# 키워드 개수를 계산하는 함수
def get_keyword_count(keywords):
    return len(keywords.split())

# 추출된 키워드를 데이터 프레임에 추가 (top_n을 각 행의 keywords 단어 개수로 설정)
df_filtered_13['extracted_keywords'] = df_filtered_13.apply(
    lambda row: textrank_keywords(
        row['title'], 
        row['abstract'], 
        top_n=get_keyword_count(row['keywords']), 
        beta=0.5
    ) if pd.notnull(row['abstract']) else [], 
    axis=1
)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_13[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_13 = apply_metrics(df_filtered_13)

# 최종 결과 출력
df_result13 = df_filtered_13[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 1.4 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Glove
time.sleep(3)
df_filtered_14 = df_filtered.copy()

# 단어 간 유사도 계산 함수 (GloVe 사용)
def word_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    else:
        return 0.0  # 임베딩이 없는 경우 유사도를 0으로 설정

# 유사도 행렬 계산 함수
def compute_similarity_matrix(words, embeddings, epsilon=1e-5):
    size = len(words)
    similarity_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i != j:
                similarity = word_similarity(words[i], words[j], embeddings)
                similarity_matrix[i][j] = similarity if similarity > epsilon else epsilon
    return similarity_matrix

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        for word in words:
            if word in ['sort of', 'kind of', 'a little', 'rather']:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        for word in words:
            if word in ['maybe', 'possibly', 'could', 'might']:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# TF 계산 함수 (Word Position Importance 반영)
def calculate_tf_with_position(title, abstract, beta=0.5):
    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    doc_length = len(words_title) + len(words_abstract)
    
    # 제목에 있는 단어의 TF는 1, 초록에 있는 단어의 TF는 β로 설정
    tf = {}
    
    for word in words_title:
        tf[word] = tf.get(word, 0) + 1 / doc_length
    
    for word in words_abstract:
        tf[word] = tf.get(word, 0) + beta / doc_length
    
    return tf

# Textrank 키워드 추출 함수 with GloVe (Word Position Importance 반영)
def textrank_keywords_glove_with_position(title, abstract, top_n=5, beta=0.5, embeddings=None):
    # 제목과 초록을 합쳐서 문장으로 분할
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    
    sentences = [s for s, w in weighted_sentences]
    weights = [w for s, w in weighted_sentences]
    
    words = word_tokenize(text.lower())
    
    # 단어의 TF 값 계산 (Word Position Importance 반영)
    tf = calculate_tf_with_position(title, abstract, beta=beta)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 계산
    similarity_matrix = compute_similarity_matrix(words, embeddings)
    
    # 유사도 행렬에 가중치 적용
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
            similarity_matrix[i][j] *= (weights[i] + weights[j]) / 2  # 가중치 적용
    
    # 그래프 생성 및 PageRank 계산
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    ranked_words = sorted(((scores[i], word) for i, word in enumerate(words)), reverse=True)
    
    # 상위 키워드 추출
    keywords = []
    seen_words = set()
    for _, word in ranked_words:
        if word not in seen_words and word.isalnum():
            keywords.append(word)
            seen_words.add(word)
        if len(keywords) >= top_n:
            break
    
    return keywords

# 추출된 키워드를 데이터 프레임에 추가 (top_n을 각 행의 keywords 단어 개수로 설정)
df_filtered_14['extracted_keywords'] = df_filtered_14.apply(lambda row: textrank_keywords_glove_with_position(
        row['title'], row['abstract'], top_n=get_keyword_count(row['keywords']), beta=0.5,  # β 값은 필요에 따라 조정 가능
        embeddings=glove_embeddings), axis=1)

# 데이터 프레임 출력 
print(df_filtered_14[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_14 = apply_metrics(df_filtered_14)

# 최종 결과 출력
df_result14 = df_filtered_14[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 2. textrank + TP-CoGlo-TextRank(GLove)
time.sleep(3)
df_filtered_2 = df_filtered.copy()

# 단어 간 유사도 계산 함수 (GloVe 사용)
def word_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    else:
        return 0.0  # 임베딩이 없는 경우 유사도를 0으로 설정

# 유사도 행렬 계산 with noise  # 데이터 프레임 크기가 작은 경우 인위적으로 noise 추가
def compute_similarity_matrix(words, embeddings, epsilon=1e-5):
    size = len(words)
    similarity_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i != j:
                similarity = word_similarity(words[i], words[j], embeddings)
                similarity_matrix[i][j] = similarity if similarity > epsilon else epsilon
    return similarity_matrix

# # without noise
# def compute_similarity_matrix(words, embeddings):
#     size = len(words)
#     similarity_matrix = np.zeros((size, size))
    
#     for i in range(size):
#         for j in range(size):
#             if i != j:
#                 similarity_matrix[i][j] = word_similarity(words[i], words[j], embeddings)
#     return similarity_matrix

# TP-CoGlo-TextRank 키워드 추출 함수
def tp_coglo_textrank(text, top_n=5, embeddings=None):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    # 유사도 행렬 계산
    similarity_matrix = compute_similarity_matrix(words, embeddings)
    # 그래프 생성 및 PageRank 계산
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, tol=1e-5, max_iter=2000, dangling=None)  # max_iter를 더 늘리고 tol을 더 낮춤
    # 각 단어의 점수를 계산하여 정렬
    ranked_words = sorted(((scores[i], word) for i, word in enumerate(words)), reverse=True)
    # 상위 키워드 추출
    keywords = []
    seen_words = set()
    for _, word in ranked_words:
        if word not in seen_words and word.isalnum():  # 단어가 이미 본 것이 아니고, 알파벳/숫자로만 이루어진 경우
            keywords.append(word)
            seen_words.add(word)
        if len(keywords) >= top_n:
            break
    return keywords

# 각 행의 'keywords'에서 단어 개수를 계산하여 'num_keywords' 열 생성
df_filtered_2['num_keywords'] = df_filtered_2['keywords'].apply(lambda x: len(x.split()))

# 'num_keywords'를 top_n으로 사용하여 'extracted_keywords' 생성
df_filtered_2['extracted_keywords'] = df_filtered_2.apply(lambda row: tp_coglo_textrank(row['abstract'], top_n=row['num_keywords'], embeddings=glove_embeddings) if pd.notnull(row['abstract']) else [], axis=1)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_2[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_2 = apply_metrics(df_filtered_2)

# 최종 결과 출력
df_result2 = df_filtered_2[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 3. textrank + Watts-Strogatz model
time.sleep(3)
df_filtered_3 = df_filtered.copy()
df_filtered_3.dtypes

# Watts-Strogatz 그래프를 생성하는 함수
def construct_ws_graph(words, p=0.1, k=4):
    size = len(words)
    
    # k 값이 단어 수보다 큰 경우 조정
    if k > size:
        k = max(1, size // 2)  # k를 단어 수의 절반 이하로 조정
    
    # Initialize a regular ring lattice
    graph = nx.watts_strogatz_graph(size, k, p)
    
    # Add edges based on co-occurrence within a window size
    for i in range(size):
        for j in range(i + 1, min(i + k, size)):
            if words[i] != words[j]:
                graph.add_edge(i, j)
    
    return graph

# 텍스트에서 중요 단어를 추출하는 함수
def calculate_ws_weight(text, top_n=5):
    words = word_tokenize(text.lower())
    graph = construct_ws_graph(words, p=0.1, k=4)
    ws_scores = nx.pagerank(graph)
    
    # words의 인덱스가 아닌, 실제 단어로 키워드를 매핑
    ranked_words = sorted(((ws_scores[i], word) for i, word in enumerate(words) if i in ws_scores), reverse=True)
    keywords = [word for _, word in ranked_words[:top_n]]
    return keywords

# 각 행의 'keywords'에서 단어 개수를 계산하여 'num_keywords' 열 생성
df_filtered_3['num_keywords'] = df_filtered_3['keywords'].apply(lambda x: len(x.split()))

# 데이터 프레임에 추출된 키워드를 추가
df_filtered_3['extracted_keywords'] = df_filtered_3.apply(lambda row: calculate_ws_weight(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [], axis=1)

# 데이터 프레임 출력 
print(df_filtered_3[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_3 = apply_metrics(df_filtered_3)

# 최종 결과 출력
df_result3 = df_filtered_3[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 4.1 textrank + Infomap
time.sleep(3)
df_filtered_41 = df_filtered.copy()
df_filtered_41.dtypes

# Infomap을 사용한 Textrank 키워드 추출 함수 (임베딩 제외)
def infomap_textrank_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # 유사도 행렬 계산 (단순 단어 공통성 기반)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        words_i = word_tokenize(sentence_i.lower())
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_j = word_tokenize(sentence_j.lower())
            # 단순 단어 일치 개수를 유사도로 계산
            similarity = len(set(words_i) & set(words_j))
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 모듈별로 중요한 단어 선택
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)
    
    return list(set(keywords))

# DataFrame에 'keywords'의 단어 개수를 계산하여 'num_keywords' 열 추가
df_filtered_41['num_keywords'] = df_filtered_41['keywords'].apply(lambda x: len(x.split()))

# Infomap을 사용하여 키워드를 추출하면서 각 행의 'num_keywords'를 top_n으로 지정
df_filtered_41['extracted_keywords'] = df_filtered_41.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# num_keywords 열은 필요 없으면 제거
df_filtered_41.drop(columns=['num_keywords'], inplace=True)

# 데이터 프레임 출력 
print(df_filtered_41[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_41 = apply_metrics(df_filtered_41)

# 최종 결과 출력
df_result41 = df_filtered_41[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 4.2 textrank + term frequency, term postion, word co-occurence + Infomap
df_filtered_42 = df_filtered.copy()
df_filtered_42.dtypes

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 계산 함수
def calculate_co_occurrence(words, window_size=2):
    co_occurrence = Counter()
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Infomap을 사용하여 커뮤니티별로 중요한 단어를 추출하는 함수 정의
def infomap_keywords_extraction(title, abstract, top_n=5, beta=0.5):
    if not title or not abstract or title.strip() == '' or abstract.strip() == '':
        return []

    # 텍스트를 문장으로 분리하고, 전체 단어로 분리
    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    words = words_title + words_abstract

    # TF 값 계산
    tf = calculate_tf(abstract)

    # 공출현 계산
    co_occurrence = calculate_co_occurrence(words)

    # 단어를 정수로 매핑
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # Infomap 알고리즘 초기화
    infomap = Infomap()

    # 노드와 엣지를 Infomap 구조에 추가
    for i, word in enumerate(words):
        for j in range(i + 1, len(words)):
            weight = 1 if word in words_title else beta # 제목에 있는 단어일 경우 가중치를 1로 설정
            if (word, words[j]) in co_occurrence:
                infomap.add_link(word_to_id[word], word_to_id[words[j]], weight)
    
    # Infomap 알고리즘 실행
    if infomap.num_nodes > 0:  # 노드가 있을 경우에만 실행
        infomap.run()
    else:
        return []

    # 각 모듈(커뮤니티)별 단어를 모아서 저장
    module_words = {}
    for node in infomap.iterTree():
        if node.isLeaf:
            module_id = node.moduleId
            word = id_to_word[node.physicalId]
            if module_id not in module_words:
                module_words[module_id] = []
            module_words[module_id].append(word)

    # 각 모듈에서 가장 중요한 단어 확인
    extracted_keywords = []
    for words in module_words.values():
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(top_n)  # 빈도가 높은 상위 top_n개 단어 선택
        extracted_keywords.extend([word for word, freq in most_common_words])

    return extracted_keywords

# 각 행의 keywords에서 단어 개수를 계산하는 함수
def count_keywords(keywords):
    return len(keywords.split())

# DataFrame에 적용
df_filtered_42['num_keywords'] = df_filtered_42['keywords'].apply(count_keywords)

# Infomap을 사용하여 키워드를 추출하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_42['extracted_keywords'] = df_filtered_42.apply(
    lambda row: infomap_keywords_extraction(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [], axis=1)

# 데이터 프레임 출력 
print(df_filtered_42[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_42 = apply_metrics(df_filtered_42)

# 최종 결과 출력
df_result42 = df_filtered_42[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 4.3 textrank + Infomap + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges
df_filtered_43 = df_filtered.copy()
df_filtered_43.dtypes

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'likely', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2): # window size 조절
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i+1, min(i+1+window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank + Infomap 기반 키워드 추출 함수
def infomap_textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    
    sentences = [s for s, w in weighted_sentences]
    weights = [w for s, w in weighted_sentences]
    
    words = word_tokenize(text.lower())
    
    # 단어의 TF 값 계산
    tf = calculate_tf(text)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # 유사도 행렬 계산
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # word position importance 반영
                        weight_i = 1 if word_i in word_tokenize(title.lower()) else beta
                        weight_j = 1 if word_j in word_tokenize(title.lower()) else beta
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
            similarity_matrix[i][j] = similarity * ((weights[i] + weights[j]) / 2)  # 가중치 적용
    
    # Infomap 알고리즘 초기화 및 적용
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 모듈별로 중요한 단어 선택
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)
    
    return list(set(keywords))

# 키워드 개수를 계산하는 함수
def get_keyword_count(keywords):
    return len(keywords.split())

# 추출된 키워드를 데이터 프레임에 추가 (top_n을 각 행의 keywords 단어 개수로 설정)
df_filtered_43['extracted_keywords'] = df_filtered_43.apply(
    lambda row: infomap_textrank_keywords(
        row['title'], 
        row['abstract'], 
        top_n=get_keyword_count(row['keywords']), 
        beta=0.5
    ) if pd.notnull(row['abstract']) else [], 
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_43[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_43 = apply_metrics(df_filtered_43)

# 최종 결과 출력
df_result43 = df_filtered_43[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 5.1 textrank + Infomap + GloVe 
df_filtered_51 = df_filtered.copy()
df_filtered_51.dtypes

# GloVe 임베딩을 사용한 Textrank + Infomap 기반 키워드 추출 함수
def infomap_textrank_keywords(title, abstract, top_n=5, embeddings=None):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # 문장 내 단어들의 GloVe 임베딩을 평균하여 문장 벡터 계산
    def sentence_embedding(sentence):
        words = word_tokenize(sentence.lower())
        word_vectors = [embeddings[word] for word in words if word in embeddings]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(100)  # 임베딩이 없는 단어가 있을 경우 0 벡터 사용
    
    # 각 문장에 대해 벡터화
    sentence_vectors = [sentence_embedding(sentence) for sentence in sentences]

    # 유사도 행렬 계산 (코사인 유사도 기반)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = cosine_similarity(sentence_vectors[i], sentence_vectors[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 모듈별로 중요한 단어 선택
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)
    
    return list(set(keywords))

# DataFrame에 'keywords'의 단어 개수를 계산하여 'num_keywords' 열 추가
df_filtered_51['num_keywords'] = df_filtered_51['keywords'].apply(lambda x: len(x.split()))

# Infomap과 GloVe 임베딩을 사용하여 키워드를 추출하면서 각 행의 'num_keywords'를 top_n으로 지정
df_filtered_51['extracted_keywords'] = df_filtered_51.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], embeddings=glove_embeddings) if pd.notnull(row['abstract']) else [],
    axis=1
)

# num_keywords 열은 필요 없으면 제거
df_filtered_51.drop(columns=['num_keywords'], inplace=True)

# 데이터 프레임 출력 
print(df_filtered_51[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_51 = apply_metrics(df_filtered_51)

# 최종 결과 출력
df_result51 = df_filtered_51[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 5.2 textrank + Infomap + GloVe 
df_filtered_52 = df_filtered.copy()
df_filtered_52.dtypes

# GloVe 임베딩을 사용한 Infomap 기반 키워드 추출 함수 정의
def infomap_keywords_extraction_glove(title, abstract, top_n=5, beta=0.5, embeddings=None):
    if not title or not abstract or title.strip() == '' or abstract.strip() == '':
        return []

    # 텍스트를 문장으로 분리하고, 전체 단어로 분리
    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    words = words_title + words_abstract

    # TF 값 계산
    tf = calculate_tf(abstract)

    # 공출현 계산
    co_occurrence = calculate_co_occurrence(words)

    # 단어를 정수로 매핑
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # 문장 간 임베딩 기반 유사도 계산
    similarity_matrix = np.zeros((len(words), len(words)))
    
    for word1 in words:
        for word2 in words:
            if word1 in embeddings and word2 in embeddings:
                similarity_matrix[word_to_id[word1]][word_to_id[word2]] = cosine_similarity(embeddings[word1], embeddings[word2])
            else:
                similarity_matrix[word_to_id[word1]][word_to_id[word2]] = 0  # 임베딩이 없는 경우 유사도는 0

    # Infomap 알고리즘 초기화
    infomap = Infomap()

    # 노드와 엣지를 Infomap 구조에 추가
    for i, word in enumerate(words):
        for j in range(i + 1, len(words)):
            weight = 1 if word in words_title else beta  # 제목에 있는 단어일 경우 가중치를 1로 설정
            if (word, words[j]) in co_occurrence:
                infomap.add_link(word_to_id[word], word_to_id[words[j]], similarity_matrix[word_to_id[word]][word_to_id[words[j]]])
    
    # Infomap 알고리즘 실행
    if infomap.num_nodes > 0:  # 노드가 있을 경우에만 실행
        infomap.run()
    else:
        return []

    # 각 모듈(커뮤니티)별 단어를 모아서 저장
    module_words = {}
    for node in infomap.iterTree():
        if node.isLeaf:
            module_id = node.moduleId
            word = id_to_word[node.physicalId]
            if module_id not in module_words:
                module_words[module_id] = []
            module_words[module_id].append(word)

    # 각 모듈에서 가장 중요한 단어 확인
    extracted_keywords = []
    for words in module_words.values():
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(top_n)  # 빈도가 높은 상위 top_n개 단어 선택
        extracted_keywords.extend([word for word, freq in most_common_words])

    return extracted_keywords

# 각 행의 keywords에서 단어 개수를 계산하는 함수
def count_keywords(keywords):
    return len(keywords.split())

# DataFrame에 적용
df_filtered_52['num_keywords'] = df_filtered_52['keywords'].apply(count_keywords)

# Infomap과 GloVe 임베딩을 사용하여 키워드를 추출하면서 각 행의 'num_keywords'를 top_n으로 지정
df_filtered_52['extracted_keywords'] = df_filtered_52.apply(
    lambda row: infomap_keywords_extraction_glove(row['title'], row['abstract'], top_n=row['num_keywords'], embeddings=glove_embeddings) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_52[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_52 = apply_metrics(df_filtered_52)

# 최종 결과 출력
df_result52 = df_filtered_52[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 5.3 textrank + term frequency, term postion, word co-occurence + Infomap + Double Negation, Mitigation, and Hedges + GloVe
df_filtered_53 = df_filtered.copy()
df_filtered_53.dtypes

# GloVe 임베딩을 적용한 Textrank + Infomap 기반 키워드 추출 함수
def infomap_textrank_keywords_glove(title, abstract, top_n=5, beta=0.5, embeddings=None):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    
    sentences = [s for s, w in weighted_sentences]
    weights = [w for s, w in weighted_sentences]
    
    words = word_tokenize(text.lower())
    
    # 단어의 TF 값 계산
    tf = calculate_tf(text)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # GloVe 임베딩을 사용하여 유사도 행렬 계산
    for i, sentence_i in enumerate(sentences):
        words_i = word_tokenize(sentence_i.lower())
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_j = word_tokenize(sentence_j.lower())
            similarity = 0
            # GloVe 임베딩을 사용해 단어 간 유사도 계산
            for word_i in words_i:
                for word_j in words_j:
                    if word_i in embeddings and word_j in embeddings:
                        similarity += cosine_similarity(embeddings[word_i], embeddings[word_j])
            similarity_matrix[i][j] = similarity * ((weights[i] + weights[j]) / 2)  # 가중치 적용
    
    # Infomap 알고리즘 초기화 및 적용
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 모듈별로 중요한 단어 선택
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)
    
    return list(set(keywords))

# 키워드 개수를 계산하는 함수
def get_keyword_count(keywords):
    return len(keywords.split())

# GloVe 임베딩을 적용한 키워드 추출을 DataFrame에 적용
df_filtered_53['extracted_keywords'] = df_filtered_53.apply(
    lambda row: infomap_textrank_keywords_glove(
        row['title'], 
        row['abstract'], 
        top_n=get_keyword_count(row['keywords']), 
        beta=0.5, 
        embeddings=glove_embeddings
    ) if pd.notnull(row['abstract']) else [], 
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_53[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_53 = apply_metrics(df_filtered_53)

# 최종 결과 출력
df_result53 = df_filtered_53[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 6.1 textrank + Infomap + Hierarchical
df_filtered_61 = df_filtered.copy()
df_filtered_61.dtypes

# 공출현 그래프 생성 함수 (단순 단어 연결로만 처리, term frequency, term position, co-occurrence 배제)
def hierarchical_infomap_textrank_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # 유사도 행렬 계산 (단순 단어 공통성 기반)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i, sentence_i in enumerate(sentences):
        words_i = word_tokenize(sentence_i.lower())
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_j = word_tokenize(sentence_j.lower())
            # 단순 단어 일치 개수를 유사도로 계산
            similarity = len(set(words_i) & set(words_j))
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # 계층적 구조의 Infomap을 사용하여 모듈 찾기
    infomap = Infomap()

    for i in range(len(sentences)):
        infomap.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # 계층적 구조를 반영하여 모듈별로 중요한 단어 선택
    keywords = []
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)

    return list(set(keywords))

# DataFrame에 'keywords'의 단어 개수를 계산하여 'num_keywords' 열 추가
df_filtered_61['num_keywords'] = df_filtered_61['keywords'].apply(lambda x: len(x.split()))

# 계층적 Infomap을 사용하여 키워드를 추출하면서 각 행의 'num_keywords'를 top_n으로 지정
df_filtered_61['extracted_keywords'] = df_filtered_61.apply(
    lambda row: hierarchical_infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_61[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_61 = apply_metrics(df_filtered_61)

# 최종 결과 출력
df_result61 = df_filtered_61[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 6.2 textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical
df_filtered_62 = df_filtered.copy()

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 계산 함수
def calculate_co_occurrence(words, window_size=2):
    co_occurrence = Counter()
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
    return co_occurrence

# 계층적 Infomap 기반 Textrank 키워드 추출 함수
def hierarchical_infomap_textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)

    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    words = words_title + words_abstract

    # TF 값 계산
    tf = calculate_tf(abstract)

    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(words)

    # 유사도 행렬 계산
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i, sentence_i in enumerate(sentences):
        words_i = word_tokenize(sentence_i.lower())
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            # TF 값과 공출현 빈도를 기반으로 유사도 계산
            similarity = sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # 제목에 있으면 가중치 1, 초록에 있으면 beta 가중치 적용
                        weight_i = 1 if word_i in words_title else beta
                        weight_j = 1 if word_j in words_title else beta
                        similarity += co_occurrence[(word_i, word_j)] * weight_i * weight_j
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # 계층적 구조의 Infomap을 사용하여 모듈 찾기
    infomap = Infomap()

    for i in range(len(sentences)):
        infomap.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # 계층적 구조를 반영하여 모듈별로 중요한 단어 선택
    keywords = []
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)

    return list(set(keywords))

# DataFrame에 'keywords'의 단어 개수를 계산하여 'num_keywords' 열 추가
df_filtered_62['num_keywords'] = df_filtered_62['keywords'].apply(lambda x: len(x.split()))

# 계층적 Infomap을 사용하여 키워드를 추출하면서 각 행의 'num_keywords'를 top_n으로 지정
df_filtered_62['extracted_keywords'] = df_filtered_62.apply(
    lambda row: hierarchical_infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_62[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_62 = apply_metrics(df_filtered_62)

# 최종 결과 출력
df_result62 = df_filtered_62[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 6.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Hierarchical
df_filtered_63 = df_filtered.copy()

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(sentences):
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 부정어 간 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'likely', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# 공출현 그래프 생성 함수
def create_co_occurrence_graph(sentences, window_size=2):
    graph = nx.Graph()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                if graph.has_edge(word, words[j]):
                    graph[word][words[j]]['weight'] += 1
                else:
                    graph.add_edge(word, words[j], weight=1)
    return graph

# 확장된 Infomap 기반 키워드 추출 함수 (Double Negation, Mitigation, and Hedges Weighting 포함)
def hierarchical_infomap_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(sent_tokenize(text))
    
    sentences = [s for s, w in weighted_sentences]  # 가중치 적용 후 문장 리스트
    weights = [w for s, w in weighted_sentences]    # 가중치 리스트

    # 공출현 그래프 생성
    graph = create_co_occurrence_graph(sentences)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # Infomap 알고리즘 적용 (계층적 구조 반영)
    infomap = Infomap()

    for edge in graph.edges(data=True):
        node1 = word_to_id[edge[0]]
        node2 = word_to_id[edge[1]]
        weight = edge[2]['weight']
        infomap.addLink(node1, node2, weight)

    infomap.run()

    # 각 노드가 속한 모듈 정보를 트리 구조에서 추출
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            node_name = id_to_word[node.node_id]
            module_assignments[node_name] = node.moduleIndex()

    # 모듈별로 단어 점수를 계산
    module_scores = {}
    for node, module_id in module_assignments.items():
        # node가 제목에 있으면 가중치 1, 초록에 있으면 가중치 beta
        if node in word_tokenize(title.lower()):
            node_score = sum([graph[node][nbr]['weight'] * 1 for nbr in graph.neighbors(node)])
        else:
            node_score = sum([graph[node][nbr]['weight'] * beta for nbr in graph.neighbors(node)])

        if module_id in module_scores:
            module_scores[module_id].append((node, node_score))
        else:
            module_scores[module_id] = [(node, node_score)]

    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [node for node, score in sorted_nodes[:top_n]]
        hierarchical_keywords.extend(keywords)

    return list(set(hierarchical_keywords))

# 각 행의 'keywords'에서 단어 개수를 계산하는 함수
def count_keywords(keywords):
    return len(keywords.split())

# DataFrame에 적용하여 'num_keywords' 열 추가
df_filtered_63['num_keywords'] = df_filtered_63['keywords'].apply(count_keywords)

# 추출된 키워드를 데이터 프레임에 추가하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_63['extracted_keywords'] = df_filtered_63.apply(
    lambda row: hierarchical_infomap_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5) if pd.notnull(row['abstract']) else [],
    axis=1)

# 데이터 프레임 출력 
print(df_filtered_63[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_63 = apply_metrics(df_filtered_63)

# 최종 결과 출력
df_result63 = df_filtered_63[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 7.1 textrank + Infomap + Hierarchical + GloVe
df_filtered_71 = df_filtered.copy()

# 공출현 그래프 생성 함수 (GloVe 임베딩 기반 유사도 사용)
def hierarchical_infomap_textrank_keywords(title, abstract, top_n=5, embeddings=None):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)

    # 유사도 행렬 계산 (GloVe 임베딩 기반)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            similarity = sentence_similarity(sentence_i, sentence_j, embeddings)
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # 계층적 구조의 Infomap을 사용하여 모듈 찾기
    infomap = Infomap()

    for i in range(len(sentences)):
        infomap.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # 계층적 구조를 반영하여 모듈별로 중요한 단어 선택
    keywords = []
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)

    return list(set(keywords))

# DataFrame에 'keywords'의 단어 개수를 계산하여 'num_keywords' 열 추가
df_filtered_71['num_keywords'] = df_filtered_71['keywords'].apply(lambda x: len(x.split()))

# GloVe 임베딩을 사용하여 계층적 Infomap을 사용한 키워드 추출
df_filtered_71['extracted_keywords'] = df_filtered_71.apply(
    lambda row: hierarchical_infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], embeddings=glove_embeddings) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_71[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_71 = apply_metrics(df_filtered_71)

# 최종 결과 출력
df_result71 = df_filtered_71[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 7.2 textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical + GloVe 
df_filtered_72 = df_filtered.copy()

# TF 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 계산 함수
def calculate_co_occurrence(words, window_size=2):
    co_occurrence = Counter()
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
    return co_occurrence

# 계층적 Infomap 기반 GloVe 적용 Textrank 키워드 추출 함수
def hierarchical_infomap_textrank_keywords(title, abstract, top_n=5, beta=0.5, embeddings=None):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)

    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    words = words_title + words_abstract

    # TF 값 계산
    tf = calculate_tf(abstract)

    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(words)

    # 유사도 행렬 계산 (GloVe 임베딩 기반)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            # GloVe 임베딩을 기반으로 문장 간 유사도 계산
            similarity = sentence_similarity(sentence_i, sentence_j, embeddings)
            # TF 값과 공출현 빈도를 기반으로 유사도 보정
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity += sum(tf[word] for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        weight_i = 1 if word_i in words_title else beta
                        weight_j = 1 if word_j in words_title else beta
                        similarity += co_occurrence[(word_i, word_j)] * weight_i * weight_j
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # 계층적 구조의 Infomap을 사용하여 모듈 찾기
    infomap = Infomap()

    for i in range(len(sentences)):
        infomap.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # 계층적 구조를 반영하여 모듈별로 중요한 단어 선택
    keywords = []
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)

    return list(set(keywords))

# DataFrame에 'keywords'의 단어 개수를 계산하여 'num_keywords' 열 추가
df_filtered_72['num_keywords'] = df_filtered_72['keywords'].apply(lambda x: len(x.split()))

# GloVe 임베딩을 사용하여 계층적 Infomap을 사용한 키워드 추출
df_filtered_72['extracted_keywords'] = df_filtered_72.apply(
    lambda row: hierarchical_infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], embeddings=glove_embeddings) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_72[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_72 = apply_metrics(df_filtered_72)

# 최종 결과 출력
df_result72 = df_filtered_72[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 7.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Hierarchical + GloVe
df_filtered_73 = df_filtered.copy()

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(sentences):
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 부정어 간 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'likely', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# 공출현 그래프 생성 함수
def create_co_occurrence_graph(sentences, window_size=2):
    graph = nx.Graph()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                if graph.has_edge(word, words[j]):
                    graph[word][words[j]]['weight'] += 1
                else:
                    graph.add_edge(word, words[j], weight=1)
    return graph

# 확장된 Infomap 기반 GloVe를 포함한 키워드 추출 함수 (Double Negation, Mitigation, and Hedges Weighting 포함)
def hierarchical_infomap_keywords(title, abstract, top_n=5, beta=0.5, embeddings=None):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(sent_tokenize(text))
    
    sentences = [s for s, w in weighted_sentences]  # 가중치 적용 후 문장 리스트
    weights = [w for s, w in weighted_sentences]    # 가중치 리스트

    # 공출현 그래프 생성
    graph = create_co_occurrence_graph(sentences)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # Infomap 알고리즘 적용 (계층적 구조 반영)
    infomap = Infomap()

    for edge in graph.edges(data=True):
        node1 = word_to_id[edge[0]]
        node2 = word_to_id[edge[1]]
        weight = edge[2]['weight']
        infomap.addLink(node1, node2, weight)

    infomap.run()

    # 각 노드가 속한 모듈 정보를 트리 구조에서 추출
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            node_name = id_to_word[node.node_id]
            module_assignments[node_name] = node.moduleIndex()

    # 모듈별로 단어 점수를 계산
    module_scores = {}
    for node, module_id in module_assignments.items():
        # node가 제목에 있으면 가중치 1, 초록에 있으면 가중치 beta
        if node in word_tokenize(title.lower()):
            node_score = sum([graph[node][nbr]['weight'] * 1 for nbr in graph.neighbors(node)])
        else:
            node_score = sum([graph[node][nbr]['weight'] * beta for nbr in graph.neighbors(node)])

        if module_id in module_scores:
            module_scores[module_id].append((node, node_score))
        else:
            module_scores[module_id] = [(node, node_score)]

    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [node for node, score in sorted_nodes[:top_n]]
        hierarchical_keywords.extend(keywords)

    return list(set(hierarchical_keywords))

# 각 행의 'keywords'에서 단어 개수를 계산하는 함수
def count_keywords(keywords):
    return len(keywords.split())

# DataFrame에 적용하여 'num_keywords' 열 추가
df_filtered_73['num_keywords'] = df_filtered_73['keywords'].apply(count_keywords)

# GloVe 임베딩을 사용하여 키워드를 추출하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_73['extracted_keywords'] = df_filtered_73.apply(
    lambda row: hierarchical_infomap_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5, embeddings=glove_embeddings) if pd.notnull(row['abstract']) else [],
    axis=1)

# 데이터 프레임 출력 
print(df_filtered_73[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_73 = apply_metrics(df_filtered_73)

# 최종 결과 출력
df_result73 = df_filtered_73[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 8.1 textrank + Infomap + Multi Entropy
df_filtered_81 = df_filtered.copy()

# 엔트로피 계산 함수
def calculate_entropy(prob_dist):
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))

# 계층적 Infomap 기반 Multi Entropy Textrank 키워드 추출 함수
def hierarchical_infomap_textrank_keywords_multi_entropy(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # 각 문장의 단어 분포를 사용한 엔트로피 계산
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        words_i = word_tokenize(sentence_i.lower())
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_j = word_tokenize(sentence_j.lower())
            # 단순 단어 일치 개수를 유사도로 계산
            similarity = len(set(words_i) & set(words_j))
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 계층적 모듈 탐색
    infomap = Infomap()

    for i in range(len(sentences)):
        infomap.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # 모듈 내 엔트로피와 가중치 적용
    module_scores = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            prob_dist = np.array(list(word_freq.values())) / np.sum(list(word_freq.values()))
            entropy = calculate_entropy(prob_dist)
            module_id = node.moduleIndex()

            # Word Position 가중치 적용
            word_weights = []
            for word in words:
                if word in word_tokenize(title.lower()):
                    word_weights.append(1)  # 제목에 포함된 단어
                elif word in word_tokenize(abstract.lower()):
                    word_weights.append(beta)  # 초록에 포함된 단어
                else:
                    word_weights.append(0)

            # 엔트로피 기반의 모듈 가중치 계산
            p_i_star = sum(word_freq.values()) / len(words)
            H_Q = calculate_entropy(prob_dist)  # 모듈 간 엔트로피
            H_Pi = entropy  # 모듈 내 엔트로피
            module_weight = p_i_star * (H_Q + H_Pi) * np.mean(word_weights)

            if module_id in module_scores:
                module_scores[module_id].append((sentence_idx, module_weight))
            else:
                module_scores[module_id] = [(sentence_idx, module_weight)]

    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [word_tokenize(sentences[node].lower()) for node, _ in sorted_nodes[:top_n]]
        hierarchical_keywords.extend([word for words in keywords for word in words])
    
    return list(set(hierarchical_keywords))

# 단어 개수를 세는 함수
def count_keywords(keywords):
    return len(keywords.split())

# 각 행의 keywords 개수로 top_n 설정
df_filtered_81['num_keywords'] = df_filtered_81['keywords'].apply(count_keywords)

# 확장된 Infomap 기반 Textrank 키워드를 추가
df_filtered_81['extracted_keywords'] = df_filtered_81.apply(
    lambda row: hierarchical_infomap_textrank_keywords_multi_entropy(
        row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5
    ), axis=1)

# 데이터 프레임 출력
print(df_filtered_81[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_81 = apply_metrics(df_filtered_81)

# 최종 결과 출력
df_result81 = df_filtered_81[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 8.2 textrank + term frequency, term postion, word co-occurence + Infomap + Multi Entropy
df_filtered_82 = df_filtered.copy()

# 공출현 계산 함수
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# 엔트로피 계산 함수
def calculate_entropy(prob_dist):
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))

# 확장된 Infomap 기반 키워드 추출 함수 (확장된 방정식 반영 및 Word Position 반영)
def infomap_textrank_keywords_extended(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # 공출현 기반 유사도 행렬 계산
    co_occurrence = calculate_co_occurrence(sentences)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(co_occurrence[(word, word)] for word in common_words)
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 각 모듈의 엔트로피를 계층적으로 계산 및 가중치 적용
    module_scores = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            prob_dist = np.array(list(word_freq.values())) / np.sum(list(word_freq.values()))
            entropy = calculate_entropy(prob_dist)
            module_id = node.moduleIndex()

            # Word Position 가중치 적용
            word_weights = []
            for word in words:
                if word in word_tokenize(title.lower()):
                    word_weights.append(1)  # 제목에 포함된 단어
                elif word in word_tokenize(abstract.lower()):
                    word_weights.append(beta)  # 초록에 포함된 단어
                else:
                    word_weights.append(0)

            # 확장 방정식에 따른 가중치 적용
            p_i_star = sum(word_freq.values()) / len(words)
            H_Q = calculate_entropy(prob_dist)
            H_Pi = entropy
            module_weight = p_i_star * (H_Q + H_Pi) * np.mean(word_weights)

            if module_id in module_scores:
                module_scores[module_id].append((sentence_idx, module_weight))
            else:
                module_scores[module_id] = [(sentence_idx, module_weight)]
    
    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [word_tokenize(sentences[node].lower()) for node, _ in sorted_nodes[:top_n]]
        hierarchical_keywords.extend([word for words in keywords for word in words])
    
    return list(set(hierarchical_keywords))

# 단어 개수를 세는 함수
def count_keywords(keywords):
    return len(keywords.split())

# 각 행의 keywords 개수로 top_n 설정
df_filtered_82['num_keywords'] = df_filtered_82['keywords'].apply(count_keywords)

# DataFrame에 확장된 Infomap 기반 Textrank 키워드를 추가
df_filtered_82['extracted_keywords'] = df_filtered_82.apply(
    lambda row: infomap_textrank_keywords_extended(
        row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5
    ), axis=1)

# 데이터 프레임 출력 
print(df_filtered_82[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_82 = apply_metrics(df_filtered_82)

# 최종 결과 출력
df_result82 = df_filtered_82[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 8.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Multi Entropy
df_filtered_83 = df_filtered.copy()

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(sentences):
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 부정어 간 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'likely', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# 공출현 계산 함수
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# 엔트로피 계산 함수
def calculate_entropy(prob_dist):
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))

# 확장된 Infomap 기반 키워드 추출 함수 (Double Negation, Mitigation, and Hedges Weighting 포함)
def infomap_textrank_keywords_extended(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(sent_tokenize(text))  # 가중치 적용된 문장

    sentences = [s for s, w in weighted_sentences]  # 가중치가 적용된 문장 리스트
    weights = [w for s, w in weighted_sentences]  # 각 문장의 가중치 리스트
    
    # 공출현 기반 유사도 행렬 계산
    co_occurrence = calculate_co_occurrence(sentences)
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            common_words = set(words_i) & set(words_j)
            similarity = sum(co_occurrence[(word, word)] for word in common_words)
            similarity_matrix[i][j] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 각 모듈의 엔트로피를 계층적으로 계산 및 가중치 적용
    module_scores = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            prob_dist = np.array(list(word_freq.values())) / np.sum(list(word_freq.values()))
            entropy = calculate_entropy(prob_dist)
            module_id = node.moduleIndex()

            # Word Position 및 문장 가중치 적용
            word_weights = []
            for word in words:
                if word in word_tokenize(title.lower()):
                    word_weights.append(1)  # 제목에 포함된 단어
                elif word in word_tokenize(abstract.lower()):
                    word_weights.append(beta)  # 초록에 포함된 단어
                else:
                    word_weights.append(0)

            # 확장 방정식에 따른 가중치 적용
            p_i_star = sum(word_freq.values()) / len(words)
            H_Q = calculate_entropy(prob_dist)
            H_Pi = entropy
            module_weight = p_i_star * (H_Q + H_Pi) * np.mean(word_weights) * np.mean(weights)  # 문장 가중치 반영

            if module_id in module_scores:
                module_scores[module_id].append((sentence_idx, module_weight))
            else:
                module_scores[module_id] = [(sentence_idx, module_weight)]
    
    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [word_tokenize(sentences[node].lower()) for node, _ in sorted_nodes[:top_n]]
        hierarchical_keywords.extend([word for words in keywords for word in words])
    
    return list(set(hierarchical_keywords))

# 단어 개수를 세는 함수
def count_keywords(keywords):
    return len(keywords.split())

# 각 행의 keywords 개수로 top_n 설정
df_filtered_83['num_keywords'] = df_filtered_83['keywords'].apply(count_keywords)

# DataFrame에 확장된 Infomap 기반 Textrank 키워드를 추가 (Double Negation, Mitigation, and Hedges Weighting 포함)
df_filtered_83['extracted_keywords'] = df_filtered_83.apply(
    lambda row: infomap_textrank_keywords_extended(
        row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5
    ), axis=1)

# 데이터 프레임 출력 
print(df_filtered_83[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_83 = apply_metrics(df_filtered_83)

# 최종 결과 출력
df_result83 = df_filtered_83[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 9.1 textrank + Infomap + Multi Entropy + GloVe 
df_filtered_91 = df_filtered.copy()

# GloVe 기반 문장 임베딩 계산 함수
def sentence_embedding(sentence, embeddings, dim=100):
    words = word_tokenize(sentence.lower())
    word_embeddings = [embeddings[word] for word in words if word in embeddings]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(dim)

# 엔트로피 계산 함수
def calculate_entropy(prob_dist):
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))

# 계층적 Infomap 기반 Multi Entropy Textrank 키워드 추출 함수 (GloVe 임베딩 적용)
def hierarchical_infomap_textrank_keywords_multi_entropy_glove(title, abstract, glove_embeddings, top_n=5, beta=0.5, glove_dim=100):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # GloVe 임베딩을 기반으로 문장 간 유사도 계산
    sentence_embeddings = [sentence_embedding(sentence, glove_embeddings, dim=glove_dim) for sentence in sentences]
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 계층적 모듈 탐색
    infomap = Infomap()

    for i in range(len(sentences)):
        infomap.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # 모듈 내 엔트로피와 가중치 적용
    module_scores = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            prob_dist = np.array(list(word_freq.values())) / np.sum(list(word_freq.values()))
            entropy = calculate_entropy(prob_dist)
            module_id = node.moduleIndex()

            # Word Position 가중치 적용
            word_weights = []
            for word in words:
                if word in word_tokenize(title.lower()):
                    word_weights.append(1)  # 제목에 포함된 단어
                elif word in word_tokenize(abstract.lower()):
                    word_weights.append(beta)  # 초록에 포함된 단어
                else:
                    word_weights.append(0)

            # 엔트로피 기반의 모듈 가중치 계산
            p_i_star = sum(word_freq.values()) / len(words)
            H_Q = calculate_entropy(prob_dist)  # 모듈 간 엔트로피
            H_Pi = entropy  # 모듈 내 엔트로피
            module_weight = p_i_star * (H_Q + H_Pi) * np.mean(word_weights)

            if module_id in module_scores:
                module_scores[module_id].append((sentence_idx, module_weight))
            else:
                module_scores[module_id] = [(sentence_idx, module_weight)]

    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [word_tokenize(sentences[node].lower()) for node, _ in sorted_nodes[:top_n]]
        hierarchical_keywords.extend([word for words in keywords for word in words])
    
    return list(set(hierarchical_keywords))

# 단어 개수를 세는 함수
def count_keywords(keywords):
    return len(keywords.split())

# 각 행의 keywords 개수로 top_n 설정
df_filtered_91['num_keywords'] = df_filtered_91['keywords'].apply(count_keywords)

# 확장된 Infomap 기반 Textrank 키워드를 추가 (GloVe 임베딩 적용)
df_filtered_91['extracted_keywords'] = df_filtered_91.apply(
    lambda row: hierarchical_infomap_textrank_keywords_multi_entropy_glove(
        row['title'], row['abstract'], glove_embeddings, top_n=row['num_keywords'], beta=0.5
    ), axis=1)

# 데이터 프레임 출력
print(df_filtered_91[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_91 = apply_metrics(df_filtered_91)

# 최종 결과 출력
df_result91 = df_filtered_91[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 9.2 textrank + term frequency, term postion, word co-occurence + Infomap + Multi level Entropy + GloVe 
df_filtered_92 = df_filtered.copy()

# GloVe 기반 문장 임베딩 계산 함수
def sentence_embedding(sentence, embeddings, dim=100):
    words = word_tokenize(sentence.lower())
    word_embeddings = [embeddings[word] for word in words if word in embeddings]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(dim)

# 엔트로피 계산 함수
def calculate_entropy(prob_dist):
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))

# 확장된 Infomap 기반 키워드 추출 함수 (GloVe 임베딩 적용 및 확장 방정식 반영)
def infomap_textrank_keywords_extended_glove(title, abstract, glove_embeddings, top_n=5, beta=0.5, glove_dim=100):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # GloVe 임베딩을 기반으로 문장 간 유사도 계산
    sentence_embeddings = [sentence_embedding(sentence, glove_embeddings, dim=glove_dim) for sentence in sentences]
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 계층적 모듈 탐색
    infomap = Infomap()

    for i in range(len(sentences)):
        infomap.add_node(i)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # 모듈 내 엔트로피와 가중치 적용
    module_scores = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            prob_dist = np.array(list(word_freq.values())) / np.sum(list(word_freq.values()))
            entropy = calculate_entropy(prob_dist)
            module_id = node.moduleIndex()

            # Word Position 가중치 적용
            word_weights = []
            for word in words:
                if word in word_tokenize(title.lower()):
                    word_weights.append(1)  # 제목에 포함된 단어
                elif word in word_tokenize(abstract.lower()):
                    word_weights.append(beta)  # 초록에 포함된 단어
                else:
                    word_weights.append(0)

            # 엔트로피 기반의 모듈 가중치 계산
            p_i_star = sum(word_freq.values()) / len(words)
            H_Q = calculate_entropy(prob_dist)  # 모듈 간 엔트로피
            H_Pi = entropy  # 모듈 내 엔트로피
            module_weight = p_i_star * (H_Q + H_Pi) * np.mean(word_weights)

            if module_id in module_scores:
                module_scores[module_id].append((sentence_idx, module_weight))
            else:
                module_scores[module_id] = [(sentence_idx, module_weight)]

    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [word_tokenize(sentences[node].lower()) for node, _ in sorted_nodes[:top_n]]
        hierarchical_keywords.extend([word for words in keywords for word in words])
    
    return list(set(hierarchical_keywords))

# 단어 개수를 세는 함수
def count_keywords(keywords):
    return len(keywords.split())

# 각 행의 keywords 개수로 top_n 설정
df_filtered_92['num_keywords'] = df_filtered_92['keywords'].apply(count_keywords)

# 확장된 Infomap 기반 Textrank 키워드를 추가 (GloVe 임베딩 적용)
df_filtered_92['extracted_keywords'] = df_filtered_92.apply(
    lambda row: infomap_textrank_keywords_extended_glove(
        row['title'], row['abstract'], glove_embeddings, top_n=row['num_keywords'], beta=0.5
    ), axis=1)

# 데이터 프레임 출력
print(df_filtered_92[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_92 = apply_metrics(df_filtered_92)

# 최종 결과 출력
df_result92 = df_filtered_92[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 9.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Multi Entropy + GloVe 
df_filtered_93 = df_filtered.copy()

# GloVe 기반 문장 임베딩 계산 함수
def sentence_embedding(sentence, embeddings, dim=100):
    words = word_tokenize(sentence.lower())
    word_embeddings = [embeddings[word] for word in words if word in embeddings]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(dim)

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(sentences):
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 부정어 간 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'likely', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# 엔트로피 계산 함수
def calculate_entropy(prob_dist):
    return -np.sum(prob_dist * np.log(prob_dist + 1e-9))

# 확장된 Infomap 기반 키워드 추출 함수 (GloVe 임베딩 적용 및 Double Negation, Mitigation, and Hedges Weighting 포함)
def infomap_textrank_keywords_extended_glove(title, abstract, glove_embeddings, top_n=5, beta=0.5, glove_dim=100):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(sent_tokenize(text))  # 가중치 적용된 문장

    sentences = [s for s, w in weighted_sentences]  # 가중치가 적용된 문장 리스트
    weights = [w for s, w in weighted_sentences]  # 각 문장의 가중치 리스트
    
    # GloVe 임베딩을 기반으로 문장 간 유사도 계산
    sentence_embeddings = [sentence_embedding(sentence, glove_embeddings, dim=glove_dim) for sentence in sentences]
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap을 사용하여 모듈 찾기
    infomap = Infomap()
    
    for i in range(len(sentences)):
        infomap.add_node(i)
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()
    
    # 각 모듈의 엔트로피를 계층적으로 계산 및 가중치 적용
    module_scores = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            prob_dist = np.array(list(word_freq.values())) / np.sum(list(word_freq.values()))
            entropy = calculate_entropy(prob_dist)
            module_id = node.moduleIndex()

            # Word Position 및 문장 가중치 적용
            word_weights = []
            for word in words:
                if word in word_tokenize(title.lower()):
                    word_weights.append(1)  # 제목에 포함된 단어
                elif word in word_tokenize(abstract.lower()):
                    word_weights.append(beta)  # 초록에 포함된 단어
                else:
                    word_weights.append(0)

            # 확장 방정식에 따른 가중치 적용
            p_i_star = sum(word_freq.values()) / len(words)
            H_Q = calculate_entropy(prob_dist)
            H_Pi = entropy
            module_weight = p_i_star * (H_Q + H_Pi) * np.mean(word_weights) * np.mean(weights)  # 문장 가중치 반영

            if module_id in module_scores:
                module_scores[module_id].append((sentence_idx, module_weight))
            else:
                module_scores[module_id] = [(sentence_idx, module_weight)]
    
    # 각 모듈에서 상위 top_n 노드를 추출
    hierarchical_keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        keywords = [word_tokenize(sentences[node].lower()) for node, _ in sorted_nodes[:top_n]]
        hierarchical_keywords.extend([word for words in keywords for word in words])
    
    return list(set(hierarchical_keywords))

# 단어 개수를 세는 함수
def count_keywords(keywords):
    return len(keywords.split())

# 각 행의 keywords 개수로 top_n 설정
df_filtered_93['num_keywords'] = df_filtered_93['keywords'].apply(count_keywords)

# DataFrame에 확장된 Infomap 기반 Textrank 키워드를 추가 (Double Negation, Mitigation, Hedges Weighting 및 GloVe 임베딩 포함)
df_filtered_93['extracted_keywords'] = df_filtered_93.apply(
    lambda row: infomap_textrank_keywords_extended_glove(
        row['title'], row['abstract'], glove_embeddings, top_n=row['num_keywords'], beta=0.5
    ), axis=1)

# 데이터 프레임 출력 
print(df_filtered_93[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_93 = apply_metrics(df_filtered_93)

# 최종 결과 출력
df_result93 = df_filtered_93[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 10.1 textrank + Infomap + jaccard
##  Jaccard 유사도는 단어의 빈도와 위치를 반영하지 않음
# Jaccard 유사도는 단순히 두 집합에서 단어의 존재 여부만을 고려하여 교집합과 합집합을 계산합니다.
# 이 때문에, 단어가 텍스트에 몇 번 등장했는지 또는 단어가 문장의 어느 위치에 있는지는 Jaccard 유사도에 반영되지 않습니다.
# Glove 임베딩, 단어 빈도 (TF), 단어 위치 (Position), 공출현(Word Co-occurrence) 등과는 달리, Jaccard 유사도는 단어 간의 의미적 유사성이나 중요도를 고려하지 않기 때문에 결과에 큰 차이를 만들지 못합니다.
df_filtered_101 = df_filtered.copy()
df_filtered_101 .dtypes

# Preprocess the abstract (remove stopwords and tokenize)
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]  # Only keep alphanumeric words
    return words

# Compute Jaccard similarity between two sets of words
def jaccard_similarity(words1, words2):
    set1, set2 = set(words1), set(words2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

# Extract keywords using Infomap and TextRank without GloVe
def infomap_textrank_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words_list = [preprocess_text(sent) for sent in sentences]
    
    # Compute similarity matrix using Jaccard similarity
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = jaccard_similarity(words_list[i], words_list[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
        
    # Build a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Use Infomap to find clusters in the graph
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()

    # Extract top keywords from each cluster
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = words_list[sentence_idx]
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)
    
    return list(set(keywords))

# Jaccard 유사도 적용 과정:
# 토큰화 및 전처리:
# preprocess_text 함수로 각 문장을 토큰화하고, 단어를 전처리(알파벳 및 숫자만 유지)합니다.
# 유사도 행렬 생성:
# 각 문장의 단어 리스트(words_list)를 기반으로, 두 문장 간의 Jaccard 유사도를 계산합니다.
# jaccard_similarity 함수는 두 문장의 단어 집합 사이의 교집합과 합집합을 계산하여 유사도를 반환합니다.
# 이렇게 계산된 Jaccard 유사도는 similarity_matrix에 저장됩니다.
# 그래프 구축:
# 계산된 Jaccard 유사도를 바탕으로 similarity_matrix로부터 그래프를 생성합니다. 각 문장은 노드가 되고, 유사도 값은 노드 간의 링크로 표현됩니다.
# 이 그래프를 기반으로 Infomap 알고리즘을 사용하여 클러스터링 및 키워드 추출을 수행합니다.
# 결론적으로, Jaccard 유사도는 두 문장 간의 유사도를 계산하는 데 사용되었으며, 유사도 행렬을 구축하는 데 적용되었습니다.


# Add 'num_keywords' for setting top_n dynamically
df_filtered_101['num_keywords'] = df_filtered_101['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using Infomap and Jaccard similarity
df_filtered_101['extracted_keywords'] = df_filtered_101.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_101[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_101 = apply_metrics(df_filtered_101)

# 최종 결과 출력
df_result101 = df_filtered_101[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 10.2 textrank + term frequency, term postion, word co-occurence + Infomap + jcd
##  Jaccard 유사도는 단어의 빈도와 위치를 반영하지 않음
# Jaccard 유사도는 단순히 두 집합에서 단어의 존재 여부만을 고려하여 교집합과 합집합을 계산합니다.
# 이 때문에, 단어가 텍스트에 몇 번 등장했는지 또는 단어가 문장의 어느 위치에 있는지는 Jaccard 유사도에 반영되지 않습니다.
# Glove 임베딩, 단어 빈도 (TF), 단어 위치 (Position), 공출현(Word Co-occurrence) 등과는 달리, Jaccard 유사도는 단어 간의 의미적 유사성이나 중요도를 고려하지 않기 때문에 결과에 큰 차이를 만들지 못합니다.
df_filtered_102 = df_filtered.copy()
df_filtered_102.dtypes

# Preprocess the abstract (remove stopwords and tokenize)
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]  # Only keep alphanumeric words
    return words

# Compute term frequency (TF)
def calculate_tf(text):
    words = preprocess_text(text)
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# Compute co-occurrence of words within a window size
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = preprocess_text(sentence)
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Compute similarity matrix using TF, co-occurrence, and position
def calculate_similarity_matrix(sentences, title, beta=0.5):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # Preprocess title and sentences
    title_words = preprocess_text(title)
    sentence_words = [preprocess_text(sent) for sent in sentences]
    
    # Calculate term frequency (TF) for each sentence
    tf_list = [calculate_tf(sent) for sent in sentences]
    
    # Calculate co-occurrence matrix for sentences
    co_occurrence = calculate_co_occurrence(sentences)
    
    for i, words_i in enumerate(sentence_words):
        for j, words_j in enumerate(sentence_words):
            if i == j:
                continue
            # Find common words between two sentences
            common_words = set(words_i) & set(words_j)
            
            # Calculate similarity based on TF and co-occurrence
            similarity = sum(tf_list[i].get(word, 0) * tf_list[j].get(word, 0) for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # Apply term position importance based on title and abstract
                        weight_i = 1 if word_i in title_words else beta
                        weight_j = 1 if word_j in title_words else beta
                        similarity += co_occurrence[(word_i, word_j)] * weight_i * weight_j
            similarity_matrix[i][j] = similarity
    
    return similarity_matrix

# Extract keywords using Infomap and TextRank
def infomap_textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # Compute similarity matrix
    similarity_matrix = calculate_similarity_matrix(sentences, title, beta)
    
    # Build a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Use Infomap to find clusters in the graph
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()

    # Extract top keywords from each cluster
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = preprocess_text(sentences[sentence_idx])
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)
    
    return list(set(keywords))

# Add 'num_keywords' for setting top_n dynamically
df_filtered_102['num_keywords'] = df_filtered_102['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using Infomap and term frequency, term position, and word co-occurrence
df_filtered_102['extracted_keywords'] = df_filtered_102.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_102[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_102 = apply_metrics(df_filtered_102)

# 최종 결과 출력
df_result102 = df_filtered_102[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]

# 데이터 프레임 출력 
print(df_filtered_102[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_102 = apply_metrics(df_filtered_102)

# 최종 결과 출력
df_result102 = df_filtered_102[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 10.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + jcd
df_filtered_103 = df_filtered.copy()

# Preprocess the abstract (remove stopwords and tokenize)
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]  # Only keep alphanumeric words
    return words

# Compute term frequency (TF)
def calculate_tf(text):
    words = preprocess_text(text)
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# Compute co-occurrence of words within a window size
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = preprocess_text(sentence)
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(sentences):
    weighted_sentences = []
    
    for sentence in sentences:
        words = preprocess_text(sentence)
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 부정어 간 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# Compute similarity matrix using TF, co-occurrence, position, and weights
def calculate_similarity_matrix(sentences, title, beta=0.5):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Preprocess title and sentences, and apply sentence weights
    title_words = preprocess_text(title)
    weighted_sentences = apply_weights(sentences)
    sentence_words = [preprocess_text(sent) for sent, _ in weighted_sentences]
    sentence_weights = [weight for _, weight in weighted_sentences]
    
    # Calculate term frequency (TF) for each sentence
    tf_list = [calculate_tf(sent) for sent, _ in weighted_sentences]
    
    # Calculate co-occurrence matrix for sentences
    co_occurrence = calculate_co_occurrence([sent for sent, _ in weighted_sentences])
    
    for i, words_i in enumerate(sentence_words):
        for j, words_j in enumerate(sentence_words):
            if i == j:
                continue
            # Find common words between two sentences
            common_words = set(words_i) & set(words_j)
            
            # Calculate similarity based on TF, co-occurrence, and sentence weights
            similarity = sum(tf_list[i].get(word, 0) * tf_list[j].get(word, 0) for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # Apply term position importance based on title and abstract
                        weight_i = 1 if word_i in title_words else beta
                        weight_j = 1 if word_j in title_words else beta
                        similarity += co_occurrence[(word_i, word_j)] * weight_i * weight_j * sentence_weights[i] * sentence_weights[j]
            similarity_matrix[i][j] = similarity
    
    return similarity_matrix

# Extract keywords using Infomap and TextRank with Double Negation, Mitigation, and Hedges Weighting
def infomap_textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # Compute similarity matrix
    similarity_matrix = calculate_similarity_matrix(sentences, title, beta)
    
    # Build a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Use Infomap to find clusters in the graph
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()

    # Extract top keywords from each cluster
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = preprocess_text(sentences[sentence_idx])
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)
    
    return list(set(keywords))

# Add 'num_keywords' for setting top_n dynamically
df_filtered_103['num_keywords'] = df_filtered_103['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using Infomap and term frequency, term position, word co-occurrence, and Double Negation, Mitigation, and Hedges Weighting
df_filtered_103['extracted_keywords'] = df_filtered_103.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_103[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_103 = apply_metrics(df_filtered_103)

# 최종 결과 출력
df_result103 = df_filtered_103[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 11.1 textrank + Infomap + NetMRF
df_filtered_111 = df_filtered.copy()
df_filtered_111.dtypes

# MRF 초기화 및 최적화 함수 (NetMRF 적용)
def initialize_mrf(similarity_matrix):
    normalized_sim_matrix = normalize(similarity_matrix, axis=1, norm='l1')
    return normalized_sim_matrix

def apply_mrf(similarity_matrix, num_iterations=1000, threshold=0.01):
    n = similarity_matrix.shape[0]
    states = np.random.choice([0, 1], size=n)  # 문장 상태 초기화 (0: 선택되지 않음, 1: 선택됨)
    
    for _ in range(num_iterations):
        for i in range(n):
            neighbors = similarity_matrix[i]
            prob_select = np.dot(neighbors, states)  # 이웃의 상태에 따른 확률 계산
            if prob_select > threshold:  # 임계값에 따라 상태 결정
                states[i] = 1
            else:
                states[i] = 0
    return states

# NetMRF 기반 단어 추출 (공출현 행렬 기반으로 MRF 최적화 적용)
def infomap_netmrf_keywords_extraction(text, top_n=5, num_iterations=1000, threshold=0.01):
    if not text or text.strip() == '':
        return []
    
    words = word_tokenize(text.lower())
    co_occurrence = calculate_co_occurrence(words)
    
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    similarity_matrix = np.zeros((len(word_to_id), len(word_to_id)))
    for (word1, word2), weight in co_occurrence.items():
        similarity_matrix[word_to_id[word1]][word_to_id[word2]] = weight
    
    # MRF 초기화 및 최적화 적용
    mrf_probabilities = initialize_mrf(similarity_matrix)
    optimized_states = apply_mrf(mrf_probabilities, num_iterations, threshold)
    
    # 최적화된 상태에 따라 선택된 단어들만 추출
    selected_words = [id_to_word[i] for i, state in enumerate(optimized_states) if state == 1]
    
    # Infomap을 통해 최종적으로 중요한 단어 추출
    infomap = Infomap()
    for word1 in selected_words:
        for word2 in selected_words:
            if word1 != word2 and co_occurrence[(word1, word2)] > 0:
                infomap.add_link(word_to_id[word1], word_to_id[word2], co_occurrence[(word1, word2)])
    
    if infomap.num_nodes > 0:
        infomap.run()
    else:
        return []
    
    module_words = {}
    for node in infomap.iterTree():
        if node.isLeaf:
            module_id = node.moduleId
            word = id_to_word[node.physicalId]
            if module_id not in module_words:
                module_words[module_id] = []
            module_words[module_id].append(word)
    
    extracted_keywords = []
    for words in module_words.values():
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(top_n)
        extracted_keywords.extend([word for word, _ in most_common_words])
    
    return extracted_keywords

# DataFrame에 적용 (NetMRF를 사용하여 키워드 추출)
df_filtered_111['num_keywords'] = df_filtered_111['keywords'].apply(count_keywords)

df_filtered_111['extracted_keywords'] = df_filtered_111.apply(
    lambda row: infomap_netmrf_keywords_extraction(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_111[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_111 = apply_metrics(df_filtered_111)

# 최종 결과 출력
df_result111 = df_filtered_111[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 11.2 textrank + term frequency, term postion, word co-occurence + Infomap + NetMRF
df_filtered_112 = df_filtered.copy()
df_filtered_112.dtypes

# TF 계산 함수 (abstract에서의 term frequency 계산)
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 계산 함수
def calculate_co_occurrence(words, window_size=2):
    co_occurrence = Counter()
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
    return co_occurrence

# MRF 초기화 및 최적화 함수 (NetMRF 적용)
def initialize_mrf(similarity_matrix):
    normalized_sim_matrix = normalize(similarity_matrix, axis=1, norm='l1')
    return normalized_sim_matrix

def apply_mrf(similarity_matrix, num_iterations=1000, threshold=0.01):
    n = similarity_matrix.shape[0]
    states = np.random.choice([0, 1], size=n)  # 문장 상태 초기화 (0: 선택되지 않음, 1: 선택됨)
    
    for _ in range(num_iterations):
        for i in range(n):
            neighbors = similarity_matrix[i]
            prob_select = np.dot(neighbors, states)  # 이웃의 상태에 따른 확률 계산
            if prob_select > threshold:  # 임계값에 따라 상태 결정
                states[i] = 1
            else:
                states[i] = 0
    return states

# NetMRF 기반 단어 추출 (공출현 행렬 기반으로 MRF 최적화 적용, TF와 term position 반영)
def infomap_netmrf_keywords_extraction(title, abstract, top_n=5, num_iterations=1000, threshold=0.01, beta=0.5):
    if not title or not abstract or title.strip() == '' or abstract.strip() == '':
        return []
    
    # 단어 토큰화
    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    words = words_title + words_abstract
    
    # TF 계산 (abstract 기반)
    tf = calculate_tf(abstract)

    # 공출현 계산
    co_occurrence = calculate_co_occurrence(words)

    # 단어를 정수로 매핑
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(word_to_id), len(word_to_id)))
    for (word1, word2), weight in co_occurrence.items():
        similarity_matrix[word_to_id[word1]][word_to_id[word2]] = weight
    
    # MRF 초기화 및 최적화 적용
    mrf_probabilities = initialize_mrf(similarity_matrix)
    optimized_states = apply_mrf(mrf_probabilities, num_iterations, threshold)
    
    # 최적화된 상태에 따라 선택된 단어들만 추출
    selected_words = [id_to_word[i] for i, state in enumerate(optimized_states) if state == 1]
    
    # Infomap 알고리즘 초기화
    infomap = Infomap()
    for word1 in selected_words:
        for word2 in selected_words:
            if word1 != word2 and co_occurrence[(word1, word2)] > 0:
                infomap.add_link(word_to_id[word1], word_to_id[word2], co_occurrence[(word1, word2)])

    if infomap.num_nodes > 0:
        infomap.run()
    else:
        return []

    # 각 모듈(커뮤니티)별 단어를 모아서 저장
    module_words = {}
    for node in infomap.iterTree():
        if node.isLeaf:
            module_id = node.moduleId
            word = id_to_word[node.physicalId]
            if module_id not in module_words:
                module_words[module_id] = []
            module_words[module_id].append(word)

    # 각 모듈에서 TF 점수를 반영하여 중요한 단어 추출
    extracted_keywords = []
    for words in module_words.values():
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(top_n)
        extracted_keywords.extend([word for word, _ in most_common_words])

    return extracted_keywords

# DataFrame에 적용 (NetMRF를 사용하여 키워드 추출)
df_filtered_112['num_keywords'] = df_filtered_112['keywords'].apply(count_keywords)

df_filtered_112['extracted_keywords'] = df_filtered_112.apply(
    lambda row: infomap_netmrf_keywords_extraction(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_112[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_112 = apply_metrics(df_filtered_112)

# 최종 결과 출력
df_result112 = df_filtered_112[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 11.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + NetMRF
df_filtered_113 = df_filtered.copy()
df_filtered_113.dtypes

# Preprocess the abstract (remove stopwords and tokenize)
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]  # Only keep alphanumeric words
    return words

# Compute term frequency (TF)
def calculate_tf(text):
    words = preprocess_text(text)
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# Compute co-occurrence of words within a window size
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = preprocess_text(sentence)
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(sentences):
    weighted_sentences = []
    
    for sentence in sentences:
        words = preprocess_text(sentence)
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 부정어 간 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# Compute similarity matrix using TF, co-occurrence, position, and weights
def calculate_similarity_matrix(sentences, title, beta=0.5):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Preprocess title and sentences, and apply sentence weights
    title_words = preprocess_text(title)
    weighted_sentences = apply_weights(sentences)
    sentence_words = [preprocess_text(sent) for sent, _ in weighted_sentences]
    sentence_weights = [weight for _, weight in weighted_sentences]
    
    # Calculate term frequency (TF) for each sentence
    tf_list = [calculate_tf(sent) for sent, _ in weighted_sentences]
    
    # Calculate co-occurrence matrix for sentences
    co_occurrence = calculate_co_occurrence([sent for sent, _ in weighted_sentences])
    
    for i, words_i in enumerate(sentence_words):
        for j, words_j in enumerate(sentence_words):
            if i == j:
                continue
            # Find common words between two sentences
            common_words = set(words_i) & set(words_j)
            
            # Calculate similarity based on TF, co-occurrence, and sentence weights
            similarity = sum(tf_list[i].get(word, 0) * tf_list[j].get(word, 0) for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # Apply term position importance based on title and abstract
                        weight_i = 1 if word_i in title_words else beta
                        weight_j = 1 if word_j in title_words else beta
                        similarity += co_occurrence[(word_i, word_j)] * weight_i * weight_j * sentence_weights[i] * sentence_weights[j]
            similarity_matrix[i][j] = similarity
    
    return similarity_matrix

# Extract keywords using Infomap and TextRank with Double Negation, Mitigation, and Hedges Weighting
def infomap_textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # Compute similarity matrix
    similarity_matrix = calculate_similarity_matrix(sentences, title, beta)
    
    # Build a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Use Infomap to find clusters in the graph
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()

    # Extract top keywords from each cluster
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = preprocess_text(sentences[sentence_idx])
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)
    
    return list(set(keywords))

# Add 'num_keywords' for setting top_n dynamically
df_filtered_113['num_keywords'] = df_filtered_113['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using Infomap and term frequency, term position, word co-occurrence, and Double Negation, Mitigation, and Hedges Weighting
df_filtered_113['extracted_keywords'] = df_filtered_113.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_113[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_113 = apply_metrics(df_filtered_113)

# 최종 결과 출력
df_result113 = df_filtered_113[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 12.1 textrank + Infomap + NetMRF + GloVe 
df_filtered_121 = df_filtered.copy()

# GloVe 임베딩을 사용한 유사도 계산 함수
def glove_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    return 0.0

# MRF 초기화 및 최적화 함수 (NetMRF 적용)
def initialize_mrf(similarity_matrix):
    normalized_sim_matrix = normalize(similarity_matrix, axis=1, norm='l1')
    return normalized_sim_matrix

def apply_mrf(similarity_matrix, num_iterations=1000, threshold=0.01):
    n = similarity_matrix.shape[0]
    states = np.random.choice([0, 1], size=n)  # 문장 상태 초기화 (0: 선택되지 않음, 1: 선택됨)
    
    for _ in range(num_iterations):
        for i in range(n):
            neighbors = similarity_matrix[i]
            prob_select = np.dot(neighbors, states)  # 이웃의 상태에 따른 확률 계산
            if prob_select > threshold:  # 임계값에 따라 상태 결정
                states[i] = 1
            else:
                states[i] = 0
    return states

# NetMRF 기반 단어 추출 (GloVe 유사도 및 공출현 행렬 기반으로 MRF 최적화 적용)
def infomap_netmrf_keywords_extraction(text, top_n=5, glove_embeddings=None, num_iterations=1000, threshold=0.01):
    if not text or text.strip() == '':
        return []
    
    words = word_tokenize(text.lower())
    co_occurrence = calculate_co_occurrence(words)
    
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    similarity_matrix = np.zeros((len(word_to_id), len(word_to_id)))
    
    # GloVe 임베딩을 사용하여 유사도 계산
    for word1 in word_to_id:
        for word2 in word_to_id:
            if word1 != word2:
                similarity = glove_similarity(word1, word2, glove_embeddings)
                if similarity > 0:
                    similarity_matrix[word_to_id[word1]][word_to_id[word2]] = similarity
    
    # MRF 초기화 및 최적화 적용
    mrf_probabilities = initialize_mrf(similarity_matrix)
    optimized_states = apply_mrf(mrf_probabilities, num_iterations, threshold)
    
    # 최적화된 상태에 따라 선택된 단어들만 추출
    selected_words = [id_to_word[i] for i, state in enumerate(optimized_states) if state == 1]
    
    # Infomap을 통해 최종적으로 중요한 단어 추출
    infomap = Infomap()
    for word1 in selected_words:
        for word2 in selected_words:
            if word1 != word2 and co_occurrence[(word1, word2)] > 0:
                infomap.add_link(word_to_id[word1], word_to_id[word2], co_occurrence[(word1, word2)])
    
    if infomap.num_nodes > 0:
        infomap.run()
    else:
        return []
    
    module_words = {}
    for node in infomap.iterTree():
        if node.isLeaf:
            module_id = node.moduleId
            word = id_to_word[node.physicalId]
            if module_id not in module_words:
                module_words[module_id] = []
            module_words[module_id].append(word)
    
    extracted_keywords = []
    for words in module_words.values():
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(top_n)
        extracted_keywords.extend([word for word, _ in most_common_words])
    
    return extracted_keywords

# DataFrame에 적용 (NetMRF를 사용하여 GloVe 임베딩으로 키워드 추출)
df_filtered_121['num_keywords'] = df_filtered_121['keywords'].apply(count_keywords)

df_filtered_121['extracted_keywords'] = df_filtered_121.apply(
    lambda row: infomap_netmrf_keywords_extraction(
        row['abstract'], top_n=row['num_keywords'], glove_embeddings=glove_embeddings
    ) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_121[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_121 = apply_metrics(df_filtered_121)

# 최종 결과 출력
df_result121 = df_filtered_121[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 12.2 textrank + term frequency, term postion, word co-occurence + Infomap + NetMRF + GloVe 
df_filtered_122 = df_filtered.copy()
df_filtered_122.dtypes

# GloVe 임베딩을 사용한 유사도 계산 함수
def glove_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    return 0.0

# TF 계산 함수 (abstract에서의 term frequency 계산)
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 계산 함수
def calculate_co_occurrence(words, window_size=2):
    co_occurrence = Counter()
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
    return co_occurrence

# MRF 초기화 및 최적화 함수 (NetMRF 적용)
def initialize_mrf(similarity_matrix):
    normalized_sim_matrix = normalize(similarity_matrix, axis=1, norm='l1')
    return normalized_sim_matrix

def apply_mrf(similarity_matrix, num_iterations=1000, threshold=0.01):
    n = similarity_matrix.shape[0]
    states = np.random.choice([0, 1], size=n)  # 문장 상태 초기화 (0: 선택되지 않음, 1: 선택됨)
    
    for _ in range(num_iterations):
        for i in range(n):
            neighbors = similarity_matrix[i]
            prob_select = np.dot(neighbors, states)  # 이웃의 상태에 따른 확률 계산
            if prob_select > threshold:  # 임계값에 따라 상태 결정
                states[i] = 1
            else:
                states[i] = 0
    return states

# NetMRF 기반 단어 추출 (GloVe 임베딩 및 공출현 행렬 기반으로 MRF 최적화 적용)
def infomap_netmrf_keywords_extraction(title, abstract, top_n=5, num_iterations=1000, threshold=0.01, beta=0.5, glove_embeddings=None):
    if not title or not abstract or title.strip() == '' or abstract.strip() == '':
        return []
    
    # 단어 토큰화
    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    words = words_title + words_abstract
    
    # TF 계산 (abstract 기반)
    tf = calculate_tf(abstract)

    # 공출현 계산
    co_occurrence = calculate_co_occurrence(words)

    # 단어를 정수로 매핑
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(word_to_id), len(word_to_id)))
    
    # GloVe 임베딩을 사용한 유사도 및 공출현 계산
    for word1 in word_to_id:
        for word2 in word_to_id:
            if word1 != word2:
                glove_sim = glove_similarity(word1, word2, glove_embeddings)
                co_occurrence_sim = co_occurrence.get((word1, word2), 0)
                similarity_matrix[word_to_id[word1]][word_to_id[word2]] = glove_sim + co_occurrence_sim
    
    # MRF 초기화 및 최적화 적용
    mrf_probabilities = initialize_mrf(similarity_matrix)
    optimized_states = apply_mrf(mrf_probabilities, num_iterations, threshold)
    
    # 최적화된 상태에 따라 선택된 단어들만 추출
    selected_words = [id_to_word[i] for i, state in enumerate(optimized_states) if state == 1]
    
    # Infomap 알고리즘 초기화
    infomap = Infomap()
    for word1 in selected_words:
        for word2 in selected_words:
            if word1 != word2 and co_occurrence.get((word1, word2), 0) > 0:
                infomap.add_link(word_to_id[word1], word_to_id[word2], co_occurrence.get((word1, word2)))

    if infomap.num_nodes > 0:
        infomap.run()
    else:
        return []

    # 각 모듈(커뮤니티)별 단어를 모아서 저장
    module_words = {}
    for node in infomap.iterTree():
        if node.isLeaf:
            module_id = node.moduleId
            word = id_to_word[node.physicalId]
            if module_id not in module_words:
                module_words[module_id] = []
            module_words[module_id].append(word)

    # 각 모듈에서 TF 점수를 반영하여 중요한 단어 추출
    extracted_keywords = []
    for words in module_words.values():
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(top_n)
        extracted_keywords.extend([word for word, _ in most_common_words])

    return extracted_keywords

# DataFrame에 적용 (NetMRF와 GloVe 임베딩을 사용하여 키워드 추출)
df_filtered_122['num_keywords'] = df_filtered_122['keywords'].apply(count_keywords)

df_filtered_122['extracted_keywords'] = df_filtered_122.apply(
    lambda row: infomap_netmrf_keywords_extraction(
        row['title'], row['abstract'], top_n=row['num_keywords'], glove_embeddings=glove_embeddings
    ) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_122[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_122 = apply_metrics(df_filtered_122)

# 최종 결과 출력
df_result122 = df_filtered_122[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


#### 12.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + NetMRF + GloVe 
df_filtered_123 = df_filtered.copy()
df_filtered_123.dtypes

# GloVe 임베딩을 사용한 유사도 계산 함수
def glove_similarity(word1, word2, embeddings):
    if word1 in embeddings and word2 in embeddings:
        return cosine_similarity(embeddings[word1], embeddings[word2])
    return 0.0

# Double Negation, Mitigation, and Hedges Weighting 적용 함수
def apply_weights(sentences):
    weighted_sentences = []
    
    for sentence in sentences:
        words = preprocess_text(sentence)
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 부정어 간 거리가 길수록 가중치 증가

        # Mitigation (완화 표현) 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges (완충 표현) 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # 완충 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# Compute similarity matrix using GloVe, TF, co-occurrence, and position
def calculate_similarity_matrix(sentences, title, glove_embeddings, beta=0.5):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Preprocess title and sentences, and apply sentence weights
    title_words = preprocess_text(title)
    weighted_sentences = apply_weights(sentences)
    sentence_words = [preprocess_text(sent) for sent, _ in weighted_sentences]
    sentence_weights = [weight for _, weight in weighted_sentences]
    
    # Calculate term frequency (TF) for each sentence
    tf_list = [calculate_tf(sent) for sent, _ in weighted_sentences]
    
    # Calculate co-occurrence matrix for sentences
    co_occurrence = calculate_co_occurrence([sent for sent, _ in weighted_sentences])
    
    for i, words_i in enumerate(sentence_words):
        for j, words_j in enumerate(sentence_words):
            if i == j:
                continue
            # Find common words between two sentences
            common_words = set(words_i) & set(words_j)
            
            # Calculate similarity based on GloVe embeddings and co-occurrence
            similarity = sum(tf_list[i].get(word, 0) * tf_list[j].get(word, 0) for word in common_words)
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        # Apply term position importance based on title and abstract
                        weight_i = 1 if word_i in title_words else beta
                        weight_j = 1 if word_j in title_words else beta
                        glove_sim = glove_similarity(word_i, word_j, glove_embeddings)
                        similarity += (glove_sim + co_occurrence[(word_i, word_j)]) * weight_i * weight_j * sentence_weights[i] * sentence_weights[j]
            similarity_matrix[i][j] = similarity
    
    return similarity_matrix

# Extract keywords using Infomap and GloVe-based similarity
def infomap_textrank_keywords(title, abstract, top_n=5, glove_embeddings=None, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
    # Compute similarity matrix
    similarity_matrix = calculate_similarity_matrix(sentences, title, glove_embeddings, beta)
    
    # Build a graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Use Infomap to find clusters in the graph
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    
    infomap.run()

    # Extract top keywords from each cluster
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = preprocess_text(sentences[sentence_idx])
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)
    
    return list(set(keywords))

# Add 'num_keywords' for setting top_n dynamically
df_filtered_123['num_keywords'] = df_filtered_123['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using Infomap and GloVe embeddings with Double Negation, Mitigation, and Hedges Weighting
df_filtered_123['extracted_keywords'] = df_filtered_123.apply(
    lambda row: infomap_textrank_keywords(
        row['title'], row['abstract'], top_n=row['num_keywords'], glove_embeddings=glove_embeddings, beta=0.5
    ) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_123[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_123 = apply_metrics(df_filtered_123)

# 최종 결과 출력
df_result123 = df_filtered_123[['precision', 'recall', 'f1', 'rouge1', 'rougeL', 'FM_Index', 'ARI', 'MCC', 'Bray_Curtis']]


print(df_result11) #### 1.1 textrank
print(df_result12) #### 1.2 textrank + term frequency, term postion, word co-occurence
print(df_result13) #### 1.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting
print(df_result14) #### 1.4 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Glove
print(df_result2) #### 2. textrank + TP-CoGlo-TextRank(GLove)
print(df_result3) #### 3. textrank + Watts-Strogatz model
print(df_result41) #### 4.1 textrank + Infomap
print(df_result42) #### 4.2 textrank + term frequency, term postion, word co-occurence + Infomap
print(df_result43) #### 4.3 textrank + Infomap + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges
print(df_result51) #### 5.1 textrank + Infomap + GloVe 
print(df_result52) #### 5.2 textrank + term frequency, term postion, word co-occurence + Infomap + GloVe
print(df_result53) #### 5.3 textrank + term frequency, term postion, word co-occurence + Infomap + Double Negation, Mitigation, and Hedges + GloVe
print(df_result61) #### 6.1 textrank + Infomap + Hierarchical
print(df_result62) #### 6.2 textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical
print(df_result63) #### 6.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Hierarchical
print(df_result71) #### 7.1 textrank + Infomap + Hierarchical + GloVe 
print(df_result72) #### 7.2 textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical + GloVe 
print(df_result73) #### 7.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Hierarchical + GloVe 
print(df_result81) #### 8.1 textrank + Infomap + Multi Entropy
print(df_result82) #### 8.2 textrank + term frequency, term postion, word co-occurence + Infomap + Multi Entropy
print(df_result83) #### 8.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Multi Entropy
print(df_result91) #### 9.1 textrank + Infomap + Multi Entropy + GloVe
print(df_result92) #### 9.2 textrank + term frequency, term postion, word co-occurence + Infomap + Multi level Entropy + GloVe 
print(df_result101) #### 10.1 textrank + Infomap + jaccard
print(df_result102) #### 10.2 textrank + term frequency, term postion, word co-occurence + Infomap + jcd
print(df_result103) #### 10.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + jcd
print(df_result111) #### 11.1 textrank + Infomap + NetMRF
print(df_result112) #### 11.2 textrank + term frequency, term postion, word co-occurence + Infomap + NetMRF
print(df_result113) #### 11.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + NetMRF
print(df_result121) #### 12.1 textrank + Infomap + NetMRF + GloVe 
print(df_result122) #### 12.2 textrank + term frequency, term postion, word co-occurence + Infomap + NetMRF + GloVe
print(df_result123) #### 12.3 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + NetMRF + GloVe 


# 각 DataFrame의 평균 계산 함수
def calculate_means(df):
    means = df.mean()
    return means

# 각 DataFrame의 평균 계산
means_result11 = calculate_means(df_result11)
means_result12 = calculate_means(df_result12)
means_result13 = calculate_means(df_result13)
means_result14 = calculate_means(df_result14)
means_result2 = calculate_means(df_result2)
means_result3 = calculate_means(df_result3)
means_result41 = calculate_means(df_result41)
means_result42 = calculate_means(df_result42)
means_result43 = calculate_means(df_result43)
means_result51 = calculate_means(df_result51)
means_result52 = calculate_means(df_result52)
means_result53 = calculate_means(df_result53)
means_result61 = calculate_means(df_result61)
means_result62 = calculate_means(df_result62)
means_result63 = calculate_means(df_result63)
means_result71 = calculate_means(df_result71)
means_result72 = calculate_means(df_result72)
means_result73 = calculate_means(df_result73)
means_result81 = calculate_means(df_result81)
means_result82 = calculate_means(df_result82)
means_result83 = calculate_means(df_result83)
means_result91 = calculate_means(df_result91)
means_result92 = calculate_means(df_result92)
means_result101 = calculate_means(df_result101)
means_result102 = calculate_means(df_result102)
means_result103 = calculate_means(df_result103)
means_result111 = calculate_means(df_result111)
means_result112 = calculate_means(df_result112)
means_result113 = calculate_means(df_result113)
means_result121 = calculate_means(df_result121)
means_result122 = calculate_means(df_result122)
means_result123 = calculate_means(df_result123)

# 평균 결과를 사전으로 변환
means_dict = {
    "M01 textrank"  : means_result11,
"M02 textrank + term frequency, term postion, word co-occurence"  : means_result12,
"M03 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting "  : means_result13,
"M04 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Glove"  : means_result14,
"M05 textrank + TP-CoGlo-TextRank(GLove)"  : means_result2,
"M06 textrank + Watts-Strogatz model"  : means_result3,
"M07 textrank + Infomap"  : means_result41,
"M08 textrank + term frequency, term postion, word co-occurence + Infomap"  : means_result42,
"M09 textrank + Infomap + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges"  : means_result43,
"M10 textrank + Infomap + GloVe "  : means_result51,
"M11 textrank + term frequency, term postion, word co-occurence + Infomap + GloVe"  : means_result52,
"M12 textrank + term frequency, term postion, word co-occurence + Infomap + Double Negation, Mitigation, and Hedges + GloVe"  : means_result53,
"M13 textrank + Infomap + Hierarchical"  : means_result61,
"M14 textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical"  : means_result62,
"M15 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Hierarchical"  : means_result63,
"M16 textrank + Infomap + Hierarchical + GloVe "  : means_result71,
"M17 textrank + term frequency, term postion, word co-occurence + Infomap + Hierarchical + GloVe "  : means_result72,
"M18 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Hierarchical + GloVe "  : means_result73,
"M19 textrank + Infomap + Multi Entropy"  : means_result81,
"M20 textrank + term frequency, term postion, word co-occurence + Infomap + Multi Entropy"  : means_result82,
"M21 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Multi Entropy"  : means_result83,
"M22 textrank + Infomap + Multi Entropy + GloVe "  : means_result91,
"M23 textrank + term frequency, term postion, word co-occurence + Infomap + Multi level Entropy + GloVe "  : means_result92,
"M24 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + Multi Entropy + GloVe "  : means_result101,
"M25 textrank + Infomap + jcd"  : means_result102,
"M26 textrank + term frequency, term postion, word co-occurence + Infomap + jcd"  : means_result103,
"M27 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + jcd"  : means_result111,
"M28 textrank + Infomap + NetMRF"  : means_result112,
"M29 textrank + term frequency, term postion, word co-occurence + Infomap + NetMRF"  : means_result113,
"M30 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + NetMRF"  : means_result121,
"M31 textrank + Infomap + NetMRF + GloVe "  : means_result122,
"M32 textrank + term frequency, term postion, word co-occurence + Infomap + NetMRF + GloVe "  : means_result123,
"M33 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + NetMRF + GloVe "  : means_result123
}

# 사전을 DataFrame으로 변환
summary_df = pd.DataFrame(means_dict)

# 전치 (Transpose)하여 인덱스가 행으로, 열이 컬럼으로 되게 변환
summary_df = summary_df.T

# summary_df를 CSV 파일로 저장
summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result_1000_1006_pre.csv', index=True)


# 1. Precision (정밀도) 용도: 정밀도는 모델이 얼마나 정확하게 예측했는지를 나타냅니다. 즉, 예측한 결과가 실제로 얼마나 맞았는지를 평가할 때 사용됩니다.

# 2. Recall (재현율) 용도: 재현율은 모델이 실제로 있어야 할 정답을 얼마나 많이 찾았는지를 평가합니다. 특히, 누락을 줄이는 것이 중요한 경우에 유용합니다.

# 3. F1 Score (F1 점수) 용도: F1 점수는 정밀도와 재현율 간의 균형이 중요한 상황에서 유용합니다. 특히, 둘 중 하나에 치우친 모델을 평가하는 데 좋습니다.

# 4. ROUGE-1 & ROUGE-L (루즈-1 및 루즈-L) ROUGE-1은 추출된 단어들에서 단일 단어 수준에서 일치하는지를 평가합니다.
# ROUGE-L: 가장 긴 공통 연속 부분 문자열(LCS)을 기반으로, 추출된 문장과 참 문장 사이의 구조적 유사성을 평가합니다.

# 5. FM Index (Fowlkes–Mallows Index) 용도: 클러스터링 또는 군집화를 평가할 때, 참 레이블과의 일치도를 측정하는 데 사용됩니다. 정밀도와 재현율의 기하 평균을 사용해 클러스터의 품질을 평가합니다.

# 6. ARI (Adjusted Rand Index) 용도: 클러스터링 결과가 우연히 잘 맞아떨어졌는지, 실제로 유의미한 군집화를 했는지를 평가할 때 사용됩니다. ARI는 조정된 값이므로, 랜덤하게 군집을 생성한 것과 실제 군집화 성능을 비교하는 데 적합합니다.

# 7. MCC (Matthews Correlation Coefficient) 용도: 정확도, 정밀도, 재현율을 모두 고려하여 예측의 전체적인 성능을 평가합니다. 특히, 데이터가 불균형한 경우에도 신뢰성 있게 성능을 평가할 수 있는 지표입니다.

# 8. Bray-Curtis Dissimilarity (브레이-커티스 비유사도) 용도: 생태학이나 데이터 마이닝에서 두 분포의 차이를 측정하는 데 사용됩니다. 샘플 간의 비유사도를 측정하는 데 유용합니다.


