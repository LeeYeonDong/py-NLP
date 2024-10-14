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
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
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
from sklearn.metrics import average_precision_score, fowlkes_mallows_score, adjusted_rand_score, matthews_corrcoef
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import braycurtis


# 파일 경로 설정
# file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined100.csv'
file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined500.csv'
# file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined1000.csv'
#file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined2000.csv'
#file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined5000.csv'
#file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined10000.csv'
#file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined15000.csv'
#file_path = 'D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\dblp_v14_random_sample_combined20000.csv'


# CSV 파일 불러오기
df_filtered = pd.read_csv(file_path)
df_filtered = df_filtered.astype(str)
df_filtered.dtypes
df_filtered = df_filtered[['id', 'title', 'keywords', 'year', 'abstract', 'authors']]
df_filtered = df_filtered.dropna(subset=['id', 'title', 'keywords', 'year', 'abstract', 'authors'])
df_filtered['keywords']
df_filtered['keywords'] = df_filtered['keywords'].str.replace(',', '', regex=False)

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

# 각 DataFrame의 평균 계산 함수
def calculate_means(df):
    means = df.mean()
    return means

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
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

# FM Index, Bray-Curtis, IoU 계산 함수
def calculate_additional_metrics(extracted, actual):
    extracted_set = set(extracted)
    actual_set = set(actual)

    # 이진화된 벡터로 변환 (키워드 존재 여부에 따라 1, 0으로 구분)
    extracted_binary = [1 if word in extracted_set else 0 for word in actual]
    actual_binary = [1] * len(actual)

    if len(extracted_binary) == len(actual_binary):
        fm_index = fowlkes_mallows_score(actual_binary, extracted_binary)
        bray_curtis = braycurtis(extracted_binary, actual_binary)
        iou_score = jaccard_score(actual_binary, extracted_binary)
    else:
        fm_index, bray_curtis, iou_score = 0, 1, 0  # 다른 크기의 경우 처리

    return fm_index, bray_curtis, iou_score

# DataFrame에 적용하여 모든 메트릭스 계산
def apply_metrics(df):
    # Precision, Recall, F1, ROUGE-1 계산
    df['metrics'] = df.apply(lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split()), axis=1)
    df[['precision', 'recall', 'f1']] = pd.DataFrame(df['metrics'].tolist(), index=df.index)

    df['rouge'] = df.apply(lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split()), axis=1)
    df['rouge1'] = df['rouge'].apply(lambda x: x['rouge1'].fmeasure)

    # FM Index, Bray-Curtis, IoU 계산
    df['additional_metrics'] = df.apply(lambda row: calculate_additional_metrics(row['extracted_keywords'], row['keywords'].split()), axis=1)
    df[['FM_Index', 'Bray_Curtis', 'IoU']] = pd.DataFrame(df['additional_metrics'].tolist(), index=df.index)

    return df


#### M01 textrank
start_time = time.time()

time.sleep(3)
df_filtered_01 = df_filtered.copy()
df_filtered_01.dtypes

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
df_filtered_01['num_keywords'] = df_filtered_01['keywords'].apply(lambda x: len(x.split()))

# 'num_keywords'를 top_n으로 사용하여 'extracted_keywords' 생성
df_filtered_01['extracted_keywords'] = df_filtered_01.apply(
    lambda row: textrank_keywords(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_01[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_01 = apply_metrics(df_filtered_01)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_01['time'] = total_time

# 최종 결과 출력
df_filtered_01.dtypes
df_result01 = df_filtered_01[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]

means_result01 = calculate_means(df_result01)



#### M02 textrank + term frequency, term postion, word co-occurence
start_time = time.time()

time.sleep(3)
df_filtered_02 = df_filtered.copy()
df_filtered_02.dtypes

# TF calculation function
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# Co-occurrence calculation function
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank keyword extraction function
def textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Calculate term frequency (TF)
    tf = calculate_tf(text)
    
    # Calculate word co-occurrence
    co_occurrence = calculate_co_occurrence(sentences)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            
            # Find common words between two sentences
            common_words = set(words_i) & set(words_j)
            similarity = sum(tf[word] for word in common_words)
            
            # Incorporate co-occurrence and term position
            for word_i in words_i:
                for word_j in words_j:
                    if (word_i, word_j) in co_occurrence:
                        weight_i = 1 if word_i in word_tokenize(title.lower()) else beta
                        weight_j = 1 if word_j in word_tokenize(title.lower()) else beta
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
                        similarity_matrix[i][j] += similarity * weight_i * weight_j  # Adjust for term position

    # Build similarity graph and apply PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Rank sentences and extract top keywords
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences:
        keywords.extend(word_tokenize(sentence.lower()))
        if len(set(keywords)) >= top_n:
            break
    
    return list(set(keywords))[:top_n]

# Example dataframe usage
start_time = time.time()

# Assume df_filtered is pre-defined with 'abstract' and 'keywords' columns
df_filtered_02 = df_filtered.copy()

# Calculate number of keywords in 'keywords' column
df_filtered_02['num_keywords'] = df_filtered_02['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using textrank_keywords function
df_filtered_02['extracted_keywords'] = df_filtered_02.apply(
    lambda row: textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# Print result
print(df_filtered_02[['abstract', 'keywords', 'extracted_keywords']])

# Assume apply_metrics is a custom function to calculate metrics on the DataFrame
time.sleep(3)
df_filtered_02 = apply_metrics(df_filtered_02)

end_time = time.time()
total_time = end_time - start_time
df_filtered_02['time'] = total_time

# Final result output
df_result02 = df_filtered_02[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]

means_result02 = calculate_means(df_result02)



#### M03 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting 
start_time = time.time()

time.sleep(3)
df_filtered_03 = df_filtered.copy()
df_filtered_03.dtypes

# Function to apply weights for Double Negation, Mitigation, and Hedges
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # Default weight

        # Double Negation Weighting
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nowhere', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # Increase weight if multiple negations found

        # Mitigation Weighting
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # Increase weight if mitigation words are present

        # Hedges Weighting
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'likely', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # Increase weight if hedge words are present

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# TF calculation function
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# Co-occurrence calculation function
def calculate_co_occurrence(sentences, window_size=2): # window size adjustable
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Textrank keyword extraction function with Double Negation, Mitigation, and Hedges Weighting
def textrank_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    
    sentences = [s for s, w in weighted_sentences]  # Extract sentences
    weights = [w for s, w in weighted_sentences]    # Extract corresponding weights
    
    words = word_tokenize(text.lower())
    
    # Calculate TF
    tf = calculate_tf(text)
    
    # Calculate word co-occurrence
    co_occurrence = calculate_co_occurrence(sentences)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # Calculate similarity matrix
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
                        weight_i = 1 if word_i in word_tokenize(title.lower()) else beta
                        weight_j = 1 if word_j in word_tokenize(title.lower()) else beta
                        similarity += co_occurrence[(word_i, word_j)] / sum(co_occurrence[(word_i, word)] for word in words)
            
            # Apply sentence weights
            similarity_matrix[i][j] = similarity * ((weights[i] + weights[j]) / 2)
    
    # Build similarity graph and apply PageRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Extract keywords from top ranked sentences
    keywords = []
    for score, sentence in ranked_sentences[:top_n]:
        keywords.extend(word_tokenize(sentence.lower()))
    
    return list(set(keywords))

# Function to calculate the number of keywords from the 'keywords' column
def get_keyword_count(keywords):
    return len(keywords.split())

# DataFrame example usage (df_filtered is assumed to be pre-defined with 'abstract' and 'keywords' columns)
df_filtered_03 = df_filtered.copy()

# Extract keywords using textrank_keywords function (adjusting top_n based on number of existing keywords)
df_filtered_03['extracted_keywords'] = df_filtered_03.apply(
    lambda row: textrank_keywords(
        row['title'], 
        row['abstract'], 
        top_n=get_keyword_count(row['keywords']), 
        beta=0.5
    ) if pd.notnull(row['abstract']) else [], 
    axis=1
)

# Print the DataFrame result
print(df_filtered_03[['abstract', 'keywords', 'extracted_keywords']])

# Assume apply_metrics is a custom function to calculate metrics on the DataFrame
time.sleep(3)
df_filtered_03 = apply_metrics(df_filtered_03)

end_time = time.time()
total_time = end_time - start_time
df_filtered_03['time'] = total_time

# Final result output
df_result03 = df_filtered_03[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result03 = calculate_means(df_result03)



# M04 textrank + TP-CoGlo-TextRank
start_time = time.time()

time.sleep(3)
df_filtered_04 = df_filtered.copy()
df_filtered_04.dtypes

# GloVe 유사도 계산 함수
def calculate_glove_similarity(word1, word2):
    try:
        # GloVe에서 벡터를 가져옵니다
        vector1 = glove_embeddings.get(word1)
        vector2 = glove_embeddings.get(word2)
        
        # 두 벡터가 존재하고, 모두 1차원 벡터일 때 코사인 유사도 계산
        if vector1 is not None and vector2 is not None and len(vector1.shape) == 1 and len(vector2.shape) == 1:
            vector1 = vector1.reshape(1, -1)  # 벡터를 2차원으로 변환
            vector2 = vector2.reshape(1, -1)  # 벡터를 2차원으로 변환
            return cosine_similarity(vector1, vector2)[0][0]
        else:
            return 0.0  # 벡터가 존재하지 않으면 유사도를 0으로 반환
    except Exception as e:
        print(f"Error calculating similarity between {word1} and {word2}: {e}")
        return 0.0  # 예외 발생 시 유사도 0으로 반환

# 단어 빈도 계산 함수 (TF)
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 빈도 계산 함수
def calculate_co_occurrence(sentences, window_size=2):
    co_occurrence = Counter()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                co_occurrence[(word, words[j])] += 1
                co_occurrence[(words[j], word)] += 1
    return co_occurrence

# TP-CoGlo-TextRank 키워드 추출 함수
def tp_coglo_textrank_keywords(title, abstract, top_n=5, beta=0.5, gamma=0.2, lambda_param=0.3):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # 단어의 TF 계산
    tf = calculate_tf(text)
    
    # 공출현 빈도 계산
    co_occurrence = calculate_co_occurrence(sentences)
    
    # 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    # 유사도 행렬 계산 (공출현 빈도와 GloVe 유사도 결합)
    for i, sentence_i in enumerate(sentences):
        for j, sentence_j in enumerate(sentences):
            if i == j:
                continue
            words_i = word_tokenize(sentence_i.lower())
            words_j = word_tokenize(sentence_j.lower())
            
            # 공출현 기반 유사도
            co_occurrence_similarity = sum(tf[word] for word in set(words_i) & set(words_j))
            
            # GloVe 기반 유사도
            glove_similarity = sum(calculate_glove_similarity(word_i, word_j) for word_i in words_i for word_j in words_j)
            
            # 가중치를 적용한 유사도
            similarity = gamma * co_occurrence_similarity + (1 - gamma) * glove_similarity
            similarity_matrix[i][j] = similarity

    # TextRank 알고리즘 적용
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # 상위 순위 문장에서 키워드 추출
    keywords = []
    for score, sentence in ranked_sentences[:top_n]:
        keywords.extend(word_tokenize(sentence.lower()))
    
    return list(set(keywords))

# 키워드 개수 계산 함수
def get_keyword_count(keywords):
    return len(keywords.split())

# TP-CoGlo-TextRank로 키워드 추출
df_filtered_04['extracted_keywords'] = df_filtered_04.apply(
    lambda row: tp_coglo_textrank_keywords(
        row['title'], 
        row['abstract'], 
        top_n=get_keyword_count(row['keywords']),
        beta=0.5,
        gamma=0.2,  # GloVe 비율
        lambda_param=0.3  # 단어 빈도와 위치 가중치
    ) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 결과 출력
print(df_filtered_04[['abstract', 'keywords', 'extracted_keywords']])

# Metrics 적용 (apply_metrics 함수가 이미 정의되어 있다고 가정)
time.sleep(3)
df_filtered_04 = apply_metrics(df_filtered_04)

end_time = time.time()
total_time = end_time - start_time
df_filtered_04['time'] = total_time

# Final result output
df_result04 = df_filtered_04[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result04 = calculate_means(df_result04)


# M05 textrank + Watts-Strogatz model
start_time = time.time()

time.sleep(3)
df_filtered_05 = df_filtered.copy()
df_filtered_05.dtypes

# 클러스터링 계수 계산 함수
def calculate_clustering_coefficient(graph, node):
    neighbors = list(nx.neighbors(graph, node))
    if len(neighbors) < 2:
        return 0.0
    subgraph = graph.subgraph(neighbors)
    possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
    actual_edges = subgraph.number_of_edges()
    return actual_edges / possible_edges if possible_edges > 0 else 0.0

# 평균 경로 길이 계산 함수
def calculate_average_path_length(graph, node):
    lengths = nx.single_source_shortest_path_length(graph, node)
    return sum(lengths.values()) / len(lengths) if len(lengths) > 1 else 0.0

# 종합 가중치 계산 함수 (Watts-Strogatz 특성 + TF-IDF + 품사 + 위치)
def calculate_ws_weight(graph, node, clustering_coeff, avg_path_length, tfidf_score, pos_weight, loc_weight):
    if avg_path_length > 0:  # Avoid division by zero
        ws_weight = (clustering_coeff / avg_path_length) * pos_weight * loc_weight * tfidf_score
    else:
        ws_weight = clustering_coeff * pos_weight * loc_weight * tfidf_score  # Fallback if avg_path_length is 0
    return ws_weight

# WS-TextRank 키워드 추출 함수
def ws_textrank_keywords(text, top_n=5):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # TF-IDF 계산
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.max(axis=0).toarray()[0]))
    
    # 단어 네트워크 그래프 생성
    graph = nx.Graph()
    for i, sentence in enumerate(sentences):
        words_in_sentence = word_tokenize(sentence.lower())
        for word in words_in_sentence:
            if word not in graph:
                graph.add_node(word)
        for j, word in enumerate(words_in_sentence):
            for k in range(j + 1, len(words_in_sentence)):
                if not graph.has_edge(word, words_in_sentence[k]):
                    graph.add_edge(word, words_in_sentence[k])
    
    # 각 단어의 클러스터링 계수와 평균 경로 길이 계산
    ws_scores = {}
    for node in graph.nodes:
        clustering_coeff = calculate_clustering_coefficient(graph, node)
        avg_path_length = calculate_average_path_length(graph, node)
        tfidf_score = tfidf_scores.get(node, 0.0)
        pos_weight = 1.0  # 단순화 위해 기본값 사용, 필요 시 품사에 따른 가중치 추가 가능
        loc_weight = 1.0  # 위치 가중치 (필요 시 추가 구현 가능)
        
        # 종합 가중치 계산
        ws_weight = calculate_ws_weight(graph, node, clustering_coeff, avg_path_length, tfidf_score, pos_weight, loc_weight)
        ws_scores[node] = ws_weight

    # WS-TextRank 가중치로 상위 top_n 키워드 추출
    ranked_keywords = sorted(ws_scores.items(), key=lambda item: item[1], reverse=True)
    return [word for word, score in ranked_keywords[:top_n]]

# 각 행의 'keywords'에서 단어 개수를 계산하여 'num_keywords' 열 생성
df_filtered_05['num_keywords'] = df_filtered_05['keywords'].apply(lambda x: len(x.split()))

# 'num_keywords'를 top_n으로 사용하여 'extracted_keywords' 생성
df_filtered_05['extracted_keywords'] = df_filtered_05.apply(
    lambda row: ws_textrank_keywords(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 결과 출력
print(df_filtered_05[['abstract', 'keywords', 'extracted_keywords']])

# 결과를 메트릭에 맞춰 계산
df_filtered_05 = apply_metrics(df_filtered_05)

# 처리 시간 계산
end_time = time.time()
total_time = end_time - start_time
df_filtered_05['time'] = total_time

# 최종 결과 출력
df_result05 = df_filtered_05[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result05 = calculate_means(df_result05)



#### M06 textrank + Infomap
start_time = time.time()

time.sleep(3)
df_filtered_06 = df_filtered.copy()
df_filtered_06.dtypes

# Infomap 모듈화 적용 함수
def apply_infomap(graph):
    # Infomap 모듈화 과정 구현 (네트워크에서 모듈 구조를 식별)
    communities = nx.community.greedy_modularity_communities(graph)
    return communities

# Textrank + Infomap 키워드 추출 함수
def textrank_infomap_keywords(text, top_n=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()
    
    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap 알고리즘을 이용한 모듈화 적용
    communities = apply_infomap(nx_graph)
    
    # 커뮤니티 내에서 TextRank 알고리즘 적용
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences:
        keywords.extend(word_tokenize(sentence.lower()))
        if len(set(keywords)) >= top_n:
            break
    
    return list(set(keywords))[:top_n]

# 각 행의 'keywords'에서 단어 개수를 계산하여 'num_keywords' 열 생성
df_filtered_06['num_keywords'] = df_filtered_06['keywords'].apply(lambda x: len(x.split()))

# 'num_keywords'를 top_n으로 사용하여 'extracted_keywords' 생성
df_filtered_06['extracted_keywords'] = df_filtered_06.apply(
    lambda row: textrank_infomap_keywords(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_06[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_06 = apply_metrics(df_filtered_06)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_06['time'] = total_time

# 최종 결과 출력
df_result06 = df_filtered_06[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result06 = calculate_means(df_result06)



# M07 textrank + term frequency, term postion, word co-occurence + Infomap
start_time = time.time()

time.sleep(3)
df_filtered_07 = df_filtered.copy()
df_filtered_07.dtypes

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

# Infomap을 사용한 커뮤니티 모듈화 및 Textrank 결합 키워드 추출 함수
def textrank_infomap_keywords(title, abstract, top_n=5, beta=0.5):
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
            weight = 1 if word in words_title else beta  # 제목에 있는 단어일 경우 가중치 1 적용
            if (word, words[j]) in co_occurrence:
                infomap.add_link(word_to_id[word], word_to_id[words[j]], weight)

    # Infomap 실행
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

# 데이터프레임 복사 및 키워드 개수 계산
df_filtered_07 = df_filtered.copy()
df_filtered_07['num_keywords'] = df_filtered_07['keywords'].apply(count_keywords)

# Infomap을 사용하여 키워드를 추출하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_07['extracted_keywords'] = df_filtered_07.apply(
    lambda row: textrank_infomap_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [], axis=1)

# 결과 출력
print(df_filtered_07[['abstract', 'keywords', 'extracted_keywords']])

# 메트릭 계산 (apply_metrics가 이미 정의되어 있다고 가정)
df_filtered_07 = apply_metrics(df_filtered_07)

# 처리 시간 기록
end_time = time.time()
total_time = end_time - start_time
df_filtered_07['time'] = total_time

# 최종 결과 출력
df_result07 = df_filtered_07[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result07 = calculate_means(df_result07)


# M08 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges + Infomap
start_time = time.time()

time.sleep(3)
df_filtered_08 = df_filtered.copy()
df_filtered_08.dtypes

# Double Negation, Mitigation, Hedges 가중치 적용 함수
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
            weight += distance / len(words)  # 여러 부정 표현일수록 가중치 증가

        # Mitigation 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'to some extent', 'moderately', 'fairly', 'in part', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 표현 발견 시 가중치 증가

        # Hedges 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate', 'presumably', 'arguably']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # Hedges 표현 발견 시 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

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

# Infomap을 사용한 커뮤니티 모듈화 및 Textrank 결합 키워드 추출 함수
def textrank_infomap_keywords(title, abstract, top_n=5, beta=0.5):
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

    # Double Negation, Mitigation, Hedges 가중치 적용
    weighted_sentences = apply_weights(title + ' ' + abstract)
    
    # 단어를 정수로 매핑
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # Infomap 알고리즘 초기화
    infomap = Infomap()

    # 노드와 엣지를 Infomap 구조에 추가
    for i, word in enumerate(words):
        for j in range(i + 1, len(words)):
            weight = 1 if word in words_title else beta  # 제목에 있는 단어일 경우 가중치 1 적용
            if (word, words[j]) in co_occurrence:
                infomap.add_link(word_to_id[word], word_to_id[words[j]], weight)

    # Infomap 실행
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

# 데이터프레임 복사 및 키워드 개수 계산
df_filtered_08 = df_filtered.copy()
df_filtered_08['num_keywords'] = df_filtered_08['keywords'].apply(count_keywords)

# Infomap을 사용하여 키워드를 추출하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_08['extracted_keywords'] = df_filtered_08.apply(
    lambda row: textrank_infomap_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [], axis=1)

# 결과 출력
print(df_filtered_08[['abstract', 'keywords', 'extracted_keywords']])

# 메트릭 계산 (apply_metrics가 이미 정의되어 있다고 가정)
df_filtered_08 = apply_metrics(df_filtered_08)

# 처리 시간 기록
end_time = time.time()
total_time = end_time - start_time
df_filtered_08['time'] = total_time

# 최종 결과 출력
df_result08 = df_filtered_08[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result08 = calculate_means(df_result08)



#### M09 textrank + Infomap + 2-layer
start_time = time.time()

time.sleep(3)
df_filtered_09 = df_filtered.copy()
df_filtered_09.dtypes

# 공출현 그래프 없이 단순 유사도 기반 네트워크 생성 함수
def create_simple_graph(sentences):
    graph = nx.Graph()
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    for word_list in words:
        for i, word in enumerate(word_list):
            for j in range(i + 1, len(word_list)):
                if graph.has_edge(word, word_list[j]):
                    graph[word][word_list[j]]['weight'] += 1
                else:
                    graph.add_edge(word, word_list[j], weight=1)
    return graph

# 단순 Infomap 기반 키워드 추출 함수 (TF, Term Position, Co-occurrence 배제)
def simplified_infomap_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)

    # 단순 유사도 기반 그래프 생성
    graph = create_simple_graph(sentences)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # Infomap 알고리즘 적용
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
        node_score = sum([graph[node][nbr]['weight'] for nbr in graph.neighbors(node)])

        if module_id in module_scores:
            module_scores[module_id].append((node, node_score))
        else:
            module_scores[module_id] = [(node, node_score)]

    # 각 모듈에서 상위 top_n 노드를 추출
    keywords = []
    for module_id, nodes in module_scores.items():
        sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
        top_keywords = [node for node, score in sorted_nodes[:top_n]]
        keywords.extend(top_keywords)

    return list(set(keywords))

# 각 행의 'keywords'에서 단어 개수를 계산하는 함수
df_filtered_09['num_keywords'] = df_filtered_09['keywords'].apply(lambda x: len(x.split()))

# 추출된 키워드를 데이터 프레임에 추가하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_09['extracted_keywords'] = df_filtered_09.apply(
    lambda row: simplified_infomap_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1)

# 필요에 따라 'num_keywords' 열을 제거할 수 있습니다
df_filtered_09.drop(columns=['num_keywords'], inplace=True)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_09[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_09 = apply_metrics(df_filtered_09)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_09['time'] = total_time

# 최종 결과 출력
df_result09 = df_filtered_09[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result09 = calculate_means(df_result09)


# M10 textrank + term frequency, term position, word co-occurrence + Infomap + 2-layer
start_time = time.time()

time.sleep(3)
df_filtered_10 = df_filtered.copy()

# Term Frequency 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    doc_length = len(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 그래프 생성 함수 (Term Frequency 반영)
def create_co_occurrence_graph(sentences, tf, window_size=2):
    graph = nx.Graph()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                weight = tf[word] * tf[words[j]]  # TF 값을 가중치로 반영
                if graph.has_edge(word, words[j]):
                    graph[word][words[j]]['weight'] += weight
                else:
                    graph.add_edge(word, words[j], weight=weight)
    return graph

# 확장된 Infomap 기반 키워드 추출 함수 (Term Position 반영)
def hierarchical_infomap_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    tf = calculate_tf(text)  # TF 계산

    # 공출현 그래프 생성 (TF 반영)
    graph = create_co_occurrence_graph(sentences, tf)

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

    # 모듈별로 단어 점수를 계산 (Term Position 반영)
    module_scores = {}
    for node, module_id in module_assignments.items():
        # node가 제목에 있으면 가중치 1, 초록에 있으면 가중치 beta
        if node in word_tokenize(title.lower()):
            node_score = sum([graph[node][nbr]['weight'] * 1 for nbr in graph.neighbors(node)])
        else:
            node_score = sum([graph[node][nbr]['weight'] * beta for nbr in graph.neighbors(node)])  # Term Position 반영

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
df_filtered_10['num_keywords'] = df_filtered_10['keywords'].apply(lambda x: len(x.split()))

# 추출된 키워드를 데이터 프레임에 추가하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_10['extracted_keywords'] = df_filtered_10.apply(
    lambda row: hierarchical_infomap_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5) if pd.notnull(row['abstract']) else [],
    axis=1)

# 필요에 따라 'num_keywords' 열을 제거할 수 있습니다
df_filtered_10.drop(columns=['num_keywords'], inplace=True)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_10[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_10 = apply_metrics(df_filtered_10)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_10['time'] = total_time

# 최종 결과 출력
df_result10 = df_filtered_10[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result10 = calculate_means(df_result10)



# M11 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + 2-layer
start_time = time.time()

time.sleep(3)
df_filtered_11 = df_filtered.copy()

# Double Negation, Mitigation, and Hedges 가중치 적용 함수
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 복수의 부정이 있는 경우 가중치 증가

        # Mitigation 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'moderately', 'fairly', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 단어가 있는 경우 가중치 증가

        # Hedges 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # Hedge 단어가 있는 경우 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# Term Frequency 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    doc_length = len(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 그래프 생성 함수 (TF 반영)
def create_co_occurrence_graph(sentences, tf, window_size=2):
    graph = nx.Graph()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                weight = tf[word] * tf[words[j]]  # TF 값으로 가중치 반영
                if graph.has_edge(word, words[j]):
                    graph[word][words[j]]['weight'] += weight
                else:
                    graph.add_edge(word, words[j], weight=weight)
    return graph

# 확장된 Infomap 기반 키워드 추출 함수 (Double Negation, Mitigation, Hedges Weighting 포함)
def hierarchical_infomap_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    sentences = [s for s, w in weighted_sentences]  # 문장 추출
    tf = calculate_tf(text)  # TF 계산

    # 공출현 그래프 생성 (TF 반영)
    graph = create_co_occurrence_graph(sentences, tf)

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

    # 모듈별로 단어 점수를 계산 (가중치 반영)
    module_scores = {}
    for node, module_id in module_assignments.items():
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
df_filtered_11['num_keywords'] = df_filtered_11['keywords'].apply(lambda x: len(x.split()))

# 추출된 키워드를 데이터 프레임에 추가하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_11['extracted_keywords'] = df_filtered_11.apply(
    lambda row: hierarchical_infomap_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5) if pd.notnull(row['abstract']) else [],
    axis=1)

# 필요에 따라 'num_keywords' 열을 제거할 수 있습니다
df_filtered_11.drop(columns=['num_keywords'], inplace=True)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_11[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_11 = apply_metrics(df_filtered_11)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_11['time'] = total_time

# 최종 결과 출력
df_result11 = df_filtered_11[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result11 = calculate_means(df_result11)



# M12 textrank + Infomap + 3-layer
start_time = time.time()

time.sleep(3)
df_filtered_12 = df_filtered.copy()
df_filtered_12.dtypes

# 2계층 구조에서 Infomap 적용 함수 (재귀적 계층 탐지)
def hierarchical_infomap_2layer_textrank_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    
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
    
    # Infomap 모듈화 1계층 적용
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    infomap.run()

    # 1계층 모듈에서 키워드 추출
    keywords = []
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            module_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(module_keywords)
    
    # 2계층 구조 (하위 모듈)로 확장하여 추가 키워드 추출
    for module in infomap.tree:
        if not module.isLeaf:
            sub_infomap = Infomap()
            sub_sentences = [sentences[node.node_id] for node in module.children if node.isLeaf]
            
            # 하위 모듈에서 유사도 행렬 생성
            sub_similarity_matrix = np.zeros((len(sub_sentences), len(sub_sentences)))
            for i, sentence_i in enumerate(sub_sentences):
                words_i = word_tokenize(sentence_i.lower())
                for j, sentence_j in enumerate(sub_sentences):
                    if i == j:
                        continue
                    words_j = word_tokenize(sentence_j.lower())
                    sub_similarity = len(set(words_i) & set(words_j))
                    sub_similarity_matrix[i][j] = sub_similarity
            
            # 하위 모듈 Infomap 적용
            sub_graph = nx.from_numpy_array(sub_similarity_matrix)
            for i in range(len(sub_sentences)):
                sub_infomap.add_node(i)
            for i in range(len(sub_sentences)):
                for j in range(i + 1, len(sub_sentences)):
                    if sub_similarity_matrix[i][j] > 0:
                        sub_infomap.add_link(i, j, sub_similarity_matrix[i][j])
            sub_infomap.run()

            # 2계층의 하위 모듈에서 키워드 추출
            for sub_node in sub_infomap.tree:
                if sub_node.isLeaf:
                    sub_sentence_idx = sub_node.node_id
                    sub_words = word_tokenize(sub_sentences[sub_sentence_idx].lower())
                    sub_word_freq = Counter(sub_words)
                    sub_module_keywords = [word for word, _ in sub_word_freq.most_common(top_n)]
                    keywords.extend(sub_module_keywords)

    return list(set(keywords))

# DataFrame에 'keywords'의 단어 개수를 계산하여 'num_keywords' 열 추가
df_filtered_12['num_keywords'] = df_filtered_12['keywords'].apply(lambda x: len(x.split()))

# 계층적 Infomap 2계층 구조를 사용하여 키워드를 추출하면서 각 행의 'num_keywords'를 top_n으로 지정
df_filtered_12['extracted_keywords'] = df_filtered_12.apply(
    lambda row: hierarchical_infomap_2layer_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력
print(df_filtered_12[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_12 = apply_metrics(df_filtered_12)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_12['time'] = total_time

# 최종 결과 출력
df_result12 = df_filtered_12[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result12 = calculate_means(df_result12)



# M13 textrank + term frequency, term postion, word co-occurence + Infomap + 3-layer
start_time = time.time()

time.sleep(3)
df_filtered_13 = df_filtered.copy()

# Term Frequency 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    doc_length = len(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 그래프 생성 함수 (TF 및 Term Position 반영)
def create_co_occurrence_graph(sentences, tf, window_size=2):
    graph = nx.Graph()
    for sentence_idx, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                weight = tf[word] * tf[words[j]]  # TF 값을 가중치로 반영
                position_weight = 1 / (sentence_idx + 1)  # Term position 반영 (앞에 나올수록 가중치 큼)
                final_weight = weight * position_weight  # 최종 가중치
                
                if graph.has_edge(word, words[j]):
                    graph[word][words[j]]['weight'] += final_weight
                else:
                    graph.add_edge(word, words[j], weight=final_weight)
    return graph

# 2계층 Infomap 기반 키워드 추출 함수 (TF, Term Position, Co-occurrence 반영)
def hierarchical_infomap_2layer_keywords(title, abstract, top_n=5):
    text = title + ' ' + abstract
    sentences = sent_tokenize(text)
    tf = calculate_tf(text)  # TF 계산

    # 공출현 그래프 생성 (TF 및 Term Position 반영)
    graph = create_co_occurrence_graph(sentences, tf)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # 1계층 Infomap 적용
    infomap = Infomap()

    for edge in graph.edges(data=True):
        node1 = word_to_id[edge[0]]
        node2 = word_to_id[edge[1]]
        weight = edge[2]['weight']
        infomap.addLink(node1, node2, weight)

    infomap.run()

    # 1계층 모듈에서 키워드 추출
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            node_name = id_to_word[node.node_id]
            module_assignments[node_name] = node.moduleIndex()

    # 2계층 구조로 확장하여 추가 키워드 추출
    hierarchical_keywords = []
    for module_id in set(module_assignments.values()):
        module_nodes = [node for node, mod_id in module_assignments.items() if mod_id == module_id]

        # 하위 모듈에서 유사도 행렬 생성
        sub_similarity_matrix = np.zeros((len(module_nodes), len(module_nodes)))
        for i, node_i in enumerate(module_nodes):
            words_i = word_tokenize(node_i.lower())
            for j, node_j in enumerate(module_nodes):
                if i == j:
                    continue
                words_j = word_tokenize(node_j.lower())
                sub_similarity = len(set(words_i) & set(words_j))
                sub_similarity_matrix[i][j] = sub_similarity

        # 2계층 Infomap 적용
        sub_graph = nx.from_numpy_array(sub_similarity_matrix)
        sub_infomap = Infomap()

        for i in range(len(module_nodes)):
            sub_infomap.addNode(i)
        for i in range(len(module_nodes)):
            for j in range(i + 1, len(module_nodes)):
                if sub_similarity_matrix[i][j] > 0:
                    sub_infomap.addLink(i, j, sub_similarity_matrix[i][j])

        sub_infomap.run()

        # 2계층에서 키워드 추출
        sub_module_assignments = {}
        for sub_node in sub_infomap.tree:
            if sub_node.isLeaf:
                sub_node_name = id_to_word[sub_node.node_id]
                sub_module_assignments[sub_node_name] = sub_node.moduleIndex()

        # 모듈별로 단어 점수를 계산
        module_scores = {}
        for node, sub_module_id in sub_module_assignments.items():
            node_score = sum([graph[node][nbr]['weight'] for nbr in graph.neighbors(node)])

            if sub_module_id in module_scores:
                module_scores[sub_module_id].append((node, node_score))
            else:
                module_scores[sub_module_id] = [(node, node_score)]

        # 각 모듈에서 상위 top_n 단어 추출
        for sub_module_id, nodes in module_scores.items():
            sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
            top_keywords = [node for node, score in sorted_nodes[:top_n]]
            hierarchical_keywords.extend(top_keywords)

    return list(set(hierarchical_keywords))

# 각 행의 'keywords'에서 단어 개수를 계산하는 함수
df_filtered_13['num_keywords'] = df_filtered_13['keywords'].apply(lambda x: len(x.split()))

# 계층적 Infomap 2계층 구조를 사용하여 키워드를 추출하면서 각 행의 'num_keywords'를 top_n으로 지정
df_filtered_13['extracted_keywords'] = df_filtered_13.apply(
    lambda row: hierarchical_infomap_2layer_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_13[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_13 = apply_metrics(df_filtered_13)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_13['time'] = total_time

# 최종 결과 출력
df_result13 = df_filtered_13[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result13 = calculate_means(df_result13)



# M14 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + 3-layer
start_time = time.time()

time.sleep(3)
df_filtered_14 = df_filtered.copy()
df_filtered_14.dtypes

# Double Negation, Mitigation, and Hedges 가중치 적용 함수
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # 기본 가중치

        # Double Negation 가중치 적용
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # 복수의 부정이 있는 경우 가중치 증가

        # Mitigation 가중치 적용
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'moderately', 'fairly', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # 완화 단어가 있는 경우 가중치 증가

        # Hedges 가중치 적용
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'appear', 'likely', 'suggest', 'indicate']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # Hedge 단어가 있는 경우 가중치 증가

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# Term Frequency 계산 함수
def calculate_tf(text):
    words = word_tokenize(text.lower())
    word_counts = Counter(words)
    doc_length = len(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# 공출현 그래프 생성 함수 (TF 반영)
def create_co_occurrence_graph(sentences, tf, window_size=2):
    graph = nx.Graph()
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + 1 + window_size, len(words))):
                weight = tf[word] * tf[words[j]]  # TF 값으로 가중치 반영
                if graph.has_edge(word, words[j]):
                    graph[word][words[j]]['weight'] += weight
                else:
                    graph.add_edge(word, words[j], weight=weight)
    return graph

# 2계층 Infomap 기반 키워드 추출 함수 (Double Negation, Mitigation, Hedges Weighting 포함)
def hierarchical_infomap_2layer_keywords(title, abstract, top_n=5, beta=0.5):
    text = title + ' ' + abstract
    weighted_sentences = apply_weights(text)
    sentences = [s for s, w in weighted_sentences]  # 문장 추출
    tf = calculate_tf(text)  # TF 계산

    # 공출현 그래프 생성 (TF 반영)
    graph = create_co_occurrence_graph(sentences, tf)

    # 노드를 정수로 매핑
    word_to_id = {word: idx for idx, word in enumerate(graph.nodes())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # 1계층 Infomap 적용
    infomap = Infomap()

    for edge in graph.edges(data=True):
        node1 = word_to_id[edge[0]]
        node2 = word_to_id[edge[1]]
        weight = edge[2]['weight']
        infomap.addLink(node1, node2, weight)

    infomap.run()

    # 1계층 모듈에서 키워드 추출
    module_assignments = {}
    for node in infomap.tree:
        if node.isLeaf:
            node_name = id_to_word[node.node_id]
            module_assignments[node_name] = node.moduleIndex()

    # 2계층 구조로 확장하여 추가 키워드 추출
    hierarchical_keywords = []
    for module_id in set(module_assignments.values()):
        module_nodes = [node for node, mod_id in module_assignments.items() if mod_id == module_id]

        # 하위 모듈에서 유사도 행렬 생성
        sub_similarity_matrix = np.zeros((len(module_nodes), len(module_nodes)))
        for i, node_i in enumerate(module_nodes):
            words_i = word_tokenize(node_i.lower())
            for j, node_j in enumerate(module_nodes):
                if i == j:
                    continue
                words_j = word_tokenize(node_j.lower())
                sub_similarity = len(set(words_i) & set(words_j))
                sub_similarity_matrix[i][j] = sub_similarity

        # 2계층 Infomap 적용
        sub_graph = nx.from_numpy_array(sub_similarity_matrix)
        sub_infomap = Infomap()

        for i in range(len(module_nodes)):
            sub_infomap.addNode(i)
        for i in range(len(module_nodes)):
            for j in range(i + 1, len(module_nodes)):
                if sub_similarity_matrix[i][j] > 0:
                    sub_infomap.addLink(i, j, sub_similarity_matrix[i][j])

        sub_infomap.run()

        # 2계층에서 키워드 추출
        sub_module_assignments = {}
        for sub_node in sub_infomap.tree:
            if sub_node.isLeaf:
                sub_node_name = id_to_word[sub_node.node_id]
                sub_module_assignments[sub_node_name] = sub_node.moduleIndex()

        # 모듈별로 단어 점수를 계산
        module_scores = {}
        for node, sub_module_id in sub_module_assignments.items():
            node_score = sum([graph[node][nbr]['weight'] for nbr in graph.neighbors(node)])

            if sub_module_id in module_scores:
                module_scores[sub_module_id].append((node, node_score))
            else:
                module_scores[sub_module_id] = [(node, node_score)]

        # 각 모듈에서 상위 top_n 단어 추출
        for sub_module_id, nodes in module_scores.items():
            sorted_nodes = sorted(nodes, key=lambda item: item[1], reverse=True)
            top_keywords = [node for node, score in sorted_nodes[:top_n]]
            hierarchical_keywords.extend(top_keywords)

    return list(set(hierarchical_keywords))

# 각 행의 'keywords'에서 단어 개수를 계산하는 함수
df_filtered_14['num_keywords'] = df_filtered_14['keywords'].apply(lambda x: len(x.split()))

# 추출된 키워드를 데이터 프레임에 추가하면서 각 행의 keywords 개수를 top_n으로 지정
df_filtered_14['extracted_keywords'] = df_filtered_14.apply(
    lambda row: hierarchical_infomap_2layer_keywords(row['title'], row['abstract'], top_n=row['num_keywords'], beta=0.5) if pd.notnull(row['abstract']) else [],
    axis=1)

# 필요에 따라 'num_keywords' 열을 제거할 수 있습니다
df_filtered_14.drop(columns=['num_keywords'], inplace=True)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_14[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_14 = apply_metrics(df_filtered_14)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_14['time'] = total_time

# 최종 결과 출력
df_result14 = df_filtered_14[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result14 = calculate_means(df_result14)


#### M15 textrank + Infomap + jaccard
##  Jaccard 유사도는 단어의 빈도와 위치를 반영하지 않음
# Jaccard 유사도는 단순히 두 집합에서 단어의 존재 여부만을 고려하여 교집합과 합집합을 계산합니다.
# 이 때문에, 단어가 텍스트에 몇 번 등장했는지 또는 단어가 문장의 어느 위치에 있는지는 Jaccard 유사도에 반영되지 않습니다.
# Glove 임베딩, 단어 빈도 (TF), 단어 위치 (Position), 공출현(Word Co-occurrence) 등과는 달리, Jaccard 유사도는 단어 간의 의미적 유사성이나 중요도를 고려하지 않기 때문에 결과에 큰 차이를 만들지 못합니다.
start_time = time.time()

time.sleep(3)
df_filtered_15 = df_filtered.copy()
df_filtered_15.dtypes

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

# Add 'num_keywords' for setting top_n dynamically
df_filtered_15['num_keywords'] = df_filtered_15['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using Infomap and Jaccard similarity
df_filtered_15['extracted_keywords'] = df_filtered_15.apply(
    lambda row: infomap_textrank_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 필요에 따라 'num_keywords' 열을 제거할 수 있습니다
df_filtered_15.drop(columns=['num_keywords'], inplace=True)

# 데이터 프레임 출력 (처음 5행)
print(df_filtered_15[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_15 = apply_metrics(df_filtered_15)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_15['time'] = total_time

# 최종 결과 출력
df_result15 = df_filtered_15[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result15 = calculate_means(df_result15)



# M16 textrank + Infomap + NetMRF
start_time = time.time()

time.sleep(3)
df_filtered_16 = df_filtered.copy()
df_filtered_16.dtypes

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

# Textrank + Infomap 키워드 추출 함수 (Jaccard similarity 적용)
def textrank_infomap_jaccard_keywords(text, top_n=5):
    sentences = sent_tokenize(text)
    words_list = [preprocess_text(sent) for sent in sentences]
    
    # Jaccard 유사도를 이용해 유사도 행렬 계산
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = jaccard_similarity(words_list[i], words_list[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # Infomap 알고리즘을 이용한 모듈화 적용
    communities = nx.community.greedy_modularity_communities(nx_graph)
    
    # 커뮤니티 내에서 TextRank 알고리즘 적용
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    keywords = []
    for score, sentence in ranked_sentences:
        keywords.extend(word_tokenize(sentence.lower()))
        if len(set(keywords)) >= top_n:
            break
    
    return list(set(keywords))[:top_n]

# 각 행의 'keywords'에서 단어 개수를 계산하여 'num_keywords' 열 생성
df_filtered_16['num_keywords'] = df_filtered_16['keywords'].apply(lambda x: len(x.split()))

# 'num_keywords'를 top_n으로 사용하여 'extracted_keywords' 생성
df_filtered_16['extracted_keywords'] = df_filtered_16.apply(
    lambda row: textrank_infomap_jaccard_keywords(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# 데이터 프레임 출력 
print(df_filtered_16[['abstract', 'keywords', 'extracted_keywords']])

# 데이터프레임에서 모든 메트릭을 계산하여 최종 결과 반환
time.sleep(3)
df_filtered_16 = apply_metrics(df_filtered_16)

end_time = time.time()

# 처리 시간 계산 후 time 열에 추가
total_time = end_time - start_time
df_filtered_16['time'] = total_time

# 최종 결과 출력
df_result16 = df_filtered_16[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result16 = calculate_means(df_result16)



# M17 textrank + Infomap + NetMRF
start_time = time.time()

time.sleep(3)
df_filtered_17 = df_filtered.copy()
df_filtered_17.dtypes

# Step 1: NetMRF Energy Function
def netmrf_energy(graph, communities, alpha=0.5):
    energy = 0.0
    for node in graph.nodes():
        node_community = communities[node]
        for neighbor in graph.neighbors(node):
            neighbor_community = communities[neighbor]
            edge_weight = graph[node][neighbor]['weight']

            # Pairwise potential: encourage nodes in the same community to have strong connections
            if node_community == neighbor_community:
                energy -= alpha * edge_weight  # Decrease energy for strong intra-community edges
            else:
                energy += (1 - alpha) * edge_weight  # Increase energy for inter-community edges
    
    return energy

# Step 2: Apply NetMRF Optimization
def apply_netmrf_optimization(graph):
    # Step 2.1: Initialize random communities
    communities = {node: np.random.choice([0, 1]) for node in graph.nodes()}  # 2 communities initially

    # Step 2.2: Iteratively optimize the communities
    for _ in range(10):  # Fixed number of iterations for simplicity
        for node in graph.nodes():
            # Calculate energy for both community assignments (0 and 1)
            current_community = communities[node]
            energy_0 = netmrf_energy(graph, {**communities, node: 0})
            energy_1 = netmrf_energy(graph, {**communities, node: 1})

            # Assign to the community with lower energy
            if energy_0 < energy_1:
                communities[node] = 0
            else:
                communities[node] = 1
    
    return communities

# Step 3: Integrate NetMRF into Infomap + Textrank Pipeline
def textrank_infomap_netmrf_keywords(text, top_n=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()

    # Step 3.1: Build a network graph from similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # Step 3.2: Apply Infomap for initial community detection
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # Step 3.3: Apply NetMRF Optimization
    communities = apply_netmrf_optimization(nx_graph)

    # Step 3.4: Extract keywords from the communities
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)

    return list(set(keywords))

# Step 4: Example usage of the integrated model (M17 with NetMRF optimization)
df_filtered_17 = df_filtered.copy()
df_filtered_17['num_keywords'] = df_filtered_17['keywords'].apply(lambda x: len(x.split()))

# Extract keywords using Textrank + Infomap + NetMRF
df_filtered_17['extracted_keywords'] = df_filtered_17.apply(
    lambda row: textrank_infomap_netmrf_keywords(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# Display the final DataFrame
print(df_filtered_17[['abstract', 'keywords', 'extracted_keywords']])

# Step 5: Apply metrics and calculate time
time.sleep(3)
df_filtered_17 = apply_metrics(df_filtered_17)

end_time = time.time()
total_time = end_time - start_time
df_filtered_17['time'] = total_time

# Output final result
df_result17 = df_filtered_17[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]


means_result17 = calculate_means(df_result17)



# M18 textrank + term frequency, term position, word co-occurrence + Infomap + NetMRF
start_time = time.time()

df_filtered_18 = df_filtered.copy()

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

# NetMRF optimization
def apply_netmrf_optimization(graph):
    # Apply NetMRF optimization to refine communities
    communities = {node: np.random.choice([0, 1]) for node in graph.nodes()}  # Initialize random communities
    # Here you would include the logic to refine the community assignments
    # through NetMRF optimization techniques
    return communities

# Textrank + Infomap + NetMRF 최적화 키워드 추출 함수
def textrank_infomap_netmrf_keywords(title, abstract, top_n=5, beta=0.5):
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

    # 유사도 행렬 계산
    vectorizer = TfidfVectorizer()
    sentences = sent_tokenize(title + ' ' + abstract)
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()

    # 네트워크 그래프 생성
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # Infomap 적용
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])

    infomap.run()

    # NetMRF 최적화 적용
    communities = apply_netmrf_optimization(nx_graph)

    # 모듈에서 상위 키워드 추출
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)

    return list(set(keywords))

# Apply the keyword extraction function to the dataframe
df_filtered_18['num_keywords'] = df_filtered_18['keywords'].apply(lambda x: len(x.split()))
df_filtered_18['extracted_keywords'] = df_filtered_18.apply(
    lambda row: textrank_infomap_netmrf_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [], 
    axis=1
)

# Display the results
print(df_filtered_18[['abstract', 'keywords', 'extracted_keywords']])

# Process and calculate metrics
df_filtered_18 = apply_metrics(df_filtered_18)

end_time = time.time()
df_filtered_18['time'] = end_time - start_time

# Output the final metrics
df_result18 = df_filtered_18[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]

means_result18 = calculate_means(df_result18)



# M19 Textrank + term frequency, term position, word co-occurrence + Double Negation, Mitigation, Hedges + Infomap + NetMRF
start_time = time.time()

time.sleep(3)
df_filtered_19 = df_filtered.copy()
df_filtered_19.dtypes

# Step 1: Apply Double Negation, Mitigation, Hedges Weights
def apply_weights(text):
    sentences = sent_tokenize(text)
    weighted_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        weight = 1.0  # Default weight

        # Double Negation weight
        negation_indices = [i for i, word in enumerate(words) if word in ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'none']]
        if len(negation_indices) > 1:
            distance = negation_indices[-1] - negation_indices[0]
            weight += distance / len(words)  # Higher weight for multiple negations

        # Mitigation weight
        mitigation_words = ['sort of', 'kind of', 'a little', 'rather', 'somewhat', 'partly', 'slightly', 'moderately', 'just']
        for word in words:
            if word in mitigation_words:
                weight += 0.5  # Increase weight for mitigation words

        # Hedges weight
        hedges_words = ['maybe', 'possibly', 'could', 'might', 'perhaps', 'seem', 'likely', 'suggest', 'indicate']
        for word in words:
            if word in hedges_words:
                weight += 0.2  # Increase weight for hedge words

        weighted_sentences.append((sentence, weight))
    
    return weighted_sentences

# Step 2: Term Frequency (TF) Calculation
def calculate_tf(text):
    words = word_tokenize(text.lower())
    doc_length = len(words)
    word_counts = Counter(words)
    tf = {word: count / doc_length for word, count in word_counts.items()}
    return tf

# Step 3: Co-occurrence Calculation
def calculate_co_occurrence(words, window_size=2):
    co_occurrence = Counter()
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
    return co_occurrence

# Step 4: Textrank + Infomap + NetMRF Optimization for Keyword Extraction
def textrank_infomap_netmrf_keywords(title, abstract, top_n=5, beta=0.5):
    if not title or not abstract or title.strip() == '' or abstract.strip() == '':
        return []

    # Preprocessing and applying weights (Double Negation, Mitigation, Hedges)
    weighted_sentences = apply_weights(title + ' ' + abstract)
    words_title = word_tokenize(title.lower())
    words_abstract = word_tokenize(abstract.lower())
    words = words_title + words_abstract

    # Term frequency (TF) and Co-occurrence calculation
    tf = calculate_tf(abstract)
    co_occurrence = calculate_co_occurrence(words)

    # Map words to IDs
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # Create a similarity matrix using TF-IDF
    vectorizer = TfidfVectorizer()
    sentences = sent_tokenize(title + ' ' + abstract)
    X = vectorizer.fit_transform(sentences)
    similarity_matrix = (X * X.T).toarray()

    # Step 4.1: Build a network graph from the similarity matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)

    # Step 4.2: Apply Infomap for initial community detection
    infomap = Infomap()
    for i in range(len(sentences)):
        infomap.add_node(i)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if similarity_matrix[i][j] > 0:
                infomap.add_link(i, j, similarity_matrix[i][j])
    infomap.run()

    # Step 4.3: Apply NetMRF Optimization
    communities = apply_netmrf_optimization(nx_graph)

    # Step 4.4: Extract keywords from communities
    keywords = []
    for node in infomap.tree:
        if node.is_leaf:
            sentence_idx = node.node_id
            words = word_tokenize(sentences[sentence_idx].lower())
            word_freq = Counter(words)
            top_keywords = [word for word, _ in word_freq.most_common(top_n)]
            keywords.extend(top_keywords)

    return list(set(keywords))

# Step 5: NetMRF Optimization (Energy Minimization)
def netmrf_energy(graph, communities, alpha=0.5):
    energy = 0.0
    for node in graph.nodes():
        node_community = communities[node]
        for neighbor in graph.neighbors(node):
            neighbor_community = communities[neighbor]
            edge_weight = graph[node][neighbor]['weight']

            # Pairwise potential: encourage nodes in the same community to have strong connections
            if node_community == neighbor_community:
                energy -= alpha * edge_weight  # Decrease energy for intra-community edges
            else:
                energy += (1 - alpha) * edge_weight  # Increase energy for inter-community edges
    return energy

def apply_netmrf_optimization(graph):
    communities = {node: np.random.choice([0, 1]) for node in graph.nodes()}  # Initialize random communities
    for _ in range(10):  # Fixed number of iterations
        for node in graph.nodes():
            energy_0 = netmrf_energy(graph, {**communities, node: 0})
            energy_1 = netmrf_energy(graph, {**communities, node: 1})
            communities[node] = 0 if energy_0 < energy_1 else 1
    return communities

# Applying the integrated model to the dataset
df_filtered_19['num_keywords'] = df_filtered_19['keywords'].apply(lambda x: len(x.split()))

# Apply Textrank + Infomap + NetMRF optimization for keyword extraction
df_filtered_19['extracted_keywords'] = df_filtered_19.apply(
    lambda row: textrank_infomap_netmrf_keywords(row['title'], row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [], axis=1)

# Display the DataFrame
print(df_filtered_19[['abstract', 'keywords', 'extracted_keywords']])

# Step 6: Apply metrics and calculate processing time
df_filtered_19 = apply_metrics(df_filtered_19)

end_time = time.time()
df_filtered_19['time'] = end_time - start_time

# Output the final results
df_result19 = df_filtered_19[['precision', 'recall', 'f1', 'rouge1', 'FM_Index', 'Bray_Curtis', 'IoU', 'time']]

# Calculate the means
means_result19 = calculate_means(df_result19)




means_result01 = calculate_means(df_result01)
means_result02 = calculate_means(df_result02)
means_result03 = calculate_means(df_result03)
means_result04 = calculate_means(df_result04)
means_result05 = calculate_means(df_result05)
means_result06 = calculate_means(df_result06)
means_result07 = calculate_means(df_result07)
means_result08 = calculate_means(df_result08)
means_result09 = calculate_means(df_result09)
means_result10 = calculate_means(df_result10)
means_result11 = calculate_means(df_result11)
means_result12 = calculate_means(df_result12)
means_result13 = calculate_means(df_result13)
means_result14 = calculate_means(df_result14)
means_result15 = calculate_means(df_result15)
means_result16 = calculate_means(df_result16)
means_result17 = calculate_means(df_result17)
means_result18 = calculate_means(df_result18)
means_result19 = calculate_means(df_result19)

# 평균 결과를 사전으로 변환
means_dict = {
"M01 textrank" : means_result01 ,
"M02 textrank + term frequency, term postion, word co-occurence" : means_result02 ,
"M03 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting " : means_result03 ,
"M04 textrank + TP-CoGlo-TextRank" : means_result04 ,
"M05 textrank + Watts-Strogatz model" : means_result05 ,
"M06 textrank + Infomap" : means_result06 ,
"M07 textrank + term frequency, term postion, word co-occurence + Infomap" : means_result07 ,
"M08 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges + Infomap" : means_result08 ,
"M09 textrank + Infomap + 2-layer" : means_result09 ,
"M10 textrank + term frequency, term postion, word co-occurence + Infomap + 2-layer" : means_result10 ,
"M11 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + 2-layer" : means_result11 ,
"M12 textrank + Infomap + 3-layer" : means_result12 ,
"M13 textrank + term frequency, term postion, word co-occurence + Infomap + 3-layer" : means_result13 ,
"M14 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + 3-layer" : means_result14 ,
"M15 textrank + Infomap + jaccard" : means_result15 ,
"M16 textrank + Infomap + jaccard + 2-layer" : means_result16 ,
"M17 textrank + Infomap + NetMRF" : means_result17 ,
"M18 textrank + term frequency, term postion, word co-occurence + Infomap + NetMRF" : means_result18 ,
"M19 textrank + term frequency, term postion, word co-occurence + Double Negation, Mitigation, and Hedges Weighting + Infomap + NetMRF" : means_result19 
}


# 사전을 DataFrame으로 변환
summary_df = pd.DataFrame(means_dict)

# 전치 (Transpose)하여 인덱스가 행으로, 열이 컬럼으로 되게 변환
summary_df = summary_df.T

# summary_df를 CSV 파일로 저장
# summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_100.csv', index=True)
summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_500.csv', index=True)
# summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_1000.csv', index=True)
# summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_2000.csv', index=True)
#summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_5000.csv', index=True)
# summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_10000.csv', index=True)
#summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_15000.csv', index=True)
# summary_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\summary_df_result19_20000.csv', index=True)
