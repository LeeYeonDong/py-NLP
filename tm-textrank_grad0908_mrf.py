import numpy as np
import pandas as pd
import networkx as nx
from infomap import Infomap
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import normalize
from rouge_score import rouge_scorer

# CSV 파일 불러오기
df_filtered_infomap_mrf = df_filtered.copy()
df_filtered_infomap_mrf.dtypes

# 공출현 계산 함수 (윈도우 크기 증가)
def calculate_co_occurrence(words, window_size=5):  # 윈도우 크기를 5로 증가
    co_occurrence = Counter()
    for i, word in enumerate(words):
        for j in range(i + 1, min(i + 1 + window_size, len(words))):
            co_occurrence[(word, words[j])] += 1
            co_occurrence[(words[j], word)] += 1
    return co_occurrence

# MRF 초기화 (Markov Random Field)
def initialize_mrf(similarity_matrix):
    normalized_sim_matrix = normalize(similarity_matrix, axis=1, norm='l1')
    return normalized_sim_matrix

# MRF 최적화 (Gibbs Sampling 방식 사용)
def apply_mrf(similarity_matrix, num_iterations=2000):  # 반복 횟수 증가
    n = similarity_matrix.shape[0]
    states = np.random.choice([0, 1], size=n)  # 문장 상태 초기화 (0: 선택되지 않음, 1: 선택됨)
    
    for _ in range(num_iterations):
        for i in range(n):
            neighbors = similarity_matrix[i]
            prob_select = np.dot(neighbors, states)  # 이웃의 상태에 따른 확률 계산
            # 임계값을 더 높은 값으로 설정하여 중요한 문장만 선택되도록 조정
            if prob_select > 0.1:  
                states[i] = 1
            else:
                states[i] = 0
    return states

# TF-IDF를 적용하여 단어의 중요도를 계산하는 함수
def calculate_tfidf(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

# Infomap을 사용하여 MRF 최적화 후 중요한 단어 추출
def infomap_mrf_keywords_extraction(text, corpus, top_n=5):
    if not text or text.strip() == '':
        return []
    
    # 텍스트를 단어로 분리
    words = word_tokenize(text.lower())
    
    # 공출현 계산
    co_occurrence = calculate_co_occurrence(words)
    
    # 단어를 정수로 매핑
    word_to_id = {word: i for i, word in enumerate(set(words))}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    # 공출현을 기반으로 유사도 행렬 생성
    similarity_matrix = np.zeros((len(word_to_id), len(word_to_id)))
    for (word1, word2), weight in co_occurrence.items():
        similarity_matrix[word_to_id[word1]][word_to_id[word2]] = weight
    
    # MRF 초기화 및 최적화 적용
    mrf_probabilities = initialize_mrf(similarity_matrix)
    optimized_states = apply_mrf(mrf_probabilities)
    
    # 최적화된 상태에 따라 선택된 단어들만 추출
    selected_words = [id_to_word[i] for i, state in enumerate(optimized_states) if state == 1]
    
    # 전체 코퍼스에서 TF-IDF 계산
    tfidf_matrix, feature_names = calculate_tfidf(corpus)
    
    # TF-IDF 스코어 반영하여 중요한 단어 선택
    tfidf_scores = {feature_names[i]: tfidf_matrix[0, i] for i in range(len(feature_names))}
    
    # Infomap 알고리즘 초기화
    infomap = Infomap()
    
    # 선택된 단어로 그래프 구성
    for word1 in selected_words:
        for word2 in selected_words:
            if word1 != word2 and co_occurrence[(word1, word2)] > 0:
                infomap.add_link(word_to_id[word1], word_to_id[word2], co_occurrence[(word1, word2)])
    
    # Infomap 실행
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
    
    # 각 모듈에서 TF-IDF 점수에 따라 중요한 단어 선택
    extracted_keywords = []
    for words in module_words.values():
        word_freq = Counter(words)
        most_common_words = word_freq.most_common(top_n)  # 빈도가 높은 상위 top_n개 단어 선택
        sorted_words = sorted(most_common_words, key=lambda x: tfidf_scores.get(x[0], 0), reverse=True)
        extracted_keywords.extend([word for word, _ in sorted_words[:top_n]])
    
    return extracted_keywords

# 각 행의 keywords에서 단어 개수를 계산하는 함수
def count_keywords(keywords):
    return len(keywords.split())

# DataFrame에 적용
df_filtered_infomap_mrf['num_keywords'] = df_filtered_infomap_mrf['keywords'].apply(count_keywords)

# 코퍼스 데이터를 전체적으로 처리하여 TF-IDF 계산에 사용
corpus = df_filtered_infomap_mrf['abstract'].fillna('').tolist()

# MRF 기반 최적화된 문장을 사용하여 키워드를 추출
df_filtered_infomap_mrf['extracted_keywords'] = df_filtered_infomap_mrf.apply(
    lambda row: infomap_mrf_keywords_extraction(row['abstract'], corpus, top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# num_keywords 열 제거
df_filtered_infomap_mrf.drop(columns=['num_keywords'], inplace=True)

# 결과 출력 (처음 5행)
print(df_filtered_infomap_mrf[['abstract', 'keywords', 'extracted_keywords']])

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

# ROUGE 점수 계산 함수
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def calculate_rouge(extracted, actual):
    extracted_text = ' '.join(extracted)
    actual_text = ' '.join(actual)
    scores = scorer.score(actual_text, extracted_text)
    return scores

# Precision, Recall, F1 계산 및 데이터 프레임에 추가
df_filtered_infomap_mrf['metrics'] = df_filtered_infomap_mrf.apply(
    lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split()), axis=1
)

df_filtered_infomap_mrf[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_mrf['metrics'].tolist(), index=df_filtered_infomap_mrf.index)

# ROUGE 점수 계산 및 데이터 프레임에 추가
df_filtered_infomap_mrf['rouge'] = df_filtered_infomap_mrf.apply(
    lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split()), axis=1
)
df_filtered_infomap_mrf['rouge1'] = df_filtered_infomap_mrf['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap_mrf['rougeL'] = df_filtered_infomap_mrf['rouge'].apply(lambda x: x['rougeL'].fmeasure)

# 최종 결과 추출
df_result_infomap_mrf = df_filtered_infomap_mrf[['precision', 'recall', 'f1', 'rouge1', 'rougeL']]

# 결과 출력
print(df_result_infomap_mrf)

# 각 DataFrame의 평균 계산 함수
def calculate_means(df):
    means = df.mean()
    return means

means_result_infomap_mrf = calculate_means(df_result_infomap_mrf)