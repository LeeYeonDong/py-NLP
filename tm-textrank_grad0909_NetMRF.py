import numpy as np
from sklearn.preprocessing import normalize

#### 16. textrank + Infomap + NetMRF
df_filtered_infomap_NetMRF = df_filtered.copy()
df_filtered_infomap_NetMRF.dtypes

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
df_filtered_infomap_NetMRF['num_keywords'] = df_filtered_infomap_NetMRF['keywords'].apply(count_keywords)

df_filtered_infomap_NetMRF['extracted_keywords'] = df_filtered_infomap_NetMRF.apply(
    lambda row: infomap_netmrf_keywords_extraction(row['abstract'], top_n=row['num_keywords']) if pd.notnull(row['abstract']) else [],
    axis=1
)

# num_keywords 열 제거
df_filtered_infomap_NetMRF.drop(columns=['num_keywords'], inplace=True)

# 결과 출력 (처음 5행)
print(df_filtered_infomap_NetMRF[['abstract', 'keywords', 'extracted_keywords']])


# Calculate precision, recall, and f1 metrics
df_filtered_infomap_NetMRF['metrics'] = df_filtered_infomap_NetMRF.apply(
    lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split()), axis=1
)
df_filtered_infomap_NetMRF[['precision', 'recall', 'f1']] = pd.DataFrame(df_filtered_infomap_NetMRF['metrics'].tolist(), index=df_filtered_infomap_NetMRF.index)

# Calculate ROUGE scores
df_filtered_infomap_NetMRF['rouge'] = df_filtered_infomap_NetMRF.apply(
    lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split()), axis=1
)
df_filtered_infomap_NetMRF['rouge1'] = df_filtered_infomap_NetMRF['rouge'].apply(lambda x: x['rouge1'].fmeasure)
df_filtered_infomap_NetMRF['rougeL'] = df_filtered_infomap_NetMRF['rouge'].apply(lambda x: x['rougeL'].fmeasure)

# Final DataFrame with results
df_result_infomap_NetMRF = df_filtered_infomap_NetMRF[['precision', 'recall', 'f1', 'rouge1', 'rougeL']]

# Output results
print(df_result_infomap_NetMRF)

# 각 DataFrame의 평균 계산 함수
def calculate_means(df):
    means = df.mean()
    return means

means_result_infomap_NetMRF = calculate_means(df_result_infomap_NetMRF)
means_result_infomap_NetMRF 