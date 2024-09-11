import numpy as np
import pandas as pd

# MRF 적용 후 메트릭 평가를 위한 함수
def evaluate_metrics_with_threshold(df, thresholds, num_iterations=1000):
    results = []
    for threshold in thresholds:
        print(f'Calculating for threshold: {threshold:.2f}')
        
        # NetMRF 기반 키워드 추출
        df['extracted_keywords'] = df.apply(
            lambda row: infomap_netmrf_keywords_extraction(
                row['abstract'], top_n=row['num_keywords'], threshold=threshold
            ) if pd.notnull(row['abstract']) else [],
            axis=1
        )
        
        # Precision, Recall, F1 계산
        df['metrics'] = df.apply(
            lambda row: calculate_metrics(row['extracted_keywords'], row['keywords'].split()), axis=1
        )
        df[['precision', 'recall', 'f1']] = pd.DataFrame(df['metrics'].tolist(), index=df.index)
        
        # ROUGE 점수 계산
        df['rouge'] = df.apply(
            lambda row: calculate_rouge(row['extracted_keywords'], row['keywords'].split()), axis=1
        )
        df['rouge1'] = df['rouge'].apply(lambda x: x['rouge1'].fmeasure)
        df['rougeL'] = df['rouge'].apply(lambda x: x['rougeL'].fmeasure)
        
        # 평균 성능 측정
        avg_precision = df['precision'].mean()
        avg_recall = df['recall'].mean()
        avg_f1 = df['f1'].mean()
        avg_rouge1 = df['rouge1'].mean()
        avg_rougeL = df['rougeL'].mean()
        
        # 결과 저장
        results.append({
            'threshold': threshold,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'rouge1': avg_rouge1,
            'rougeL': avg_rougeL
        })
    
    return pd.DataFrame(results)

# 평가할 threshold 범위 설정 (0.0부터 0.5까지 0.01 간격으로)
thresholds = np.arange(0.0, 0.51, 0.01)

# 각 행의 keywords에서 단어 개수를 계산하는 함수
def count_keywords(keywords):
    return len(keywords.split())

# DataFrame에 적용
df_filtered_infomap_NetMRF['num_keywords'] = df_filtered_infomap_NetMRF['keywords'].apply(count_keywords)

# 코퍼스 데이터를 전체적으로 처리하여 사용
corpus = df_filtered_infomap_NetMRF['abstract'].fillna('').tolist()

# Grid Search 실행
results_df = evaluate_metrics_with_threshold(df_filtered_infomap_NetMRF, thresholds)

# 결과 출력
print(results_df)

# 최종 결과를 CSV로 저장 (필요시)
results_df.to_csv('D:\\대학원\\논문\\textrank\\rawdata\\dblp_v14.tar\\grid_search_results.csv', index=False)