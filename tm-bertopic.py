# pip install numpy # ctrl + k + c : 주석처리 ctrl + k + u : 주석해제
import numpy as np
# pip install pandas
import pandas as pd
from bertopic import BERTopic

## dynamic topic modeling
import re

# Prepare data
trump = pd.read_csv("D:/대학원/논문/Topic Modeling/trump.csv", encoding = "utf-8")

# preprocessing data
trump.text = trump.apply(lambda row: re.sub(r"http\S+", "", row.text).lower(), 1)
trump.text = trump.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.text.split())), 1)
trump.text = trump.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.text).split()), 1)
trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
timestamps = trump.date.to_list()
tweets = trump.text.to_list()

# BERTopic
topic_model = BERTopic(min_topic_size = 70, n_gram_range = (1,3), verbose = True, calculate_probabilities = True)
topics, probabilities = topic_model.fit_transform(trump["text"].to_list())

# corresponding timestamps, and the related topics
# bins
# topics_over_time = topic_model.topics_over_time(tweets, timestamps, nr_bins = 20)

# Parameters
# Tuning
# topics_over_time = topic_model.topics_over_time(tweets, timestamps, global_tuning=True, evolution_tuning=True, nr_bins=20)
# Datetime format
topics_over_time_date = topic_model.topics_over_time(tweets, timestamps, datetime_format="%Y/%m/%d", nr_bins = 20)

# Visualization 1
topic_model.visualize_topics_over_time(topics_over_time_date, top_n_topics = 20).show()
topic_model.visualize_topics_over_time(topics_over_time_date, topics = [9, 10, 11, 15, 8, 12]).show()

# Visualization 2
topic_model.get_topic_info().head(20) # -1 = outlier

topic_nr = topic_model.get_topic_info().iloc[6]["Topic"] # select a frequent topic
topic_model.get_topic(topic_nr)

topic_model.visualize_topics(top_n_topics = 20).show()

# Visualization 3
topic_model.visualize_topics_over_time(topics_over_time_date, top_n_topics = 10).show()

# Visualization 4
topic_model.visualize_barchart(top_n_topics = 20).show()

# Visualization 5
topic_model.visualize_term_rank().show()

# Visualization 6
topic_model.visualize_hierarchy(top_n_topics = 20).show()

# Visualization 7
topic_model.visualize_heatmap(top_n_topics = 20).show()

# Visualization 8
topic_model.estimate()
topic_model.visualize_distribution(topic_model.probabilities_[0], min_probability = 0.015).show()
topic_model.visualize_distribution(probabilities[0]).show()



## interactive topic modeling
# load data
from sklearn.datasets import fetch_20newsgroups
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

docs.info()

# 두 가지 출력인 토픽(topics)과 확률(probabilities)이 만들어집니다. 토픽의 값은 단순히 할당된 토픽을 나타냅니다. 반면에 확률은 문서가 일어날 수 있는 토픽에 포함될 가능성
model = BERTopic()
topics, probabilities = model.fit_transform(docs)

# 상대적 빈도(relative frequency)로 생성된 토픽에 액세스
model.get_topic_freq().head()

# 두 번째로 빈번하게 생성된 주제, 즉 Topic 49. -1은 outlier
model.get_topic(49)



