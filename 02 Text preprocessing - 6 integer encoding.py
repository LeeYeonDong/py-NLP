# pip install numpy
import numpy as np
# pip install pandas
import pandas as pd
# dplython
import pandas as pd
from dplython import (DplyFrame, X, diamonds, select, sift,
  sample_n, sample_frac, head, arrange, mutate, group_by,
  summarize, DelayFunction)

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords

# loading data [txt data]
f = open("D:/대학원/논문/커뮤니케이션학과/reply_paste.txt","r")
reply_text = f.readline()

# 문장 토큰화
sentences = sent_tokenize(reply_text)
print(sentences)

vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    # 단어 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄인다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거한다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거한다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    preprocessed_sentences.append(result) 
print(preprocessed_sentences)

vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
# 기본 딕셔러니 -> Key1:Value1
# items :쌍
# item[0]은 dict의 key, item[1]은 dict의 value
print(vocab_sorted)

# 정수 부여
word_to_index = {}
i = 0
for (word, freq) in vocab_sorted :
    if freq > 2 : # 빈도수가 작은 단어는 제외.
        i = i + 1
        word_to_index[word] = i
word_to_index

vocab_size = 5

# 인덱스가 5 초과인 단어 제거
words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

# 해당 단어에 대한 인덱스 정보를 삭제
for w in words_frequency:
    del word_to_index[w]
print(word_to_index)

word_to_index['OOV'] = len(word_to_index) + 1
print(word_to_index)

encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴.
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            # 만약 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴.
            encoded_sentence.append(word_to_index["OOV"])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)

## data frame
# sort
vocab_df = pd.DataFrame.from_dict(data = vocab, orient="index")
vocab_df

vocab_df = DplyFrame(vocab_df)
vocab_df.columns = ["Freq"]

vocab_df = vocab_df.sort_values("Freq", ascending = False) >> sift(X.Freq > 2)
vocab_df["word"] = vocab_df.index.to_list()
vocab_df 

# integer encoding
integer = []
for i in range(1,len(vocab_df)+1):
    integer.append(i)
    
vocab_df["encoding"] = integer
vocab_df = vocab_df >> sift(X.encoding <= 5) >> select(X.word,X.encoding)
vocab_df

df2 = pd.DataFrame({'word' : ['OOV'], 'encoding' : ['6']})
vocab_df = pd.concat([vocab_df,df2], ignore_index = False)
vocab_df.set_index("word", inplace = True)
vocab_df

vocab_dict = vocab_df.to_dict("dict")
vocab_dict

vocab_index = vocab_dict["encoding"]
print(vocab_index)

encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴.
            encoded_sentence.append(vocab_index[word])
        except KeyError:
            # 만약 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴.
            encoded_sentence.append(vocab_index["OOV"])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)

# 케라스(Keras)의 텍스트 전처리
from tensorflow.keras.preprocessing.text import Tokenizer
preprocessed_sentences

tokenizer = Tokenizer()
# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences) 
print(tokenizer.word_index)

# texts_to_sequences()는 입력으로 들어온 코퍼스에 대해서 각 단어를 이미 정해진 인덱스로 변환합니다.
print(tokenizer.texts_to_sequences(preprocessed_sentences))
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(preprocessed_sentences) 

print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))

# 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
tokenizer.fit_on_texts(preprocessed_sentences)
print(tokenizer.texts_to_sequences(preprocessed_sentences))