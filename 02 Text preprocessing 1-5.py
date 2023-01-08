# pip install numpy # ctrl + k + c : 주석처리 ctrl + k + u : 주석해제
import numpy as np
# pip install pandas
import pandas as pd
# dplython
import pandas as pd
from dplython import (DplyFrame, X, diamonds, select, sift,
  sample_n, sample_frac, head, arrange, mutate, group_by,
  summarize, DelayFunction)

## loading data
df_ytb1 = pd.read_csv("D:/대학원/논문/커뮤니케이션학과/유튜브.csv", encoding = "utf-8")
df_ytb1 = DplyFrame(df_ytb1)
print(df_ytb1.index)
print(df_ytb1)

# loading packages
from nltk import word_tokenize
from nltk import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# 토큰화
# word_tokenize
df_ytb1["token_ytb"] = df_ytb1.apply(lambda row: word_tokenize(row['댓글_ytb']), axis=1)
df_ytb1["token_ytb"]
댓글_ytb = df_ytb1 >> select(X.댓글_ytb)
댓글_ytb 

# wordPunctTokenizer
df_ytb1.apply(lambda row: WordPunctTokenizer().tokenize(row['댓글_ytb']), axis=1)

# 케라스의 text_to_word_sequence는 기본적으로 모든 알파벳을 소문자로 바꾸면서 마침표나 컴마, 느낌표 등의 구두점을 제거
df_ytb1.apply(lambda row: text_to_word_sequence(row['댓글_ytb']), axis=1)

# Penn Treebank Tokenization
from nltk import TreebankWordTokenizer
TreebankWordTokenizer = TreebankWordTokenizer()
df_ytb1.apply(lambda row: TreebankWordTokenizer.tokenize(row['댓글_ytb']), axis=1)

# 문장 토큰화(Sentence Tokenization)
from nltk.tokenize import sent_tokenize

df_ytb1_댓글_doc = pd.read_csv("D:/대학원/논문/커뮤니케이션학과/df_ytb1_댓글_doc.csv", encoding = "utf-8")
df_ytb1_댓글_doc = DplyFrame(df_ytb1_댓글_doc)
df_ytb1_댓글_doc

df_ytb1_댓글_doc.apply(lambda row: sent_tokenize(row['댓글_doc']), axis=1)

# 문장 토큰화(Sentence Tokenization) - 한국어
# pip install kss
import kss
text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))

# 품사 태깅(Part-of-speech tagging)
# NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습
from nltk import word_tokenize
from nltk import pos_tag
댓글_ytb_tag = df_ytb1.apply(lambda row: pos_tag(word_tokenize(row['댓글_ytb'])), axis=1)
댓글_ytb_tag.to_csv("D:/대학원/논문/커뮤니케이션학과/댓글_ytb_tag.csv", index = False)

from konlpy.tag import Okt
from konlpy.tag import Kkma
okt = Okt()
kkma = Kkma()
print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 

# 표제어 추출(Lemmatization)
댓글_ytb_tag_df = pd.read_csv("D:/대학원/논문/커뮤니케이션학과/댓글_ytb_tag_df.csv", encoding = "utf-8")
댓글_ytb_tag_df

from nltk import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

댓글_ytb_tag_df.loc[:, ["token"]].apply(lambda row: lemmatizer.lemmatize(row['token']), axis=1)

# 어간 추출(Stemming)
from nltk import PorterStemmer
from nltk import word_tokenize
stemmer = PorterStemmer()
댓글_ytb_tag_df.loc[:, ["token"]].apply(lambda row : stemmer.stem(row["token"]), axis = 1)
words = ['formalize', 'allowance', 'electricical']
[stemmer.stem(word) for word in words]

from nltk.stem import LancasterStemmer
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
댓글_ytb_tag_df.loc[:, ["token"]].apply(lambda row : lancaster_stemmer.stem(row["token"]), axis = 1)

# 불용어(Stopword)
from nltk.corpus import stopwords
from nltk import word_tokenize 
from konlpy.tag import Okt
stop_words = stopwords.words("english")
len(stop_words)
stop_words
stop_words.__class__ # class 확인

stop_words_set = set(stopwords.words('english')) # 집합, 중복되지 않는 원소(unique) 얻을 때
stop_words_set.__class__

list(stop_words_set).__class__ # set을 list로 변환
list(stop_words_set)

word_tokens = 댓글_ytb_tag_df.loc[:, ["token"]].apply(lambda row : word_tokenize(row["token"]), axis = 1)
word_tokens

result = []
for word in word_tokens :
    if word not in stop_words :
        result.append(word)
result

stop_words_ko = pd.read_csv("D:/대학원/논문/주제모형/한글_불용어.csv", encoding = "utf-8")
stop_words_ko = list(stop_words_ko)
okt = Okt()
댓글_ytb_tag_df["reply"].__class__
댓글_ytb_tag_df.loc[:, ["reply"]].__class__

댓글_ytb_unique = DplyFrame(댓글_ytb_tag_df["reply"].unique())

댓글_ytb_unique.columns = ["reply"]
댓글_ytb_unique

댓글_ytb_unique = 댓글_ytb_unique.loc[:, ["reply"]].apply(lambda row : word_tokenize(row["reply"]), axis = 1)
댓글_ytb_unique

result = []
for word in 댓글_ytb_unique :
    if word not in stop_words_ko :
        result.append(word)
result