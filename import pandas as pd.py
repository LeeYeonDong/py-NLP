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


df_filtered = df_filtered.astype(str)
df_filtered.dtypes
df_filtered = df_filtered[['id', 'title', 'keywords', 'year', 'abstract', 'authors']]
df_filtered = df_filtered.dropna(subset=['id', 'title', 'keywords', 'year', 'abstract', 'authors'])
