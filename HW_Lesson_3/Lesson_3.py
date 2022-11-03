import numpy as np
import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import annoy
from gensim.models import Word2Vec, FastText
import gensim.downloader as api
import pickle
from tqdm import tqdm_notebook
from tqdm.notebook import trange, tqdm
import re


def preprocess_txt(line):
    spls = re.sub(r"https?://[^,\s]+,?", " ", line)
    spls = re.sub(r'\d+\s?', " ", spls).strip()
    spls = re.sub(r'(\b\w+)\s+\1', r'\1', spls)
    spls = re.sub(r'[^а-яА-Яё ]', ' ', spls)
    spls = " ".join(re.sub(html_chars, ' ', spls).split())
    spls = "".join(i if i not in exclude else ' ' for i in spls.strip()).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls


def get_response(question, index, model, index_map):
    question = preprocess_txt(question)
    vector = np.zeros(300)
    norm = 0
    for word in question:
        if word in model.wv:
            vector += model.wv[word]
            norm += 1
    if norm > 0:
        vector = vector / norm
    answers = index.get_nns_by_vector(vector, 3, )
    return [index_map[i] for i in answers]



# print(api.info()['models'].keys())
#
# word_vectors = api.load("glove-wiki-gigaword-100")  # загрузим предтренированные вектора слов из gensim-data
# # выведим слово наиболее близкое к 'woman', 'king' и далекое от 'man'
# result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
# print("{}: {:.4f}".format(*result[0]))
#
# # выведем лишнее слово
# print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
#
# print(word_vectors.doesnt_match("black green summer brown".split()))
#
# # определим схожесть между словами
# similarity = word_vectors.similarity('woman', 'man')
# print(similarity)
#
# similarity = word_vectors.similarity('human', 'man')
# print(similarity)
#
# similarity = word_vectors.similarity('bee', 'man')
# print(similarity)
#
# # найдем top-3 самых близких слов
# result = word_vectors.similar_by_word("man", topn=3)
# print(result)
#
# result = word_vectors.similar_by_word("cat", topn=3)
# print(result)
#
# result = word_vectors.similar_by_word("mouth", topn=3)
# print(result)

# Simple chat-bot example

# counter_all = 0
# counter_filter = 0
# with open("prepared_answers.txt", "r", encoding='utf-8') as f:
#     for line in tqdm_notebook(f):
#         counter_all += 1
#         spls = line.split("\t")
#         if len(spls[0].split()) < 2 or len(spls[1].split()) < 3 or len(spls[0].split()) > 15:
#             continue
#
#         counter_filter += 1
#

assert True

#Small preprocess of the answers

question = None
written = False

# with open("prepared_answers.txt", "w", encoding='utf-8') as fout:
#     with open("J:/Storage_for_ML/Otvety.txt", "r", encoding='utf-8', errors="ignore") as fin:
#         for line in tqdm_notebook(fin):
#             if line.startswith("---") or line == '\n':
#                 written = False
#                 continue
#             if not written and question is not None:
#                 fout.write(question.replace("\t", " ").strip() + "\t" + line.replace("\t", " "))
#                 written = True
#                 question = None
#                 continue
#             if not written:
#                 question = line.strip()
#                 continue


assert True

# Preprocess for models fitting

sentences = []
html_tag = {'</p>', '<br>', '<p>', '</li>', '<li>', '</ul>', '<ul>', '</div>', '<div>',
                 '</h2>', '<h2>', '</h1>', '<h1>', '<h3>', '</h3>', '</h4>', '<h4>', '<h5>', '</h5>',
                 '</body>', '<body>', '</html>', '<html>', '<form>', '</title>', '<title>',
              '</layer>', '<layer>', '</iframe>', '<iframe>', '</form>', '</span>', '<span>',
              '<input>', '</input>', '</comment>', '<comment>', '</textarea>', '<textarea>',
              '</ilayer>', '<ilayer>', '<head>', '</head>'}

html_chars = '|'.join(map(re.escape, html_tag))

removed_chars = {'•', '⁈', '⁉', '․', '‥', '…', '‧', '★', '☆', '☝', '☭', '♁',
                 '♀', '♂', '☹', '☺', '☻', '♠', '♡', '♢', '♣', '♤', '—',
                 '♥', '♦', '♧', '♩', '♪', '♫', '♬', '♛', '♚', '♜', '♞'}

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation).union(removed_chars)
c = 0

with open("J:/Storage_for_ML/Otvety.txt", "r", encoding='utf-8', errors="ignore") as fin:
    for line in tqdm_notebook(fin):
        spls = preprocess_txt(line)
        if spls:
            sentences.append(spls)
            c += 1
        if c > 100000:
            break


sentences = [i for i in sentences if len(i) > 2]

print(sentences[0])

modelW2V = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=1)

modelFT = FastText(sentences=sentences, vector_size=300, min_count=1, window=5, workers=8)

w2v_index = annoy.AnnoyIndex(300, 'angular')
ft_index = annoy.AnnoyIndex(300, 'angular')

index_map = {}
counter = 0

with open("prepared_answers.txt", "r", encoding='utf-8') as f:
    for line in tqdm_notebook(f):
        n_w2v = 0
        n_ft = 0
        spls = line.split("\t")
        index_map[counter] = spls[1]
        question = preprocess_txt(spls[0])

        vector_w2v = np.zeros(300)
        vector_ft = np.zeros(300)
        for word in question:
            if word in modelW2V.wv:
                vector_w2v += modelW2V.wv[word]
                n_w2v += 1
            if word in modelFT.wv:
                vector_ft += modelFT.wv[word]
                n_ft += 1
        if n_w2v > 0:
            vector_w2v = vector_w2v / n_w2v
        if n_ft > 0:
            vector_ft = vector_ft / n_ft
        w2v_index.add_item(counter, vector_w2v)
        ft_index.add_item(counter, vector_ft)

        counter += 1

        if counter > 100000:
            break

w2v_index.build(10)
ft_index.build(10)

TEXT = "какой город самы красивый"

print(get_response(TEXT, w2v_index, modelW2V, index_map))

print(get_response(TEXT, ft_index, modelFT, index_map))

print()

