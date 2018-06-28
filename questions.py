import os
import sys

import scipy as sp

from sklearn.feature_extraction.text import CountVectorizer

from pymystem3 import Mystem
mystem = Mystem()

RUSSIAN_STOPWORDS = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 
'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 
'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 
'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 
'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 
'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 
'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 
'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 
'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 
'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 
'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 
'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 
'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 
'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 
'конечно', 'всю', 'между']


# from utils import DATA_DIR

# TOY_DIR = os.path.join(DATA_DIR, "toy")
# questions = [open(os.path.join(TOY_DIR, f)).read() for f in os.listdir(TOY_DIR)]

second_list = open('data_ques.csv', 'r')

questions = []
for line in second_list:
    questions.append(line)

# print(questions)

new_post = input('Введите текст: ')

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')


class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ((''.join(mystem.lemmatize(w))) for w in analyzer(doc))

# vectorizer = CountVectorizer(min_df=1, stop_words='english',
# preprocessor=stemmer)
vectorizer = StemmedCountVectorizer(min_df=1, stop_words=RUSSIAN_STOPWORDS)

from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words='english', decode_error='ignore')

X_train = vectorizer.fit_transform(questions)

num_samples, num_features = X_train.shape
# print("#samples: %d, #features: %d" % (num_samples, num_features))

new_post_vec = vectorizer.transform([new_post])
# print(new_post_vec, type(new_post_vec))
# print(new_post_vec.toarray())
# print(vectorizer.get_feature_names())


def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized

    return sp.linalg.norm(delta.toarray())

dist = dist_norm

best_dist = sys.maxsize
best_i = None

for i in range(0, num_samples):
    post = questions[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist(post_vec, new_post_vec)

    # print("=== Вопрос %i схожесть=%.2f: %s" % (i, d, post))

    if d < best_dist:
        best_dist = d
        best_i = i

print(f'Лучшее совпадение с вопросом {best_i},\nТекст вопроса: "{questions[best_i]}"')