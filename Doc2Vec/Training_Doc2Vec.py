# -*- coding: utf-8 -*-

import os
import sys
import MeCab
import collections
from gensim import models
from gensim.models.doc2vec import LabeledSentence

INPUT_DOC_DIR = './Wikipedia/JUMP/'
# gs://doc2vec_takakuwa/

OUTPUT_MODEL = './doc2vec_JUMP.model'
# gs://doc2vec_takakuwa
PASSING_PRECISION = 93

# 全てのファイルのリストを取得
def get_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

# ファイルから文章を返す
def read_document(path):
    # with open(path, 'r', encoding='sjis', errors='ignore') as f: only Python3?
    with open(path, 'r') as f:
        return f.read()

# ファイルから単語のリストを取得
def corpus_to_sentences(corpus):
    docs = [read_document(x) for x in corpus] #Fileから文章を詠みこむ
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        sys.stdout.write('\r前処理中 {} / {}'.format(idx, len(corpus))) #処理が全体のいくつ終わったか表示
        yield split_into_words(doc, name) # 文章から単語に分解して返す関数へfor文の度に送る

# 文章から単語に分解して返す
def split_into_words(doc, name=''): # 例　doc= '''グラウンド からは 野球 部 の 掛け声 が 響い て き て いる。'''
    mecab = MeCab.Tagger("-Ochasen")
    lines = mecab.parse(doc).splitlines() #単語ごとに品詞分解　＋　改行コードで区切りリスト化 // 例）[グラウンド	グラウンド	グラウンド	名詞-一般		, から	カラ	から	助詞-格助詞-一般 /n ... ]
    words = []
    for line in lines:
        chunks = line.split('\t') #空白で区切りリスト化 // 例)　['グラウンド', 'グラウンド', 'グラウンド', '名詞-一般', '', '']
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):  # 動詞・形容詞・名詞(数詞以外)を抽出。chunks[3]は品詞
            words.append(chunks[0])
    return LabeledSentence(words=words, tags=[name])  # 文章を構成する動詞・形容詞・名詞(数詞以外)とその文章のラベル（ファイル名）をタグ付 例)　LabeledSentence(['グラウンド', '野球', '部', '掛け声', '響い', 'き', 'いる'], ['a'])

# 学習
def train(sentences):
    model = models.Doc2Vec(size=400, alpha=0.0015, sample=1e-4, min_count=1, workers=4, )
    model.build_vocab(sentences) #Build vocabulary from a sequence of sentences (can be a once-only generator stream)
    for x in range(Sim_times):
        print(x)
        model.train(sentences ,epochs=model.iter, total_examples=model.corpus_count)
        ranks = []
        # 評価は、学習した文章のうち100個で類似の文章を検索し、最も類似度の高い文章が自分自身だった回数で行う
        for doc_id in range(Test_times):
            inferred_vector = model.infer_vector(sentences[doc_id].words) # Infer a vector for given post-bulk training document.
            # len(model.docvecs)の数（おそらくsentencesの数＋２）だけ、sentences[doc_id].wordsに近いと思われる、学習データ内の文章を、確率が最も高い順に並べたリストを返す　[(ラベル名,確率),...]
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs)) #topnのデフォルトは10
            rank = [docid for docid, sim in sims].index(sentences[doc_id].tags[0]) # simsのラベル名のみのリストを作り　＋　sentences[doc_id].tags[0]とそれが一致するindexを返す　⇨　0の場合、正解を意味する。
            ranks.append(rank)
        if collections.Counter(ranks)[0] >= PASSING_PRECISION: #リストranksから、要素の値が「0」のものの数がPASSING_PRECISION以上の場合
            break
    return model

# size: ベクトル化した際の次元数
# alpha: 学習率
# sample: 単語を無視する際の頻度の閾値 あまりに高い頻度で出現する単語は意味のない単語である可能性が高いので、無視することがあります。その閾値を設定します。
# min_count: 学習に使う単語の最低出現回数 sampleとは逆に、頻度が少なすぎる単語もその文章を表現するのに適切でない場合があるので無視することがあります。
# workers: 学習時のスレッド数

if __name__ == '__main__': #直接プログラムを回した場合起動
    corpus = list(get_all_files(INPUT_DOC_DIR))
    if len(corpus)<30:
        Sim_times = len(corpus)
        Test_times = len(corpus)
    else:
        Sim_times = 30
        Test_times = 10
    sentences = list(corpus_to_sentences(corpus))
    model = train(sentences)
    model.save(OUTPUT_MODEL)