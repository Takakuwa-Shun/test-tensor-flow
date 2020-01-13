# -*- coding: utf-8 -*-

import Training_Doc2Vec
from gensim import models
from gensim.models.doc2vec import DocvecsArray
import os



# 似た文章を探す(文章で検索)
def search_similar_texts_from_text(words):
    x = model.infer_vector(words) # Infer a vector for given post-bulk training document.
    most_similar_texts = model.docvecs.most_similar([x]) # xに近いと思われる、学習データ内の文章を、確率が最も高い順に並べたリストを返す　[(ラベル名,確率),...]
    for similar_text in most_similar_texts:
        print('パス: {}, 確率 : {}'.format(similar_text[0], similar_text[1]))

# 似た単語を探す(文章で検索)
def search_similar_words_from_text(words):
    for word in words:
        print()
        print(word + ':')
        for result in model.most_similar(positive=word, topn=10):
            print(result[0])

# 似た文章を探す(作者・作品で検索)
def search_similar_texts_from_author_or_work(path):
    most_similar_texts = model.docvecs.most_similar(path)
    # print(str(most_similar_texts).decode("string-escape"))
    print('似ている文章順に')
    for similar_text in most_similar_texts:
        # print(similar_text[0])
       print('タイトル: {}, 確率 : {}'.format(similar_text[0],similar_text[1]))

# 言葉の足し引き
def Word_Addition_Subtraction(Positive_word,Negative_word):
    most_similar_texts = model.most_similar(positive=Positive_word, negative=Negative_word)
    print('似ている文章順に')
    for similar_text in most_similar_texts:
        # print(similar_text[0])
       print('タイトル: {}, 確率 : {}'.format(similar_text[0],similar_text[1]))


    # 学習した文章の類似度を測定
def Search_Similarity(doc1,doc2):
    similarity = model.similarity(doc1, doc2)
    print(similarity)

if __name__ == '__main__':
    # 文章で検索する場合
    # model = models.Doc2Vec.load('doc2vec_JUMP.model')
    # search_doc = '''。'''
    # words = Training_Doc2Vec.split_into_words(search_doc).words
    # search_similar_texts_from_text(words)
    #
    # # ファイル名で検索する場合
    model = models.Doc2Vec.load('doc2vec_JUMP.model')
    dir = "./Wikipedia/JUMP/"
    file ="ONE PIECE"
    DOC_PATH = os.path.join(dir, file + ".txt")
    search_similar_texts_from_author_or_work(DOC_PATH)
    #
    # # 言葉の足し引きで検索する場合
    # model = models.doc2vec.DocvecsArray.load('doc2vec_JUMP.model')
    # Positive_word = "テニスの王子様"
    # Negative_word = "テニス"
    # Word_Addition_Subtraction(Positive_word,Negative_word)

    # 文章の類似度を測定
    # model = models.doc2vec.DocvecsArray.load('doc2vec_JUMP.model')
    # # model = models.Doc2Vec.load('doc2vec_JUMP.model')
    # dir = "./Wikipedia/JUMP/"
    # doc1 = "テニスの王子様"
    # doc2 = "ONE PIECE"
    # DOC_PATH1 = os.path.join(dir, doc1 + ".txt")
    # DOC_PATH2 = os.path.join(dir, doc2 + ".txt")
    # Search_Similarity(doc1,doc2)