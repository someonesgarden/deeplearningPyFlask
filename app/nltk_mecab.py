#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import codecs
import os
import sys
import importlib
path2proj = "/Users/user/PycharmProjects/deeplearningFlask"
sys.path.append(path2proj)

import MeCab
from lib.chasen import *
importlib.reload(sys)
sys.setdefaultencoding('utf-8')
# sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
import re, pprint
import nltk
from nltk.corpus.reader import *
from nltk.corpus.reader.util import *
from nltk.text import Text
from xml.dom.minidom import parse
from math import log

from nltk.book import *

def getNoun(words):
   noun = []
   tagger = MeCab.Tagger( "-Ochasen" )
   node = tagger.parseToNode( words.encode( "utf-8" ) )
   while node:
      if node.feature.split(",")[0] == "名詞":
         replace_node = re.sub( re.compile( "[!-/:-@[-`{-~]" ), "", node.surface )
         if replace_node != "" and replace_node != " ":
            noun.append( replace_node )
      node = node.next
   return noun

def getTopKeywords(TF,n):
   list = sorted( TF.items(), key=lambda x:x[1], reverse=True )
   return list[0:n]

def calcTFIDF( N,TF, DF ):
   tfidf = TF * log( N / DF )
   return tfidf

def pp(obj):
    pp = pprint.PrettyPrinter(indent=4, width=160)
    str = pp.pformat(obj)
    return re.sub(r"\\u([0-9a-f]{4})", lambda x: chr(int("0x"+x.group(1),16)), str)


def load_eng_book():
    from nltk.corpus import gutenberg
    emma = gutenberg.words('austen-emma.txt')

    for fileid in gutenberg.fileids():
        print(gutenberg.sents(fileid))
        num_chars = len(gutenberg.raw(fileid))
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
        print(int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab), fileid)

data = {
    u"スクリプト言語":
    {u"Perl": u"パール",
    u"Python": u"パイソン",
    u"Ruby": u"ルビー"},
    u"関数型言語":
    {u"Erlang": u"アーラング",
    u"Haskell": u"ハスケル",
    u"Lisp": u"リスプ"}
}

print(pp(data))

jp_sent_tokenizer = nltk.RegexpTokenizer(u'[^　「」！？。]*[！？。]')

jp_chartype_tokenizer = nltk.RegexpTokenizer(u'([ぁ-んー]+|[ァ-ンー]+|[\u4e00-\u9FFF]+|[^ぁ-んァ-ンー\u4e00-\u9FFF]+)')

ginga = PlaintextCorpusReader("data/nltk/", r'gingatetsudono_yoru.txt', encoding='utf-8',
                              para_block_reader=read_line_block,
                              sent_tokenizer=jp_sent_tokenizer,
                              word_tokenizer=jp_chartype_tokenizer)

# print ginga.raw()

# print '/'.join(ginga.words()[0:50])

jeita = ChasenCorpusReader('/Users/user/nltk_data/corpora/jeita/', '.*chasen', encoding='utf-8')

# print '/'.join(jeita.words()[22100:22130])

# print '\nEOS\n'.join(['\n'.join("%s/%s" % (w[0],w[1].split('\t')[2]) for w in sent) for sent in jeita.tagged_sents()[2170:2173]])

ginga_t = Text(w.encode('utf-8') for w in ginga.words())

# print ginga_t.concordance("川")

print(0)

from lib.knbc import *
from nltk.corpus.util import LazyCorpusLoader
root = nltk.data.find('/Users/user/nltk_data/corpora/knbc/corpus1')
fileids = [f for f in find_corpus_fileids(FileSystemPathPointer(root), ".*") if re.search(r"\d\-\d\-[\d]+\-[\d]+", f)]

print(1)

def _knbc_fileids_sort(x):
    cells = x.split('-')
    return (cells[0], int(cells[1]), int(cells[2]), int(cells[3]))

print(2)
knbc = LazyCorpusLoader('knbc/corpus1', KNBCorpusReader, sorted(fileids, key=_knbc_fileids_sort), encoding='euc-jp')

# print knbc.fileids()
mecab = MeCab.Tagger()

# print mecab.parse("僕は人間が大好きですが、猫も好きなのです")

print(3)



N = 1675757
tf = {}

df = {}
dom = parse("data/wiki/jawiki-latest-pages-articles1.xml")
print("dom")
print(dom)
text = dom.getElementsByTagName("text")

for i, text in enumerate(text):
    print(i)
    print(text)
    df_list = []
    noun = getNoun(text.childNodes[0].data)

    for word in noun:
        try:
            tf[word] = tf[word] + 1
        except KeyError:
            tf[word] = 1
    for word in noun:
        try:
            if word in df_list:
                continue
            df[word] = df[word] + 1
        except KeyError:
            df[word] = 1

tfidf = {}
for k,v in getTopKeywords( tf, 100 ):
    tfidf[k] = calcTFIDF(N,tf[k],df[k])
for k,v in getTopKeywords( tfidf, 100):
    print(k,v)

print(-1)