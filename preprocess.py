from asyncore import read
from cgitb import text
from posixpath import split
import gensim
import os
import pickle
from nltk.tree import Tree
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import logging
import numpy as np
# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
WORD2VEC_MODEL = "../wiki_all.model/wiki_all.sent.split.model"
TWITTER_NO_OOV_DATASET = os.path.join('input', 'twitter', 'noov.csv')
TWITTER_OOV_DATASET = os.path.join('input', 'twitter', 'oov.csv')
IMDB_ROOT = os.path.join('input', 'imdb')
IMDB_COIN_PATH = os.path.join(IMDB_ROOT, 'COIN.pickle')

IMDB_NO_OOV_DATASET = os.path.join('input', 'imdb', 'noov.csv')
IMDB_OOV_DATASET = os.path.join('input', 'imdb', 'oov.csv')

WIN_SIZE = list(range(5,17,2))

W2V_SIZE = 400
dataset_filename = 'training.1600000.processed.noemoticon.csv'
dataset_path = os.path.join("input",dataset_filename)

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens).strip()

CLEANED_DATASET_PATH = os.path.join('input', 'cleaned.csv')

def build_coin_twitter():
    w2v_model = gensim.models.word2vec.Word2Vec.load(WORD2VEC_MODEL)
    if not os.path.exists(TWITTER_OOV_DATASET) or not os.path.exists(TWITTER_NO_OOV_DATASET):
        if not os.path.exists(CLEANED_DATASET_PATH):
            logging.info('Loading dataset...')
            df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
            logging.info('Cleaning dataset...')
            df.text = df.text.apply(lambda x: preprocess(x))
            df = df.dropna(subset=['text'])
            df.to_csv(CLEANED_DATASET_PATH, index=False, header=False)
        else:
            logging.info('Loading pre-cleaned dataset...')
            df = pd.read_csv(CLEANED_DATASET_PATH, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
            # nan has to be dropped twice, need to know why
            df.dropna(subset=['text'], inplace=True)

        def filter_oov(row):
            for word in row['text'].split(' '):
                if word not in w2v_model.wv.vocab:
                    return False
            return True


        #df = pd.DataFrame(d, columns=['Name', 'Age', 'Score'])
        # m = df.apply(filter_fn, axis=1)
        oovFilter = df.apply(filter_oov, axis=1)
        noov = df[oovFilter]
        oov = df[~oovFilter]
        logging.info(f"# tweets without oov-words: {len(noov)} ")
        logging.info(f"# tweets with oov-words: {len(oov)} ")
        
        logging.info('Generating auxiliary oov postfix...')
        def oov_aux_postfix(text):
            words = text.split()
            ret = [[] for _ in WIN_SIZE]
            for word in words:
                for w, l in zip(WIN_SIZE, ret):
                    if word not in w2v_model.wv.vocab:
                        l.append(word + '_' + str(w))
                    else:
                        l.append(word)
            ret = [' '.join(l) for l in ret]
            return ret
        oov['win5'], oov['win7'], oov['win9'], oov['win11'], oov['win13'], oov['win15'] = zip(*oov['text'].map(oov_aux_postfix))
        noov.to_csv(TWITTER_NO_OOV_DATASET, index=False, header=False)
        oov.to_csv(TWITTER_OOV_DATASET, index=False, header=False)
    else:
        oov = pd.read_csv(TWITTER_OOV_DATASET, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
        noov = pd.read_csv(TWITTER_NO_OOV_DATASET, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

    logging.info('Building COIN vectors...')
    coinDict = {}
    for text in oov['text']:
        words = text.split()
        L = len(words)
        for i, word in enumerate(words):
            if word not in w2v_model.wv.vocab:
                for winSize in WIN_SIZE:
                    buff = np.zeros((W2V_SIZE,))
                    cnt = 0
                    context = words[max(0, i-winSize): min(L, i+winSize+1)]
                    for c in context:
                        if c in w2v_model.wv.vocab:
                            buff += w2v_model.syn1neg[w2v_model.wv.vocab[c].index]
                            cnt += 1
                    if cnt > 0:
                        buff /= cnt
                        coinDict.setdefault(word, {}).setdefault(winSize, [])
                        coinDict[word][winSize].append(buff)
    for word, coinWin in coinDict.items():
        for k, v in coinWin.items():
            coinWin[k] = np.average(np.asarray(v), axis=0)
    with open('COIN.pickle'.format(winSize), 'wb') as f:
        pickle.dump(coinDict, f)
        
def build_coin_imdb():
    w2v_model = gensim.models.word2vec.Word2Vec.load(WORD2VEC_MODEL)
    df_train = pd.read_csv(os.path.join(IMDB_ROOT, 'train.csv'), encoding ='utf-8', index_col=0)
    df_test = pd.read_csv(os.path.join(IMDB_ROOT, 'test.csv'), encoding ='utf-8', index_col=0)
    if not os.path.exists(IMDB_OOV_DATASET) or not os.path.exists(IMDB_NO_OOV_DATASET):
        def select_oov(row):
            for word in row['text'].split(' '):
                if word not in w2v_model.wv.vocab:
                    return True
            return False
        
        oovSelector = df_test.apply(select_oov, axis=1)
        oov = df_test[oovSelector]
        
        noovSelector = ~df_train.apply(select_oov, axis=1)
        noov = df_train[noovSelector]
        
        logging.info(f"# reviews without oov-words: {len(noov)} ")
        logging.info(f"# reviews with oov-words: {len(oov)} ")
        
        logging.info('Generating auxiliary oov postfix...')
        def oov_aux_postfix(text):
            words = text.split()
            ret = [[] for _ in WIN_SIZE]
            for word in words:
                for w, l in zip(WIN_SIZE, ret):
                    if word not in w2v_model.wv.vocab:
                        l.append(word + '_' + str(w))
                    else:
                        l.append(word)
            ret = [' '.join(l) for l in ret]
            return ret
        oov['win5'], oov['win7'], oov['win9'], oov['win11'], oov['win13'], oov['win15'] = zip(*oov['text'].map(oov_aux_postfix))
        noov.to_csv(IMDB_NO_OOV_DATASET, index=False, header=True)
        oov.to_csv(IMDB_OOV_DATASET, index=False, header=True)
    else:
        oov = pd.read_csv(TWITTER_OOV_DATASET, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
        noov = pd.read_csv(TWITTER_NO_OOV_DATASET, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

    logging.info('Building COIN vectors...')
    coinDict = {}
    for df in [df_test, df_train]:
        for text in df['text']:
            words = text.split()
            L = len(words)
            for i, word in enumerate(words):
                if word not in w2v_model.wv.vocab:
                    for winSize in WIN_SIZE:
                        buff = np.zeros((W2V_SIZE,))
                        cnt = 0
                        context = words[max(0, i-winSize): min(L, i+winSize+1)]
                        for c in context:
                            if c in w2v_model.wv.vocab:
                                buff += w2v_model.syn1neg[w2v_model.wv.vocab[c].index]
                                cnt += 1
                        if cnt > 0:
                            buff /= cnt
                            coinDict.setdefault(word, {}).setdefault(winSize, [])
                            coinDict[word][winSize].append(buff)
    for word, coinWin in coinDict.items():
        for k, v in coinWin.items():
            coinWin[k] = np.average(np.asarray(v), axis=0)
    with open(IMDB_COIN_PATH, 'wb') as f:
        pickle.dump(coinDict, f)

def preporcess_imdb():
    # split the test and train into oov.csv and noov.csv
    
    data = {'trainpos':[], 'trainneg': [], 'testpos': [], 'testneg': []}
    for f1 in ['train', 'test']:
        for f2 in ['pos', 'neg']:
            dir = os.path.join(IMDB_ROOT, f1, f2)
            for file in os.listdir(dir):
                if file.split('.')[-1] != 'txt':
                    continue
                with open(os.path.join(dir, file), 'r', encoding='utf-8') as f:
                    data[f1+f2].append(preprocess(f.read()))
    train = pd.DataFrame(data={'target': [1] * len(data['trainpos']) + [0] * len(data['trainneg']), 'text': data['trainpos'] + data['trainneg']})
    test = pd.DataFrame(data={'target': [1] * len(data['testpos']) + [0] * len(data['testneg']), 'text': data['testpos'] + data['testneg']})
    train.to_csv(os.path.join(IMDB_ROOT, 'train.csv'))
    test.to_csv(os.path.join(IMDB_ROOT, 'test.csv'))
if __name__ == '__main__':
    preporcess_imdb()
    build_coin_imdb()