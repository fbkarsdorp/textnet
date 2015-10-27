import glob

from string import punctuation
from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import textnet

def time_analyzer(filename, token_type='word', ngram_size=1):
    print(filename)
    with open(filename, encoding='latin-1') as infile:
        for line in infile:
            token, lemma, _, _ = line.strip().split("\t")
            n = len(token)
            token = token.lower()
            if not all(e in punctuation for e in lemma) and lemma not in ('-lrb-', '-rrb-'):
                if token_type == 'word':
                    yield token
                else:
                    for i in range(n):
                        if (i + ngram_size - 1) < n:
                            yield token[i: i + ngram_size]


decades = '1970s', '1980s'
metafiles = ['/vol/bigdata/corpora/TIME/%s.csv' % decade for decade in decades]
metainf = {}
for meta in metafiles:
    for line in open(meta, encoding='latin-1'):
        fields = line.strip().split(';')
        if len(fields) < 3: continue
        idnumber, year, date = fields[:3]
        if len(date.split('/')[0]) < 4:
            metainf[idnumber] = pd.datetime.strptime(date, "%d/%m/%y")
        else:
            try:
                metainf[idnumber] = pd.datetime.strptime(date, "%Y/%m/%d")
            except ValueError:
                print(date)

filenames = [f for decade in decades 
               for f in glob.glob("/vol/bigdata/corpora/TIME/rich_texts_txt/%s/*.tag.txt" % decade) 
               if metainf[f.split('/')[-1].replace('.tag.txt', '')] is not None]

filenames.sort(key=lambda x: metainf[x.split('/')[-1].replace(".tag.txt", "")])
time_dates = np.array([metainf[f.split('/')[-1].replace('.tag.txt', '')] for f in filenames])

vectorizer = TfidfVectorizer(input='filename', analyzer=partial(time_analyzer, token_type='char', ngram_size=4), min_df=2)
time_magazine = vectorizer.fit_transform(filenames)

neighbors = textnet.bootstrap_neighbors(
    time_magazine, time_dates, sigma=0, n_iter=100, all_min=True, grouped_pairwise=True, groupby=1)

running_statistics = textnet.statistics.evolving_graph_statistics(
    neighbors, time_dates, groupby=pd.TimeGrouper(freq='M'), sigma=0.3)

sigma_statistics = textnet.statistics.eval_sigmas(neighbors, step_size=0.05)

