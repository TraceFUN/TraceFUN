import os
import re
import random
import argparse
import shutil
import collections
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from gensim import corpora, models, matutils, similarities

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stops = set(stopwords.words('english'))

def setup_seed(seed):
    random.seed(seed)  # set random seed for python
    np.random.seed(seed)  # set random seed for numpy

def text_to_word_list(document):
    # 分词+小写
    document_tokenized = [word.lower() for word in document.split()]
    # 去除停用词
    stop_words = stopwords.words('english')
    document_no_stopwords = [word for word in document_tokenized if word not in stop_words]
    # 去除标点符号
    document_no_punctuations = []
    for word in document_no_stopwords:
        word = re.sub('[\,\.\:\;\?\(\)\[\]\&\!\*\@\#\$\%\=\/\+\'\{\}\<\>\-\`]', ' ', word)
        document_no_punctuations.extend(word.split())

    return document_no_punctuations

class Prediction:

    def __init__(self, standard_dir):
        self.dataset_dir = standard_dir
        self.similarity_lists_dir = os.path.join(standard_dir, 'Similarity_Lists')

    def run(self):
        self.load_data()
        logger.info('build VSM')
        documents = [doc for doc in self.source.values()] + [doc for doc in self.target.values()]
        self.build_VSM(documents)
        logger.info('Start predicting similarity to source artifacts')
        wait_predict_df = self.process_data(self.source_ids_in_unlabeled, self.source_ids_in_labeled)
        self.predict('source', wait_predict_df, self.source)
        logger.info('Start predicting similarity of target artifacts')
        wait_predict_df = self.process_data(self.target_ids_in_unlabeled, self.target_ids_in_labeled)
        self.predict('target', wait_predict_df, self.target)

    def load_data(self):
        logger.info('1. load data')

        source_df = pd.read_csv(os.path.join(self.dataset_dir, 'source.csv'))
        target_df = pd.read_csv(os.path.join(self.dataset_dir, 'target.csv'))
        result_df = pd.read_csv(os.path.join(self.dataset_dir, 'result.csv'))

        # Since the id of the source artifact is a number,
        # every time the data is read from the csv file,
        # the data type should be converted to a string and modified according to its own data.
        source_df = source_df.astype({'id': 'int'})
        source_df = source_df.astype({'id': 'str'})
        result_df = result_df.astype({'s_id': 'int'})
        result_df = result_df.astype({'s_id': 'str'})

        # The document corresponding to the artifact id
        self.source = {id: text_to_word_list(doc) for id, doc in source_df[['id', 'doc']].values}
        self.target = {id: text_to_word_list(doc) for id, doc in target_df[['id', 'doc']].values}
        # all id
        self.source_ids = source_df.id.to_list()
        self.target_ids = target_df.id.to_list()
        # labeled id
        self.source_ids_in_labeled = list({}.fromkeys(result_df.s_id.to_list()).keys())
        self.target_ids_in_labeled = list({}.fromkeys(result_df.t_id.to_list()).keys())
        # unlabeled id
        self.source_ids_in_unlabeled = list(
            {}.fromkeys([id for id in self.source_ids if id not in self.source_ids_in_labeled]).keys())
        self.target_ids_in_unlabeled = list(
            {}.fromkeys([id for id in self.target_ids if id not in self.target_ids_in_labeled]).keys())

        logger.info(f'labeled source artifacts：{len(self.source_ids_in_labeled)}')
        logger.info(f'unlabeled source artifacts：{len(self.source_ids_in_unlabeled)}')
        logger.info(f'labeled target artifacts：{len(self.target_ids_in_labeled)}')
        logger.info(f'unlabeled target artifacts：{len(self.target_ids_in_unlabeled)}')

        self.s_to_t = collections.defaultdict(set)  # Target artifact group for each source artifact marked
        self.t_to_s = collections.defaultdict(set)  # The source artifact group for each target artifact marked
        for s_id, t_id in result_df[['s_id', 't_id']].values:
            self.s_to_t[s_id].add(t_id)
            self.t_to_s[t_id].add(s_id)

        self.result_str = [f'{source_id}_{target_id}' for source_id, target_id in result_df[['s_id', 't_id']].values]

    def process_data(self, ids_in_unlabeled, ids_in_labeled):
        logger.info('2. Data process')
        wait_predict_data_list = list()

        for unlabeled_id, labeled_id in itertools.product(ids_in_unlabeled, ids_in_labeled):
            wait_predict_data_list.append([unlabeled_id, labeled_id])

        wait_predict_data_df = pd.DataFrame(wait_predict_data_list, columns=['id1', 'id2'])

        return wait_predict_data_df

    def predict(self, artifact_type, wait_predict_df, documents):
        logger.info('3. Data prediction')
        predict_data = [[index, row['id1'], row['id2']]for index, row in wait_predict_df.iterrows()]
        for index, id1, id2 in tqdm(predict_data, '预测数据'):
            pred = self.similarity(documents[id1], documents[id2])
            wait_predict_df.loc[index, 'pred'] = pred
        if not os.path.exists(self.similarity_lists_dir):
            os.mkdir(self.similarity_lists_dir)
        similarity_list_path = os.path.join(self.similarity_lists_dir, f'{artifact_type}_similarity_list_by_vsm.csv')
        wait_predict_df.to_csv(similarity_list_path, index=False)

    def build_VSM(self, documents):
        dictionary = corpora.Dictionary(documents)  # generate dict
        corpus = [dictionary.doc2bow(doc) for doc in documents]  # generate corpus
        tfidf_model = models.TfidfModel(corpus, id2word=dictionary)  # build vsm
        logger.info("Finish building VSM model")
        self.dictionary = dictionary
        self.vsm = tfidf_model

    def similarity(self, source, target):
        source_bow = self.dictionary.doc2bow(source)
        target_bow = self.dictionary.doc2bow(target)
        source_vec = self.vsm[source_bow]
        target_vec = self.vsm[target_bow]
        return matutils.cossim(source_vec, target_vec)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--standard_dir", default="", type=str,
        help="The input data dir. Should contain the source.csv, target.csv and result.scv files for the task.")
    return parser.parse_args()

def main():
    setup_seed(76)
    args = get_args()
    predict = Prediction(args.standard_dir)
    predict.run()

if __name__ == '__main__':
    main()