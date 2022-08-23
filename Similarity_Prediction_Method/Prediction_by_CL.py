import os
import re
import argparse
import random
import shutil
import itertools
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import collections
from gensim.models import KeyedVectors

import tensorflow as tf
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adadelta
import keras.backend as K

# Add this code when the computer memory is insufficient
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
# ------------------------------------------------------
def setup_seed(seed):
    random.seed(seed)  # set random seed for python
    np.random.seed(seed)  # set random seed for numpy
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")

current_file_path = os.path.dirname(os.path.abspath(__file__))
stops = set(stopwords.words('english'))

def text_to_word_list(text):
    '''
    Pre process and convert texts to a list of words
    '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

class Prediction:

    def __init__(self, standard_dir):
        self.dataset_dir = standard_dir
        self.similarity_lists_dir = os.path.join(standard_dir, 'Similarity_Lists')
        self.max_seq_length = 1000
        self.embedding_dim = 300
        self.EMBEDDING_FILE = os.path.join(current_file_path, f'glove.6B.{self.embedding_dim}d_format.txt')
        # Model variables
        self.n_hidden = 50
        self.gradient_clipping_norm = 1.25

    def run(self):
        self.load_data()
        self.run_source()
        self.run_target()

    def run_source(self):
        self.build_embedding_matrix(self.source)
        wait_predict_data, wait_predict_df = self.process_data(self.source_ids_in_unlabeled, self.source_ids_in_labeled, self.source)
        self.predict('source', wait_predict_data, wait_predict_df)

    def run_target(self):
        self.build_embedding_matrix(self.target)
        wait_predict_data, wait_predict_df = self.process_data(self.target_ids_in_unlabeled, self.target_ids_in_labeled, self.target)
        self.predict('target', wait_predict_data, wait_predict_df)

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
        self.source = {id: doc for id, doc in source_df[['id', 'doc']].values}
        self.target = {id: doc for id, doc in target_df[['id', 'doc']].values}
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

    def build_embedding_matrix(self, artifact_dict):
        logger.info('2. build embedding matrix')
        vocabulary = dict()
        inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
        logger.info('wait load word2vec pre-trained model...')
        word2vec = KeyedVectors.load_word2vec_format(self.EMBEDDING_FILE, binary=False)
        logger.info('load word2vec pre-trained model success')

        logger.info('Start converting words to indices')
        for id, doc in artifact_dict.items():
            a2n = []
            for word in text_to_word_list(doc):
                if word in stops and word not in word2vec:
                    continue
                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    a2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    a2n.append(vocabulary[word])
            artifact_dict[id] = a2n

        embeddings = 1 * np.random.randn(len(vocabulary) + 1, self.embedding_dim)
        embeddings[0] = 0  # So that the padding will be ignored
        logger.info(f'The amount of words{len(vocabulary)}， matrix{len(vocabulary) + 1}*{self.embedding_dim}')

        # Build the embedding matrix
        logger.info('start build matrix')
        for word, index in tqdm(vocabulary.items(), desc='build the embedding matrix'):
            if word in word2vec:
                embeddings[index] = word2vec.get_vector(word)

        del word2vec

        self.embeddings = embeddings

    def process_data(self, ids_in_unlabeled, ids_in_labeled, documents):
        logger.info('3. Data process')
        wait_predict_data_list = list()

        for unlabeled_id, labeled_id in itertools.product(ids_in_unlabeled, ids_in_labeled):
            wait_predict_data_list.append([unlabeled_id, documents[unlabeled_id], labeled_id, documents[labeled_id]])

        wait_predict_data_df = pd.DataFrame(wait_predict_data_list, columns=['id1', 'doc1', 'id2', 'doc2'])
        wait_predict_data_dict = {'left': wait_predict_data_df.doc1, 'right': wait_predict_data_df.doc2}

        # Zero padding
        for dataset, side in itertools.product([wait_predict_data_dict], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=self.max_seq_length)
        logger.info('Zero padding')

        # Make sure everything is ok
        assert wait_predict_data_dict['left'].shape == wait_predict_data_dict['right'].shape

        return wait_predict_data_dict, wait_predict_data_df

    def predict(self, artifact_type, wait_predict_data, wait_predict_df):
        logger.info('4. labeling unlabeled data')

        logger.info('Text Information Representation')
        # 1) Text Input Layer
        artifact_input_1 = layers.Input(shape=(self.max_seq_length,), dtype='int32', name='artifact_input_1')
        artifact_input_2 = layers.Input(shape=(self.max_seq_length,), dtype='int32', name='artifact_input_2')

        # 2) Embedding Layer
        embedding_layer = layers.Embedding(
            input_dim=len(self.embeddings),
            output_dim=self.embedding_dim,
            weights=[self.embeddings],
            input_length=self.max_seq_length,
            trainable=False,
            name='embedding_layer'
        )
        artifact_embedding_1 = embedding_layer(artifact_input_1)
        artifact_embedding_2 = embedding_layer(artifact_input_2)

        # 3) Shared LSTM
        shared_lstm = layers.CuDNNLSTM(self.n_hidden, name='LSTM')
        artifact_lstm_1 = shared_lstm(artifact_embedding_1)
        artifact_lstm_2 = shared_lstm(artifact_embedding_2)

        # 4) Malstm Distance Layer
        # K.clear_session()
        mahanttan_layer = layers.Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)))
        mahanttan_distance = mahanttan_layer([artifact_lstm_1, artifact_lstm_2])
        logger.info('Malstm Distance Layer')

        # 5) Build the model
        model = Model(inputs=[artifact_input_1, artifact_input_2], outputs=[mahanttan_distance])
        model.compile(loss=self.loss(1), optimizer='RMSprop', metrics=['accuracy'])
        model.summary()

        # 6) Load weights
        model_path = os.path.join(current_file_path, 'model', 'model_0_weights.h5')
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        model.summary()

        # 7) predict
        pred = model.predict([wait_predict_data['left'], wait_predict_data['right']])
        wait_predict_df['pred'] = [i[0] for i in pred]
        if not os.path.exists(self.similarity_lists_dir):
            os.mkdir(self.similarity_lists_dir)
        similarity_list_path = os.path.join(self.similarity_lists_dir, f'{artifact_type}_similarity_list_by_cl.csv')
        wait_predict_df[['id1', 'id2', 'pred']].to_csv(similarity_list_path, index=False)

    def euclidean_distance(self, vects):
        """Find the Euclidean distance between two vectors.

        Arguments:
            vects: List containing two tensors of same length.

        Returns:
            Tensor containing euclidean distance
            (as floating point value) between vectors.
        """

        x, y = vects
        sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

    def loss(self, margin=1):
        """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

      Arguments:
          margin: Integer, defines the baseline for distance for which pairs
                  should be classified as dissimilar. - (default is 1).

      Returns:
          'constrastive_loss' function with data ('margin') attached.
      """

        # Contrastive loss = mean( (1-true_value) * square(prediction) +
        #                         true_value * square( max(margin-prediction, 0) ))
        def contrastive_loss(y_true, y_pred):
            """Calculates the constrastive loss.

          Arguments:
              y_true: List of labels, each label is of type float32.
              y_pred: List of predictions of same length as of y_true,
                      each label is of type float32.

          Returns:
              A tensor containing constrastive loss as floating point value.
          """

            square_pred = tf.math.square(y_pred)
            margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
            return tf.math.reduce_mean(
                (1 - y_true) * square_pred + (y_true) * margin_square
            )

        return contrastive_loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--standard_dir", default="", type=str,
        help="The input data dir. Should contain the source.csv, target.csv and result.scv files for the task.")
    parser.add_argument(
        "--task_type", default="", type=str,
        help="Type of the task.")
    return parser.parse_args()

def main():
    setup_seed(76)
    args = get_args()
    logger.info(f'{args.standard_dir} --- {args.task_type}')
    predict = Prediction(args.standard_dir)
    predict.load_data()
    if args.task_type == 'source':
        predict.run_source()
    else:
        assert (args.task_type == 'target')
        predict.run_target()

if __name__ == '__main__':
    main()

