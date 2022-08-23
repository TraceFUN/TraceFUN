import os
import re
import time
import random
import datetime
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import tensorflow as tf
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
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

class Train:

    def __init__(self):
        self.artifacts_cols = ['doc1', 'doc2']
        self.seed = 76
        self.model_dir = os.path.join(current_file_path, 'model')
        # Model variables
        self.n_hidden = 50
        self.gradient_clipping_norm = 1.25
        self.batch_size = 64
        self.n_epoch = 100
        # pre-trained word2vec path
        self.embedding_dim = 300
        self.EMBEDDING_FILE = os.path.join(current_file_path, f'glove.6B.{self.embedding_dim}d_format.txt')
        self.GLOVE_FILE = os.path.join(current_file_path, f'glove.6B.{self.embedding_dim}d.txt')

    def run(self):
        '''
        no cross-validation
        '''
        self.MODEL_NO = 0
        self.load_data()
        self.build_embedding_matrix()
        X_train, Y_train, X_test, Y_test = self.process_data()
        self.train([X_train['left'], X_train['right']], Y_train)
        self.test(X_test, Y_test)
        logger.info('---End---')

    def load_data(self):
        logger.info('1. Load data and shuffle')
        train_df = pd.read_csv('train.csv')
        # train_df = train_df.sample(frac=1)
        # train_df = train_df.reset_index()
        # train_df.drop(['index'], axis=1, inplace=True)
        self.train_df = train_df



    def build_embedding_matrix(self):
        logger.info('2. Build embedding matrix')
        vocabulary = dict()
        inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
        logger.info('wait load word2vec...')
        if not os.path.exists(self.EMBEDDING_FILE):
            glove2word2vec(self.GLOVE_FILE, self.EMBEDDING_FILE)
        word2vec = KeyedVectors.load_word2vec_format(self.EMBEDDING_FILE, binary=False)  # load pre-trained word2vec
        logger.info('load word2vec success')

        logger.info('start converting words to vectors')
        for index, row in tqdm(self.train_df.iterrows(), desc='train data'):

            for artifact in self.artifacts_cols:

                a2n = []  # a2n -> artifact numbers representation
                for word in text_to_word_list(row[artifact]):

                    # Check for unwanted words
                    if word in stops and word not in word2vec:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        a2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        a2n.append(vocabulary[word])

                # Replace artifacts as word to artifact as number representation
                self.train_df.at[index, artifact] = a2n

        embeddings = 1 * np.random.randn(len(vocabulary) + 1, self.embedding_dim)
        embeddings[0] = 0  # So that the padding will be ignored
        logger.info(f'words count:{len(vocabulary)}, matrix shape:{len(vocabulary) + 1}*{self.embedding_dim}')

        logger.info('start build the embedding matrix')
        for word, index in tqdm(vocabulary.items(), desc='build the embedding matrix'):
            if word in word2vec:
                embeddings[index] = word2vec.get_vector(word)

        del word2vec

        self.embeddings = embeddings

    def process_data(self):
        logger.info('3. Split the dataset and process the data')
        self.max_seq_length = max(
            self.train_df.doc1.map(lambda x: len(x)).max(),
            self.train_df.doc2.map(lambda x: len(x)).max()
        )

        logger.info(f'document has at most words:{self.max_seq_length}')

        X = self.train_df[self.artifacts_cols]
        Y = self.train_df[['label']]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)

        # Split to dicts
        X_train = {'left': X_train.doc1, 'right': X_train.doc2}
        X_test = {'left': X_test.doc1, 'right': X_test.doc2}

        # Convert labels to their numpy representations
        Y_train = Y_train.label.astype(float)
        Y_test = Y_test.label.astype(float)
        # Y_train = Y_train.label
        # Y_test = Y_test.label

        # Zero padding
        for dataset,  side in itertools.product([X_train, X_test], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=self.max_seq_length)
        logger.info('zero padding')

        # Make sure everything is ok
        assert X_train['left'].shape == X_train['right'].shape
        assert len(X_train['left']) == len(Y_train)

        return X_train, Y_train, X_test, Y_test

    def train(self, X, Y):
        logger.info('4. Model training')

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

        callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

        # Train the model
        logger.info('Train the model')
        training_start_time = time.time()
        model_history = model.fit(
            X, Y,
            batch_size=self.batch_size,
            epochs=self.n_epoch,
            validation_split=0.2,
            shuffle=True,
            callbacks=[callback]
        )
        logger.info("Training time finished.\n{} epochs in {}".format(self.n_epoch, datetime.timedelta(seconds=time.time() - training_start_time)))

        self.model = model
        self.model_history = model_history

    def test(self, X_test, Y_test):
        logger.info('5. model evaluation')
        Y_pred = self.model.predict([X_test['left'], X_test['right']])

        accuracy = accuracy_score(Y_test == 1, Y_pred >= 0.5)
        precision = precision_score(Y_test == 1, Y_pred >= 0.5, average=None)
        recall = recall_score(Y_test == 1, Y_pred >= 0.5, average=None)
        f_measure = f1_score(Y_test == 1, Y_pred >= 0.5, average=None)

        logger.info(f"model_{self.MODEL_NO}: test accuracy: {accuracy}")
        logger.info(f"model_{self.MODEL_NO}: test precision p: {precision[1]}")
        logger.info(f"model_{self.MODEL_NO}: test recall p: {recall[1]}")
        logger.info(f"model_{self.MODEL_NO}: test f1 score p: {f_measure[1]}")
        logger.info(f"model_{self.MODEL_NO}: test precision n: {precision[0]}")
        logger.info(f"model_{self.MODEL_NO}: test recall n: {recall[0]}")
        logger.info(f"model_{self.MODEL_NO}: test f1 score n: {f_measure[0]}")

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # Save model and model weights
        logger.info('Save the model')
        self.model.save(os.path.join(self.model_dir, f'model_{self.MODEL_NO}.h5'))
        self.model.save_weights(os.path.join(self.model_dir, f'model_{self.MODEL_NO}_weights.h5'))

        # Record test result
        with open(os.path.join(self.model_dir, f'result_{self.MODEL_NO}.txt'), 'w') as f:
            f.write(f"model_{self.MODEL_NO}: test accuracy: {accuracy}\n")
            f.write(f"model_{self.MODEL_NO}: test precision p: {precision[1]}\n")
            f.write(f"model_{self.MODEL_NO}: test recall p: {recall[1]}\n")
            f.write(f"model_{self.MODEL_NO}: test f1 score p: {f_measure[1]}\n")
            f.write(f"model_{self.MODEL_NO}: test precision n: {precision[0]}\n")
            f.write(f"model_{self.MODEL_NO}: test recall n: {recall[0]}\n")
            f.write(f"model_{self.MODEL_NO}: test f1 score n: {f_measure[0]}\n")

        # Plot accuracy
        plt.plot(self.model_history.history['accuracy'])
        plt.plot(self.model_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(self.model_dir, f'accuracy_{self.MODEL_NO}.png'))
        # plt.show()

        # Plot loss
        plt.cla()
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(os.path.join(self.model_dir, f'loss_{self.MODEL_NO}.png'))
        # plt.show()

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


if __name__ == '__main__':
    setup_seed(76)
    train = Train()
    train.run()
