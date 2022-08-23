import os
from collections import defaultdict
import pandas as pd
import numpy as np
import logging
import itertools

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")

class generation:

    def __init__(self):
        pass

    def run(self, dataset_dir):
        self.load_data(dataset_dir)
        # build similarity matrix
        source_similar_artifacts = [sim_ids for sim_ids in self.t_to_s.values()]  # similar source artifacts group
        source_matrix = self.similarity_matrix(self.source_ids, source_similar_artifacts)
        target_similar_artifacts = [sim_ids for sim_ids in self.s_to_t.values()]  # similar target artifacts group
        target_matrix = self.similarity_matrix(self.target_ids, target_similar_artifacts)
        # generate train data
        source_train_data_df = self.generate_train_data(self.source_ids, source_matrix, self.source_documents)
        source_train_data_df.to_csv(os.path.join(dataset_dir, 'source_train_data.csv'), index=False)
        target_train_data_df = self.generate_train_data(self.target_ids, target_matrix, self.target_documents)
        target_train_data_df.to_csv(os.path.join(dataset_dir, 'target_train_data.csv'), index=False)

    def load_data(self, dataset_dir):
        source_df = pd.read_csv(os.path.join(dataset_dir, 'source.csv'))  # source artifacts
        target_df = pd.read_csv(os.path.join(dataset_dir, 'target.csv'))  # target artifacts
        self.result_df = pd.read_csv(os.path.join(dataset_dir, 'result.csv'))  # traceability links

        # artifacts documents
        self.source_documents = {str(id).strip(): document for id, document in source_df[['id', 'doc']].values}
        self.target_documents = {str(id).strip(): document for id, document in target_df[['id', 'doc']].values}

        self.s_to_t = defaultdict(set)  # target artifacts which linked to the same source artifact
        self.t_to_s = defaultdict(set)  # source artifacts which linked to the same target artifact
        for source_id, target_id in self.result_df[['s_id', 't_id']].values:
            source_id = str(source_id).strip()
            target_id = str(target_id).strip()
            self.s_to_t[source_id].add(target_id)
            self.t_to_s[target_id].add(source_id)

        source_ids = [str(id).strip() for id in self.result_df.s_id.values]  # all labeled source artifacts
        self.source_ids = list(set(source_ids))  # deduplication
        self.source_ids.sort()
        target_ids = [str(id).strip() for id in self.result_df.t_id.values]  # all labeled target artifacts
        self.target_ids = list(set(target_ids))  # deduplication
        self.target_ids.sort()

    def similarity_matrix(self, artifact_ids, similar_artifacts):
        '''
        build a software artifact similarity matrix
        '''
        matrix = dict()
        for id1 in artifact_ids:
            labels = []
            for id2 in artifact_ids:
                if id1 == id2:
                    label = -1
                else:
                    label = 0
                    for sim_ids in similar_artifacts:
                        if id1 in sim_ids and id2 in sim_ids:
                            label = 1
                labels.append(label)
            matrix[id1] = labels
        matrix_df = pd.DataFrame(matrix, index=artifact_ids)
        return matrix_df

    def generate_train_data(self, artifact_ids, matrix, artifact_documents):
        train_data = list()
        # 2 length tuples, in sorted order, no repeated elements
        for id1, id2 in itertools.combinations(artifact_ids, 2):
            train_data.append([id1, id2, matrix.loc[id1, id2]])

        train_data_df = pd.DataFrame(train_data, columns=['id1', 'id2', 'label'])
        train_data_df['doc1'] = [artifact_documents.get(id1) for id1 in train_data_df.id1.values]
        train_data_df['doc2'] = [artifact_documents.get(id2) for id2 in train_data_df.id2.values]
        train_data_df.dropna(inplace=True)

        return train_data_df

def main(datasets_dir):
    for dirpath, dirnames, files in os.walk(datasets_dir):
        for dirname in dirnames:
            if dirname == 'standard':
                standard_dir = os.path.join(dirpath, dirname)  # Folder path converted to standard data
                g = generation()
                g.run(standard_dir)
                logger.info(f'generate 【{standard_dir}】')

    # Combine all training data together
    df_list = list()
    for dirpath, dirnames, files in os.walk(datasets_dir):
        for dirname in dirnames:
            if dirname == 'standard':
                standard_dir = os.path.join(dirpath, dirname)  # Folder path converted to standard data
                # Here only the source artifact training data is used
                source_train_data_path = os.path.join(standard_dir, 'source_train_data.csv')
                if not os.path.exists(source_train_data_path):
                    raise Exception("can't find the source_train_data.csv")
                source_train_data_df = pd.read_csv(source_train_data_path)
                df_list.append(source_train_data_df)
                logger.info(f'append 【{standard_dir}】')
    df = pd.concat(df_list)
    df.to_csv('train.csv', index=False)

if __name__ == '__main__':
    # Parameters that need to be modified according to their own conditions
    datasets_dir = r'..\data\CoEST'  # Path to all datasets
    # ---------------------------------------------------------------------
    main(datasets_dir)

