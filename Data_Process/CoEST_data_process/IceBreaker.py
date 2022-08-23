import os
import shutil
import pandas as pd
import numpy as np


class Format:

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.standard_path = os.path.join(dataset_dir, 'standard')
        self.create_folder(self.standard_path)

    def create_folder(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def transform_artifacts_from_csv(self, file_name, save_name):
        raw_df = pd.read_csv(os.path.join(self.dataset_dir, file_name))
        df = pd.DataFrame([raw_df.columns.values], columns=['id', 'doc'])
        raw_df.columns = ['id', 'doc']
        df = pd.concat([df, raw_df])
        df.to_csv(os.path.join(self.standard_path, save_name), index=False)

    def transform_TL_from_csv(self, file_name):
        raw_df = pd.read_csv(os.path.join(self.dataset_dir, file_name))
        df = pd.DataFrame([raw_df.columns.values], columns=['s_id', 't_id'])
        raw_df.columns = ['s_id', 't_id']
        df = pd.concat([df, raw_df])
        df.to_csv(os.path.join(self.standard_path, 'result.csv'), index=False)

def main(datasets_dir):
    instance = Format(os.path.join(datasets_dir, 'IceBreaker'))
    instance.transform_artifacts_from_csv(file_name='Requirements.csv', save_name='source.csv')
    instance.transform_artifacts_from_csv(file_name='ClassDiagram.csv', save_name='target.csv')
    instance.transform_TL_from_csv(file_name='Requirements2ClassMatrix.csv')
    print('IceBreaker OK')

if __name__ == '__main__':
    datasets_dir = ''
    main(datasets_dir)