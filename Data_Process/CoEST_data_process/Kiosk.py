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

    def transform_TL_from_csv_Kiosk(self, file_name):
        raw_df1 = pd.read_csv(os.path.join(self.dataset_dir, file_name))
        raw_df2 = pd.DataFrame([raw_df1.columns.values], columns=['id_pair'])
        raw_df1.columns = ['id_pair']
        df = pd.concat([raw_df1, raw_df2])
        id_pairs = df.id_pair.values.tolist()
        id_pairs = [pair.split(',') for pair in id_pairs]
        res = list()
        for i in id_pairs:
            if len(i) > 2:
                i = i[:-1]
            res.append(i)
        pd.DataFrame(res, columns=['s_id', 't_id']).to_csv(os.path.join(self.standard_path, 'result.csv'), index=False)

def main(datasets_dir):
    instance = Format(os.path.join(datasets_dir, 'Kiosk'))
    instance.transform_artifacts_from_csv(file_name='Requirements1.csv', save_name='source.csv')
    instance.transform_artifacts_from_csv(file_name='ProcessSpecs.csv', save_name='target.csv')
    instance.transform_TL_from_csv_Kiosk(file_name='Matrix1.csv')
    print('Kiosk OK')

if __name__ == '__main__':
    datasets_dir = ''
    main(datasets_dir)