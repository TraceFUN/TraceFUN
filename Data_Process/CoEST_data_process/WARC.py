import os
import re
import shutil
import chardet
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

    def get_artifact_from_txt(self, dir_name, save_name):
        txt_dir = os.path.join(self.dataset_dir, dir_name)
        res = list()
        for file_name in os.listdir(txt_dir):
            if not file_name.endswith('.txt'):
                continue
            file_path = os.path.join(txt_dir, file_name)
            with open(file_path, 'rb') as f:
                content = f.read()
                encode = chardet.detect(content).get('encoding')
                content = content.decode(encode)
            content = content.split('-')[1].strip()
            res.append([file_name.replace('.txt', ''), content])
        pd.DataFrame(res, columns=['id', 'doc']).to_csv(os.path.join(self.standard_path, save_name), index=False)

    def get_link_list_from_txt(self, file_name):
        with open(os.path.join(self.dataset_dir, file_name), 'rb') as f:
            content = f.read()
            encode = chardet.detect(content).get('encoding')
            content = content.decode(encode)
        res = list()
        for row in content.split('\n'):
            if row.find('%') != -1:
                continue
            ids = re.split(r'[\t\r\s]', row)
            ids = list(map(lambda x: x.replace('.txt', '').strip(), ids))
            s_id = list(filter(lambda x: not x.startswith('SRS') and len(x) > 0, ids))[0]
            t_ids = list(filter(lambda x: x.startswith('SRS'), ids))
            for t_id in t_ids:
                res.append([s_id, t_id])
        return res

    def transform_TL_WARC(self):
        file_name_list = ['FRStoSRS.txt', 'NFRtoSRS.txt']
        res = list()
        for file_name in file_name_list:
            link_list = self.get_link_list_from_txt(file_name)
            res.extend(link_list)
        pd.DataFrame(res, columns=['s_id', 't_id']).to_csv(os.path.join(self.standard_path, 'result.csv'), index=False)

    def transform_artifacts_WARC(self):
        self.get_artifact_from_txt(dir_name='FRS', save_name='FRS.csv')
        self.get_artifact_from_txt(dir_name='NFR', save_name='NFR.csv')
        self.get_artifact_from_txt(dir_name='SRS', save_name='SRS.csv')
        source1 = pd.read_csv(os.path.join(self.standard_path, 'FRS.csv'))
        source2 = pd.read_csv(os.path.join(self.standard_path, 'NFR.csv'))
        pd.concat([source1, source2]).to_csv(os.path.join(self.standard_path, 'source.csv'), index=False)
        shutil.copy(os.path.join(self.standard_path, 'SRS.csv'), os.path.join(self.standard_path, 'target.csv'))

def main(datasets_dir):
    instance = Format(os.path.join(datasets_dir, 'WARC'))
    instance.transform_artifacts_WARC()
    instance.transform_TL_WARC()
    print('WARC OK')

if __name__ == '__main__':
    datasets_dir = ''
    main(datasets_dir)