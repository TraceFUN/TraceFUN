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

    def transform_artifacts(self):
        artifact_dir_name_dict = {
            'UC': '1 - use cases',
            'ID': '2 - Interaction diagrams',
            'TC': '3 - test cases',
            'CC': '4 - class description',
        }
        res = list()
        artifacts_dir = os.path.join(self.dataset_dir, '2 - docs (English)')
        for artifact_type, artifact_dir_name in artifact_dir_name_dict.items():
            artifact_dir = os.path.join(artifacts_dir, artifact_dir_name)
            for file_name in os.listdir(artifact_dir):
                file_path = os.path.join(artifact_dir, file_name)
                with open(file_path, 'rb') as f:
                    content = f.read()
                    encode = chardet.detect(content).get('encoding')
                    content = content.decode(encode)
                    content = re.sub('[\s\n\r\t]', ' ' ,content)
                artifact_id = f'{artifact_type}{file_name.replace(".txt", "")}'
                res.append([artifact_id, content])
        df = pd.DataFrame(res, columns=['id', 'doc'])
        df.to_csv(os.path.join(self.standard_path, 'source.csv'), index=False)
        df.to_csv(os.path.join(self.standard_path, 'target.csv'), index=False)

    def transform_TL(self):
        result_dir = os.path.join(self.dataset_dir, 'oracle')
        res = list()
        for file_name in os.listdir(result_dir):
            if file_name == 'SimpleUC_CC.txt':
                continue
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, 'r') as f:
                content = f.readlines()
            for row in content:
                ids = re.split('[:\s\n\t\r]', row)
                ids = list(filter(lambda x: len(x) > 0, ids))
                ids = list(map(lambda x: x.replace('.txt', '').strip(), ids))
                s_id, t_ids = ids[0], ids[1:]
                source_prefix, target_prefix = file_name.replace('.txt', '').split('_')
                for t_id in t_ids:
                    res.append([source_prefix + s_id, target_prefix + t_id])
        pd.DataFrame(res, columns=['s_id', 't_id']).to_csv(os.path.join(self.standard_path, 'result.csv'), index=False)

def main(datasets_dir):
    instance = Format(os.path.join(datasets_dir, 'EasyClinic'))
    instance.transform_artifacts()
    instance.transform_TL()
    print('EasyClinic ok')

if __name__ == '__main__':
    datasets_dir = ''
    main(datasets_dir)