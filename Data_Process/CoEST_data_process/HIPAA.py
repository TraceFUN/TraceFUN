import os
import re
import shutil
import chardet
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


class Format:

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.standard_path = os.path.join(dataset_dir, 'standard')
        self.create_folder(self.standard_path)

        self.HIPAA_List = ['1Care2x', '2CCHIT', '3ClearHealth', '4Consultations', '5iTrust', '6TrialImplementations',
                      '7PatientOS', '8PracticeOne', '9Soren', '10WorldVista']

    def create_folder(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def transform_artifacts(self):
        res = list()
        for file_name in self.HIPAA_List:
            artifacts = self.get_artifacts_from_xml(file_name)
            res.extend(artifacts)
        pd.DataFrame(res, columns=['id', 'doc']).to_csv(os.path.join(self.standard_path, 'target.csv'), index=False)

        source_artifacts = self.get_artifacts_from_xml('HIPAA')
        pd.DataFrame(source_artifacts, columns=['id', 'doc']).to_csv(os.path.join(self.standard_path, 'source.csv'), index=False)

    def transform_TL(self):
        res = list()
        for file_name in self.HIPAA_List:
            with open(os.path.join(self.dataset_dir, f'{file_name}.txt'), 'r') as f:
                content = f.readlines()
            for row in content:
                ids = re.split('[,\s\t\r]', row)
                ids = list(filter(lambda x: len(x) > 0, ids))
                s_id, t_ids = ids[0], ids[1:]
                for t_id in t_ids:
                    res.append([s_id.strip(), t_id.strip()])
        pd.DataFrame(res, columns=['s_id', 't_id']).to_csv(os.path.join(self.standard_path, 'result.csv'), index=False)

    def get_artifacts_from_xml(self, file_name):
        artifacts_branch='artifacts'
        artifact_branch='artifact'
        artifact_id='art_id'
        artifact_doc='art_title'

        res = list()
        xml_tree = ET.parse(os.path.join(self.dataset_dir, f'{file_name}.xml'))
        # xml_root = xml_tree.getroot()
        # artifacts = xml_root.find(artifacts_branch).findall(artifact_branch)
        artifacts = xml_tree.findall(artifact_branch)

        for artifact in artifacts:
            id_text = artifact.find(artifact_id).text.strip()
            doc_text = artifact.find(artifact_doc).text.strip()
            res.append([id_text, doc_text])

        return res

def main(datasets_dir):
    instance = Format(os.path.join(datasets_dir, 'HIPAA'))
    instance.transform_artifacts()
    instance.transform_TL()
    print('HIPAA ok')

if __name__ == '__main__':
    datasets_dir = ''
    main(datasets_dir)