import os
import shutil
import chardet
import collections
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

'''
standard format
source.csv: columns=[id, doc]
target.csv: columns=[id, doc]
result.csv: columns=[s_id, t_id]
'''

class Format:

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.standard_path = os.path.join(dataset_dir, 'standard')
        self.create_folder(self.standard_path)

    def create_folder(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def common_run(self, source, target, result, path_flag):
        self.transform_artifacts_from_xml(file_name=source, save_name='source.csv', path_flag=path_flag)
        self.transform_artifacts_from_xml(file_name=target, save_name='target.csv', path_flag=path_flag)
        self.transform_TL_from_xml(file_name=result)


    def transform_artifacts_from_xml(self, file_name, save_name,
                            artifacts_branch='artifacts',
                            artifact_branch='artifact',
                            artifact_id='id',
                            artifact_doc='content',
                            path_flag=True):
        res = list()
        xml_tree = ET.parse(os.path.join(self.dataset_dir, f'{file_name}.xml'))
        xml_root = xml_tree.getroot()
        artifacts = xml_root.find(artifacts_branch).findall(artifact_branch)

        if path_flag:
            for artifact in artifacts:
                id_text = artifact.find(artifact_id).text.strip()
                doc_text = artifact.find(artifact_doc).text.strip()
                filepath = os.path.join(self.dataset_dir, doc_text)
                with open(filepath, 'rb') as f:
                    content = f.read()
                    encode = chardet.detect(content).get('encoding')
                    content = content.decode(encode)
                res.append([id_text, content.strip()])
        else:
            for artifact in artifacts:
                id_text = artifact.find(artifact_id).text.strip()
                doc_text = artifact.find(artifact_doc).text.strip()
                res.append([id_text, doc_text])

        df = pd.DataFrame(res, columns=['id', 'doc'])
        df.to_csv(os.path.join(self.standard_path, save_name), index=False)

    def transform_TL_from_xml(self, file_name,
                     links_branch='links',
                     link_branch='link',
                     source_id='source_artifact_id',
                     target_id='target_artifact_id'):
        res = list()
        xml_tree = ET.parse(os.path.join(self.dataset_dir, f'{file_name}.xml'))
        xml_root = xml_tree.getroot()
        links = xml_root.find(links_branch).findall(link_branch)

        for link in links:
            source = link.find(source_id).text.strip()
            target = link.find(target_id).text.strip()
            res.append([source, target])
        df = pd.DataFrame(res, columns=['s_id', 't_id'])
        df.to_csv(os.path.join(self.standard_path, 'result.csv'), index=False)

def main(datasets_dir):
    dataset_detail = {
        'Albergate': ['source_req', 'target_code', 'answer_req_code', True],
        'CCHIT': ['source2', 'target2', 'answer2', False],
        'CM1-NASA': ['CM1-sourceArtifacts', 'CM1-targetArtifacts', 'CM1-answerSet', False],
        'eANCI': ['source_req', 'target_code', 'answer_req_code', False],
        # 'EBT': ['source_Req', 'target_CC', 'answer_Req_CC', False],
        'eTOUR': ['source_req', 'target_code', 'answer_req_code', True],
        'GANNT': ['source_req', 'target_req', 'answer_req_req', True],
        'iTrust': ['source_req', 'target_code', 'answer_req_code', True],
        'SMOS': ['source_req', 'target_code', 'answer_req_code', True],
    }

    for dataset_name, deatil in dataset_detail.items():
        source, target, result, path_flag = deatil
        Format(os.path.join(datasets_dir, dataset_name)) \
            .common_run(source, target, result, path_flag)
        print(f'{dataset_name} ok')

    EBT = Format(os.path.join(datasets_dir, 'EBT'))
    EBT.transform_artifacts_from_xml(file_name='source_Req', save_name='source.csv', path_flag=False)
    EBT.transform_artifacts_from_xml(file_name='target_CC', save_name='target.csv', path_flag=True)
    EBT.transform_TL_from_xml(file_name='answer_Req_CC')
    print('EBT ok')

if __name__ == '__main__':
    datasets_dir = ''
    main(datasets_dir)
