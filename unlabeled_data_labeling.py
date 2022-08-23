import os
import random
import pandas as pd
import numpy as np
import collections

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")

class Labeling:

    def __init__(self, standard_dir):
        self.dataset_dir = standard_dir

    def run(self, count, frac, methods):
        self.output_dir = os.path.join(self.dataset_dir, f'output_{frac}')

        for m in methods:
            similarity_lists_dir = os.path.join(self.dataset_dir, 'Similarity_Lists')
            source_sim_list_path = os.path.join(similarity_lists_dir, f'source_similarity_list_by_{m}.csv')
            target_sim_list_path = os.path.join(similarity_lists_dir, f'target_similarity_list_by_{m}.csv')
            self.source_sim_list_df = pd.read_csv(source_sim_list_path)
            self.target_sim_list_df = pd.read_csv(target_sim_list_path)
            logger.info(f'----------frac:{frac}: use {m} sim list labeling----------')
            self.labeling(m, count, frac / 100, self.source_sim_list_df, self.target_sim_list_df)

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
        self.source = {id: document for id, document in source_df[['id', 'doc']].values}
        self.target = {id: document for id, document in target_df[['id', 'doc']].values}
        # all id
        self.source_ids = source_df.id.to_list()
        self.target_ids = target_df.id.to_list()
        # labeled id
        self.source_ids_in_labeled = list({}.fromkeys(result_df.s_id.to_list()).keys())
        self.target_ids_in_labeled = list({}.fromkeys(result_df.t_id.to_list()).keys())
        # unlabeled id
        self.source_ids_in_unlabeled = list({}.fromkeys([id for id in self.source_ids if id not in self.source_ids_in_labeled]).keys())
        self.target_ids_in_unlabeled = list({}.fromkeys([id for id in self.target_ids if id not in self.target_ids_in_labeled]).keys())

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

    def labeling(self, sim_func, original_link_count, frac, source_sim_df, target_sim_df):
        need_add_link_count = int(original_link_count * frac)
        logger.info(f'Start labeling, the number of original links: {original_link_count}, the threshold: {frac}, the number of new links to be added: {need_add_link_count}')

        # Predicted data for source and target artifacts
        source_sim_df = source_sim_df.copy()
        target_sim_df = target_sim_df.copy()
        source_sim_df.sort_values('pred', ascending=False, inplace=True)
        target_sim_df.sort_values('pred', ascending=False, inplace=True)
        # Since the id of the source artifact is a number,
        # every time the data is read from the csv file,
        # the data type should be converted to a string and modified according to its own data.
        source_sim_df = source_sim_df.astype({'id1': 'int', 'id2': 'int'})
        source_sim_df = source_sim_df.astype({'id1': 'str', 'id2': 'str'})

        '**********Generate traceability links from unlabeled data**********'
        logger.info('Generate traceability links from unlabeled data')
        link_from_source_data = list()  # Traceability links generated from similar source artifacts
        link_from_target_data = list()  # Traceability links generated from similar target artifacts
        similar_source_index, similar_target_index = 0, 0  # Index of similar pairs of source and target artifacts
        source_sim_pred, target_sim_pred = 1, 1  # Minimum similarity of source and target artifacts
        while need_add_link_count > 0:  # Exit the loop when the number of new links to be added is less than or equal to 0
            similar_source = source_sim_df.iloc[similar_source_index]  # Similar source artifacts
            similar_target = target_sim_df.iloc[similar_target_index]  # Similar target artifacts
            source_pred = float(similar_source['pred'])  # Similarity of similar source artifacts
            target_pred = float(similar_target['pred'])  # Similarity of similar target artifacts
            if source_pred > target_pred:  # Use source artifact kick to generate traceability links when similar source artifacts are more similar
                links = self.create_link_from_source(similar_source['id1'], similar_source['id2'])
                link_from_source_data.extend(links)
                similar_source_index += 1  # Source Artifact Similar Pair Index+1
                need_add_link_count -= len(links)  # The number of new links that need to be added minus the number of new links added in this loop
                source_sim_pred = source_pred
            elif source_pred < target_pred:  # Use source artifact kick to generate traceability links when similar target artifacts are more similar
                links = self.create_link_from_target(similar_target['id1'], similar_target['id2'])
                link_from_target_data.extend(links)
                similar_target_index += 1  # Target Artifact Artifact Similar Pair Index+1
                need_add_link_count -= len(links)  # The number of new links that need to be added minus the number of new links added in this loop
                target_sim_pred = target_pred
            else:
                assert (source_pred == target_pred)  # The similarity between the two workpieces is equal
                ls_f_s = self.create_link_from_source(similar_source['id1'], similar_source['id2'])
                ls_f_t = self.create_link_from_target(similar_target['id1'], similar_target['id2'])
                link_from_source_data.extend(ls_f_s)
                link_from_target_data.extend(ls_f_t)
                similar_source_index += 1  # Source Artifact Similar Pair Index+1
                similar_target_index += 1  # Target Artifact Artifact Similar Pair Index+1
                need_add_link_count = need_add_link_count - len(ls_f_s) - len(ls_f_t)  # The number of new links that need to be added minus the number of new links added in this loop
                source_sim_pred, target_sim_pred = source_pred, target_pred
        logger.info(f'The minimum source artifact similarity used for identification is: {source_sim_pred}')
        logger.info(f'The minimum target artifact similarity used for identification is: {target_sim_pred}')
        # deduplication
        link_from_source_data = set(link_from_source_data)
        link_from_target_data = set(link_from_target_data)
        # count
        link_from_source_num = len(link_from_source_data)
        link_from_target_num = len(link_from_target_data)
        logger.info(f'Added number of links via source artifact similarity:{link_from_source_num}')
        logger.info(f'Added number of links by target artifact similarity:{link_from_target_num}')
        # take union
        new_links = list(link_from_source_data.union(link_from_target_data))
        # deduplication
        new_links = list({}.fromkeys(new_links).keys())
        # Convert from id string to id list
        new_links = [id_str.split('_') for id_str in new_links]
        logger.info(f'Number of new tag links：{len(new_links)}')
        # save labeled link file
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        csv_path = os.path.join(self.output_dir, f'labeling_result_by_{sim_func}.csv')
        pd.DataFrame(new_links, columns=['s_id', 't_id']).to_csv(csv_path, index=False)

    def labeling_by_random(self, original_link_count, frac, source_sim_df, target_sim_df):
        need_add_link_count = int(original_link_count * frac)
        logger.info(f'Start labeling, the number of original links: {original_link_count}, the threshold: {frac}, the number of new links to be added: {need_add_link_count}')

        # Out-of-order prediction data
        source_sim_df = source_sim_df.copy()

        # Since the id of the source artifact is a number, every time the data is read from the csv file, the data type should be converted to a string and modified according to its own data.
        source_sim_df = source_sim_df.astype({'id1': 'int', 'id2': 'int'})
        source_sim_df = source_sim_df.astype({'id1': 'str', 'id2': 'str'})

        target_sim_df = target_sim_df.copy()
        source_sim_df['type'] = 'source'
        target_sim_df['type'] = 'target'
        df = pd.concat([source_sim_df, target_sim_df])
        df = df.sample(frac=1)

        # random links list
        index = 0
        link_from_source_num, link_from_target_num = 0, 0
        random_links = list()
        while need_add_link_count > 0:  # Exit the loop when the number of new links to be added is less than or equal to 0
            id1, id2, pred, type = df.iloc[index].values.tolist()
            if type == 'source':
                links = self.create_link_from_source(id1, id2)
                random_links.extend(links)
                link_from_source_num += len(links)
            elif type == 'target':
                links = self.create_link_from_target(id1, id2)
                random_links.extend(links)
                link_from_target_num += len(links)
            else:
                raise Exception('artifact type error')
            need_add_link_count -= len(links)
            index += 1
        # deduplication
        random_links = list({}.fromkeys(random_links).keys())
        # Convert from id string to id list
        random_links = [id_str.split('_') for id_str in random_links]
        logger.info(f'Add random links：{len(random_links)}')
        logger.info(f'Added number of links via source artifact similarity:{link_from_source_num}')
        logger.info(f'Added number of links by target artifact similarity:{link_from_target_num}')
        # save labeled link file
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        random_csv_path = os.path.join(self.output_dir, 'labeling_result_by_random.csv')
        pd.DataFrame(random_links, columns=['s_id', 't_id']).to_csv(random_csv_path, index=False)

    def create_link_from_source(self, id1, id2):
        links = list()
        # All target artifact ids linked to source artifact id1
        s_id = id2
        for t_id in self.s_to_t.get(id1, []):
            id_str = f'{s_id}_{t_id}'
            if id_str not in self.result_str:
                links.append(id_str)
        # All target artifact ids linked to source artifact id2
        s_id = id1
        for t_id in self.s_to_t.get(id2, []):
            id_str = f'{s_id}_{t_id}'
            if id_str not in self.result_str:
                links.append(id_str)
        return links


    def create_link_from_target(self, id1, id2):
        links = list()
        # All source artifact ids linked to target artifact id1
        t_id = id2
        for s_id in self.t_to_s.get(id1, []):
            id_str = f'{s_id}_{t_id}'
            if id_str not in self.result_str:
                links.append(id_str)
        # All source artifact ids linked to target artifact id2
        t_id = id1
        for s_id in self.t_to_s.get(id2, []):
            id_str = f'{s_id}_{t_id}'
            if id_str not in self.result_str:
                links.append(id_str)
        return links

def main(TLR_data_dir):
    train_data_count = {
        'flask': 480,
        'pgcli': 338,
        'keras': 352,
    }
    methods = ['vsm', 'cl']
    for dataset_name, count in train_data_count.items():
        logger.info(f'---{dataset_name} start---')
        standard_dir = os.path.join(TLR_data_dir, 'standard', dataset_name)
        labeling = Labeling(standard_dir)
        labeling.load_data()
        for frac in [110, 80, 50, 20, 5]:
            labeling.run(count, frac, methods)

if __name__ == '__main__':
    TLR_data_dir = r'data\TLR'
    main(TLR_data_dir)


