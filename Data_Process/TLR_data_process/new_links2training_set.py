import os
import shutil
import pandas as pd
import numpy as np

'''After identifying unlabeled data, add new links to the training set'''
def add_data2train_data(data_type, add_data_path, work_data_dir):
    add_data_df = pd.read_csv(add_data_path)
    add_data_df.rename(columns={'s_id': 'issue_id', 't_id': 'commit_id'}, inplace=True)

    for group in range(1, 6):
        train_dir = work_data_dir + os.sep + str(group) + os.sep + 'train'
        link_df = pd.read_csv(os.path.join(train_dir, f'link_original_{group}.csv'))
        df = pd.concat([link_df, add_data_df])
        df.to_csv(os.path.join(train_dir, f'link_{data_type}_{group}.csv'), index=False)

def copy_original_data(work_data_dir):
    for group in range(1, 6):
        train_dir = work_data_dir + os.sep + str(group) + os.sep + 'train'
        link_original_path = os.path.join(train_dir, f'link_original_{group}.csv')
        if not os.path.exists(link_original_path):
            shutil.copy(os.path.join(train_dir, 'link.csv'), os.path.join(train_dir, f'link_original_{group}.csv'))

if __name__ == '__main__':
    TLR_data_dir = r'..\..\data\TLR'
    print('Processing...')
    # ----------Add the new links to the training set----------
    for dataset_name in ['flask','pgcli','keras']:
        for frac in [5, 20, 50, 80, 110]:
            standard_dir = os.path.join(TLR_data_dir, 'standard', dataset_name)
            work_data_dir = os.path.join(TLR_data_dir, 'work_data_by_frac', f'work_data_{frac}', dataset_name)
            output_dir = os.path.join(TLR_data_dir, 'standard', dataset_name, f'output_{frac}')

            copy_original_data(work_data_dir)

            vsm_labeling_result = os.path.join(output_dir, 'labeling_result_by_vsm.csv')
            cl_labeling_result = os.path.join(output_dir, 'labeling_result_by_cl.csv')
            random_labeling_result = os.path.join(output_dir, 'labeling_result_by_random.csv')

            add_data2train_data('vsm', vsm_labeling_result, work_data_dir)
            add_data2train_data('cl', cl_labeling_result, work_data_dir)
            add_data2train_data('random', random_labeling_result, work_data_dir)

    print('Done!')
