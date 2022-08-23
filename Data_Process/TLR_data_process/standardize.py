import os
import re
import shutil
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

'''Convert data to standard format and perform 5-fold stratified cross-validation份'''
class Process:

    def __init__(self, TLR_data_dir):
        self.TLR_data_dir = TLR_data_dir
        self.dataset_dir_list = {
            'flask': self.TLR_data_dir + os.sep + 'pallets' + os.sep + 'flask',
            'pgcli': self.TLR_data_dir + os.sep + 'dbcli' + os.sep + 'pgcli',
            'keras': self.TLR_data_dir + os.sep + 'keras-team' + os.sep + 'keras'
        }

    def create_folder(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def run(self):
        standard_dir = os.path.join(self.TLR_data_dir, 'standard')
        if not os.path.exists(standard_dir):
            os.mkdir(standard_dir)

        work_data_dir = os.path.join(self.TLR_data_dir, 'work_data')
        if not os.path.exists(standard_dir):
            os.mkdir(standard_dir)

        # Convert standard format data
        for dataset_name, dataset_dir in self.dataset_dir_list.items():
            dataset_in_standard = os.path.join(standard_dir, dataset_name)
            self.create_folder(dataset_in_standard)
            self.formatting2standard(dataset_dir, dataset_in_standard)

        # split_TLR_data_with_cross_validation
        for dataset_name, dataset_dir in self.dataset_dir_list.items():
            dataset_in_work_data = os.path.join(work_data_dir, dataset_name)
            self.create_folder(dataset_in_work_data)
            self.split_TLR_data_with_cross_validation(dataset_dir, dataset_in_work_data)

    def formatting2standard(self, dataset_dir, standard_dir):
        source_df = pd.read_csv(dataset_dir + os.sep + 'issue.csv')
        target_df = pd.read_csv(dataset_dir + os.sep + 'commit.csv')
        result_df = pd.read_csv(dataset_dir + os.sep + 'clean_link.csv')

        source_df.fillna('', inplace=True)
        source_df['doc'] = source_df['issue_desc'] + ' ' + source_df['issue_comments']
        source_df.rename(columns={'issue_id': 'id'}, inplace=True)
        source_df = source_df[['id', 'doc']]
        source_df.dropna(axis=0, inplace=True)

        target_df.fillna('', inplace=True)
        target_df['doc'] = target_df['summary'] + ' ' + target_df['diff']
        target_df.rename(columns={'commit_id': 'id'}, inplace=True)
        target_df = target_df[['id', 'doc']]
        target_df.dropna(axis=0, inplace=True)

        result_df.rename(columns={'issue_id': 's_id', 'commit_id': 't_id'}, inplace=True)
        result_df = result_df[['s_id', 't_id']]
        result_df.dropna(axis=0, inplace=True)

        source_df = source_df.astype({'id': 'int'})
        source_df = source_df.astype({'id': 'str'})
        result_df = result_df.astype({'s_id': 'int'})
        result_df = result_df.astype({'s_id': 'str'})

        source_df.to_csv(os.path.join(standard_dir, 'source.csv'), index=False)
        target_df.to_csv(os.path.join(standard_dir, 'target.csv'), index=False)
        result_df.to_csv(os.path.join(standard_dir, 'result.csv'), index=False)

    def split_TLR_data_with_cross_validation(self, dataset_dir, work_data_dir):
        shutil.copy(os.path.join(dataset_dir, 'issue.csv'), os.path.join(work_data_dir, 'issue.csv'))
        shutil.copy(os.path.join(dataset_dir, 'commit.csv'), os.path.join(work_data_dir, 'commit.csv'))

        link_df = pd.read_csv(os.path.join(dataset_dir, 'clean_link.csv'))
        link_df.drop(['Unnamed: 0'], inplace=True, axis=1)
        link_df['label'] = 1

        X = link_df[['issue_id', 'commit_id']]
        Y = link_df[['label']]

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)

        num = 1
        for train, test in kfold.split(X, Y):
            dataset_group_dir = os.path.join(work_data_dir, str(num))
            train_dir = os.path.join(dataset_group_dir, 'train')
            valid_dir = os.path.join(dataset_group_dir, 'valid')
            test_dir = os.path.join(dataset_group_dir, 'test')
            os.mkdir(dataset_group_dir)
            os.mkdir(train_dir)
            os.mkdir(valid_dir)
            os.mkdir(test_dir)

            test_df = X.iloc[test]
            X_train, X_valid, Y_train, Y_valid = train_test_split(X.iloc[train], Y.iloc[train], test_size=0.2)

            X_train.to_csv(os.path.join(train_dir, 'link.csv'), index=False)
            X_valid.to_csv(os.path.join(valid_dir, 'link.csv'), index=False)
            test_df.to_csv(os.path.join(test_dir, 'link.csv'), index=False)

            num += 1

'''Preprocessing Software Artifacts Text'''
def data_preprocess(dataset_dir):
    issue_path = os.path.join(dataset_dir, 'issue.csv')
    commit_path = os.path.join(dataset_dir, 'commit.csv')

    issue_df = pd.read_csv(issue_path)
    commit_df = pd.read_csv(commit_path)

    issue_df['issue_desc'] = issue_df['issue_desc'].apply(iss_desc)
    issue_df['issue_comments'] = issue_df['issue_comments'].apply(iss_comments)
    issue_df.drop(['Unnamed: 0'], inplace=True, axis=1)

    commit_df['summary'] = commit_df['summary'].apply(cm_summary)
    commit_df['diff'] = commit_df['diff'].apply(cm_diff)
    commit_df.drop(['Unnamed: 0'], inplace=True, axis=1)

    os.rename(issue_path, os.path.join(dataset_dir, 'issue_original.csv'))
    os.rename(commit_path, os.path.join(dataset_dir, 'commit_original.csv'))
    issue_df.to_csv(os.path.join(dataset_dir, 'issue.csv'), index=False)
    commit_df.to_csv(os.path.join(dataset_dir, 'commit.csv'), index=False)

def iss_desc(desc):
    if pd.isnull(desc):
        desc = ""
    desc = re.sub("<!-.*->", "", desc)
    desc = re.sub("```.*```", "", desc, flags=re.DOTALL)
    desc = " ".join(word_tokenize(desc))
    return desc

def iss_comments(comments):
    return " ".join(word_tokenize(comments.split("\n")[0]))  # use only the first comment (title)

def cm_diff(diff):
    diff = diff.replace('{','[').replace('}',']')
    diff_sents = eval(diff)
    # if len(diff_sents) < 5:
    #     return
    diff_tokens = []
    for sent in diff_sents:
        sent = sent.strip("+- ")
        diff_tokens.extend(word_tokenize(sent))
    diff = " ".join(diff_tokens)
    return diff

def cm_summary(summary):
    return " ".join(word_tokenize(summary))

'''To save space, add and delete artifact text data to each folder'''
def copy_artifacts(dataset_dir):
    issue_path = os.path.join(dataset_dir, 'issue.csv')
    commit_path = os.path.join(dataset_dir, 'commit.csv')
    for group in range(1, 6):
        dataset_group_dir = os.path.join(dataset_dir, str(group))
        train_dir = os.path.join(dataset_group_dir, 'train')
        valid_dir = os.path.join(dataset_group_dir, 'valid')
        test_dir = os.path.join(dataset_group_dir, 'test')
        for path in [train_dir, valid_dir, test_dir]:
            shutil.copy(issue_path, os.path.join(path, 'issue.csv'))
            shutil.copy(commit_path, os.path.join(path, 'commit.csv'))

def del_artifacts(dataset_dir):
    for group in range(1, 6):
        dataset_group_dir = os.path.join(dataset_dir, str(group))
        train_dir = os.path.join(dataset_group_dir, 'train')
        valid_dir = os.path.join(dataset_group_dir, 'valid')
        test_dir = os.path.join(dataset_group_dir, 'test')
        for path in [train_dir, valid_dir, test_dir]:
            os.remove(os.path.join(path, 'issue.csv'))
            os.remove(os.path.join(path, 'commit.csv'))

def copy_work_data(TLR_data_dir):
    work_data_dir = os.path.join(TLR_data_dir, 'work_data')
    work_data_by_frac_dir = os.path.join(TLR_data_dir, 'work_data_by_frac')
    shutil.rmtree(work_data_by_frac_dir, ignore_errors=True)
    os.mkdir(work_data_by_frac_dir)
    for frac in [5, 20, 50, 80, 110]:
        frac_dir = os.path.join(work_data_by_frac_dir, f'work_data_{frac}')
        shutil.copytree(work_data_dir, frac_dir)

if __name__ == '__main__':
    TLR_data_dir = r'..\..\data\TLR'
    print('Processing...')
    # ----------Convert Standard Format and Cross Validation----------
    Process(TLR_data_dir).run()

    # ----------Artifact Text Preprocessing----------
    for dataset_name in ['flask', 'pgcli', 'keras']:
        dataset_dir = os.path.join(TLR_data_dir, 'work_data', dataset_name)
        data_preprocess(dataset_dir)

    # ----------Add remove artifact text to each folder----------
    for dataset_name in ['flask', 'pgcli', 'keras']:
        dataset_dir = os.path.join(TLR_data_dir, 'work_data', dataset_name)
        copy_artifacts(dataset_dir)
        # del_artifacts(dataset_dir)

    # ----------copy work_data by frac----------
    copy_work_data(TLR_data_dir)
    print('Done!')

