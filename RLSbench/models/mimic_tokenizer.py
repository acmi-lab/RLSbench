import os
import pickle
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
import logging

import random

import numpy as np
import pandas as pd


logger = logging.getLogger("label_shift")

"""
Reference: https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sample_year(anchor_year_group):
    year_min = int(anchor_year_group[:4])
    year_max = int(anchor_year_group[-4:])
    assert year_max - year_min == 2
    return random.randint(year_min, year_max)


def assign_readmission_label(row):
    curr_subject_id = row.subject_id
    curr_admittime = row.admittime

    next_row_subject_id = row.next_row_subject_id
    next_row_admittime = row.next_row_admittime

    if curr_subject_id != next_row_subject_id:
        label = 0
    elif (next_row_admittime - curr_admittime).days > 15:
        label = 0
    else:
        label = 1

    return label


def diag_icd9_to_3digit(icd9):
    if icd9.startswith('E'):
        if len(icd9) >= 4:
            return icd9[:4]
        else:
            print(icd9)
            return icd9
    else:
        if len(icd9) >= 3:
            return icd9[:3]
        else:
            print(icd9)
            return icd9


def diag_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def diag_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + diag_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + diag_icd10_to_3digit(icd[6:])
    else:
        raise


def list_join(lst):
    return ' <sep> '.join(lst)


def proc_icd9_to_3digit(icd9):
    if len(icd9) >= 3:
        return icd9[:3]
    else:
        return icd9


def proc_icd10_to_3digit(icd10):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        return icd10


def proc_icd_to_3digit(icd):
    if icd[:4] == 'ICD9':
        return 'ICD9_' + proc_icd9_to_3digit(icd[5:])
    elif icd[:5] == 'ICD10':
        return 'ICD10_' + proc_icd10_to_3digit(icd[6:])
    else:
        raise


def process_mimic_data(data_dir):
    set_seed(seed=42)

    for file in ['patients.csv', 'diagnoses_icd.csv', 'procedures_icd.csv']:
        if not os.path.isfile(os.path.join(data_dir, file)):
            raise ValueError(f'Please download {file} to {data_dir}')

    # Patients
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    patients['real_anchor_year_sample'] = patients.anchor_year_group.apply(lambda x: sample_year(x))
    patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'real_anchor_year_sample']]
    patients = patients.dropna().reset_index(drop=True)
    admissions = pd.read_csv(os.path.join(data_dir, 'admissions.csv'))
    admissions['admittime'] = pd.to_datetime(admissions['admittime']).dt.date
    admissions = admissions[['subject_id', 'hadm_id', 'race', 'admittime', 'hospital_expire_flag']]
    admissions = admissions.dropna()
    admissions['mortality'] = admissions.hospital_expire_flag
    admissions = admissions.sort_values(by=['subject_id', 'hadm_id', 'admittime'])
    admissions['next_row_subject_id'] = admissions.subject_id.shift(-1)
    admissions['next_row_admittime'] = admissions.admittime.shift(-1)
    admissions['readmission'] = admissions.apply(lambda x: assign_readmission_label(x), axis=1)
    admissions = admissions[['subject_id', 'hadm_id', 'race', 'admittime', 'mortality', 'readmission']]
    admissions = admissions.dropna().reset_index(drop=True)

    # Diagnoses ICD
    diagnoses_icd = pd.read_csv(os.path.join(data_dir, 'diagnoses_icd.csv'))
    diagnoses_icd = diagnoses_icd.dropna()
    diagnoses_icd = diagnoses_icd.drop_duplicates()
    diagnoses_icd = diagnoses_icd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'])
    diagnoses_icd['icd_code'] = diagnoses_icd.apply(lambda x: f'ICD{x.icd_version}_{x.icd_code}', axis=1)
    diagnoses_icd['icd_3digit'] = diagnoses_icd.icd_code.apply(lambda x: diag_icd_to_3digit(x))
    diagnoses_icd = diagnoses_icd.groupby(['subject_id', 'hadm_id']).agg({'icd_3digit': list_join}).reset_index()
    diagnoses_icd = diagnoses_icd.rename(columns={'icd_3digit': 'diagnoses'})

    # Procedures ICD
    procedures_icd = pd.read_csv(os.path.join(data_dir, 'procedures_icd.csv'))
    procedures_icd = procedures_icd.dropna()
    procedures_icd = procedures_icd.drop_duplicates()
    procedures_icd = procedures_icd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'])
    procedures_icd['icd_code'] = procedures_icd.apply(lambda x: f'ICD{x.icd_version}_{x.icd_code}', axis=1)
    procedures_icd['icd_3digit'] = procedures_icd.icd_code.apply(lambda x: proc_icd_to_3digit(x))
    procedures_icd = procedures_icd.groupby(['subject_id', 'hadm_id']).agg({'icd_3digit': list_join}).reset_index()
    procedures_icd = procedures_icd.rename(columns={'icd_3digit': 'procedure'})

    # Merge
    df = admissions.merge(patients, on='subject_id', how='inner')
    df['real_admit_year'] = df.apply(lambda x: x.admittime.year - x.anchor_year + x.real_anchor_year_sample, axis=1)
    df['age'] = df.apply(lambda x: x.admittime.year - x.anchor_year + x.anchor_age, axis=1)
    df = df[['subject_id', 'hadm_id',
             'admittime', 'real_admit_year',
             'age', 'gender', 'race',
             'mortality', 'readmission']]
    df = df.merge(diagnoses_icd, on=['subject_id', 'hadm_id'], how='inner')
    df = df.merge(procedures_icd, on=['subject_id', 'hadm_id'], how='inner')
    df.to_csv(os.path.join(data_dir, 'data_preprocessed.csv'))

    # Cohort Selection
    processed_file = os.path.join(data_dir, 'processed_mimic_data.csv')
    df = df[df.age.apply(lambda x: (x >= 18) & (x <= 89))]
    df.to_csv(processed_file, index=False)
    return processed_file


class MIMICStay:

    def __init__(self,
                 icu_id,
                 icu_timestamp,
                 mortality,
                 readmission,
                 age,
                 gender,
                 race):
        self.icu_id = icu_id    # str
        self.icu_timestamp = icu_timestamp  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.race = race  # str

        self.diagnosis = []     # list of tuples (timestamp in min (int), diagnosis (str))
        self.treatment = []     # list of tuples (timestamp in min (int), treatment (str))

    def __repr__(self):
        return f'MIMIC ID-{self.icu_id}, mortality-{self.mortality}, readmission-{self.readmission}'


def get_stay_dict(save_dir):
    mimic_dict = {}
    input_path = process_mimic_data(save_dir)
    fboj = open(input_path)
    name_list = fboj.readline().strip().split(',')
    for eachline in fboj:
        t = eachline.strip().split(',')
        tempdata = {eachname: t[idx] for idx, eachname in enumerate(name_list)}
        mimic_value = MIMICStay(icu_id=tempdata['hadm_id'],
                                 icu_timestamp=tempdata['real_admit_year'],
                                 mortality=tempdata['mortality'],
                                 readmission=tempdata['readmission'],
                                 age=tempdata['age'],
                                 gender=tempdata['gender'],
                                 race=tempdata['race'])
        mimic_value.diagnosis = tempdata['diagnoses'].split(' <sep> ')
        mimic_value.treatment = tempdata['procedure'].split(' <sep> ')
        mimic_dict[tempdata['hadm_id']] = mimic_value

    pickle.dump(mimic_dict, open(os.path.join(save_dir, 'mimic_stay_dict.pkl'), 'wb'))
    
class Vocabulary(object):

    def __init__(self):
        self.word2idx = {'<pad>': 0, '<cls>': 1, '<unk>': 2}
        self.idx2word = {0: '<pad>', 1: '<cls>', 2: '<unk>'}
        assert len(self.word2idx) == len(self.idx2word)
        self.idx = len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def vocab_construction(all_words, output_filename):
    vocab = Vocabulary()
    for word in all_words:
        vocab.add_word(word)
    logger.info(f"Mimic Vocab len: {len(vocab)}")

    # sanity check
    assert set(vocab.word2idx.keys()) == set(vocab.idx2word.values())
    assert set(vocab.word2idx.values()) == set(vocab.idx2word.keys())
    for word in vocab.word2idx.keys():
        assert word == vocab.idx2word[vocab(word)]

    pickle.dump(vocab, open(output_filename, 'wb'))
    return

def build_vocab_mimic(vocab_dir):
    mimic_dict_path = os.path.join(vocab_dir, 'mimic_stay_dict.pkl') 
    if not os.path.exists(mimic_dict_path):
        logger.info(f"Dumping mimic_stay_dict at {mimic_dict_path} ...")
        get_stay_dict(vocab_dir)
    all_icu_stay_dict = pickle.load(open(mimic_dict_path,'rb'))
    all_codes = []
    for icu_id in all_icu_stay_dict.keys():
        for code in all_icu_stay_dict[icu_id].treatment:
            all_codes.append(code)
        for code in all_icu_stay_dict[icu_id].diagnosis:
            all_codes.append(code)
    all_codes = list(set(all_codes))
    vocab_construction(all_codes, os.path.join(vocab_dir, 'mimic_vocab.pkl'))
    
def to_index(sequence, vocab, prefix='', suffix=''):
    """ convert code to index """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    return sequence


class MIMICTokenizer:
    def __init__(self, data_dir):
        self.vocab_dir = os.path.join(data_dir, 'mimic')
        if not os.path.exists(os.path.join(self.vocab_dir, 'mimic_vocab.pkl')):
            build_vocab_mimic(self.vocab_dir)
        self.code_vocabs, self.code_vocabs_size = self._load_code_vocabs()
        self.type_vocabs, self.type_vocabs_size = self._load_type_vocabs()

    def _load_code_vocabs(self):

        vocabs = pickle.load(open(os.path.join(self.vocab_dir, 'mimic_vocab.pkl'), 'rb'))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_type_vocabs(self):
        vocabs = Vocabulary()
        for word in ['dx', 'tr']:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def get_code_vocabs_size(self):
        return self.code_vocabs_size

    def get_type_vocabs_size(self):
        return self.type_vocabs_size

    def __call__(self,
                 batch_codes: List[str],
                 batch_types: List[str],
                 padding=True,
                 prefix='<cls>',
                 suffix=''):

        # to tensor
        batch_codes = [torch.tensor(to_index(c, self.code_vocabs, prefix=prefix, suffix=suffix)) for c in batch_codes]
        batch_types = [torch.tensor(to_index(t, self.type_vocabs, prefix=prefix, suffix=suffix)) for t in batch_types]

        # padding
        if padding:
            batch_codes = pad_sequence(batch_codes, batch_first=True)
            batch_types = pad_sequence(batch_types, batch_first=True)

        return batch_codes, batch_types
