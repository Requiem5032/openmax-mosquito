import glob
import math
import tensorflow as tf
import numpy as np
import pandas as pd

from utils.data_helper import *


class JHUDataset(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))

        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return np.asarray(batch_x), np.asarray(batch_y)

    def get_all(self):
        return np.asarray(self.x), np.asarray(self.y)
    
    def get_class_num(self):
        return len(np.unique(np.asarray(self.y), axis=0))


def prepare_data(fold_num, ratio=[0.7, 0.15, 0.15]):
    split_df = split_data(fold_num)
    unique_genus, unique_species = get_unique_species(fold_num)
    species_map = get_data_map()
    row_species_map = row_to_species(split_df, unique_species)

    output_df = pd.DataFrame(
        columns=['ID', 'Genus', 'Species', 'Split', 'Genus_Name', 'Species_Name'])

    for i in range(len(split_df)):
        source_path = species_map[split_df.loc[i, 'folder']]
        img_list = glob.glob(source_path + '*m.jpg')
        target_species = row_species_map[i]
        target_genus = unique_genus.index(
            unique_species[target_species].split(' ')[0])
        genus_name = unique_genus[target_genus]
        if genus_name == 'mosquito':
            species_name = 'mosquito'
        else:
            species_name = unique_species[target_species].split(' ')[1]

        if split_df.loc[i, 'fold'] == 'tr/v/t':
            train_list, val_list, test_list = split_img_list(img_list, ratio)
            output_df = extend_df(output_df, train_list, target_genus,
                                  target_species, 'train', genus_name, species_name)
            output_df = extend_df(output_df, val_list, target_genus,
                                  target_species, 'val', genus_name, species_name)
            output_df = extend_df(output_df, test_list, target_genus,
                                  target_species, 'test', genus_name, species_name)
        else:
            output_df = extend_df(output_df, img_list, target_genus,
                                  target_species, 'ext_test', genus_name, species_name)
    return output_df


def validate_split(split_df, species_map):
    for i in range(len(split_df)):
        unk = split_df.loc[i, 'unknown']
        genus_known = split_df.loc[i, 'genus']
        species_known = split_df.loc[i, 'species']
        assert pd.isnull(unk) + pd.isnull(genus_known) + \
            pd.isnull(species_known) == 2, f'Error at row {i}'
        assert split_df.loc[i, 'folder'] in species_map, 'Error at row {}, {} not found'.format(
            i, split_df.loc[i, 'folder'])
        file_num = len(
            glob.glob(species_map[split_df.loc[i, 'folder']] + '*m.*'))
        if file_num != split_df.loc[i, 'm']:
            print('Index {}, files found in {}: {}, files according to split: {}'.format(
                i, species_map[split_df.loc[i, 'folder']], file_num, split_df.loc[i, 'm']))


def create_dataset(df, batch_size, mode):
    class_num = len(get_species_map(df))
    df_known = filter_value(df, 'Species', range(class_num-5, class_num), include=False
                            ).reset_index(drop=True)
    class_num_known = len(get_species_map(df_known))

    if mode == 'train':
        output_df = filter_value(
            df_known, 'Split', ['train'], include=True).reset_index(drop=True)
    elif mode == 'val':
        output_df = filter_value(
            df_known, 'Split', ['val'], include=True).reset_index(drop=True)
    elif mode == 'test':
        output_df = filter_value(
            df_known, 'Split', ['test'], include=True).reset_index(drop=True)
    elif mode == 'test_unknown':
        output_df = filter_value(
            df, 'Split', ['test'], include=True).reset_index(drop=True)
    else:
        return None

    output_df = pd.DataFrame(data={
        'img': output_df['ID'],
        'label': output_df['Species'],
    })

    output_df['img'] = output_df['img'].map(lambda x: process_img_path(x))
    output_df['label'] = output_df['label'].map(
        lambda x: encode_one_hot(x, class_num_known))

    img_list = []
    label_list = []

    for i in range(len(output_df)):
        img_list.append(output_df['img'][i])
    for i in range(len(output_df)):
        label_list.append(output_df['label'][i])

    img_list = np.asarray(img_list, dtype=np.float32)
    label_list = np.asarray(label_list, dtype=np.float32)

    output_ds = JHUDataset(x=img_list, y=label_list, batch_size=batch_size)
    return output_ds
