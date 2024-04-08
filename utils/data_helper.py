import sys
import re
import glob
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

IMG_DIM = 299


def process_img_path(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.xception.preprocess_input(img)
    img = cv2.resize(img, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_LINEAR)
    return img


def encode_one_hot(label, class_num):
    one_hot_arr = np.zeros(class_num, dtype=np.float32)
    if label < class_num:
        one_hot_arr[label] = 1.
    return one_hot_arr


def get_data_map():
    species_map = {}
    genus_dir = glob.glob('data/*/')
    pattern = re.compile(r'([^/]*)(?=/$)')
    for genus in genus_dir:
        genus_name = pattern.search(genus)
        species_dir = glob.glob(genus + '*/')

        for species in species_dir:
            species_name = pattern.search(species)
            gen_sp = ' '.join([genus_name.group(), species_name.group()])
            sub_species_dir = glob.glob(species + '*/')
            species_map[gen_sp] = species

            for sub_species in sub_species_dir:
                sub_species_name = pattern.search(sub_species)
                gen_sp_subsp = ' '.join(
                    [genus_name.group(), species_name.group(), sub_species_name.group()])
                species_map[gen_sp_subsp] = sub_species
    return species_map


def get_species_map(df):
    class_map = {}
    for i in range(len(df)):
        class_map[df.loc[i, 'Species']] = df.loc[i, 'Species_Name']
    return class_map


def rename_species(name):
    pattern = re.compile(r'([^ ]*)(?=$)')
    if pd.isnull(name):
        return name
    assert len(name.split(' ')) > 1
    species = pattern.search(name)
    new_name = name.replace(species.group(), name.replace(' ', '_'))
    return new_name


def row_to_species(split_df, unique_species):
    row_species_map = {}
    for i in range(len(split_df)):
        idx = False
        if not pd.isnull(split_df.loc[i, 'unknown']):
            row_species_map[i] = unique_species.index(
                split_df.loc[i, 'unknown'])
            idx = True
        if not pd.isnull(split_df.loc[i, 'genus']):
            if idx:
                raise ValueError('Resetting directory to species map')
            row_species_map[i] = unique_species.index(split_df.loc[i, 'genus'])
            idx = True
        if not pd.isnull(split_df.loc[i, 'species']):
            if idx:
                raise ValueError('Resetting directory to species map')
            row_species_map[i] = unique_species.index(
                split_df.loc[i, 'species'])
    return row_species_map


def split_helper(id_to_sample, ratio):
    df = pd.DataFrame(data=list(id_to_sample.keys()))
    train_list, val_list, test_list = np.split(
        df.sample(frac=1), [int(ratio[0]*len(df)), int((ratio[0]+ratio[1])*len(df))])
    return train_list, val_list, test_list


def join_list(id_to_sample, id_list):
    joined_list = []
    for id in id_list:
        joined_list.extend(id_to_sample[id])
    return joined_list


def split_img_list(img_list, ratio):
    id_to_sample = {}
    pattern = re.compile(r'(?<=-)\d*(?=_)')
    for img in img_list:
        id_match = pattern.search(img)
        id = id_match.group()
        if id in id_to_sample:
            id_to_sample[id].append(img)
        else:
            id_to_sample[id] = [img]
    train_list, val_list, test_list = split_helper(id_to_sample, ratio)
    try:
        train_list = join_list(id_to_sample, train_list[0])
    except:
        sys.exit(train_list[0])
    val_list = join_list(id_to_sample, val_list[0])
    test_list = join_list(id_to_sample, test_list[0])
    return train_list, val_list, test_list


def split_data(fold_num):
    split_df = pd.read_excel(
        'data/unk training - 1.29.2020 - reduced.xlsx', sheet_name=f'fold{fold_num}')
    split_df = split_df[split_df['ignore'] != 'YES'].reset_index(drop=True)
    split_df['genus'] = split_df['genus'].apply(rename_species)
    split_df['species'] = split_df['species'].apply(rename_species)
    return split_df


def extend_df(output_df, img_list, target_genus, target_species, split, genus_name, species_name):
    temp_df = pd.DataFrame(data=img_list, columns=['ID'])
    temp_df['Genus'] = target_genus
    temp_df['Species'] = target_species
    temp_df['Split'] = split
    temp_df['Genus_Name'] = genus_name
    temp_df['Species_Name'] = species_name
    temp_df = temp_df[output_df.columns]
    output_df = pd.concat([output_df, temp_df]).reset_index(drop=True)
    return output_df


def filter_value(df, col, values, include=False):
    if include:
        return df[df[col].isin(values)]
    else:
        return df[~df[col].isin(values)]


def get_unique_species(fold_num):
    if fold_num == 1:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_dorsalis',
            'aedes aedes_japonicus',
            'aedes aedes_sollicitans',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'culex culex_erraticus',
            'culex culex_pipiens_sl',
            'culex culex_salinarius',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 2:
        unique_species = [
            'aedes aedes_albopictus',
            'aedes aedes_dorsalis',
            'aedes aedes_japonicus',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_erraticus',
            'culex culex_salinarius',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 3:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_dorsalis',
            'aedes aedes_japonicus',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_erraticus',
            'culex culex_pipiens_sl',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 4:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_japonicus',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_vexans',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_erraticus',
            'culex culex_pipiens_sl',
            'culex culex_salinarius',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 5:
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_dorsalis',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'culex culex_pipiens_sl',
            'culex culex_salinarius',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']
    elif fold_num == 'big':
        unique_species = [
            'aedes aedes_aegypti',
            'aedes aedes_albopictus',
            'aedes aedes_atlanticus',
            'aedes aedes_canadensis',
            'aedes aedes_dorsalis',
            'aedes aedes_flavescens',
            'aedes aedes_infirmatus',
            'aedes aedes_japonicus',
            'aedes aedes_nigromaculis',
            'aedes aedes_sollicitans',
            'aedes aedes_taeniorhynchus',
            'aedes aedes_triseriatus',
            'aedes aedes_trivittatus',
            'aedes aedes_vexans',
            'anopheles anopheles_coustani',
            'anopheles anopheles_crucians',
            'anopheles anopheles_freeborni',
            'anopheles anopheles_funestus',
            'anopheles anopheles_gambiae',
            'anopheles anopheles_pseudopunctipennis',
            'anopheles anopheles_punctipennis',
            'anopheles anopheles_quadrimaculatus',
            'coquillettidia coquillettidia_perturbans',
            'culex culex_coronator',
            'culex culex_erraticus',
            'culex culex_nigripalpus',
            'culex culex_pipiens_sl',
            'culex culex_restuans',
            'culex culex_salinarius',
            'culiseta culiseta_incidens',
            'culiseta culiseta_inornata',
            'deinocerites deinocerites_cancer',
            'deinocerites deinocerites_cuba-1',
            'mansonia mansonia_titillans',
            'psorophora psorophora_ciliata',
            'psorophora psorophora_columbiae',
            'psorophora psorophora_cyanescens',
            'psorophora psorophora_ferox',
            'psorophora psorophora_pygmaea',
            'aedes aedes_spp',
            'anopheles anopheles_spp',
            'culex culex_spp',
            'psorophora psorophora_spp',
            'mosquito']

    if fold_num == 'big':
        unique_genus = [
            'aedes',
            'anopheles',
            'culex',
            'coquillettidia',
            'culiseta',
            'deinocerites',
            'mansonia',
            'psorophora',
            'mosquito'
        ]
    else:
        unique_genus = [
            'aedes',
            'anopheles',
            'culex',
            'psorophora',
            'mosquito'
        ]

    return unique_genus, unique_species
