#!/usr/bin/env python3


import os
import requests
import zipfile
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import itertools
from itertools import combinations
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr


from TGB.tgb.utils.info import DATA_URL_DICT
from TGB.tgb.linkproppred.dataset import LinkPropPredDataset
from rc_utils import load_data_df, clever_sampler,calculate_edges_checked, plot_sampler_effectiveness, find_sequences_across_range_with_threshold, plot_recurrent_samples, relative_recurrency_comp


#try:
#    import cudf
#    from numba import cuda
#    CUDA_AVAILABLE = cuda.is_available()
#    print(f'CUDA Available!! {CUDA_AVAILABLE}')
#except ImportError:
#    CUDA_AVAILABLE = False
#    print('CUDA NOT available')


wd = '/Users/riccardoconci/Local_documents/ACS submissions/GeomDL/'
os.chdir(wd)



def run_py():

    #try:
    #    import cudf
    #    from numba import cuda
    #    CUDA_AVAILABLE = cuda.is_available()
    #except ImportError:
    #    CUDA_AVAILABLE = False

    #'tgbl-review', 'tgbl-coin','tgbl-coin', 'tgbl-comment', 'tgbl-flight'
    datasets = ['tgbl-coin','tgbl-comment' ] #'tgbl-comment', 'tgbl-coin']


    for dataset_name in datasets:

        print('Starting with ',dataset_name )

        print('Preparing dataset')
        dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
        data_df, train_df, val_df, test_df = load_data_df(dataset, dataset_name)
        
        file_name = dataset_name + '_time_diff_count_df.pkl'
        if os.path.exists(file_name):
            print(f'Loading {file_name}!!')
            with open(file_name, 'rb') as file:  
                time_diff_count_df = pickle.load(file)

        else:
            subset_df = data_df[data_df['count'] >= 4]
            subset_df = subset_df[subset_df['TimeDiff'] != 0].dropna()
            time_diff_count_df = subset_df.groupby('TimeDiff').size().reset_index(name='time_diff_count')
            time_diff_count_df.sort_values('time_diff_count', inplace=True, ascending=False)
            time_diff_count_df['cumulative_count'] = time_diff_count_df['time_diff_count'].cumsum()
            with open(file_name, 'wb') as file:  # 'wb' is correct here for write-binary
                pickle.dump(time_diff_count_df, file)
            print(f'{file_name} has been saved!')
        
        subset_df = data_df[data_df['count'] >= 4]
        subset_df = subset_df[subset_df['TimeDiff'] != 0].dropna()


        print('Sampling the potential recurrent periods')
        noise_value = 10
        sub_list_length = {
            'tgbl-wiki': 42,
            'tgbl-review': 12,
            'tgbl-coin': 30,
            'tgbl-comment': 30, 
            'tgbl-flight': 21
        }
        num_samples = sub_list_length[dataset_name]
        print('num_samples', num_samples)
        sub_list = clever_sampler(time_diff_count_df, noise_value, num_samples)

        #perc_time_diff, perc_edges = calculate_edges_checked(sub_list,noise_value, time_diff_count_df )
        #plot_sampler_effectiveness(sub_list, time_diff_seq, dataset_name)

        print('Calculating the recurrent events across the train, train_val and test using the refined time diffs')
        train_subset_df = subset_df[subset_df['train_mask'] == True]
        val_subset_df = subset_df[subset_df['val_mask'] == True]
        test_subset_df = subset_df[subset_df['test_mask'] == True]
        train_val_subset_df = pd.concat([train_subset_df, val_subset_df], axis=0)

        subsets_df_list = [train_subset_df, train_val_subset_df, test_subset_df]
        subset_names = ['_train', '_train_val', '_test']

        recurrent_df_list = []

        for i in range(len(subsets_df_list)):
            subset_df_extraction = subsets_df_list[i]
            subset_name = subset_names[i]

            print('Extracting periodic sequences in ' + subset_name)

            file_name = dataset_name + subset_name + '_recurrency_df.pkl'
            if os.path.exists(file_name):
                print(f'Loading {file_name}!!')
                with open(file_name, 'rb') as file: 
                    subset_df_recurrent = pickle.load(file)
                print(subset_df_recurrent.columns)
                recurrent_df_list.append(subset_df_recurrent)
                continue 

            min_consecutive_events = 4 
            if subset_name == '_train_val':
                timestamp_threshold = val_df['Timestamp'].min()
            else:
                timestamp_threshold = None

            print('timestamp_threshold', timestamp_threshold)
            
            subset_df_recurrent = find_sequences_across_range_with_threshold(subset_df_extraction, sub_list, noise_value, min_consecutive_events, timestamp_threshold)
            print(subset_df_recurrent.columns)
            recurrent_df_list.append(subset_df_recurrent)
            
            print('Saving the recurrent sequences df! ')
            file_name_save = dataset_name + subset_name + '_recurrency_df.pkl'
            with open( file_name_save , 'wb') as file: 
                pickle.dump(subset_df_recurrent, file) 
            
        combined_recurrent_df = pd.concat(recurrent_df_list)
        combined_recurrent_df = combined_recurrent_df.groupby(['Source', 'Destination'])['Timestamps'].agg(sum).apply(sorted).reset_index()
        combined_recurrent_df = combined_recurrent_df[['Source', 'Destination', 'Timestamps']]
        combined_recurrent_df = combined_recurrent_df.explode('Timestamps').reset_index(drop=True)
        combined_recurrent_df = combined_recurrent_df.rename(columns={'Timestamps': 'Timestamp'})


        file_name_save = dataset_name + '_combined_recurrency_df.pkl'
        with open( file_name_save , 'wb') as file: 
            pickle.dump(combined_recurrent_df, file) 

        #threshold = val_df['Timestamp'].min()
        #file_path = dataset_name + '_final_results.pkl'
        #results = relative_recurrency_comp(subset_df, combined_recurrent_df, threshold, file_path)
        #print(results)



if __name__ == '__main__':
    
    run_py()








