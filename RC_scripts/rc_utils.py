import os
import pandas as pd
import pickle 
import random 
import matplotlib.pyplot as plt
import numpy as np


try:
    import cudf
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
    print('CUDA Available!!')
except ImportError:
    CUDA_AVAILABLE = False
    print('CUDA NOT available')

def check_gpu_memory():
    if CUDA_AVAILABLE:
        device = cuda.get_current_device()
        total_memory = device.total_memory
        free_memory = device.get_memory_info().free
        print(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
        print(f"Free GPU Memory: {free_memory / (1024**3):.2f} GB")
    else:
        print("CUDA is not available. Operations will run on the CPU.")



def load_data_df(dataset, dataset_name):

    file_name = dataset_name + '_data_df.pkl'

    data = dataset.full_data  #a dictioinary stores all the edge data
    
    if os.path.exists(file_name):
        print(f'Loading {file_name}!!')
        with open(file_name, 'rb') as file:
            data_df = pickle.load(file)
    else:
        data_df = pd.DataFrame([data['sources'], data['destinations'], data['timestamps']]).T
        data_df.columns = ['Source', 'Destination', 'Timestamp']

        print(f'Saving {file_name}!!')
        with open( dataset_name + '_data_df.pkl', 'wb') as file: 
            pickle.dump(data_df, file) 


    data_df['AdjustedTimeStamp'] = data_df['Timestamp'] - data_df['Timestamp'].min()
    data_df['DateTime'] = pd.to_datetime(data_df['Timestamp'], unit='s')

    #print('Finding out the most common time differences')
    data_df.sort_values(by=['Source', 'Destination', 'Timestamp'], inplace=True)
    data_df['TimeDiff'] = data_df.groupby(['Source', 'Destination'])['Timestamp'].diff()
    data_df['cumulative_count'] = data_df.groupby(['Source', 'Destination']).cumcount() + 1
    data_df['count'] = data_df.groupby(['Source', 'Destination'])['Source'].transform('count')

    # Assuming 'dataset_df' is your DataFrame and 'dataset' contains your masks
    data_df['train_mask'] = dataset.train_mask
    data_df['val_mask'] = dataset.val_mask
    data_df['test_mask'] = dataset.test_mask

    # Split the DataFrame into training, validation, and test sets
    train_df = data_df[data_df['train_mask']]
    val_df = data_df[data_df['val_mask']]
    test_df = data_df[data_df['test_mask']]


    return data_df, train_df, val_df, test_df


def clever_sampler(time_diff_count_df, noise_value, final_length):
    # Sort time_diff_count_df by 'time_diff_count' to get the top 10 most common time diffs
    top_common_time_diffs = time_diff_count_df.sort_values(by='time_diff_count', ascending=False).head(15)['TimeDiff'].tolist()
    
    # Initialize 'explored' with the top 10 most common time diffs
    explored = top_common_time_diffs.copy()

    # Generate the initial 'forbidden' ranges based on these top common time diffs
    forbidden = [(val - (noise_value/100)*val, val + (noise_value/100)*val) for val in explored]

    sequence = time_diff_count_df['TimeDiff'].sort_values(ascending=True).tolist()
    attempts = 0
    max_attempts = 1000
    total_count = time_diff_count_df['time_diff_count'].sum()

    while len(explored) < final_length and attempts < max_attempts:
        r = np.random.choice(time_diff_count_df['TimeDiff'], p=time_diff_count_df['time_diff_count']/total_count)
        if all([r < lower or r > upper for lower, upper in forbidden]):
            explored.append(r)
            upper = r + (noise_value/100)*r
            lower = r - (noise_value/100)*r
            selected_values = [x for x in sequence if lower <= x <= upper]
            
            for val in selected_values:
                if (val - (noise_value/100)*val, val + (noise_value/100)*val) not in forbidden:
                    forbidden.append((val - (noise_value/100)*val, val + (noise_value/100)*val))
        attempts += 1
    return np.sort(explored)


def is_in_any_range(number, ranges):
        # Function to check if a number falls within any range in a list of ranges
        return any(lower <= number <= upper for lower, upper in ranges)

def calculate_edges_checked(sub_list,noise_value, time_diff_count_df ):
    # Calculate the enhanced (±10%) ranges for each value in the first list
    noise_frac = noise_value/100
    enhanced_ranges = [(x - noise_frac * x, x + noise_frac * x) for x in sub_list]
    #print(enhanced_ranges)
    # Function to check if a number falls within any range in a list of ranges
    def is_in_any_range(number, ranges):
        return any(lower <= number <= upper for lower, upper in ranges)

    time_diff_seq = time_diff_count_df['TimeDiff']
    # Collect the values in the second list that fall within any of the enhanced ranges
    included_values = [x for x in time_diff_seq if is_in_any_range(x, enhanced_ranges)]

    #print(f"Values from the second list included in the enhanced first list: {included_values}")

    perc_time_diffs = len(included_values)/len(time_diff_seq) * 100
    #print(perc_time_diffs)

    filtered_df = time_diff_count_df[time_diff_count_df['TimeDiff'].isin(included_values)]
    num_edges_checked = filtered_df['time_diff_count'].sum()
    perc_edges = num_edges_checked / total_count*100
    #print(perc_edges)

    return(perc_time_diffs, perc_edges)



def find_sequences_optimized(df, n, noise_percent, k, timestamp_threshold):
    df = df.copy()
    if CUDA_AVAILABLE:
        print("CUDA available, utilizing GPU for computation.")
        print("Before converting to CuDF:")
        check_gpu_memory()  # Optional, for demonstrating GPU memory usage
        df = cudf.from_pandas(df)
        print("After converting to CuDF:")
        check_gpu_memory()
    else:
        print('Not using GPU!')

    noise_value = n * noise_percent / 100
    df.sort_values(by=['Source', 'Destination', 'Timestamp'], inplace=True)
    
    # Calculate time differences
    df['Time_Diff'] = df.groupby(['Source', 'Destination'])['Timestamp'].diff()
    within_noise = ((df['Time_Diff'] >= n - noise_value) & (df['Time_Diff'] <= n + noise_value)) | df['Time_Diff'].isna()
    
    # Sequence identification
    df['Sequence_Break'] = (~within_noise).cumsum()
    df['Group_ID'] = df.groupby(['Source', 'Destination', 'Sequence_Break']).ngroup()
    
    # Filter potential sequences
    
    
    if timestamp_threshold != None:
        potential_sequences = df[df['Timestamp'] > timestamp_threshold - n * k]
        # Ensure sequences span the threshold and meet minimum count
        valid_groups = potential_sequences.groupby('Group_ID').filter(
            lambda x: x['Timestamp'].lt(timestamp_threshold).any() and 
                    x['Timestamp'].gt(timestamp_threshold).any() and
                    len(x) >= k
        )['Group_ID'].unique()
    else:
        valid_groups = df.groupby('Group_ID').filter(
            lambda x:len(x) >= k )['Group_ID'].unique()

    #print("Valid Group IDs:", valid_groups)
    df_filtered = df[df['Group_ID'].isin(valid_groups)]
    
    # Aggregate results without N and Noise_Percentage
    results = df_filtered.groupby(['Source', 'Destination', 'Group_ID']).agg(
        Number_of_Events=('Timestamp', 'size'),
        Timestamps=('Timestamp', list)
    ).reset_index()

    # Add N and Noise_Percentage to the results DataFrame
    results['N'] = n
    results['Noise_value'] = noise_value
    
    # Drop the Group_ID column if it's not needed in the final output
    results.drop(columns='Group_ID', inplace=True)

    if CUDA_AVAILABLE:
         return results.to_pandas()
    else:
        return results
    


def find_sequences_across_range_with_threshold(df, n_range, noise_percent, k, timestamp_threshold):
    all_results = []

    for n in n_range:
        print(n)
        result_df = find_sequences_optimized(df, n, noise_percent, k, timestamp_threshold)
        if not result_df.empty:
            all_results.append(result_df)
    
    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
    else:
        final_results_df = pd.DataFrame()
    
    return final_results_df




def relative_recurrency_comp(subset_df, combined_recurrent_df, val_threshold, file_path):
    results = {}

    # Convert 'Timestamp' in combined_recurrent_df to list for compatibility in comparisons
    combined_recurrent_df['Timestamp'] = combined_recurrent_df['Timestamp'].apply(lambda x: [x])
    
    # Concatenate and sort timestamps for recurrent interactions, group by 'Source' and 'Destination'
    combined_recurrent = combined_recurrent_df.groupby(['Source', 'Destination'])['Timestamp'].sum().apply(sorted).reset_index()

    # Calculate total counts
    total_edges = len(subset_df)
    total_pairs = subset_df.groupby(['Source', 'Destination']).ngroups
    total_recurrent_edges = len(combined_recurrent_df)
    total_recurrent_pairs = combined_recurrent.groupby(['Source', 'Destination']).ngroups

    # Calculate recurrent interactions for each category
    training_only = subset_df[subset_df.groupby(['Source', 'Destination'])['Timestamp'].transform('max') < val_threshold]
    train_and_val = subset_df.groupby(['Source', 'Destination']).filter(lambda x: x['Timestamp'].min() < val_threshold and x['Timestamp'].max() > val_threshold)
    test_only = subset_df[subset_df.groupby(['Source', 'Destination'])['Timestamp'].transform('min') > val_threshold]

    # Identify recurrent interactions within each category
    training_only_recurrent = pd.merge(training_only, combined_recurrent, on=['Source', 'Destination'], how='inner')
    train_and_val_recurrent = pd.merge(train_and_val, combined_recurrent, on=['Source', 'Destination'], how='inner')
    test_only_recurrent = pd.merge(test_only, combined_recurrent, on=['Source', 'Destination'], how='inner')

    # Calculate percentages
    results['% recurrent interactions overall'] = total_recurrent_edges / total_edges * 100
    results['% recurrent interactions only in training'] = len(training_only_recurrent) / max(len(training_only), 1) * 100
    results['% recurrent interactions in train to val'] = len(train_and_val_recurrent) / max(len(train_and_val), 1) * 100
    results['% recurrent interactions only in test'] = len(test_only_recurrent) / max(len(test_only), 1) * 100

    # Calculate percentages for unique source-destination pairs
    results['% recurrent pairs overall'] = total_recurrent_pairs / total_pairs * 100
    results['% recurrent pairs only in training'] = training_only_recurrent.groupby(['Source', 'Destination']).ngroups / max(training_only.groupby(['Source', 'Destination']).ngroups, 1) * 100
    results['% recurrent pairs in train to val'] = train_and_val_recurrent.groupby(['Source', 'Destination']).ngroups / max(train_and_val.groupby(['Source', 'Destination']).ngroups, 1) * 100
    results['% recurrent pairs only in test'] = test_only_recurrent.groupby(['Source', 'Destination']).ngroups / max(test_only.groupby(['Source', 'Destination']).ngroups, 1) * 100

    # Pickle the results dictionary
    with open(file_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results



def plot_sampler_effectiveness(sub_list, time_diff_seq, dataset_name):

    # Calculate the "umbrella" range as +/- 10% for the first list
    enhanced_ranges = [(x - 0.1 * x, x + 0.1 * x) for x in sub_list]

    # Plotting the first list with their umbrellas at y-value=1, horizontally
    plt.figure(figsize=(8, 6))
    y_value_first = np.ones(len(sub_list))
    plt.errorbar(sub_list, y_value_first, xerr=[0.1 * x for x in sub_list], fmt='o', ecolor='orange', elinewidth=3, capsize=0, label='Sampled Time Diffs ±10%', linestyle="None")

    # Plotting the second list as short vertical lines at y-value=0.95, color-coded based on condition
    for x in time_diff_seq:
        # Determine color based on whether the value is within any of the enhanced ranges of the first list
        color = 'green' if is_in_any_range(x, enhanced_ranges) else 'pink'
        plt.vlines(x, 0.925, 0.975, colors=color, linewidths=1.5)  # Use thicker lines for better visibility

    # Customize the plot
    #plt.xticks(sub_list + time_diff_seq, rotation=45)
    plt.yticks([1], ['Sampled Time Diffs'])
    #plt.yticks([0], ['Time Diff data'])
    plt.xlabel('Time Diffs')
    plt.title(f'Selected Time Diff values from {dataset_name} dataset using sampling method')
    plt.ylim(0.90, 1.05)  # Adjust y-limits to zoom in on the area of interest
    plt.xticks([])
    plt.legend()
    plt.tight_layout()  # Adjust the layout
    save_title = dataset_name + '_sampled_time_diffs' 
    plt.savefig(save_title + '.jpg')


def plot_recurrent_samples(final_df, dataset_name, subset_name, validation_timestamp ):

    results_df = final_df.copy()
    if len(final_df)> 200:
        subset = random.sample(range(len(final_df) ), 200)
        results_df = results_df.iloc[subset]
        

    # Generate a unique identifier for each source-destination pair for plotting
    results_df['Source_Destination'] = results_df['Source'].astype(str) + "-" + results_df['Destination'].astype(str)
    unique_pairs = results_df['Source_Destination'].unique()
    pair_mapping = {pair: i for i, pair in enumerate(unique_pairs)}

    # Prepare data for event plot
    events = []
    for _, row in results_df.iterrows():
        pair_index = pair_mapping[row['Source_Destination']]
        timestamps = row['Timestamps']
        events.append((pair_index, timestamps))

    # Plot
    fig, ax = plt.subplots()
    for pair_index, timestamps in events:
        ax.eventplot(timestamps, lineoffsets=pair_index, linelengths=0.8, color='blue')

    # Set the y-axis labels to the source-destination pairs
    ax.set_yticks(list(pair_mapping.values()))
    ax.set_yticklabels(list(pair_mapping.keys()))

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Source-Destination Pairs')
    ax.set_title('Event Plot of Source-Destination Pairs')

    # Add a vertical line at the specified timestamp value
    if subset_name == '_train_val':
        print('validation_timestamp', validation_timestamp)
        if validation_timestamp != None:
            ax.axvline(x=validation_timestamp, color='Green', linestyle='--', label=f'Threshold: {validation_timestamp}')
    #ax.axvline(x=validation_timestamp- median_context_window_32, color='Green', linestyle='--', label=f'Context_window')
    ax.legend()
    
    save_title = dataset_name + '_recurrent_SD' + subset_name
    plt.savefig(save_title + '.jpg')
