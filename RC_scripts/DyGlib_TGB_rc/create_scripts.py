import itertools

models = ['DyRep', 'TGAT', 'GraphMixer', 'DyGFormer']
parameters = [8, 32]

# Create the grid search combinations
grid_search = list(
    itertools.product(models, parameters, parameters, parameters)
)

# Filter to keep only the combinations where num_neighbors = max_input_sequence_length = time_gap
filtered_grid_search = [
    (model, num_neighbors, time_gap, max_input_sequence_length)
    for model, num_neighbors, time_gap, max_input_sequence_length in grid_search
    if num_neighbors == time_gap == max_input_sequence_length
]

# Fixed parameters for all experiments
dataset_name = "tgbl-synthetic"
num_runs = 3
gpu = 0
num_epoch = 20

# Prepare the shell script content
script_content = "#!/bin/bash\n\n"
for model, num_neighbors, time_gap, max_input_sequence_length in filtered_grid_search:
    command = (
        f"python DyGLib_TGB/train_link_prediction.py --dataset_name {dataset_name} "
        f"--model_name {model} --max_input_sequence_length {max_input_sequence_length} "
        f"--num_neighbors {num_neighbors} --time_gap {time_gap} "
        f"--gpu {gpu} --num_epoch {num_epoch} --num_runs {num_runs}\n"
    )
    script_content += command

# Save the commands to a .sh file
script_filename = "DyGLib_TGB/run_experiments.sh"
with open(script_filename, "w") as script_file:
    script_file.write(script_content)

print(f"Saved experiment commands to {script_filename}.")
