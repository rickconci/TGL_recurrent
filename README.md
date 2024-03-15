# TGL_recurrent

Looking into the past: exploring recurring patterns in time dependencies.

This prject was done in collaboration with Kiril Bikov <https://github.com/kiril-bikov> for a Geometric Deep Learning submission as part of the MPhil in Advanced Computer Science, University of Cambridge, March 2024.

## Experiments

### Quantifying recurrence within datasets

1) download datasets either manually or through
"RC_scripts/TGB_edited/docs/tutorials/Edge_data_numpy.ipynb"

2) adjust pathnames (wd + basepath etc) within rc_main.py and rc_utils.py within RC scripts and run without required arguments:

```
python rc_main.py
```

This will both save dataset pickle files and plots if required
Plots can also be created separately onces the pickle files are saved through ideas.ipynb and final_plotting.ipynb

### Running DyGFormer and GraphMixer on interval based recurrency

1) Within DyGLib_TGB_edited_KB the utils/DataLoader.py has been modified to subselect recurrent interactions.

2) run as an example

```
 python DyGLib_TGB_edited_KB/train_link_prediction.py --dataset_name tgbl-wiki --model_name DyGFormer --max_input_sequence_length 32 --num_neighbors 32 --time_gap 32 --gpu 0 --num_epoch 20 --num_runs 3
```

### Creating synthetic dataset and running models to assess out of context recurrence

1) Move back into RC_scripts and create synthetic dataset using ```synthetic_dataset.ipynb```. Save the dataset to a 'tgbl-synthetic' folder in at the same level as the other data folders like tgbl-wiki etc.

2) Within DyGlib_TFB_rc adjust and run ```create_scripts.py``` to create the experiments .sh file.

3) run ```./run_experiments.sh```, gradually the results will be placed within a 'logs' and 'saved_results' folders.
The DataLoader.py is also edited to save the validation and test predictions for further visualisation within synthetic_dataset.ipynb.
These predictions are saved as, for example, 'saved_results/DyGFormer/tgbl-synthetic/val_synthetic_validation_data.csv'.

## Requirements

Follow <https://github.com/shenyangHuang/TGB> for installation instructions.
Base packages will be downloaded using:

```
pip install py-tgb
```

## Thank you

To the awesome TGB <https://github.com/shenyangHuang/TGB>  and DyGLib_TGB <https://github.com/yule-BUAA/DyGLib_TGB> teams for their scripts!!

