#!/bin/bash

python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name DyRep --max_input_sequence_length 8 --num_neighbors 8 --time_gap 8 --gpu 0 --num_epoch 20 --num_runs 3
python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name DyRep --max_input_sequence_length 32 --num_neighbors 32 --time_gap 32 --gpu 0 --num_epoch 20 --num_runs 3
python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name TGAT --max_input_sequence_length 8 --num_neighbors 8 --time_gap 8 --gpu 0 --num_epoch 20 --num_runs 3
python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name TGAT --max_input_sequence_length 32 --num_neighbors 32 --time_gap 32 --gpu 0 --num_epoch 20 --num_runs 3
python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name GraphMixer --max_input_sequence_length 8 --num_neighbors 8 --time_gap 8 --gpu 0 --num_epoch 20 --num_runs 3
python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name GraphMixer --max_input_sequence_length 32 --num_neighbors 32 --time_gap 32 --gpu 0 --num_epoch 20 --num_runs 3
python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name DyGFormer --max_input_sequence_length 8 --num_neighbors 8 --time_gap 8 --gpu 0 --num_epoch 20 --num_runs 3
python DyGLib_TGB/train_link_prediction.py --dataset_name tgbl-synthetic --model_name DyGFormer --max_input_sequence_length 32 --num_neighbors 32 --time_gap 32 --gpu 0 --num_epoch 20 --num_runs 3
