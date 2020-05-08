#!/bin/bash

source activate audio-proj
export PYTHONPATH=$PYTHONPATH:~/csci5980:~/csci5980/utils

python train.py --max_epochs 5 --batch_size 2 --training_csv sr44100_waveform_training_directory.csv
