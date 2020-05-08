#!/bin/bash
#PBS -l nodes=1:ppn=24:gpus=1
#PBS -l walltime=24:00:00
#PBS -l mem=100gb
#PBS -q v100
#PBS -M piehl008@umn.edu

module load singularity
cd /home/csci5980/piehl008/csci5980/

singularity exec -B logs/:/logs/,models/:/models/,/scratch.global/piehl008/:/data/ --nv shub://piehlsam/csci5980:gpu \
python train.py --max_epochs 1000 --batch_size 100 \
  --training_file /data/pickled_data/training_records_container_list.txt \
  --scale_data --training_record_byte_size 352921 \
  --shuffle_buffer_size 1000 --n_classes 3 \
  --classifier_channels 20 --decoder_blocks 2 --decoder_layers 4 \
  --encoder_blocks 1 --encoder_layers 4 --encoder_channels 32 \
  --encoder_pool 1800 --decoder_residual_channels 32 \
  --decoder_skip_channels 32
