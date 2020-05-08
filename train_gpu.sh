#!/bin/bash
#PBS -l nodes=1:ppn=24:gpus=2
#PBS -l walltime=12:00:00
#PBS -l mem=25gb
#PBS -q k40
#PBS -M piehl008@umn.edu

module load singularity
cd /home/csci5980/piehl008/csci5980/
export AWS_ACCESS_KEY_ID=38XZS5ACPIDWZXKLDS49
export AWS_SECRET_ACCESS_KEY=a0hLS8VGRrh3yIq8nl89KFj6xA7QjJRgsNaL3MAX

singularity exec -B logs/:/logs/,models/:/models/,/scratch.global/piehl008/:/data/ --nv shub://piehlsam/csci5980:gpu \
python train.py --max_epochs 1000 --batch_size 1 \
  --training_file /data/pickled_data/training_records_container_list.txt \
  --scale_data --training_record_byte_size 352921 \
  --shuffle_buffer_size 1000 --n_classes 3
