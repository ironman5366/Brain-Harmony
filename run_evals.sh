#!/bin/bash
set -e

set -a && source .s3.env

datasets=(
    "hcpya_rest1lr_gender"
    "hcpya_rest1lr_age"
    "hcpya_rest1lr_flanker"
    "hcpya_rest1lr_neofacn"
    "hcpya_rest1lr_pmat24"
)

for dataset in "${datasets[@]}"; do
    echo "Running evaluation for: $dataset"
    python -m fmri_fm_eval.main_probe brain_harmonix_f "$dataset" --overrides representation=avg_patch
done