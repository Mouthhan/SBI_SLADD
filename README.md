# SBI_SLADD

Based on SelfBlendedImages (CVPR 2022) paper's work:\
https://github.com/mapooon/SelfBlendedImages

Docker files has some problem, run author uploaded dockerhub file:\
docker pull mapooon/sbi

Try to adopt SLADD (CVPR 2022)'s adversarial training:\
https://github.com/liangchen527/SLADD

## Inferece
python3 src/inference/inference_dataset.py \
-w {weight path} \
-d CDF

## Training
python3 src/train_sbi.py \
src/configs/sbi/base.json \
-n sbi -r
