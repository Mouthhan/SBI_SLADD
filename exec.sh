#!/bin/sh

docker run -it --name sbi-mouth --gpus all --shm-size 64G \
    -v /home/viplab/Desktop/Mouth/SelfBlendedImages/:/app/ -v /home/viplab/sinica:/home/viplab/sinica \
    mapooon/sbi bash
