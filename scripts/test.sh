#!/bin/sh
TARGET=target_test
ARCH=myresnet
MODEL=./logs/model_best.pth.tar


CUDA_VISIBLE_DEVICES=0 \
python examples/test_model.py -b 256 -j 8 \
	--dataset-target ${TARGET} -a ${ARCH} --resume ${MODEL}
