#!/usr/bin/env bash

MAKEFILE_LOC = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON = python3 
FILES_TO_REMOVE = mlp.onnx input0.npy label0.npy
MODELS = mlp.onnx

# Paths:
THIRD_PARTY_LOC = $(MAKEFILE_LOC)/thirdparty

all: clean models he-man-eval 

setup: he-man-setup 

models: models.py
	$(PYTHON) models.py 

he-man-setup:
	cd $(THIRD_PARTY_LOC) ; git clone https://github.com/smile-ffg/he-man-concrete.git ; \
	cd he-man-concrete ; pip install argparse pathlib numpy tqdm ; RUSTFLAGS="-C target-cpu=native" cargo build --release ; \

he-man-eval: 
	cp $(MAKEFILE_LOC)/mlp.onnx $(THIRD_PARTY_LOC)/he-man-concrete/ ; cd $(THIRD_PARTY_LOC) ; \
	bin/he-man-concrete keyparams he-man-concrete/mlp.onnx he-man-concrete/calibration-data.zip he-man-concrete/keyparams.json he-man-concrete/mlp_calibrated.onnx ; \
	bin/he-man-concrete keygen he-man-concrete/keyparams.json demo/ ; bin/he-man-concrete encrypt he-man-concrete/ he-man-concrete/input.npy he-man-concrete/input.enc ; \
	bin/he-man-concrete inference he-man-concrete/ he-man-concrete/mlp_calibrated.onnx he-man-concrete/input.enc he-man-concrete/result.enc ; \
	bin/he-man-concrete decrypt he-man-concrete/ he-man-concrete/result.enc he-man-concrete/result.npy ; 

clean: 
	rm -f $(FILES_TO_REMOVE)