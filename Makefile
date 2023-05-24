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
	cp $(MAKEFILE_LOC)/mlp.onnx $(THIRD_PARTY_LOC)/he-man-concrete/demo/ ; cd $(THIRD_PARTY_LOC)/he-man-concrete ; \
	target/release/he-man-concrete keyparams demo/mlp.onnx demo/calibration-data.zip demo/keyparams.json demo/mlp_calibrated.onnx ; \
	target/release/he-man-concrete keygen demo/keyparams.json demo/ ; target/release/he-man-concrete encrypt demo/ demo/input.npy demo/input.enc ; \
	target/release/he-man-concrete inference demo/ demo/mlp_calibrated.onnx demo/input.enc demo/result.enc ; \
	target/release/he-man-concrete decrypt demo/ demo/result.enc demo/result.npy ; 

clean: 
	rm -f $(FILES_TO_REMOVE)