#!/usr/bin/env bash

MAKEFILE_LOC = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON = python3 
FILES_TO_REMOVE = slp.onnx mlp.onnx input0.npy label0.npy output.npy input_calib.npy output_concrete.npy
MODELS = slp.onnx mlp.onnx

# Paths:
THIRD_PARTY_LOC = $(MAKEFILE_LOC)/thirdparty

all: clean models he-man-eval concrete-eval stats

setup: he-man-setup 

models: models.py
	$(PYTHON) models.py 

he-man-setup:
	cd $(THIRD_PARTY_LOC) ; git clone https://github.com/smile-ffg/he-man-concrete.git ; \
	cd he-man-concrete ; RUSTFLAGS="-C target-cpu=native" cargo build --release ; \ 
	cd $(MAKEFILE_LOC) ;

he-man-eval: 
	cp $(MAKEFILE_LOC)/slp.onnx $(THIRD_PARTY_LOC)/he-man-concrete/demo ; cd $(THIRD_PARTY_LOC)/he-man-concrete ; \
	target/release/he-man-concrete keyparams demo/slp.onnx demo/calibration-data.zip demo/keyparams.json demo/slp_calibrated.onnx ; \
	target/release/he-man-concrete keygen demo/keyparams.json demo/ ; target/release/he-man-concrete encrypt demo/ demo/input.npy demo/input.enc ; \
	target/release/he-man-concrete inference demo/ demo/slp_calibrated.onnx demo/input.enc demo/result.enc ; \
	target/release/he-man-concrete decrypt demo/ demo/result.enc demo/result.npy ; cd $(MAKEFILE_LOC) ; \
	cd $(MAKEFILE_LOC) ;

concrete-eval:
	$(PYTHON) concreteEval.py

stats:
	$(PYTHON) stats.py

clean: 
	rm -f $(FILES_TO_REMOVE)