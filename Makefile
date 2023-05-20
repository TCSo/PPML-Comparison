#!/usr/bin/env bash

# Variables
PYTHON = python3 
FILES_TO_REMOVE = mlp.onnx input0.npy label0.npy

all: clean models

models: models.py
	$(PYTHON) models.py 

clean: 
	rm -f $(FILES_TO_REMOVE)