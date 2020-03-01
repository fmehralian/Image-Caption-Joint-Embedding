#!/usr/bin/env bash

mkdir data/icons-v1
mv ../Deep-Fashion-Joint-Embedding-Preprocessing/preprocessed_data/* data/icons-v1/*

python3 train.py
python3 test.py