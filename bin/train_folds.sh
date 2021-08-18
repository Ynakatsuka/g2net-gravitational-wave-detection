#!/bin/bash
python run.py "$@"
python run.py "$@" trainer.idx_fold=1
python run.py "$@" trainer.idx_fold=2
python run.py "$@" trainer.idx_fold=3
python run.py "$@" trainer.idx_fold=4
python run.py "$@" run=evaluate_oof
