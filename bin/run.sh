#!/bin/bash
# example
# ./bin/train_folds.sh
# ./bin/inference_folds.sh
# python src/misc/make_submission.py
# kaggle competitions submit g2net-gravitational-wave-detection -f data/output/submission/trainer\=exp001.csv -m ""

# ------------------------------------------------------------------------------
# Preprocess
# ------------------------------------------------------------------------------
# python src/misc/fold.py

# ------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------
# python src/misc/features/create_feets.py

# TODO
# python src/misc/features/create_nolds.py
# python src/misc/features/create_tsfresh.py

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

# python run.py trainer=exp002 trainer.idx_fold=1
# python run.py trainer=exp002 trainer.idx_fold=2
# python run.py trainer=exp002 trainer.idx_fold=3
# python run.py trainer=exp002 trainer.idx_fold=4
# python run.py trainer=exp002 run=evaluate_oof

python run.py +model.model.params.resize_shape=256 experiment=exp007 model.model.params.hop_length=16
python run.py +model.model.params.resize_shape=256 experiment=exp007 model.model.params.hop_length=8

python run.py augmentation=gaussian_snr trainer=long_epochs
python run.py augmentation=clip trainer=long_epochs
python run.py augmentation=clipping_distortion trainer=long_epochs
python run.py augmentation=frequency_mask trainer=long_epochs
python run.py augmentation=gain trainer=long_epochs
python run.py augmentation=gaussian_noise trainer=long_epochs
python run.py augmentation=gaussian_snr trainer=long_epochs
python run.py augmentation=impulse_response trainer=long_epochs
python run.py augmentation=pitch_shift trainer=long_epochs
python run.py augmentation=polarity_inversion trainer=long_epochs
python run.py augmentation=reverse trainer=long_epochs
python run.py augmentation=shift trainer=long_epochs
python run.py augmentation=tanh_distortion trainer=long_epochs
python run.py augmentation=time_mask trainer=long_epochs
python run.py augmentation=time_stretch trainer=long_epochs
python run.py augmentation=trim trainer=long_epochs
python run.py augmentation=gaussian_noise trainer=long_epochs +augmentation.transform.train.0.params.min_amplitude=0.0001 +augmentation.transform.train.0.params.max_amplitude=0.001
python run.py augmentation=gaussian_noise trainer=long_epochs +augmentation.transform.train.0.params.min_amplitude=0.001 +augmentation.transform.train.0.params.max_amplitude=0.005
python run.py augmentation=gaussian_snr trainer=long_epochs +augmentation.transform.train.0.params.min_snr_in_db=0 +augmentation.transform.train.0.params.max_snr_in_db=1
python run.py augmentation=gaussian_snr trainer=long_epochs +augmentation.transform.train.0.params.min_snr_in_db=0 +augmentation.transform.train.0.params.max_snr_in_db=2
python run.py augmentation=gaussian_snr trainer=long_epochs +augmentation.transform.train.0.params.min_snr_in_db=0 +augmentation.transform.train.0.params.max_snr_in_db=4
python run.py augmentation=gaussian_snr trainer=long_epochs +augmentation.transform.train.0.params.min_snr_in_db=0 +augmentation.transform.train.0.params.max_snr_in_db=8
# +augmentation.transform.train.0.params.min_sample_rate=2048 +augmentation.transform.train.0.params.max_sample_rate=8192 augmentation.transform.train.0.params.p=0.25


# python run.py augmentation=batch_noise trainer=long_epochs augmentation.batch_transform.train.0.params.p=0.25
python run.py augmentation=batch_noise trainer=long_epochs augmentation.batch_transform.train.0.params.p=0.75

python run.py dataset.dataset.0.params.normalize_mode=11
python run.py +model.model.params.resize_shape=384 experiment=exp007
python run.py +model.model.params.resize_shape=512 experiment=exp008

./bin/train_folds.sh experiment=exp002
./bin/inference_folds.sh experiment=exp002
python src/misc/make_submission.py experiment=exp002
