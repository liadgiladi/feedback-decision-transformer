# Feedback-Decision-Transformer

Author: Liad Giladi.

This project was carried out as part of my master's thesis, supervised by Dr. Gilad Katz.

A link to our paper can be found on [arXiv](TODO).

## Overview

The official codebase for [Feedback Decision Transformer: Offline Reinforcement Learning With Feedback](https://github.com/liadgiladi/feedback-decision-transformer).

TL;DR: We propose Feedback Decision Transformer (FDT), a data-driven approach that uses limited amounts of high-quality feedback at critical states to significantly improve DT’s performance.


![image info](./architecture.PNG)

## Installation

### Install Dependencies
You can install the required dependencies by running the following commands:

```shell
poetry install
poe force-torch-cuda11
poe force-jaxlib-cuda11
poetry run python -m atari_py.import_roms Roms/
```
### Downloading Datasets

Create a directory for the dataset and load the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DQN_REPLAY_DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DQN_REPLAY_DIRECTORY_NAME]
```

## Run Training 

### FDT
#### Breakout & Qbert & Seaquest
```
for seed in 123 231 312 0 42 84 64 128 256 512
do
	python run.py --gpus 1 --lr 6e-4 --states_for_feedbacks_based_on_important_states [IMPORTANT_STATES_FILE_NAME] --num_of_important_states [FEEDBACK_NUM] --feedback_regularization_lambda [FEEDBACK_LAMBDA] --wandb_project_name [WANDB_PROJECT_NAME_NAME] --conditioned_rtg [CONDITIONED_RTG] --seed $seed --generate_dataset_seed $seed --augment_only_sparse_reward_with_synthetic_oracle_feedback --disable_training_game_evaluation_callback --epochs 5 --game [GAME] --data_dir_prefix [DQN_REPLAY_DIRECTORY_NAME]
done
```

#### Pong
```
for seed in 123 231 312 0 42 84 64 128 256 512
do
	python run.py --gpus 1 --lr 6e-4 --states_for_feedbacks_based_on_important_states [IMPORTANT_STATES_FILE_NAME] --num_of_important_states [FEEDBACK_NUM] --feedback_regularization_lambda [FEEDBACK_LAMBDA] --wandb_project_name [WANDB_PROJECT_NAME_NAME] --conditioned_rtg [CONDITIONED_RTG] --seed $seed --generate_dataset_seed $seed --augment_only_sparse_reward_with_synthetic_oracle_feedback --disable_training_game_evaluation_callback --epochs 5 --game 'Pong' --batch_size 256 --context_length 50 --data_dir_prefix [DQN_REPLAY_DIRECTORY_NAME]
done
```

### DT
#### Breakout & Qbert & Seaquest
```
for seed in 123 231 312 0 42 84 64 128 256 512
do
	python run.py --gpus 1 --lr 6e-4 --wandb_project_name [WANDB_PROJECT_NAME_NAME] --conditioned_rtg [CONDITIONED_RTG] --seed $seed --generate_dataset_seed $seed --augment_only_sparse_reward_with_synthetic_oracle_feedback --disable_training_game_evaluation_callback --epochs 5 --game [GAME] --data_dir_prefix [DQN_REPLAY_DIRECTORY_NAME]
done
```

#### Pong
```
for seed in 123 231 312 0 42 84 64 128 256 512
do
	python run.py --gpus 1 --lr 6e-4 --wandb_project_name [WANDB_PROJECT_NAME_NAME] --conditioned_rtg [CONDITIONED_RTG] --seed $seed --generate_dataset_seed $seed --augment_only_sparse_reward_with_synthetic_oracle_feedback --disable_training_game_evaluation_callback --epochs 5 --game 'Pong' --batch_size 256 --context_length 50 --data_dir_prefix [DQN_REPLAY_DIRECTORY_NAME]
done
```

### BC
#### Breakout & Qbert & Seaquest
```
for seed in 123 231 312 0 42 84 64 128 256 512
do
	python run.py --model_type naive --gpus 1 --lr 6e-4 --wandb_project_name [WANDB_PROJECT_NAME_NAME] --conditioned_rtg [CONDITIONED_RTG] --seed $seed --generate_dataset_seed $seed --augment_only_sparse_reward_with_synthetic_oracle_feedback --disable_training_game_evaluation_callback --epochs 5 --game [GAME] --data_dir_prefix [DQN_REPLAY_DIRECTORY_NAME]
done
```

#### Pong
```
for seed in 123 231 312 0 42 84 64 128 256 512
do
	python run.py --model_type naive --gpus 1 --lr 6e-4 --wandb_project_name [WANDB_PROJECT_NAME_NAME] --conditioned_rtg [CONDITIONED_RTG] --seed $seed --generate_dataset_seed $seed --augment_only_sparse_reward_with_synthetic_oracle_feedback --disable_training_game_evaluation_callback --epochs 5 --game 'Pong' --batch_size 256 --context_length 50 --data_dir_prefix [DQN_REPLAY_DIRECTORY_NAME]
done
```

## Run Evaluation
### FDT
#### Breakout & Qbert
```

```
### Seaquest

#### Pong
```
for seed in 123 231 312 0 42 84 64 128 256 512
do
	python run.py --gpus 1 --eval_model --wandb_project_name_for_loading_pretrained_model [WANDB_PRETRAINED_MODEL_PROJECT_NAME] --test_seeds '[123, 231, 312, 0, 42, 84, 64, 128, 256, 512]' --lr 6e-4 --states_for_feedbacks_based_on_important_states [IMPORTANT_STATES_FILE_NAME] --num_of_important_states [FEEDBACK_NUM] --feedback_regularization_lambda [FEEDBACK_LAMBDA] --wandb_project_name [WANDB_PROJECT_NAME_NAME] --conditioned_rtg 20 --seed $seed --generate_dataset_seed $seed --augment_only_sparse_reward_with_synthetic_oracle_feedback --disable_training_game_evaluation_callback --epochs [EPOCH]  --game 'Pong' --batch_size 256 --context_length 50 --data_dir_prefix [DQN_REPLAY_DIRECTORY_NAME]
done

```

## Generate Important States For Feedback

## Oracle Feedback Generation

## Results
All hyperparameters and logs of our runs can be viewed at our TODO project.

## Citation

To cite FDT, you can use the following BibTeX entry:

```bibtex
TODO
```

## License

This project is licensed under the MIT License.


## Acknowledgments
Our code is based on the implementation of [Decision Transformer](https://github.com/scottemmons/decision-transformer). 
