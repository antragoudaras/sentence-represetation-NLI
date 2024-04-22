# Learning Sentence Representations From Natural Language Inference Data

This repository contains the code for the paper [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364).

## Structure
* `best_model_dir_1234/` contains the pre-trained models on the SNLI.
* `analysis.ipynb` contains the Entailment prediction example, results of the experiments, and discussion (for that reason it replaces the need of an actual pdf report)
* `environment_gpu/cpu.yml` allow one to install all the package depedencies used in this project
* `train_*` are the scripts used for training in Snellius



## Usage

Train a model in the SNLI using the following command:

```bash
python  train.py --encoder <encoder_type>
```
See the pyton scripts for more details

Evaluate a model using the following command:

```bash
python eval.py <checkpoint> --senteval-vocab --encoder <encoder-type> --snli --senteval
```
`--senteval-vocab` build new GloVe embeddings and creates new Vocab for each SentEval tas 
The `--snli` flag will evaluate the model on the SNLI dataset. The `--senteval` flag will evaluate the model on the SentEval datasets. See `-h` for more options such as changing the batch size or the number of epochs.

The `results_seed_` folder contain all results reported
The `results_SenteEval_seed` folder contain all results for 

## Installing Senteval 
Follow the instruction as in [Github](https://github.com/facebookresearch/SentEval).

Be carefull to change the following files to support Python > 3.10 version 
in `senteval_utils.py`, line 90:
```
if sys.version_info < (3, 10):
        expected_args = inspect.getargspec(optim_fn.__init__)[0]
    else:
        expected_args = list(inspect.signature(optim_fn.__init__).parameters.keys())
```
## Tensorboard Logs

All tensorboard logs can be found in the relevant `*tensorboard_log_dir_*`. Where each epoch/step we log the train loss, the validation loss and validation accuracy.
 

## Models

The pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1qP0iZBFZ3855miHVPoVx6xKDtfmLZhtB?usp=sharing) and [Google Drive2](https://drive.google.com/drive/folders/1ZeBUxiXtoE-RRYdJtJxlGc2GVrpfzn3I?usp=sharing). The models should be placed in the `best_model_dir_1234/` repo where the 1234 refres to the seed used.