

# Quantum Natural Language Processing : NEASQC WP 6.1
This readme gives a brief introduction to the release candidate of the [NEASQC](https://www.neasqc.eu/) work-package 6.1 on quantum natural language processing and provides guidance on how to run and compare different models (both classical natural language processing (NLP) models and quantum NLP (QNLP) models). The release candidate is a benchmarking solution that allows to:

* download and pre-process data for quantum and classical NLP model training,
* Train quantum and classical NLP models for text classification (we have set up scripts such that each model is trained 30 times for statistical analysis).
* Evaluate the trained models on held-out test sets.

We give a brief overview of the following models:
* Quantum models Beta2/3 (for Alpha3, refer to the [previous deliverable](https://www.neasqc.eu/wp-content/uploads/2023/11/NEASQC_D6_10_QNLP.pdf) and the experimental implementation [here](https://github.com/NEASQC/WP6_QNLP/tree/27f289bb50dfb9c82e5de12e2164bfad96d24b84)).
* Classical NLP models. Although, we have experimented with neural network models trained on top of pre-trained word-level (using convolutional neural networks) and sentence-level (using a feed-forward neural network) embeddings as well as a scenario without pre-trained embeddings (using a long short-term memory neural network) (see experimental results [here](https://github.com/tilde-nlp/neasqc_wp6_qnlp/tree/v2-classical-nlp-models/neasqc_wp61/doc) and [here](https://github.com/tilde-nlp/neasqc_wp6_qnlp/tree/classical-nlp-models/neasqc_wp61/doc)), in the release candidate, we focused only on sentence-level pretrained embeddings that showed to achieve better results.

## Setup

### Pre-requisites
Prior to following any steps, you should ensure that you have on your local machine and readily available:
- A copy of the repository.
- `python 3.10`, our models *might* turn out to be compatible with later versions of python but they were designed with and intended for 3.10.
- `poetry`, you can follow the instructions if needed on <a href="https://python-poetry.org/docs/#installation">the official website</a>.

### Getting started
1. Position yourself in the `final_main` branch.
2. Position yourself in the root of the repository where the files `pyproject.toml` and `poetry.lock` are located.
3. Run <pre><code>poetry install</pre></code>
4. Activate `poetry` using <pre><code>poetry shell</code></pre> More details can be found <a href="https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment">here</a>.
5. Download the `en_core_web_sm` model for `spacy` (which is used for data preparation) using the command <pre><code>python -m spacy download en_core_web_sm</pre></code>

## Workflow

The Workflow has 8 steps. Data preparation steps (Step 0 to Step 5) are common to the classical and the quantum algorithms. Model training and benchmarking is performed in Step 6 and combination of results from all experiments is performed in Step 7.

To run each step, use the corresponding bash script that is located in the directory `neasqc_wp61`. 

### Step 0 - data download

We use the following datasets from the kaggle.com. 

- `Reviews.csv` from <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/versions/1> (we will name this resource after download and processing as `amazon-fine-food-reviews.csv`)
- `labelled_newscatcher_dataset.csv` from <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset> (we will name this resource after download and processing as `labelled_newscatcher_dataset.csv`)
- `train.csv` from <https://www.kaggle.com/datasets/kk0105/ag-news> (we will name this resource after download and processing as `ag_news.csv`)
- `RAW_interactions.csv` from <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions> (we will name this resource after download and processing as `food-com-recipes-user-interactions.csv`)
- `train.csv` from <https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews> (we will name this resource as `amazon-reviews.csv`)
- `News_Category_Dataset_v3.json` from <https://www.kaggle.com/datasets/rmisra/news-category-dataset> (we will name this resource after download and processing as `huffpost-news.tsv`)

To retrive the datasets from the kaggle.com you first have to

- install the kaggle API using the command `pip install --user kaggle`
- create a Kaggle account, create an API token, make a directory `.kaggle` in the home `~` directory, and place `kaggle.json` in that directory.  
	
See instructions in <https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/>

Then, step into the `neasqc_wp61` folder and run the script `0_FetchDatasets.sh` to download the datasets and convert them into a unified format (CSV). All further scripts will be executed from this folder.

### Step 1 - Tokenising, filtering, and normalising datasets

After downloading datasets, we tokenise the data using the `spacy` tokeniser, extract class labels and text (as all other columns are not needed), and perform normalisation (e.g., decoding of HTML entitities and fixing broken Unicode symbols in some datasets). To perform this step, run the script `1_Filter6Parse.sh` passing the following parameters:

- `-i <dataset>`           Comma-separated (CSV) data file
- `-d <delimiter>`          Field delimiter symbol
- `-c <class fiels>`        Name of the class field (only if the first line in the file contains field names)
- `-t <text field>`         Name of the text field (only if the first line in the file contains field names)
- `-o <output file>`            Output file in a tab-separated format (TSV)

Examples:

```bash
bash 1_Filter6Parse.sh -i ./data/datasets/amazon-fine-food-reviews.csv -o ./data/datasets/amazon-fine-food-reviews_tok.tsv -d ',' -c 'Score' -t 'Summary'

bash 1_Filter6Parse.sh -i ./data/datasets/labelled_newscatcher_dataset.csv -o ./data/datasets/labelled_newscatcher_dataset_tok.tsv -d ';' -c 'topic' -t 'title'

bash 1_Filter6Parse.sh -i ./data/datasets/ag_news.csv -o ./data/datasets/ag_news_tok.tsv -d ','

bash 1_Filter6Parse.sh -i ./data/datasets/food-com-recipes-user-interactions.csv -o ./data/datasets/food-com-recipes-user-interactions_tok.tsv -d ',' -c 'rating' -t 'review'

bash 1_Filter6Parse.sh -i ./data/datasets/amazon-reviews.csv -o ./data/datasets/amazon-reviews_tok.tsv -d ','

bash 1_Filter6Parse.sh -i ./data/datasets/huffpost-news.csv -o ./data/datasets/huffpost-news_tok.tsv -d ','
```

### Step 2 - Re-balancing the dataset

We train classical NLP and QNLP models using balanced datasets, such that each class would be equally represented in the dataset. To perform this step, run the script `2_BalanceClasses.sh` passing the following parameters:

- `-i \<input file\>`               Tab-separated (TSV) 2-column file to rebalance.
- `-c \<classes to ignore\>`		  Optionally classes to ignore in dataset separated by `,`.

Examples:

```bash
bash 2_BalanceClasses.sh -i ./data/datasets/amazon-fine-food-reviews_tok.tsv

bash 2_BalanceClasses.sh -i ./data/datasets/labelled_newscatcher_dataset_tok.tsv -c "SCIENCE"

bash 2_BalanceClasses.sh -i ./data/datasets/ag_news_tok.tsv

bash 2_BalanceClasses.sh -i ./data/datasets/food-com-recipes-user-interactions_tok.tsv

bash 2_BalanceClasses.sh -i ./data/datasets/amazon-reviews_tok.tsv

#We drop quite a few classes from the dataset as they have rather few examples
bash 2_BalanceClasses.sh -i ./data/datasets/huffpost-news_tok.tsv -c "ARTS,ARTS & CULTURE,COLLEGE,CULTURE & ARTS,EDUCATION,ENVIRONMENT,FIFTY,GOOD NEWS,GREEN,LATINO VOICES,MEDIA,MONEY,RELIGION,SCIENCE,STYLE,TASTE,TECH,U.S. NEWS,WEIRD NEWS,WORLDPOST,category"
```
	
### Step 3 - Spliting data in train/dev/test parts

Next, we split the corpus into 3 separate (non-overlapping) parts for training, validation, and evaluation of models. To perform this step, run the script `3_SplitTrainTestDev.sh` passing the following parameters:

- `-i \<input file\>`         Tab-separated (TSV) 2-column filtered file

3 files fill be created containing suffices `_train`, `_test`, and `_dev` in their names.

Example:

```bash
bash 3_SplitTrainTestDev.sh -i ./data/datasets/amazon-fine-food-reviews_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/labelled_newscatcher_dataset_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/ag_news_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/food-com-recipes-user-interactions_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/amazon-reviews_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/huffpost-news_tok_balanced.tsv
```

### Step 4 - Acquiring embedding vectors using chosen pre-trained embedding model

We have experimented with two differentpre-trained embedding models (featuring less than 1B parameters) that have shown to allow achieving various levels of down-stream task accuracies (based on the [Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)) .

| Pre-trained model | Parameters (millions) | Embedding type | Vectors for unit | About model |
|---|---|---|---|---|
| *ember-v1* | 335 | transformer    | sentence         | 1024-dimentional sentence transformer model. <br><https://huggingface.co/llmrails/ember-v1> |
| *bert-base-uncased* | 110 | bert           | sentence         | Case insensitive model pretrained on the BookCorpus (consisting of 11,038 unpublished books) and English Wikipedia; 768-dimensional sentence transformer model.<br><https://huggingface.co/bert-base-uncased> |

BERT models are older and slower than sentence transformer models. We compare the Transformer-based models also with an older skip-gram model (for word embedding; sentence embeddings are obtained by combining individual word embeddings) using `fasttext`.

To perform this step, run the script `4_GetEmbeddings.sh` passing the following parameters:

- `-i <input file>`      input file with text examples
- `-o <output file>`     output json file with embeddings
- `-c <column>`          `2` - if 2-column input file containing class and text columns, `0` - if the whole line is a text example
- `-m <embedding name>`  Name of the embedding model
- `-t <embedding model type>`  Type of the embedding model - `fasttext`, `transformer`, or `bert`	
- `-e <embedding unit>`  Embedding unit: `sentence` or `word`	
- `-g <gpu use>`         Number of GPUs to use (from `0` to available GPUs), `-1` if CPU shall be used (default)

The `fasttext` model works only on CPU.

Embedding unit is `word` or `sentence` for the `bert` models; `sentence` for the `transformer` and `fasttext` models. 

Run this step for all 3 parts of the dataset - train, dev and test. We have set up a bash script to generate embeddings for all data sets used in this repository and the three sentence-level embedding models - `4_1_GetEmbeddingsForAllExampleSets.sh`. However, if you wish to run the script on individual data files, here is an example:

```bash
bash 4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test.tsv -o ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test_ember.json -c '2' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'
```
The output data of the vectoriser is stored using the following JSON format (and accepted as input data by the classification model training step 5 below):

```json
[
  {
    "class": "2",
    "sentence": "Late rally keeps Sonics ' win streak alive",
    "sentence_vectorized": [
      [
        1.2702344502208496,
        -1.109272517088847,
        0.729442862056898,
        -2.0236876799667654,
        -2.1608433294597083
      ]
    ]
  },
  {
    "class": "2",
    "sentence": "Fifth - Ranked Florida State Holds Off Wake Forest",
    "sentence_vectorized": [
      [
        4.42427794195838,
        2.71155631975937,
        -0.2619018355289202,
        -2.7190369563754815,
        -2.021439642325012
      ]
    ]
  }
]
```

### Step 5 - Reducing dimensions of embeddings

Since quantum computing models cannot process vectors of large dimensionality, we apply feature selection to reduce dimensions of embeddings. Embedding dimensions can be reduced using the python script `reduce_emb_dim.py` in the folder `./data/data_processing`. It uses dimensionality reduction classes from the script `dim_reduction.py`. 

Run the script using the following parameters:

- `-it \<input train file\>` 	 JSON input training file with embeddings (acquired using the script `4_GetEmbeddings.sh`)
- `-ot \<output train file\>`     JSON output training file with reduced embeddings
- `-iv \<input dev file\>` 	 JSON input validation file with embeddings (acquired using the script `4_GetEmbeddings.sh`)
- `-ov \<output dev file\>`     JSON output validation file with reduced embeddings
- `-ie \<input test file\>` 	 JSON input evaluation file with embeddings (acquired using the script `4_GetEmbeddings.sh`)
- `-oe \<output test file\>`     JSON output evaluation file with reduced embeddings
- `-n \<number of dimensions\>` Desired number of dimensions in the output vectors
- `-a \<reduction algorithm\>` Dimensionality reduction algorithm (possible values are `PCA`, `ICA`, `TSVD`, `UMAP` or `TSNE`)

Example:

`python ./data/data_processing/reduce_emb_dim.py -i ./data/datasets/reviews_filtered_train_ember.json -o ./data/datasets/reviews_filtered_train_100ember.json -n 100 -a PCA`

To reduce all datasets with `PCA` to `3`, `5`, and `8` (maximum for the quantum models) dimensions, use the script `5_ReduceDimensionsForAllDatasets.sh`.

### Step 6 - Training models

Once data is pre-processed, we can start training models. The following sub-sections document how to train both classical NLP and quantum NLP models.

#### Classical NLP

The folder `./models/classical` contains the source code of a class implementing neural network classifiers. For classical NLP, we implemented three classifiers - a shallow feed-forward neural network (FFNN; for sentence-level embeddings), a convolutional neural network (CNN; for word-level embeddings), and a bidirectional long short-term memory (LSTM) neural network (when pre-trained embeddings are not used).

To perform this step, run the script *6_TrainClassicalNNModel.sh* passing the following parameters:

- `-t \<train data file\>` JSON data file for classifier training (with embeddings) or TSV file (if not using pre-trained embeddings, acquired using the script `3_SplitTrainTestDev.sh`)
- `-d \<dev data file\>`   JSON data file for classifier validation (with embeddings) or TSV file (if not using pre-trained embeddings, acquired using the script `3_SplitTrainTestDev.sh`)
- `-f \<field\>`           Field in the JSON object by which to classify text
- `-e \<embedding unit\>`  Embedding unit: `sentence`, `word`, or `-` (if not using pre-trained embeddings)
- `-m \<model directory\>` Directory where to save the trained model (the directory will be created if not existing)
- `-g \<gpu use\>`         Number of the GPU to use (from `0` to available GPUs), `-1` if CPU should be used (default is `-1`)
	
Each model is trained 30 times (runs) for statistical analysis.

Example:

`bash ./6_TrainClassicalNNModel.sh -t ./data/datasets/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1.json -d ./data/datasets/labelled_newscatcher_dataset_tok_balanced_dev_llmrails_ember-v1.json -e ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test_llmrails_ember-v1.json -f 'class' -e 'sentence' -m ./benchmarking/results/raw/classical/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1 -g '0'`

To train classical NLP models for all datasets, all 3 sentence-level embedding models, and all reduced dimensions, run the script `6_1_TrainClassicalNNModelsForAllDatasets.sh`. This script will train 30 classifiers for each training dataset using early stopping based on the validation set. When training each classifier, the best-performing checkpoint (i.e., checkpoint achieving the highest accuracy on the validation set) will be evaluated using a test set. Results for each training dataset will be stored in separate `results.json` files. 

The format of the `results.json` files is as follows:
```json
{
  "input_args": {
    "runs": 0,
    "iterations": 0
  },
  "best_val_acc": 0,
  "best_run": 0,
  "time": [..., ...],
  "train_acc": [[..., ...], ... [..., ...]],
  "train_loss": [[..., ...], ... [..., ...]],
  "val_acc": [[..., ...], ... [..., ...]],
  "val_loss": [[..., ...], ... [..., ...]],
  "test_acc": [..., ...],
  "test_loss": [..., ...]
}
```

#### Quantum NLP

Beta 2 and 3 models follow what we call a *semi-dressed quantum circuit* (SDQC) architecture. Here, the first layer of a DQC is stripped. The classical input is handed directly to the PQC once it has been brought to the correct dimension. The input to the circuit is a dimensionality-reduced sentence embedding, i.e. a vector of size N where N is the number of qubits in the quantum circuit (the code supports up to 8 dimensions). The advantage of this model is that it relies more heavily on quantum elements as compared with a DQC. Refer to the [D6.10: WP6 QNLP Report](https://www.neasqc.eu/wp-content/uploads/2023/11/NEASQC_D6_10_QNLP.pdf) for a more detailed description of the models.

The Beta 2 and 3 model architecture is defined in `neasqc_wp61/models/quantum/beta_2_3/beta_2_3_model.py`.

A model can be trained using the script `6_TrainQuantumModel.sh`; the following parameters must be specified as command line arguments:

* `-t` : the path to the training dataset.
* `-j` : the path to the validation dataset.
* `-v` : the path to the test dataset.
* `-o` : output directory path (for model(s) and results).
* `N` : the number of qubits of the fully-connected quantum circuit.
* `-s` : the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* `-i` : the number of iterations (epochs) for the training of the model.
* `-b` : the batch size.
* `-w` : the weight decay (this can be set to 0).
* `-x` : an integer seed for result replication.
* `-p` : the `pytorch` optimiser of choice.
* `-l` : the learning rate for the optimiser.
* `-z` : the step size for the learning rate scheduler.
* `-g` : the gamma for the learning rate scheduler.
* `-r` : the number of runs of the model (each run will be initialised with a different seed determined by the -x parameter).

Example:
  ```bash
  bash 6_TrainQuantumModel.sh -t ./data/datasets/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1_3.json -j ./data/datasets/labelled_newscatcher_dataset_tok_balanced_dev_llmrails_ember-v1_3.json -v ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test_llmrails_ember-v1_3.json -p Adam -x 42 -r 1 -i 10 -N 8 -s 0.01 -b 320 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/quantum/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1_3_beta_2_3
  ```
Note that the trainer file is `neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer_tests.py` and the pipeline file is `neasqc_wp61/data/data_processing/use_beta_2_3_tests.py`.

The script `6_2_TrainQuantumModelsForAllDatasets.sh` can be used to train and evaluate QNLP models for all pre-processed datasets. Note that this may require ~5-10 hours per dataset.

### Step 7 - Combining results
Results from individual `results.json` files can be combined using the `combine-results.py` script using the following parameters:
* `-base_dir <DIR>` base directory where `results.json` files must be searched (all sub-directories will be analysed).
* `-res_dir` output directory where TSV files with the combined results will be stored.

Example:
```bash
python data/data_processing/combine-results.py ./benchmarking/results/raw/ ./benchmarking/results/analysed
```

The script will create the following files:
* `time.tsv` - average training times of the 30 runs with a 95% confidence interval for each dataset.
* `train_loss.tsv` - average training loss scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `train_acc.tsv` - average training accuracy scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `val_loss.tsv` - average validation loss scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `val_acc.tsv` - average validation accuracy scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `test_loss.tsv` - average test loss scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `test_acc.tsv` - average test accuracy scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.

Each TSV file consists of 5 columns:
* Path of the `results.json` file
* Mean score
* Lower bound (mean - 95% confidence interval)
* Upper bound (mean + 95% confidence interval)
* Confidence interval

## Results
### labelled_newscatcher_dataset
| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI |
| -------------- | ----------------- | -------------- | -------------- | ------------- | --------------- |
| Classical FFNN | bert-base-uncased | full           | 75.49±0.05     | 75.44±0.18    | 76.41±0.43      |
| Classical FFNN | ember-v1          | full           | **80.89±0.06** | 80.13±0.38    | 81.16±1.69      |
| Classical FFNN | fasttext          | full           | 76.26±0.07     | 74.43±0.68    | 74.64±0.98      |
| Classical FFNN | bert-base-uncased | 8              | 56.47±0.08     | 55.84±0.69    | 55.25±0.9       |
| QNLP Beta 2/3  | bert-base-uncased | 8              | 52.41±0.04     | 50.3±1.97     | 48.04±4.02      |
| Classical FFNN | ember-v1          | 8              | 69.14±0.09     | 69.05±0.63    | 68.73±0.89      |
| QNLP Beta 2/3  | ember-v1          | 8              | 65.38±0.03     | 64.32±1.59    | 62.42±5.26      |
| Classical FFNN | fasttext          | 8              | 55.27±0.16     | 54.21±0.93    | 54.53±1.05      |
| QNLP Beta 2/3  | fasttext          | 8              | 46.05±0.27     | 41.85±3.22    | 40.8±5.15       |
| Classical FFNN | bert-base-uncased | 5              | 44.75±0.06     | 44.2±0.5      | 44.01±0.65      |
| QNLP Beta 2/3  | bert-base-uncased | 5              | 41.3±0.06      | 39.5±1.35     | 38.75±2.73      |
| Classical FFNN | ember-v1          | 5              | 66.03±0.08     | 65.24±0.97    | 65±1.22         |
| QNLP Beta 2/3  | ember-v1          | 5              | 63.63±0.03     | 61.63±1.74    | 59.84±5.03      |
| Classical FFNN | fasttext          | 5              | 38.34±0.09     | 37.36±0.43    | 38.43±0.49      |
| QNLP Beta 2/3  | fasttext          | 5              | 33.86±0.24     | 29.65±1.35    | 29.19±3.07      |
| Classical FFNN | bert-base-uncased | 3              | 32.23±0.1      | 31.99±0.2     | 32.28±0.29      |
| QNLP Beta 2/3  | bert-base-uncased | 3              | 31.21±0.08     | 29.74±0.54    | 29.59±1.73      |
| Classical FFNN | ember-v1          | 3              | 55.2±0.08      | 55.03±0.58    | 54.7±0.85       |
| QNLP Beta 2/3  | ember-v1          | 3              | 51.98±0.13     | 50.09±1.4     | 48.92±2.9       |
| Classical FFNN | fasttext          | 3              | 36.57±0.05     | 35.46±0.26    | 36.02±0.35      |
| QNLP Beta 2/3  | fasttext          | 3              | 32.27±0.36     | 28.24±0.99    | 27.99±2.58      |

### food-com-recipes-user-interactions
| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI |
| -------------- | ----------------- | ---- | ---------- | ---------- | ---------- |
| Classical FFNN | bert-base-uncased | full | 50.94±0.11 | 49.74±0.27 | 51.63±0.58 |
| Classical FFNN | ember-v1          | full | **63.36±0.08** | 63.15±0.21 | 63.9±0.82  |
| Classical FFNN | fasttext          | full | 51.57±0.11 | 51.01±0.46 | 50.34±0.56 |
| Classical FFNN | bert-base-uncased | 8    | 36.05±0.12 | 34.39±0.32 | 35.03±0.38 |
| QNLP Beta 2/3  | bert-base-uncased | 8    | 35.04±0.06 | 33.09±0.42 | 33.42±1.57 |
| Classical FFNN | ember-v1          | 8    | 61.53±0.09 | 60.95±0.65 | 59.9±0.75  |
| QNLP Beta 2/3  | ember-v1          | 8    | 57.36±0.06 | 52.98±1.74 | 51.17±4.24 |
| Classical FFNN | fasttext          | 8    | 40.81±0.18 | 40.74±0.45 | 39.74±0.49 |
| QNLP Beta 2/3  | fasttext          | 8    | 33.83±0.31 | 31.48±1.86 | 29.1±2.44  |
| Classical FFNN | bert-base-uncased | 5    | 32.63±0.1  | 32.35±0.21 | 32.78±0.29 |
| QNLP Beta 2/3  | bert-base-uncased | 5    | 32.27±0.05 | 31.4±0.21  | 31.63±1.19 |
| Classical FFNN | ember-v1          | 5    | 61.26±0.09 | 60.66±0.48 | 59.6±0.64  |
| QNLP Beta 2/3  | ember-v1          | 5    | 57.24±0.07 | 54.22±1.24 | 52.1±3.99  |
| Classical FFNN | fasttext          | 5    | 35.36±0.12 | 35.03±0.38 | 33.89±0.36 |
| QNLP Beta 2/3  | fasttext          | 5    | 31.78±0.21 | 30.66±1.24 | 28.7±2.06  |
| Classical FFNN | bert-base-uncased | 3    | 31.26±0.11 | 31.17±0.25 | 30.64±0.27 |
| QNLP Beta 2/3  | bert-base-uncased | 3    | 30.69±0.08 | 30.34±0.26 | 30.27±0.86 |
| Classical FFNN | ember-v1          | 3    | 57.28±0.13 | 56.09±0.55 | 55.05±0.71 |
| QNLP Beta 2/3  | ember-v1          | 3    | 53.72±0.16 | 50.7±1.19  | 48.99±3.1  |
| Classical FFNN | fasttext          | 3    | 34.11±0.08 | 33.99±0.15 | 33.02±0.22 |
| QNLP Beta 2/3  | fasttext          | 3    | 31.15±0.2  | 29.46±1.22 | 27.7±1.78  |



*This project has received funding from the European Union’s Horizon
2020 research and innovation programme under grant agreement No 951821.*

