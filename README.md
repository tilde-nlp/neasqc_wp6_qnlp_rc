# Quantum Natural Language Processing : NEASQC WP 6.1

## Pre-setup
The following guide will walk you thorugh how to use our models. Prior to following those steps, you should ensure that you:
- Have a copy of the repository on your local machine.
- Have `python 3.10` installed on your local machine. Our models *might* turn out to be compatible with later versions of python but they were designed with and intended for 3.10.
- Have `poetry` installed on your local machine. You can follow the instructions on <a href="https://python-poetry.org/docs/#installation">the official website</a>.

## Setup
1. Position yourself in the `dev` branch.
2. Position yourself in the root of the repository where the files <code>pyproject.toml</code> and <code>poetry.lock</code> are located.
3. Run the command: <pre><code>$ poetry install</pre></code>. If you also want to install the dependencies to build `sphinx` documentation, use the command <pre><code>poetry install --with docs</pre></code> instead.
4. Activate the `poetry` usibg the command: <pre><code>poetry shell</code></pre>. More details can be found <a href="https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment">here</a>.

Note: as long as the formating is respected, users will be able to use our models on any dataset they like. Here for simplicity we assume datasets are placed in `neasqc_wp61/data/datasets`. The datasets we used can be found [here](https://github.com/tilde-nlp/neasqc_wp6_qnlp/tree/v2-classical-nlp-models/neasqc_wp61/data/datasets). Please note that to use these yourself you will need to add the `class` and `sentence` column headers and convert the format to `csv`.


## Models
In this section we present our main models - Alpha 3, Beta 2 and Beta 3. 

Note that each model takes in a different input, but all produce the same output: 
* A `JSON` file containing the details of all the runs (loss, accuracy, runtime, etc.)
* A `.pt` file for each run with the final weights of the model at the end of the run.

Note that all models follow the same file structure. They are divided into models, trainers and pipeline files to run.



### Alpha3

####  General overview
Alpha3 follows a dressed quantum circuit (DQC) architecture, meaning that it combines a classical network architecture with a quantum circuit. A fully-connected quantum circuit is sandwiched between linear layers. This model performs multiclass classification of natural language data. The first classical layer takes in sentence embeddings of dimension D and reduces them to an output of dimension N where N is the number of qubits of the circuit. The second classical takes the output of the quantum circuit as input (a vector of dimension N), and outputs a vector of dimension K, where K is the number of classes. The final prediction of the class is made from this vector.

The core of the model is defined in 'alpha_3_multiclass_model.py'. There are two ways to use this model, the **standard** way which relies on training the model on a *single* training dataset and evaluation it on a validation dataset, and the k-fold validation one. Each option has an model, trainer and pipeline file which straps them together.

##### Dataset formatting
For Alpha3, the dataset must be:

- In CSV format
- With 3 columns:
  * 'class' - this column will contain the numbers that represents the class of each sentence (e.g. in binary classification, this could be 0 for a negative sentence, and 1 for a positive one). The numbers should be in the range [0, C-1] where C is the total number of classes.
  * 'sentence' - this column will contain the natural language sentences that will be classified by the model.
  * 'sentence_embedding' - this column will contain the sentence embeddings (e.g. BERT, ember-v1, etc.) corresponding to each sentence. The embeddings should be in standard list/vector notation format, e.g. [a,b,...,z].

If you have a CSV file with 'class' and 'sentence' labels, and you want to add a column with the corresponding BERT embeddings, you may use our `dataset_vectoriser.py` script. From the root of the repo do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script:
```
python dataset_vectoriser.py <path-to-your-csv-dataset> -e sentence
```
This will produce a new CSV file identical to your dataset but with an additional column 'sentence_embedding' containing the embeddings for each sentence. This file will be saved to the same directory in which your dataset lives.

##### Command line arguments
The model has a number of parameters that must be specified through flags in the command line. These are:
* `-s` : an integer seed for result replication.
* `-i` : the number of iterations (epochs) for the training of the model.
* `-r` : the number of runs of the model (each run will be initialised with a different seed determined by the -s parameter).
* `-u` : the number of qubits of the fully-connected quantum circuit
* `-d` : q_delta, i.e. the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* `-p` : the <code>PyTorch</code> optimiser of choice.
* `-b` : the batch size.
* `-l`: the learning rate for the optimiser.
* `-w` : the weight decay (this can be set to 0).
* `-z` : the step size for the learning rate scheduler.
* `-g` : the gamma for the learning rate scheduler.
* `-o` : path for the output file.


#### Standard usage
The **standard** usage can be found in `alpha_3_multiclass_tests`. The trainer file is `alpha_3_multiclass_trainer_tests.py` and the pipeline is `use_alpha_3_multiclass_tests.py`.

##### Additional command line arguments for standard usage
* `-t` : the path to the training dataset.
* `-v` : the path to the test dataset.

##### Standard example
1. From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
2. Use the following command:
```
bash 6_Classify_With_Quantum_Model.sh -m alpha_3_multiclass_tests -t <path to train data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 4 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/
```



#### Cross-validation usage
In the k-fold validation use, input sentences are labelled with their corresponding split. For each split S, the the training dataset will be all other splits and the given split S will be used as validation.
The k-fold usage can be found in `alpha_3_multiclass`. The trainer file is `alpha_3_multiclass_trainer.py` and the pipeline file is `use_alpha_3_multiclass.py`.

##### Dataset formatting for cross-validation
This assumes the dataset is formatted as per standard Alpha3 format and with one additional column: `split`. The 'split' column contains numbers that indicate the split to which the sentence belongs to. For K-fold cross-validation, these numbers should be in the range [0, K-1]. Once this column is present you can run the `dataset_vectoriser.py` script as per above.

##### Additional command line arguments for cross-validation usage
* `-f` : path to the dataset containing the training and validation data with the split information as detailed before.
* `-v` : path to the test dataset.

##### Cross-validation example
1. From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
2. Use the following command:
```
bash 6_Classify_With_Quantum_Model.sh -m alpha_3_multiclass -f <path to split train and validation data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 4 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/
```




### Beta 2 

#### General overview
Beta 2 follows what we call a *semi-dressed quantum circuit* (SDQC) architecture. Here, the first layer of a DQC is stripped and the classical input in handed directly to the PQC once it has been brought to the correct dimension. The input to the circuit is a PCA-reduced sentence embedding, i.e. a vector of size N where N is the number of qubits in the quantum circuit. One starts with a sentence embedding of dimension D, reduces its dimension to N using a PCA, and this resulting vector is plugged into the quantum circuit. The advantage of this model is that it relies more heavily on quantum elements as compared with a DQC.

The Beta 2 model architecture is defined in the `beta_2_3_model.py` file. Note here that Beta3, given its very minor deviation from Beta2, is defined in the same file. See next section for more details on Beta3.

##### Dataset formatting
To run Beta 2, you must have a dataset in CSV format consisting of 4 columns:
 
* 'class' - this column will contain the numbers that represents the class of each sentence (e.g. in binary classification, this could be 0 for a negative sentence, and 1 for a positive one). The numbers should be in the range [0, C-1] where C is the total number of classes.

* 'sentence' - this column will contain the natural language sentences that will be classified by the model.

* 'sentence_embedding' - this column will contain the sentence embeddings (e.g. BERT, ember-v1, etc.) corresponding to each sentence. The embeddings should be in standard list/vector notation format, e.g. [a,b,...,z].

* 'reduced_embedding' - this column will contain the PCA-reduced sentence embeddings, in the same format as the full sentence embeddings.

Assuming you have a dataset with the first three columns (from following the instructions for Alpha 3), you can generate a new dataset with the additional 'reduced_embedding' column by using our `generate_pca_test_dataset.py` script. Simply open the script and change the path in line 5 to that of your dataset, and the path in line 18 to your desired output name and directory. Save and close. From the root of the repo do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script:
```
python generate_pca_test_dataset.py
```
This will produce a new CSV file with the additional 'reduced_embedding' column. Make sure to do this both for your traing and testing datasets. 

##### Command line arguments
The model has a number of parameters that must be specified through flags in the command line. These are:

* `-s` : an integer seed for result replication.
* `-i` : the number of iterations (epochs) for the training of the model.
* `-r` : the number of runs of the model (each run will be initialised with a different seed determined by the -s parameter).
* `-u` : the number of qubits of the fully-connected quantum circuit
* `-d` : q_delta, i.e. the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* `-p` : the <code>PyTorch</code> optimiser of choice.
* `-b` : the batch size.
* `-l` : the learning rate for the optimiser.
* `-w` : the weight decay (this can be set to 0).
* `-z` : the step size for the learning rate scheduler.
* `-g` : the gamma for the learning rate scheduler.
* `-o` : path for the output file.
* `-f` : the path to the training dataset (in the case of the **standard version**) or to the dataset containing the training and validation data (in the case of the **cross-validation version**).
* `-v` : the path to the test dataset.

#### Standard usage
The trainer file is `beta_2_3_trainer_tests.py` and the pipeline `use_beta_2_3_tests.py`.

##### Standard example
1. From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
2. Use the following command:
```
bash 6_Classify_With_Quantum_Model.sh -m beta_2_tests -f <path to train dataset> -v <path to test dataset> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```

#### Cross-validation usage
The trainer file is `beta_2_3_trainer.py` and the pipeline `use_beta_2_3.py`.

##### Dataset formatting for cross-validation
Ensure that your dataset (with the train and validation data) has an additional column: 'split'. This column contains numbers that indicate the split to which the sentence belongs to. For K-fold cross-validation, these numbers should be in the range [0, K-1]

Once this is done, you need an addtional set of columns: `reduced_embedding_i`. These columns contain the PCA-reduced embeddings, with `i` indicating that the embeddings have been reduced with a PCA that has been fitted on the training data for split `i` (that is, all splits *different from* i). If you have a dataset with all other columns, these columns are easy to add using our `generate_pca_dataset.py` script. 

Simply open the script, edit line 5 to include the path to your dataset containing the train and validation data, and edit line 30 with your desired output file path and name. Then save and close. From the root of the repository do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script with
```
python generate_pca_dataset.py 
```
This will produce a CSV file in the desired output path with the required format and columns.

For the test dataset, you do not need the 'split' columns, and you can use the `generate_pca_test_dataset.py` script, which is described in the previous section, to reduce the embeddings in the `sentence_embedding` column and add them to a new `reduced_embedding` column.

##### Cross-validation example
1. From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
2. Use the following command:
```
bash 6_Classify_With_Quantum_Model.sh -m beta_2 -f <path to split train and validation data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```




### Beta3

#### General overview
Beta3 is simply a different flavour of Beta2. Here, the vector used as input to the PQC is obtained from an adaptive-sized embedding instead of via a PCA.
The core model is defined in the file `beta_2_3_model.py`.

##### Dataset formatting
To run Beta 2, you must have a dataset in CSV format consisting of 3 columns:
 
* 'class' - this column will contain the numbers that represents the class of each sentence (e.g. in binary classification, this could be 0 for a negative sentence, and 1 for a positive one). The numbers should be in the range [0, C-1] where C is the total number of classes.

* 'sentence' - this column will contain the natural language sentences that will be classified by the model.

* 'reduced_embedding' - this column will contain the reduced fastText embeddings in standard vector format, i.e. [a,b,...,z]

Assuming you have a CSV file with the first two columns, if you want to generate the reduced fastText embeddings and add them to a new 'reduced_embedding' column, you can use our [generate_fasttext_dataset.py](/neasqc_wp61/data/data_processing/generate_fasttext_dataset.py) script. Simply open the file, edit line 7 with the path to your CSV file, edit line 25 with your desired output path, then save and close. From the root of the repo do:
```
cd neasqc_wp61/data/data_processing/
```
and then run the script:
```
python generate_fasttext_dataset.py
```
which will generate a new CSV file with the fastText embeddings in the 'reduced_embedding' column. Make sure to do this both for your training and testing datasets.

##### Command line arguments
The model has a number of parameters that must be specified through flags in the command line. These are:

* `-s` : an integer seed for result replication.
* `-i` : the number of iterations (epochs) for the training of the model.
* `-r` : the number of runs of the model (each run will be initialised with a different seed determined by the -s parameter).
* `-u` : the number of qubits of the fully-connected quantum circuit
* `-d` : q_delta, i.e. the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* `-p` : the <code>PyTorch</code> optimiser of choice.
* `-b` : the batch size.
* `-l` : the learning rate for the optimiser.
* `-w` : the weight decay (this can be set to 0).
* `-z` : the step size for the learning rate scheduler.
* `-g` : the gamma for the learning rate scheduler.
* `-o` : path for the output file.
* `-f` : the path to the training dataset (in the case of the **standard version**) or to the dataset containing the training and validation data (in the case of the **cross-validation version**).
* `-v` : the path to the test dataset.

#### Standard usage
The trainer file is `beta_2_3_trainer_tests.py` and pipeline `use_beta_2_3_tests.py`.

##### Standard example
1. From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
2. Use the following command:
```
bash 6_Classify_With_Quantum_Model.sh -m beta_3_tests -f <path to train dataset> -v <path to test dataset> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```

#### Cross-validation usage
The trainer file is `beta_2_3_trainer.py` and pipeline `use_beta_2_3.py`.

##### Dataset formatting for cross-validation
For the cross-validation version, you want the same columns as above, plus the following:
* 'split' - this column contains numbers in the range [0, K-1] where K is the number of folds in the cross-validation proceedure. This number will indicate what split the data belongs to.

If you have a dataset with the 'class', 'split' and 'sentence' column, and want to vectorise the sentences using fastText and add the result embeddings in a new 'reduced_embedding' column, you can use [generate_fasttext_dataset.py](/neasqc_wp61/data/data_processing/generate_fasttext_dataset.py) as described in the previous subsection.

##### Cross-validation example
1. From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
2. Use the following command:
```
bash 6_Classify_With_Quantum_Model.sh -m beta_3 -f <path to split train and validation data>  -v <path to test data> -p Adam -s 42 -r 1 -i 10 -u 8 -d 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```