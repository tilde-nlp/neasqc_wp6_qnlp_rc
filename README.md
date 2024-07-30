# Quantum Natural Language Processing : NEASQC WP 6.1
This readme gives a brief introduction to the models presented in the [NEASQC](https://www.neasqc.eu/) work-package 6.1 on quantum natural language processing and provides guidance on how to run each model. We give a brief overview of each of the three models here: Alpha3, Beta2 and Beta3, but encourage users to refer to the [corresponding report](https://www.neasqc.eu/wp-content/uploads/2023/11/NEASQC_D6_10_QNLP.pdf) containing an extended abstract presenting the models more in depth.





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


### Models use

#### Models outputs
Every model produces the same output:
* A `JSON` file containing the details of all the runs (loss, accuracy, runtime, etc.)
* A `.pt` file for each run with the final weights of the model at the end of the run.

#### Models arguments
When launching the training of a model with `6_run_quantum_models.sh`, the following parameters must be specified as command line arguments.

* `-m` : the name of the model
* `-d` : the path to the training dataset (in the case of the **standard version**) or to the dataset containing the training and validation data (in the case of the **cross-validation version**).
* `-t` : alternatively to `-d` some models use a path to the training file with this flag.
* `-v` : the path to the test (or validation) dataset.
* `-o` : path for the output file.
* `N` : the number of qubits of the fully-connected quantum circuit.
* `-s` : the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* `-i` : the number of iterations (epochs) for the training of the model.
* `-b` : the batch size.
* `-w` : the weight decay (this can be set to 0).
* `-x` : an integer seed for result replication.
* `-p` : the <code>PyTorch</code> optimiser of choice.
* `-l` : the learning rate for the optimiser.
* `-z` : the step size for the learning rate scheduler.
* `-g` : the gamma for the learning rate scheduler.
* `-r` : the number of runs of the model (each run will be initialised with a different seed determined by the -x parameter).


### Datasets
For simplicity we assume datasets are placed in `neasqc_wp61/data/datasets`. The datasets we used can be found [here](https://github.com/tilde-nlp/neasqc_wp6_qnlp/tree/v2-classical-nlp-models/neasqc_wp61/data/datasets). Please note that to use these yourself you will need to add the `class` and `sentence` column headers and convert the format to `.csv`.

Note that in some cases and depending on the datasets you wish to use and their naming conventions, parts of the path to the dataset might have to be adjusted in the source code.

#### Dataset formatting
In order to be compatible with our models, please ensure your dataset:
- Is in `CSV` format.
- Contains the following columns:
  - `sentence` (string) - the natural language utterance to be classified by the model.
  - `class` (integer) - the class of the sentence. Numbers should be in the range [0, K-1] where K is the total number of classes.
  - `sentence_embedding` (vector of floats) - the vector representation of the sentence obtained using some embedding (BERT, ember-v1 or other).

##### Generating and populating an embedding column (all models)
If your dataset contains only `sentence` and `class` columns but is devoide of an `embedding` one, we provide the `dataset_vectoriser.py` script to generate a BERT embedding.
* Position yourself at the root of the repository.
* Navigate to the location of the script by using <pre><code>cd neasqc_wp61/data/data_processing/ </pre></code>
* Run the script using <pre><code>python dataset_vectoriser.py PATH_TO_DATASET -e sentence </pre></code> where `PATH_TO_DATASET` is replaced by the path to your dataset.
This will produce a new `CSV` file identical to your dataset but with an additional column 'sentence_embedding' containing the embeddings for each sentence. This file will be saved to the directory where your original dataset is located.

##### Generating and populating a reduced embedding column (Beta models only)
An extra `reduced_embedding` column is needed. It will contain a compressed version of the `embedding` of a small enough dimensionality to be used as input to a quantum circuit. The compression method will depend on the model used.
Assuming your dataset already contains the basic 3 columns mentionned above, you can create a `reduced_embedding` column for Beta2. Note that we here only discuss the *standard* use of Beta models, the discussion on data pre-processing for cross-validation experiments is separate.

##### Generating Beta2 `reduced_embedding` column
* Modify the input and output paths in `neasqc_wp61/data/data_processing/generate_pca_test_dataset.py` to your desired input and output paths.
* Position yourself at the root of the repository.
* Navigate to the location of the script by using <pre><code>cd neasqc_wp61/data/data_processing/ </pre></code>
* Run <pre><code>python generate_pca_test_dataset.py </pre></code>
This will produce a new CSV file with the additional 'reduced_embedding' column. Make sure to do this both for your traing and testing datasets.

##### Generating Beta3 `reduced_embedding` column
* Modify the input and output paths in `neasqc_wp61/data/data_processing/generate_fasttext_dataset.py` to your desired input and output paths.
* Position yourself at the root of the repository.
* Navigate to the location of the script by using <pre><code>cd neasqc_wp61/data/data_processing/ </pre></code>
* Run <pre><code>python generate_fasttext_dataset.py </pre></code>
This will produce a new CSV file with the additional 'reduced_embedding' column. Make sure to do this both for your traing and testing datasets.
which will generate a new CSV file with the fastText embeddings in the 'reduced_embedding' column. Make sure to do this both for your training and testing datasets.




## Alpha3
Alpha3 follows a dressed quantum circuit (DQC) architecture, meaning that it combines a classical network architecture with a quantum circuit. A fully-connected quantum circuit is *sandwiched* between classical linear layers. This model performs multiclass classification of natural language data. The first classical layer takes in sentence embeddings of dimension D and reduces them to an output of dimension N where N is the number of qubits of the circuit. The second classical layer takes the output of the quantum circuit as input (a vector of dimension N), and outputs a vector of dimension K, where K is the number of classes. The final prediction of the class is made from this vector.

The core of the model is defined in `alpha_3_multiclass_model.py`. There are two ways to use this model, the **standard** way which relies on training the model on a *single* training dataset and evaluation it on a validation dataset, and the k-fold validation one. Each option has an model, trainer and pipeline file which ties them together.


### Standard example usage
* Position yourself at the root of the directory.
* Navigate to `neasqc_wp61` by using <pre><code> cd neasqc_wp61 </code></pre>
* Run <pre><code> bash 6_run_quantum_models.sh -m alpha_3_multiclass_tests -t PATH_TO_TRAIN  -v PATH_TO_TEST -p Adam -x 42 -r 1 -i 10 -N 4 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/  </pre></code>
Note that the trainer file is `neasqc_wp61/models/quantum/alpha/module/alpha_3_multiclass_trainer_tests.py` and the pipeline is `neasqc_wp61/data/data_processing/use_alpha_3_multiclass_tests.py`.
<!--- Runs... but are they the right files?
bash 6_run_quantum_models.sh -m alpha_3_multiclass_tests -t data/datasets/agnews_balanced_test_bert_pca.csv  -v data/datasets/agnews_balanced_test_bert_pca.csv -p Adam -x 42 -r 1 -i 10 -N 4 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/
--->


### Cross-validation usage
In the k-fold validation use, input sentences are labelled with their corresponding split. For each split S, the the training dataset will be all other splits and the given split S will be used as validation.
The k-fold usage can be found in `alpha_3_multiclass`. The trainer file is `alpha_3_multiclass_trainer.py` and the pipeline file is `use_alpha_3_multiclass.py`.

#### Dataset formatting for cross-validation
This assumes the dataset is formatted as per standard Alpha3 format and with one additional column: `split`. The `split` column contains numbers that indicate the split to which the sentence belongs to. For K-fold cross-validation, these numbers should be in the range [0, K-1]. Once this column is present you can run the `dataset_vectoriser.py` script.

#### Cross-validation example
* From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
* Run <pre><code> bash 6_run_quantum_models.sh -m alpha_3_multiclass -d PATH_TO_TRAIN -v PATH_TO_TEST -p Adam -x 42 -r 1 -i 10 -N 4 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/  </pre></code>





## Beta 2 
Beta 2 follows what we call a *semi-dressed quantum circuit* (SDQC) architecture. Here, the first layer of a DQC is stripped. The classical input is handed directly to the PQC once it has been brought to the correct dimension. The input to the circuit is a PCA-reduced sentence embedding, i.e. a vector of size N where N is the number of qubits in the quantum circuit. One starts with a sentence embedding of dimension D, reduces its dimension to N using a PCA, and this resulting vector is plugged into the quantum circuit. The advantage of this model is that it relies more heavily on quantum elements as compared with a DQC.

The Beta 2 model architecture is defined in `neasqc_wp61/models/quantum/beta_2_3/beta_2_3_model.py`. Note here that Beta3, given its very minor deviation from Beta2, is defined in the same file. See next section for more details on Beta3.


### Standard example usage
* Position yourself at the root of the directory.
* Navigate to `neasqc_wp61` by using <pre><code> cd neasqc_wp61 </code></pre>
* Run <pre><code> bash 6_run_quantum_models.sh -m beta_2_tests -d PATH_TO_TRAIN -v PATH_TO_TEST -p Adam -x 42 -r 1 -i 10 -N 8 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw  </pre></code>
Note that the trainer file is `neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer_tests.py` and the pipeline `neasqc_wp61/data/data_processing/use_beta_2_3_tests.py`.
<!--- Runs... but are they the right files?
bash 6_run_quantum_models.sh -m beta_2_tests -d data/datasets/agnews_balanced_test_bert_pca.csv -v data/datasets/agnews_balanced_test_bert_pca.csv -p Adam -x 42 -r 1 -i 10 -N 8 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
--->


### Cross-validation usage
The trainer file is `beta_2_3_trainer.py` and the pipeline `use_beta_2_3.py`.

#### Dataset formatting for cross-validation
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
For the test dataset, you do not need the `split` columns, and you can use the `generate_pca_test_dataset.py` script, which is described in the previous section, to reduce the embeddings in the `sentence_embedding` column and add them to a new `reduced_embedding` column.

#### Cross-validation example
* From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
* Run <pre><code> bash 6_run_quantum_models.sh -m beta_2 -d PATH_TO_TRAIN -v PATH_TO_TEST -p Adam -x 42 -r 1 -i 10 -N 8 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw  </pre></code>




## Beta3
Beta3 is simply a different flavour of Beta2. Here, the vector used as input to the PQC is obtained from an adaptive-sized embedding instead of via a PCA. The core model is defined in the same file as Beta2.


### Standard example usage
The trainer file is `neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer_tests.py` and pipeline `neasqc_wp61/data/data_processing/use_beta_2_3_tests.py`.
* Position yourself at the root of the directory.
* Navigate to `neasqc_wp61` by using <pre><code> cd neasqc_wp61 </code></pre>
* Run <pre><code> bash 6_run_quantum_models.sh -m beta_3_tests -d PATH_TO_TRAIN -v PATH_TO_TEST -p Adam -x 42 -r 1 -i 10 N 8 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw  </pre></code>
Trainer and pipeline are the same as for Beta2.
<!--- Runs... but are they the right files?
bash 6_run_quantum_models.sh -m beta_2_tests -d data/datasets/agnews_balanced_test_bert_pca.csv -v data/datasets/agnews_balanced_test_bert_pca.csv -p Adam -x 42 -r 1 -i 10 -N 8 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
--->


### Cross-validation usage
The trainer file is `beta_2_3_trainer.py` and pipeline `use_beta_2_3.py`.

#### Dataset formatting for cross-validation
For the cross-validation version, you want the same columns as above, plus the following:
* 'split' - this column contains numbers in the range [0, K-1] where K is the number of folds in the cross-validation proceedure. This number will indicate what split the data belongs to.

If you have a dataset with the `class`, `split` and `sentence` column, and want to vectorise the sentences using fastText and add the result embeddings in a new 'reduced_embedding' column, you can use `generate_fasttext_dataset.py` as described in the previous subsection.

#### Cross-validation example
1. From the root of the directory, navigate to `neasqc_wp61` by using: <pre><code>cd neasqc_wp61</code></pre>
2. Use the following command:
```
bash 6_run_quantum_models.sh -m beta_3 -d PATH_TO_TRAIN -v PATH_TO_TEST -p Adam -x 42 -r 1 -i 10 -N 8 -s 0.01 -b 2048 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw
```




*This project has received funding from the European Union’s Horizon
2020 research and innovation programme under grant agreement No 951821.*

