# Quantum Natural Language Processing : NEASQC WP 6.1


## Installing locally

### Obtaining a local copy of the code repository
In order to run the code locally, you will need to obtain a copy of the repository. To this end, you can either fork the repository or clone it. 

#### Cloning
We here detail the procedure to be followed for cloning.

  1. Open the code repository in your browser.
  2. Open the drop-down menu on the left. Click on the 'Switch branches/tags' button to select the <code>dev</code> branch.
  3. Click on the green code button and choose the cloning method you want to use, GitHub provides detailes steps for each method (HTTPS, SSH, etc).
  4. Open a terminal on your computer and navigate to the directory you wish to clone the repository into. 
  5. Run the following command in your terminal:
      <pre><code>$ git clone &ltcopied_url&gt</pre></code></li>
  6. Navigate into the cloned repository by using 
     <pre><code>$ cd WP6_QNLP</pre></code> </li>
  7. Run the following command in your terminal: 
      <pre><code>$ git checkout dev</pre></code></li>


### Creating a new environment and installing required packages

#### Python version
The Python version required to run the scripts and notebooks of this repository is Python 3.10. Due to the use of myQLM in one of our models, only [python.org](https://www.python.org/downloads/macos/) and `brew` python distributions are supported. `conda` and `pyenv` won't work.


1. If Python3.10 hasn't been installed (<em><strong>using brew</strong></em>) yet, or Python3.10 has been installed using any other method:
    * Install brew following the instructions detailed [here](https://brew.sh/). 
    * Run the following command on the terminal to install it on your local device.
      <pre><code>$ brew install python@3.10</pre></code>
    * By running the following command on the terminal, we make sure that we will link the recently installed Python3.10 to the environmental variable <em><strong>python3.10</em></strong>.
      <pre><code>$ brew link --overwrite python@3.10</pre></code>
      We may get an error if there was any other environmental variable named <em><strong>python3.10</em></strong>. In that case we must remove the variable from the PATH with the command: 
      <pre><code>$ unset python3.10</pre></code>
      and then use brew link command again.
2. If Python3.10 has been already installed (<em><strong>using brew</em></strong>):
    * We make sure that we have it linked to the the environmental variable <em><strong>python3.10</em></strong> using the command shown on section 1.2. A warning message will appear if we have it already linked (we can ignore it).
    * We make sure that there are no packages installed on the global Python by running the command: 
      <pre><code>$ python3.10 -m pip list</pre></code>
      In the case where there were packages installed on the global Python we should uninstall them with the command: 
      <pre><code>$ python3.10 -m pip uninstall &ltundesired package&gt</pre></code>



#### Dependencies

##### Poetry
Note: we here assume that you are happy to use `poetry`'s lightweight virtual environenment set-up. If for some reason you prefer to use an external virtual environemnt, simply activate it before using `poetry`, it will respect its precedence.
<ol>
  <li> Make sure you have <code>poetry</code> installed locally. This can be done by running  <pre><code>$ poetry --version</pre></code> in your shell and checking the output. If installed, proceed, if not, follow instructions on their official website <a href="https://python-poetry.org/docs/#installation">here</a>. </li>
  <li> <code>cd</code> to the root of the repository where the files <code>pyproject.toml</code> and <code>poetry.lock</code> are located. </li>
  <li> Run the following command in your shell: <pre><code>$ poetry install</pre></code>
  If you also want to install the dependancies used to build sphinx documentation, run the following command insted:
  <pre><code>poetry install --with docs</pre></code></li>
</ol>


##### Virtualenv
Note: As mentioned previously, one of our models uses myQLM, which will not work on a virtual env. However, all other models should work without issues.
<ol>
  <li>To create a virtual environment, go to the directory where you want to create it and run the following command in the terminal:
    <pre><code>$ python3.10 -m venv &ltenvironment_name&gt</pre></code></li>
  <li> Activate the environment (see instructions <a href="#venv_activation">here</a>). If the environment has been activated correctly its name should appear in parentheses on the left of the user name in the terminal.</li>
  <li>Ensure pip is installed. If if not, follow instructions found <a href="https://pip.pypa.io/en/stable/installation/">here</a> to install it.</li>
  <li> To install the required packages, run the command:
    <pre><code>$ python3.10 -m pip install -r requirements.txt</pre></code></li>
</ol>




#### Activating the virtual environments

##### Poetry
To activate <code>poetry</code>'s default virtual environment, simply run:
<pre><code>poetry shell</code></pre>
inside your terminal. More details can be found <a href="https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment">here</a>.

##### Virtualenv
To activate your virtualenv, simply type the following in your terminal:
<pre><code>$ source &ltenvironment_name&gt/bin/activate</pre></code>
Note that contrary to <code>poetry</code>, this virtual environment needs to be activated before you install the requirements.

## Models

In this section we present our main models - Alpha 3, Beta 2 and Beta 3. 

To learn about our other models, please see [path to archive models] and read the README file for that directory.

### Alpha 3 

#### Description

Alpha 3 follows a dressed quantum circuit(DQC) architecture, meaning that it combines a classical network architecture with a quantum circuit. A fully-connected quantum circuit is sandwiched between multi-layer perceptrons (MLPs). This model performs multiclass classification of natural language data.

The first MLP takes in sentence embeddings of dimension N and reduces them to an output of dimension Q ( < N ) where Q is the number of qubits of the circuit.

The second MLP takes the output of the quantum circuit as input (a vector of dimension Q), and outputs a vector of dimension C, where C is the number of classes. The final prediction of the class is made from this vector.

#### Files

There are two slight variations of the Alpha 3 model in this repository:

* The **standard** version, *alpha_3_multiclass_tests*. This model trains on a training dataset, and upon training, evaluates on a test dataset.

* The **cross-validation** version, *alpha_3_multiclass*, used internally for preliminary experiments. This model reads a dataset that in which sentences are labelled with their corresponding split. For each split S, the model takes all other sentences as training data, and uses the split S as validation data. This allows one to perform k-fold cross-validation on the model.

#### Datasets (standard model)

To run Alpha 3, you must have a dataset in CSV format consisting of 3 columns:
 
* 'class' - this column will contain the numbers that represents the class of each sentence (e.g. in binary classification, this could be 0 for a negative sentence, and 1 for a positive one). The numbers should be in the range [0, C-1] where C is the total number of classes.

* 'sentence' - this column will contain the natural language sentences that will be classified by the model.

* 'sentence_embedding' - this column will contain the sentence embeddings (e.g. BERT, ember-v1, etc.) corresponding to each sentence. The embeddings should be in standard list/vector notation format, e.g. [a,b,...,z].

If you have a CSV file with 'class' and 'sentence' labels, and you want to add a column with the corresponding BERT embeddings, you may use our [dataset_vectoriser.py](/neasqc_wp61/data/data_processing/dataset_vectoriser.py) script as follows:
```
python dataset_vectoriser.py <path-to-your-csv-dataset> -e sentence
```
This will produce a new CSV file identical to your dataset but with an additional column 'sentence_embedding' containing the embeddings for each sentence.

#### Datasets (cross-validation model)

If you wish to use our cross-validation version of the Alpha 3 model, simply ensure that your dataset has an additional column:
* 'split' - this column contains numbers that indicate the split to which the sentence belongs to. For K-fold cross-validation, these numbers should be in the range [0, K-1]

Adding this column is simple using the <code>pandas</code> Python library. Make sure you choose an appropriate number of splits based on the size of your dataset.

#### Running the model

The model has a number of parameters that must be specified through flags in the command line. These are:

* -s :