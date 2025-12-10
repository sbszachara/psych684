## How to Use:

### 1.) Install Environment


```conda env create -f psyc684_env.yml```

### 2.) Download Mozilla Common Voice

https://commonvoice.mozilla.org/en/datasets

Extract them anywhere. For ease of use, the compressed files and their extracted contents can remain safely in the local directory during development. They will be ignored by the .gitignore.

### 3.) Activate Environment and Run Files

The program takes the following arguments:

- the path to the folder containing 'en'.
- the sample size of sentences to use in training.
- the random seed to use for subset selection and train-test split.


```python whisper_finetune.py [path to folder]```

```python updated_WF_grafn.py [path to folder]```
