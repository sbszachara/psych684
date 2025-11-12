## How to Use:

### 1.) Install Environment


```conda env create -f psyc684_env.yml```

### 2.) Download Mozilla Common Voice

https://commonvoice.mozilla.org/en/datasets

Extrac them anywhere. For ease of use, the compressed files and their extracteed contents can remain safely in the local directory during development. They will be ignored by the .gitignore.

### 3.) Activate Environment and Run Files

The program takes the following arguments:

- the path to the folder containing 'en'.

```python whisper_finetune.py [path to folder]```