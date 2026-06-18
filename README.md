# Multiclass ADis-QSAR

A machine learning framework for multiclass acute toxicity classification of organophosphorus compounds using Activity-Difference (ADis) descriptors and molecular fingerprints.

<img width="4445" height="1771" alt="Image" src="https://github.com/user-attachments/assets/773f8283-75f0-448e-8163-bcc85a5a1ec3" />

## Previous Work
This work extends the original binary ADis-QSAR framework to multiclass toxicity classification.
Orginial ADis-QSAR repository:
https://github.com/pkj0421/ADis-QSAR


## Installation

Create and activate the conda environment:

```bash
conda env create -f ADis-QSAR-env.yaml
conda activate ADis-QSAR-env
```

## Data Preprocessing

Class information is required for dataset preparation.

The input files (`class1.tsv`, `class2.tsv`, and `class3.tsv`) are derived from `Supple.Dataset_OPcompounds(687).xlsx` and should be placed in the `dataset` directory.

The preprocessing step generates G1 sets and splits the remaining compounds into training, validation, and test sets.

### Workflow

* Input: `class1.tsv`, `class2.tsv`, `class3.tsv`
* Construct G1 using only Class 1 compounds (`--g1_size`)
* Construct G2 using:

  * Remaining Class 1 compounds
  * All Class 2 compounds
  * All Class 3 compounds
* Split G2 into training, validation, and test sets using a class-balanced ratio of 8:2:2
* Generate descriptors, apply scaling, and save the processed datasets

### Example

```bash
python Preprocessing_multi.py 
    --class1 class1.tsv 
    --class2 class2.tsv 
    --class3 class3.tsv 
    --g1_size 30 
    --o output_path
```
Example preprocessing results can be found in the Results/Preprocessing directory.

## Vary param run

This script evaluates the performance of the Multiclass ADis-QSAR framework under different parameter settings.

### Evaluated Parameters

#### Number of center structures (G1)

```text
30, 50, 80, 100
```

#### Fingerprint radius

```text
ECFP4, ECFP6
```

#### Fingerprint size (bits)

```text
256, 512
```

#### Feature scaling method

```text
MinMax 
Standard
Robust
```

### Example

```bash
python Vary_params_run_multi.py
```
Example results obtained using different parameter settings can be found in the `Results/Vary_params_results` directory.

## External Dataset Prediction

An example external dataset is provided in `Dataset/External_set.xlsx`.

Use a trained model to predict toxicity classes for external compounds:

### Example

```bash
python Predict_external_set.py 
    --external External_set.xlsx 
    --model model.pkl 
    --o output_path
```

## Contact
JeongHyeon Kim  
Chungnam National University\
[Chemoinformatics & Molecular Modeling](https://ccim.cnu.ac.kr/cim/index.do)\
Email: [wjdgus8105@naver.com](mailto:wjdgus8105@naver.com)
