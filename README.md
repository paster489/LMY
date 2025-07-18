## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/paster489/LMY.git
cd LMY
```

### 2. Create the Conda environment

Make sure you have Anaconda or Miniconda installed.

```bash
conda env create -f segformer_environment.yml
conda env create -f general_environment.yml
```

## Files Explanation

1) data_analysis.ipynb - for preliminary data analysis (conda environment per general_environment.yml)

2) preprocess.ipynb - data preprocessing (conda environment per general_environment.yml)

3) train.py - model training (conda environment per segformer_environment.yml)

4) inference.ipynb - infrence (conda environment per segformer_environment.yml)
