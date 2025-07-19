## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/paster489/LMY.git
cd LMY
```

### 2. Create the Conda environment

Make sure you have Anaconda or Miniconda installed. 

My CUDA version is 12.7, if you have other version, use https://pytorch.org/get-started/locally/ for torch installation.

```bash
conda env create -f segformer_environment.yml
conda env create -f general_environment.yml
```

## 🗂️ Files Explanation

| File / Notebook       | Description                                                    | Environment                 |
| --------------------- | -------------------------------------------------------------- | --------------------------- |
| `Summary_Task_1.pptx` | Summary presentation of **Task 1**.                            | —                           |
| `Summary_Task_2.pptx` | Summary presentation of **Task 2**.                            | —                           |
| `data_analysis.ipynb` | Preliminary data analysis notebook.                            | `general_environment.yml`   |
| `preprocess.ipynb`    | Dataset preprocessing pipeline.                                | `general_environment.yml`   |
| `train.py`            | Script for training the model.                                 | `segformer_environment.yml` |
| `inference.ipynb`     | Notebook for performing inference and visualizing predictions. | `segformer_environment.yml` |

