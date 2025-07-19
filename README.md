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

## 🗂️ Files Explanation

1) The summary of Task 1 inside the file "Summary_Task_1.pptx"

2) The summary of Task 2 inside the file "Summary_Task_2.pptx"

3) "data_analysis.ipynb" - for preliminary data analysis (conda environment per general_environment.yml)

4) "preprocess.ipynb" - data preprocessing (conda environment per general_environment.yml)

5) "train.py" - model training (conda environment per segformer_environment.yml)

6) "inference.ipynb" - infrence (conda environment per segformer_environment.yml)



| File / Notebook       | Description                                                    | Environment                 |
| --------------------- | -------------------------------------------------------------- | --------------------------- |
| `Summary_Task_1.pptx` | Summary presentation of **Task 1**.                            | —                           |
| `Summary_Task_2.pptx` | Summary presentation of **Task 2**.                            | —                           |
| `data_analysis.ipynb` | Preliminary data analysis notebook.                            | `general_environment.yml`   |
| `preprocess.ipynb`    | Dataset preprocessing pipeline.                                | `general_environment.yml`   |
| `train.py`            | Script for training the model.                                 | `segformer_environment.yml` |
| `inference.ipynb`     | Notebook for performing inference and visualizing predictions. | `segformer_environment.yml` |

