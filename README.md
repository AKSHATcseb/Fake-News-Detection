# Fake News Detection Using Machine Learning

This project implements a machine learning–based system to classify news articles as **fake** or **real**. With the rapid spread of information through online platforms, fake news has become a serious problem. This project uses supervised learning techniques to detect misleading or fabricated news.

The implementation is done in a Jupyter Notebook (`Code.ipynb`) using Python and popular data science libraries.

## Project Objectives

- To build a classification model that can distinguish between fake and real news articles.
- To preprocess and analyze textual news data using Python.
- To evaluate model performance using standard metrics such as accuracy and classification report.

## Models Used

The notebook trains and evaluates multiple machine learning models, for example:

- Logistic Regression
- Gradient Boosting Classifier
- Decision Tree Classifier
- Random Forest Classifier

## Dataset

The project uses a labelled dataset of news articles, where each record contains:

- The news text/content.
- A label indicating whether the news is **true** (real) or **false** (fake).

Make sure the dataset file is placed in the location expected by the notebook (for example, the same folder as `Code.ipynb`), and that the filename inside the notebook matches the actual dataset filename.

## Technologies and Libraries Used

The following Python libraries are imported and used in `Code.ipynb`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```

### External Libraries (need to be installed)

These must be installed in your Python environment:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install them using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Python Standard Library (no installation needed)

- string
- re

## System Requirements

**Hardware (minimum):**

- 4 GB RAM
- Intel i3 processor or equivalent
- ~500 MB free disk space

**Software:**

- Python 3
- Jupyter Notebook or JupyterLab (to run `Code.ipynb`)
- (Optional) Anaconda distribution for easier environment and package management

## How to Run the Project

1. Open terminal / command prompt / Anaconda Prompt.
2. Navigate to the project directory containing:
   - `Code.ipynb`
   - `README.md`
   - Dataset file

3. Start Jupyter:

```bash
jupyter notebook
```
or
```bash
jupyter lab
```

4. Open `Code.ipynb` in the browser.
5. Run all cells in order (`Kernel → Restart & Run All`).

## Results and Evaluation

Performance metrics used:

- Accuracy Score
- Classification Report (Precision, Recall, F1-score)

The notebook output displays comparison and final results for trained models.
