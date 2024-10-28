# 📝 Sentence Reordering with Transformers

![Python](https://img.shields.io/badge/Python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Keras](https://img.shields.io/badge/Keras-2.x-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

This repository implements a **sentence reordering model** using the Transformer architecture, inspired by the "Attention Is All You Need" paper. The project involves developing, training, and evaluating the model on text data to achieve coherent sentence ordering for tasks like summarization and generation.sec

## 📋 Table of Contents
- [🔍 Introduction](#introduction)
- [📦 Requirements](#requirements)
- [📂 Dataset](#dataset)
- [🔧 Notebook Structure](#notebook-structure)
- [⚙️ Custom Callback](#custom-callback)
- [📉 Cosine Decay Restart](#cosine-decay-restart)
- [🚀 Usage](#usage)
- [📊 Results](#results)
- [📝 Conclusion](#conclusion)
- [📚 References](#references)

## 🔍 Introduction
The objective of this project is to reorder shuffled sentences into coherent paragraphs using the **Transformer model**, which has shown great effectiveness in NLP tasks. This approach is crucial for applications such as text summarization, machine translation, and story generation.

## 📦 Requirements
To run this project, ensure you have the following packages installed:
- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install them with:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn

You can install the required packages using the following command:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

## 📂 Dataset
The dataset contains text data where each instance includes shuffled sentences. The model's task is to reorder these sentences into their original sequence. The data is split into training, validation, and test sets for comprehensive evaluation.

## 🔧 Notebook Structure
1. **Introduction**: Project overview and objectives.
2. **Data Loading and Preprocessing**: Loading and preparing data, including tokenization and padding.
3. **Model Development**: Transformer model implementation for reordering sentences.
4. **Training and Evaluation**: Model training and validation.
5. **Experiments**: Exploration of different architectures and hyperparameters.
6. **Conclusion**: Summary of results and insights.

## ⚙️ Custom Callback
A custom callback monitors validation performance during training, enabling dynamic adjustments and early stopping to avoid overfitting and improve generalization.


## 📉 Cosine Decay Restart
The project utilizes a **Cosine Decay with Restarts** learning rate schedule, which helps the model converge by periodically reducing the learning rate and restarting it to escape local minima.

## 🚀 Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/sentence-reordering-transformers.git
    ```
2. Navigate to the project directory:
    ```bash
    cd sentence-reordering-transformers
    ```
3. Open the notebook:
    ```bash
    jupyter notebook francesco_baiocchi_Sentence_Reordering.ipynb
    ```
4. Run the cells in the notebook to execute the code.
   

## 📊 Results
The model’s performance is evaluated using a custom validation callback to monitor real-time performance on the validation set. The best model achieved an average score of 0.573 on the test set.

## 📝 Conclusion
After numerous experiments and iterations, the Transformer model demonstrated average performance in the task of sentence reordering. While different architectures and parameter tuning efforts were explored, the improvements were marginal. 

## 📚 References
This project was inspired by the following papers:
- [A Comprehensive Study on Sentence Reordering](https://aclanthology.org/2021.naacl-main.134.pdf)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)


