# xAIL-ID: Explainable AI-based LSTM Model for Intrusion Detection in IoT

## Project Overview
The xAIL-ID project proposes an Intrusion Detection System (IDS) that leverages **Explainable Artificial Intelligence (XAI)** methods with a **Long Short-Term Memory (LSTM)** model to detect and classify attacks in IoT and Industrial IoT (IIoT) environments. The goal is to improve both the performance and transparency of the IDS.

The model was trained on four benchmark datasets:
- **NSL-KDD**
- **UNSW-NB15**
- **TON-IoT**
- **X-IIoTID**

Explainability was achieved using **Shapley Additive Explanations (SHAP)** and **Local Interpretable Model-agnostic Explanations (LIME)** to identify important features that influence the model’s decisions.

## Key Features
- **LSTM-based IDS** for binary and one-vs-all classification of cyberattacks in IoT environments.
- **XAI** applied using SHAP and LIME for model interpretability and transparency.
- Uses **multiple datasets** to ensure robustness across different network environments and attack types.
- **Feature selection** process reduces computational overhead while maintaining high detection accuracy by utilizing only the top 15 features of each dataset.

## Model Performance
The xAIL-ID model demonstrated high accuracy on all datasets:
- **NSL-KDD**: 99.0% Accuracy
- **UNSW-NB15**: 95.1% Accuracy
- **TON-IoT**: 98.0% Accuracy
- **X-IIoTID**: 97.7% Accuracy

## Datasets
The following datasets were used to train and evaluate the IDS:

1. **NSL-KDD**  
   A refined version of the KDD-CUP99 dataset for traditional network attacks.  
   - **Classes**: Normal, Attack (23 sub-types)
   - **Features**: 41

2. **UNSW-NB15**  
   A modern network dataset for evaluating IDS models, containing a mix of real and synthetic attack behaviors.  
   - **Classes**: Normal, Attack (9 sub-types)
   - **Features**: 47

3. **TON-IoT**  
   A large-scale IIoT dataset with telemetry data for IoT devices, focusing on cyberattacks in IIoT environments.  
   - **Classes**: Normal, Attack (9 sub-types)
   - **Features**: 43

4. **X-IIoTID**  
   A next-generation IIoT dataset capturing various modern attack types and scenarios.  
   - **Classes**: Normal, Attack (17 sub-types)
   - **Features**: 67
  
## Framework Architecture

The **xAIL-ID** framework is designed to provide a robust intrusion detection solution with explainable AI capabilities. The framework integrates several key components:

- **Data Preprocessing**: Handles the cleaning, normalization, and feature encoding of the input datasets. Class imabalance in ToN-IoT and UNSW-NB15 is addressed using SMOTE.
- **Model Training**: Utilizes an LSTM network for detecting intrusions, trained on benchmark datasets from both traditional networks and IoT/IIoT environments.
- **Explainability Module**: Leverages SHAP and LIME methods to provide local and global explanations of the model’s decisions, enhancing transparency.
- **Evaluation and Optimization**: Performs model evaluation on multiple metrics such as accuracy, precision, recall, and F1-score. Also includes a feature selection process that optimizes performance by reducing dimensionality.

This architecture ensures that the system not only provides accurate intrusion detection but also offers insight into how predictions are made, which is critical for improving trust and understanding in AI models.

<img src="/images/framework_architecture" alt="Alt Text" style="width:70%; height:auto;">

## Model Architecture

The **LSTM-based model** is the core of the xAIL-ID framework, designed to detect intrusions based on temporal patterns in network traffic. Key elements of the model architecture include:

- **Input Layer**: Takes preprocessed features from the datasets (e.g., NSL-KDD, UNSW-NB15).
- **LSTM Layers**: Three fully connected LSTM layers with 64 neurons each are used to capture temporal dependencies in the data. LSTM's inherent ability to handle sequences makes it ideal for intrusion detection in network traffic data.
- **Dropout Layers**: A dropout rate of 0.1 is applied between LSTM layers to prevent overfitting.
- **Output Layer**: Uses a Sigmoid activation function for binary classification and a SoftMax function for multi-class classification, depending on the dataset and classification task.

This architecture is optimized for detecting both known and unknown attack patterns in IoT environments.

![Model Architecture](/images/lstm_model_dark.png)

## Model Results

| Dataset      | Accuracy | Precision | Recall  | F1-Score |
|--------------|----------|-----------|---------|----------|
| **NSL-KDD**  | 99.0%    | 99.3%     | 99.3%   | 99.3%    |
| **UNSW-NB15**| 95.1%    | 98.0%     | 85.8%   | 90.8%    |
| **TON-IoT**  | 98.0%    | 98.4%     | 93.6%   | 95.8%    |
| **X-IIoTID** | 97.7%    | 99.1%     | 98.4%   | 98.7%    |

### Results with SHAP Features (Top 15)
| Dataset      | Accuracy | Precision | Recall  | F1-Score |
|--------------|----------|-----------|---------|----------|
| **NSL-KDD**  | 98.8%    | 98.7%     | 98.8%   | 98.7%    |
| **UNSW-NB15**| 97.0%    | 97.1%     | 86.1%   | 90.7%    |
| **TON-IoT**  | 97.3%    | 94.7%     | 89.4%   | 92.3%    |
| **X-IIoTID** | 97.6%    | 98.1%     | 97.4%   | 97.8%    |

### Results with LIME Features (Top 15)
| Dataset      | Accuracy | Precision | Recall  | F1-Score |
|--------------|----------|-----------|---------|----------|
| **NSL-KDD**  | 95.0%    | 96.0%     | 96.0%   | 96.0%    |
| **UNSW-NB15**| 94.0%    | 97.0%     | 82.0%   | 87.0%    |
| **TON-IoT**  | 94.0%    | 89.0%     | 80.0%   | 84.0%    |
| **X-IIoTID** | 95.0%    | 96.0%     | 95.0%   | 95.0%    |

### Results with Combined SHAP and LIME Features
| Dataset      | Accuracy | Precision | Recall  | F1-Score |
|--------------|----------|-----------|---------|----------|
| **NSL-KDD**  | 99.0%    | 99.0%     | 99.0%   | 99.0%    |
| **UNSW-NB15**| 95.1%    | 97.2%     | 85.4%   | 90.0%    |
| **TON-IoT**  | 98.0%    | 98.7%     | 90.6%   | 94.2%    |
| **X-IIoTID** | 97.7%    | 97.9%     | 97.4%   | 97.6%    |