## Sarcasm-Detection using Ensemble Learning and Deep Learning

### Objective:

The project, **`Sarcasm-Detection using Ensemble Learning and Deep Learning,`** focuses on identifying sarcasm in **`Financial News Headlines.`** It employs an **`LSTM-based model`** and baseline models **`Logistic Regression, Extreme Gradient Boosting and LightGBM model`** using **`tf-idf`** for feature representation. With practical applications in sentiment analysis and social media monitoring, the project addresses linguistic nuances in financial news. Challenges include imbalanced data and linguistic complexity.

### Dataset:

- **Entries**: **`26,709`**
- **Columns**: **`article_link, headline, is_sarcastic`**
- **Class distribution**: **`0 (14,985 instances), 1 (11,724 instances)`**
- **Data types**: Numeric and text
- **`No null values`** in the dataset; **`headline`** text converted to lowercase.

### Dataset Exploration and Preprocessing:

 **Overview:**

- **Rows**: **`26,709`**
- **Columns**: **`3 (int64: `, object: 2)`**
- **Dtype**: **`int64 (1), object (2)`**
- **Memory Usage**: 626.1+ KB

**Data Info:**

- **`Non-null`** entries in all columns
- **`Duplicate`** 1 Duplicate entry which is being dropped

**Descriptive Stats:**

- **Mean `is_sarcastic` = `0.438953`**

**Selected Data:**

- `headline` columns retained

 **Class Distribution:**

- **`Labels: 0 (14,985), 1 (11,724)`**

 **Null Values:**

- **`No null values`**

 **Duplicates Values:**
 
- **`1 Deplicate Rows`**

### Text Preprocessing: 

**Library Setup:**
- Import necessary libraries: **`nltk, spacy, string.`**

**SpaCy Model Load:**
- Load spaCy's English model **`(en_core_web_sm).`**
  
**Text Cleaning:**

- Convert text to lowercase.
- Remove punctuation and numbers.
- Download and remove English stopwords.
- Lemmatize text using spaCy.
  
**Outcome:**
- Achieve a clean and processed `headline` column in the dataset (headline_data) for improved natural language processing.

### Model Building:

  **Train Test Split:**
- Employed `stratified` split `(test_size=0.2)` for training and testing datasets

 **TF-IDF Vectorization:**
- Apply **`TfidfVectorizer(max 5000 features)`** to train and test text data.

  **Logistic Regression:**
- Train using TF-IDF features.
- Accuracy: **`77.6%`**
  
**XGBoost:**
- Implement with TF-IDF features.
- Accuracy: **`73.0%`**

**LightGBM:**
- Employ TF-IDF features.
- Accuracy: **`73.3%`**

  ### LSTM Architecture:
  
**Tokenizer:**
- Tokenize text using Tokenizer from TensorFlow.
- Convert sequences back to text for further processing.

**Embedding Layer:**
- Vocabulary size: `5000 words`.
- Output dimension: `100.`
- Input length: `Maximum sequence length.`
  
**LSTM Layer:**
- Units: `64.`
- Dropout: `0.5.`
- Recurrent Dropout: `0.5.`
  
**Dense Layer:**
- Output units: `1 (Binary classification).`
- Activation function: `Sigmoid.`
  
**Hyperparameters:**
- **Tokenizer:** Limit vocabulary size to **`5000.`**
- **Padding:** Pad sequences to ensure consistent length.
- **Maximum sequence length:** **`205`**
- **EarlyStopping:** Monitor validation accuracy. Patience: **`3 epochs.`** Minimum change: **`0.001`**
  
**Model Compilation:**
- **Loss function:** `Binary Crossentropy.`
- **Optimizer:** `Adam` with a learning rate of `0.0045.`
- **Evaluation metric:** `Accuracy.`
  
**Training:**
- **Epochs:** Train for `25 epochs.`
- **Batch Size:** Set to `64.`
  
**Evaluation:**
- Training History:
- Displayed using `plotly.express` for training and validation accuracy over epochs.

**Test Accuracy:**
- Evaluate the model on the test set.
- Achieves an accuracy of `63.42%.`
  
    
