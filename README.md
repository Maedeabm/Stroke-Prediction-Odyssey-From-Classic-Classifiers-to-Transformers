# Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers

Hey there! 👋

Ever thought about how we can predict strokes using data? 

Globally, strokes stand out as a major health threat, becoming the second most frequent cause of death. With insights from the World Health Organization, we understand that strokes make up a daunting 11% of all mortality instances. In light of this, the ability to predict strokes with precision is critically significant.

In this journey, we're mixing some old-school techniques with the latest and greatest, like CNNs and Transformers. It's a bit like comparing classic cars with electric ones. Our goal? To figure out which tool, old or new, can best tell us who's at risk of a stroke. Come join us in this exploration!

In our 'Understanding Strokes' project, we blend classic data techniques with modern giants like CNNs and Transformers. It's a quest to find the best way to spot stroke risks. A mix of old and new, all for a crucial cause!


## Dataset Snapshot: Predicting Strokes

We're diving deep into a dataset from Kaggle. This dataset has been meticulously curated to facilitate such predictions. It integrates a plethora of patient-centric attributes, spanning personal details, medical histories, and lifestyle patterns. Each data entry dives deep into individual profiles, ensuring a holistic approach to the analysis.

### Features Breakdown

- **id**: Each patient's distinct numerical tag.
- **gender**: Classifies into "Male", "Female", or "Other".
- **age**: Records the individual's age.
- **hypertension**: A binary marker; 1 for hypertension presence, 0 otherwise.
- **heart_disease**: Highlights presence (1) or absence (0) of heart ailments.
- **ever_married**: Distinguishes marital status with "No" or "Yes".
- **work_type**: Provides insights into the patient's professional sphere, encapsulating categories like "children", "Govt_job", "Never_worked", "Private", and "Self-employed".
- **Residence_type**: Denotes whether the individual is based in an "Urban" or "Rural" locale.
- **avg_glucose_level**: Represents the average glucose concentration in the bloodstream.
- **bmi**: Quantifies body fat using the body mass index metric.
- **smoking_status**: Segregates smoking behavior into "formerly smoked", "never smoked", "smokes", or "Unknown". An "Unknown" label indicates data absence.
- **stroke**: At the heart of the dataset, 1 signifies a stroke event, while 0 signals no such history.

# 🧠 Stroke Prediction with a Dash of Logistic Regression
## 📚 What's Inside?

- [Getting Started](#getting-started)
- [Model Performance Bits](#model-performance-bits)
- [Final Thoughts](#final-thoughts)
- 
## Getting Started

1. Make sure you've got these Python libraries:
   - `pandas`
   - `sklearn`
   - `matplotlib`
   - `seaborn`
2. Drop the `stroke.csv` data file in the same directory.
3. Run the script and see the magic!

## Model Performance Bits

- **Classification Report**: Some stats to show how our model is doing:
   - *Precision*: How often our positive predictions were correct.
   - *Recall*: Of all the actual positives, how many did we catch?
   - *F1-Score*: A balance between Precision and Recall.
   - *Support*: How many samples are we talking about?
   
- **Confusion Matrix**: Basically, a fancy table showing where we got things right and where we goofed up.

- **Accuracy**: A quick number to show how often we got predictions right.

<img width="685" alt="Screen Shot 2023-08-21 at 3 11 24 PM" src="https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/96ddbbe4-faef-457b-b1de-ce2daea79329">

## Final Thoughts

We're only scratching the surface here! 🚀 There's so much more to discover. Fancy tweaking some features or giving the model settings a little twist? Go for it! And if you're feeling adventurous, brace yourself because I've got Linear Discriminant Analysis (LDA) lined up for you next. Dive in and enjoy the ride! 🎢✨


# 🧠 Predicting Strokes with Linear Discriminant Analysis (LDA)

Hey there! 🌟 Welcome to our exciting journey into predicting medical outcomes using data. Today's spotlight: **Linear Discriminant Analysis**! Let's unpack this.

## Table of Contents
- [What's LDA?](#whats-lda)
- [Why Use LDA for Stroke Prediction?](#why-use-lda-for-stroke-prediction)
- [Fancy Metrics We Use](#fancy-metrics-we-use)
- [Wrapping Up](#wrapping-up)

## What's LDA?

Think of LDA like a superhero of the data world. It tries to neatly separate data points based on their categories. In our mission of predicting strokes, LDA looks at all the patient info and tries to draw the best line (or plane in geek speak) to differentiate between "might have a stroke" and "probably won't".

## Why Use LDA for Stroke Prediction?

Navigating medical data can feel like walking through a maze! 🌪️ But guess what? LDA is our compass. Instead of diving into a sea of complex numbers, LDA provides us with a clear, digestible map. This means when someone's curious about "How did the prediction come about?", we've got a neat answer!

## Fancy Metrics We Use

<img width="684" alt="Screen Shot 2023-08-21 at 3 28 42 PM" src="https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/93b56745-b172-46a2-af1a-435cbee9cbf4">

- **ROC Curve & AUC**: Imagine a graph showing our model's performance. The higher the curve (and the bigger the AUC number), the closer we are to the stars! 🌌

<img width="908" alt="Screen Shot 2023-08-21 at 3 30 27 PM" src="https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/59b8fcc6-0dfb-4f32-8a61-32f56778950f">

- **Precision-Recall Curve**: It's all about striking the right balance. We're aiming for confidence in our predictions while ensuring we're not missing out on potential cases. Higher curves = Happier predictions!

<img width="908" alt="Screen Shot 2023-08-21 at 3 31 00 PM" src="https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/1324ac87-b4e4-4d83-b4ea-001088f45f1e">

- **Log Loss**: Think of this as our model's report card. 🎓 Lower numbers mean our model is on top of its game!

<img width="912" alt="Screen Shot 2023-08-21 at 3 32 31 PM" src="https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/4f976ffc-6d46-43c5-a6f9-af73a26e6188">

## Wrapping Up

LDA isn't just a bunch of fancy jargon; it's our trusty guide in the bustling world of medical data. There are countless methods out there, but sometimes, the beauty lies in simplicity. LDA is that simple, interpretable tool in our toolkit.

# 🌳 Stroke Prediction Using Decision Trees

Dive into the world of predictive analytics with Decision Trees as we aim to predict strokes based on various health parameters.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Visual Insights](#visual-insights)
- [Feedback and Contributions](#feedback-and-contributions)

## Setup and Installation

### Prerequisites
Ensure you have the following installed:
- **Python 3.x**
- Required Libraries: 
    ```
    pandas, sklearn, matplotlib, seaborn
    ```

### Steps:

1. **Clone the Repository**:

git clone <your-repo-link>

2. **Navigate to the Directory and Install Required Libraries**:

cd <Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers>
pip install -r requirements.txt

## How to Run

Simply execute:

python decision_tree_stroke.py

## Project Structure

- **Data Loading**: Start by reading the `stroke.csv` dataset.
  
- **Preprocessing**: Handle missing values, encode categorical variables, and split data.
  
- **Model Training**: Train the Decision Tree classifier.
  
- **Evaluation**: Evaluate model performance using metrics: accuracy, confusion matrix, precision, recall, F1-score, ROC Curve, and feature importance.

## Visual Insights

![24](https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/d031d281-d9fe-41d7-b2bf-3ab8e82b3257)

- **Confusion Matrix**: Visual representation of model accuracy.

<img width="941" alt="Screen Shot 2023-08-21 at 4 06 55 PM" src="https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/78ed6b5b-70f7-4cf0-a9d0-04d538c317e3">

  
- **ROC Curve**: Graphical depiction of true positive rate versus false positive rate.

  <img width="941" alt="Screen Shot 2023-08-21 at 4 01 58 PM" src="https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/6e258ddd-8fa8-4664-b613-e177169f297e">

- **Feature Importance**: Understand which features impact the model's decisions the most.

![23](https://github.com/Maedeabm/Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers/assets/114970780/00abe8cc-a7f2-4dc2-ae87-6d67c87e74ce)

## Feedback and Contributions

Your suggestions are invaluable! 🌟 Feel free to raise issues, suggest improvements, or submit pull requests.


# 🌲 Stroke Prediction with Random Forest 🌲

Hey there! 🚀 Dive into this README to explore the mystical forest where we use the Random Forest classifier to predict strokes. Let's see the magic of ensemble learning in action!

## 📋 Table of Contents

- [Data Loading](#data-loading)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Extra Performance Metrics](#extra-performance-metrics)
- [Feature Importance](#feature-importance)
- [Conclusion](#conclusion)

## 📦 Data Loading

We kickstarted our journey by loading the dataset from Kaggle using the trusty `pandas` library.

```python
import pandas as pd
data = pd.read_csv('stroke.csv')
```

🧹 Data Preprocessing

To make sure our forest gets the best nutrients:

- Eliminated missing values
- Employed one-hot encoding for categorical data
- Segregated the data into training and testing sets

```python
data.dropna(inplace=True)
data = pd.get_dummies(data, drop_first=True)
from sklearn.model_selection import train_test_split
X = data.drop('stroke', axis=1)
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

🧙‍♂️ Model Training

With our data prepped and ready, it was time to invoke the power of the Random Forest!

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
```

📊 Model Evaluation

Post-training, we evaluated our forest's magic using:

- A classic confusion matrix
- A detailed classification report for precision, recall, and F1 scores
- Overall accuracy to gauge the forest's wisdom

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

🎯 Extra Performance Metrics

Beyond the usual metrics, we took a deeper dive to assess:

🌈 ROC Curve & AUC - Understanding the model's differentiation prowess.
🔥 Log Loss - A measure of uncertainty.
💡 Matthews Correlation Coefficient - Offers insights even with imbalanced datasets.

```python
from sklearn.metrics import roc_curve, auc, log_loss, matthews_corrcoef
y_prob = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("Log Loss:", log_loss(y_test, y_prob))
print("Matthews Correlation Coefficient (MCC):", matthews_corrcoef(y_test, y_pred))
```

🌟 Feature Importance

To understand the whisperings of the forest, we gauged the importance of features it valued.

```python
importances = rf.feature_importances_
features = X.columns
```

🎉 Conclusion 

Journeying through this enchanted forest, we witnessed the magic of the Random Forest in predicting strokes. The model's charm lies in its ensemble strength, and we hope you had fun exploring it! Feel the urge to tweak and experiment? Go ahead and amplify the magic! Best of luck, and happy coding! 🌟
