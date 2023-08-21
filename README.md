# Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers

Hey there! 👋

Ever thought about how we can predict strokes using data? 

Globally, strokes stand out as a major health threat, becoming the second most frequent cause of death. With insights from the World Health Organization, we understand that strokes make up a daunting 11% of all mortality instances. In light of this, the ability to predict strokes with precision is critically significant.

In this journey, we're mixing some old-school techniques with the latest and greatest, like CNNs and Transformers. It's a bit like comparing classic cars with electric ones. Our goal? To figure out which tool, old or new, can best tell us who's at risk of a stroke. Come join us in this exploration!

In our 'Understanding Strokes' project, we blend classic data techniques with modern giants like CNNs and Transformers. It's a quest to find the best way to spot stroke risks. A mix of old and new, all for a crucial cause!

# Dataset Snapshot: Predicting Strokes

We're diving deep into a dataset from Kaggle. This dataset has been meticulously curated to facilitate such predictions. It integrates a plethora of patient-centric attributes, spanning personal details, medical histories, and lifestyle patterns. Each data entry dives deep into individual profiles, ensuring a holistic approach to the analysis.

## Features Breakdown

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
- 
# 🧠 Stroke Prediction with a Dash of Logistic Regression

Here's a walk-through of our code adventure:

## 1. Getting Ready 🚀
First, let's get our toolbox ready. We're going to use:
- `pandas`: Our data wrangling best friend.
- `scikit-learn`: A Swiss army knife for machine learning. 


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## 2. Dive into the Data 🏊‍♂️

Time to bring in the data. Make sure you name your dataset file right (or tweak the filename in the code).


data = pd.read_csv('stroke_dataset.csv')

## 3. A Bit of Cleaning 🧼

No dataset's perfect. Here's what we're doing:

- **Tossing out rows with pesky missing values.
    Juggling with categorical values to make them fit for our model.
    Splitting our data into a training set and a test set.
    Scaling features so our model doesn't get overwhelmed by big numbers.


data = data.dropna()
data = pd.get_dummies(data, drop_first=True)
X = data.drop('stroke', axis=1)
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## 4. Model Time! 🎩

Let's train our Logistic Regression model. It's like teaching it what strokes look like based on past data.


model = LogisticRegression()
model.fit(X_train, y_train)

## 5. Test Drive 🚗

Time to see our model in action on unseen data!


y_pred = model.predict(X_test)

## 6. Report Card 📊

How did our model do? Let's check the score.


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

## Tips for the Future 🔮

To jazz things up, consider:

    Fine-tuning the settings of the model.
    Crafting new features from the data.
    Filling in missing data points instead of tossing them out.
    Looking into strategies if our data has more of one class than another.

Cheers! 🥂
