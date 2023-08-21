# Stroke-Prediction-Odyssey-From-Classic-Classifiers-to-Transformers

Hey there! 👋

Ever thought about how we can predict strokes using data? 

Globally, strokes stand out as a major health threat, becoming the second most frequent cause of death. With insights from the World Health Organization, we understand that strokes make up a daunting 11% of all mortality instances. In light of this, the ability to predict strokes with precision is critically significant.

In this journey, we're mixing some old-school techniques with the latest and greatest, like CNNs and Transformers. It's a bit like comparing classic cars with electric ones. Our goal? To figure out which tool, old or new, can best tell us who's at risk of a stroke. Come join us in this exploration!

In our 'Understanding Strokes' project, we blend classic data techniques with modern giants like CNNs and Transformers. It's a quest to find the best way to spot stroke risks. A mix of old and new, all for a crucial cause!

## 📚 What's Inside?

- [Dataset Snapshot: Predicting Strokes](#Dataset-Snapshot)
- [Stroke Prediction with a Dash of Logistic Regression](#Stroke-Prediction-with-a-Dash-of-Logistic-Regression)
- [Getting Started](#getting-started)
- [Model Performance Bits](#model-performance-bits)
- [Final Thoughts](#final-thoughts)

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
