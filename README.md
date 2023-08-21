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
