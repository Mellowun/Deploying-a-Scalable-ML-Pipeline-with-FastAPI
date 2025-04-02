# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Type: Random Forest Classifier
Parameters: n_estimators=100, random_state=42
Training Data: Modified Census Income dataset
Features Used: age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

## Intended Use
Primary intended uses: Predict whether a person makes over 50K a year based on census data
Primary intended users: Data scientists, socioeconomic researchers, policy makers
Out-of-scope use cases: Not intended for making decisions about individuals that would impact their opportunities or access to resources. Should not be used for hiring decisions or credit worthiness.
## Training Data
The model was trained on 80% of the Census Income dataset. This dataset contains demographic information such as age, education, occupation, and other features to predict whether income exceeds $50K/year.
## Evaluation Data
The model was evaluated on a 20% holdout test set from the Census Income dataset. The data contains a mix of continuous and categorical variables related to individuals' demographic and employment information.
## Metrics
Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863
## Quantitative Analyses
Performance metrics were calculated for different slices of the test data to check for disparities in model performance. Some notable observations include:
Education Level Analysis

The model performs best on individuals with higher education levels:

Doctorate (F1: 0.8793)
Professional School (F1: 0.8852)
Masters (F1: 0.8409)

The model performs poorly on individuals with lower education levels:

Preschool (F1: 0.0000)
1st-4th grade (F1: 0.0000)
7th-8th grade (F1: 0.0000)

Gender Analysis

The model performs better for males (F1: 0.6997) than females (F1: 0.6015)
For males: Precision 0.7445, Recall 0.6599
For females: Precision 0.7229, Recall 0.5150

Race Analysis

Performance is relatively consistent across racial groups:

White (F1: 0.6850)
Black (F1: 0.6667)
Asian-Pacific-Islander (F1: 0.7458)
American Indian/Eskimo (F1: 0.5556)
Other (F1: 0.8000)

Occupation Analysis

Best performance for:

Executive/Managerial positions (F1: 0.7736)
Professional Specialties (F1: 0.7778)

Worst performance for:

Private House Service (F1: 0.0000)
Farming/Fishing (F1: 0.3077)
Other Service (F1: 0.3226)

Relationship Status Analysis

Better performance for married individuals:

Husband (F1: 0.7140)
Wife (F1: 0.6953)

Lower performance for:

Own Child (F1: 0.3000)
Unmarried (F1: 0.4138)

## Ethical Considerations
The model uses demographic features like race, gender, and nationality which raises potential fairness concerns.
Predictions could reinforce existing societal biases if used inappropriately.
Special care should be taken when deploying this model to avoid discrimination.
The model should not be used for making decisions about individual opportunities without additional human oversight.
## Caveats and Recommendations
Caveats:

The model was trained on census data which may have inherent biases.
Performance varies across different demographic groups.
The data may not be representative of the current population distribution.


Recommendations:

Monitor model performance across different demographic groups when deployed.
Consider adding fairness constraints when retraining the model.
Use this model as one of many signals when making important decisions.
Periodically retrain the model with more recent data to maintain relevance.