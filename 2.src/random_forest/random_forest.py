import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Loding train, test dataset
print("LOADING TRAIN, TEST DATASET")
train_data = pd.read_csv('../../1.input/train.csv')
test_data = pd.read_csv('../../1.input/test.csv')

print("TRAIN DATA: %d"%len(train_data))
print("TEST DATA: %d"%len(test_data))

# Select features for random forest models
features = ['Pclass', 'Sex', 'SibSp', 'Parch']

# Make dummy pandas dataframe for train_X, test_X with features
train_X = pd.get_dummies(train_data[features])
train_y = train_data.Survived
test_X = pd.get_dummies(test_data[features])

# Build up random forest model using sklearn.ensemble
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
model.fit(train_X, train_y)

# Print training evaluation score
evaluation = model.score(train_X, train_y)
print("EVALUATION SCORE: %.3f"%evaluation)

# Make prediction with trained model
prediction = model.predict(test_X)

# Export prediction same with template (gender_submission.cev)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': prediction})
output.to_csv('../../3.output/random_forest_submission.csv', index = False)
print("COMPLETE TO SAVE OUTPUT")
