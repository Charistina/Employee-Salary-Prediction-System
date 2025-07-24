import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("data.csv")

# 1. Handle missing values: Replace '?' with NaN and impute
data.replace('?', np.nan, inplace=True)

# 2. Preprocess target separately
le = LabelEncoder()
data['income'] = le.fit_transform(data['income'])

# 3. Split data BEFORE preprocessing to avoid data leakage
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Identify column types
numeric_features = ['age', 'fnlwgt', 'educational-num', 
                    'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 
                        'gender', 'native-country']

# 5. Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

# 6. Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 7. Outlier handling modification: Remove only extreme outliers from training data
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.05)  # Less aggressive
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR)))]
    return df

# Get indices of remaining rows after outlier removal
X_train_filtered = remove_outliers(X_train.copy(), numeric_features)
# Get the indices of the remaining rows
remaining_indices = X_train_filtered.index
# Filter both X_train and y_train using these indices
X_train = X_train.loc[remaining_indices]
y_train = y_train.loc[remaining_indices]

# 8. Model pipeline with preprocessing
models = {
    "KNN": Pipeline([('preprocessor', preprocessor),
                     ('classifier', KNeighborsClassifier())]),
    "Logistic Regression": Pipeline([('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
    "SVM": Pipeline([('preprocessor', preprocessor),
                     ('classifier', SVC(class_weight='balanced'))]),
    "Naive Bayes": Pipeline([('preprocessor', preprocessor),
                             ('classifier', GaussianNB())]),
    "Decision Tree": Pipeline([('preprocessor', preprocessor),
                               ('classifier', DecisionTreeClassifier(class_weight='balanced'))]),
    "Random Forest": Pipeline([('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(class_weight='balanced'))])
}

# 9. Train and evaluate models with cross-validation
best_model = None
best_accuracy = 0

for name, pipeline in models.items():
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Final training and evaluation
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = pipeline

# 10. Save the entire pipeline including preprocessing
joblib.dump(best_model, "best_pipeline.pkl")
print("Best pipeline saved as best_pipeline.pkl")