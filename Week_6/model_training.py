import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Wine dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and their hyperparameter grids
models = {
    'SVC': {
        'model': SVC(),
        'param_grid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.1, 1, 'scale', 'auto']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'param_grid': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    }
}

results = []

for model_name, config in models.items():
    print(f"\n--- Training {model_name} ---")
    model = config['model']
    param_grid = config['param_grid']

    # GridSearchCV
    print("Performing GridSearchCV...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_grid_model = grid_search.best_estimator_
    y_pred_grid = best_grid_model.predict(X_test)

    accuracy_grid = accuracy_score(y_test, y_pred_grid)
    precision_grid = precision_score(y_test, y_pred_grid, average='weighted')
    recall_grid = recall_score(y_test, y_pred_grid, average='weighted')
    f1_grid = f1_score(y_test, y_pred_grid, average='weighted')

    results.append({
        'Model': model_name,
        'Tuner': 'GridSearchCV',
        'Best Params': grid_search.best_params_,
        'Accuracy': accuracy_grid,
        'Precision': precision_grid,
        'Recall': recall_grid,
        'F1-Score': f1_grid
    })

    print(f"GridSearchCV Best Params: {grid_search.best_params_}")
    print(f"GridSearchCV Accuracy: {accuracy_grid:.4f}")

    # RandomizedSearchCV (using a smaller number of iterations for demonstration)
    print("Performing RandomizedSearchCV...")
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_random_model = random_search.best_estimator_
    y_pred_random = best_random_model.predict(X_test)

    accuracy_random = accuracy_score(y_test, y_pred_random)
    precision_random = precision_score(y_test, y_pred_random, average='weighted')
    recall_random = recall_score(y_test, y_pred_random, average='weighted')
    f1_random = f1_score(y_test, y_pred_random, average='weighted')

    results.append({
        'Model': model_name,
        'Tuner': 'RandomizedSearchCV',
        'Best Params': random_search.best_params_,
        'Accuracy': accuracy_random,
        'Precision': precision_random,
        'Recall': recall_random,
        'F1-Score': f1_random
    })

    print(f"RandomizedSearchCV Best Params: {random_search.best_params_}")
    print(f"RandomizedSearchCV Accuracy: {accuracy_random:.4f}")

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print("\n--- All Results ---")
print(results_df.to_markdown(index=False))

# Select the best performing model based on F1-Score
best_model_overall = results_df.loc[results_df['F1-Score'].idxmax()]
print("\n--- Best Performing Model Overall (based on F1-Score) ---")
print(best_model_overall.to_markdown(index=False))

