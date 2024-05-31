import json
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import os

default_model_params = {
    'Logistic Regression': {'max_iter': 100, 'solver': 'lbfgs', 'C': 1.0},
    'KNN': {'n_neighbors': 5},
    'Decision Tree': {'max_depth': None},
    'Random Forest Regressor': {'n_estimators': 100},
    'Ridge Regression': {'alpha': 1.0},
}


# Define the available datasets
datasets = [
    {
        'name': 'MNIST DATA',
        'problem_type': 'classification',
    },
    {
        'name': 'IRIS DATA',
        'problem_type': 'classification',
    },
    {
        'name': 'Insurance Data',
        'problem_type': 'regression',
    },
    {
        'name': 'Red Wine Data',
        'problem_type': 'regression',
    },
]

scalers = {
    'MinMaxScaler': MinMaxScaler,
    'StandardScaler': StandardScaler,
}


def load_iris_data(split_ratio=0.2):
    iris_data = pd.read_csv('C:\\Users\\User\\senior_mlwebapp\\dataset\\iris.csv')
    X = iris_data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
    y = iris_data['variety']

    # Use train_test_split to split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)
    return X_train, X_test, y_train, y_test, "classification"


def load_insurance_data(split_ratio=0.2):
    insurance_data = pd.read_csv('C:\\Users\\User\\senior_mlwebapp\\dataset\\insurance.csv')
    X = insurance_data[['age', 'bmi', 'children']]
    y = insurance_data['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)
    return X_train, X_test, y_train, y_test, "regression"

def load_red_wine_data(split_ratio=0.2):
    red_wine_data = pd.read_csv('C:\\Users\\User\\senior_mlwebapp\\dataset\\winequality-red.csv')
    X = red_wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                       'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
    y = red_wine_data['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio)
    return X_train, X_test, y_train, y_test, "regression"


def get_datasets(request):
    if request.method == 'GET':
        return JsonResponse({'datasets': datasets})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def log_selected_dataset(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            dataset_name = data.get('datasetName')
            problem_type = data.get('problemType')

            if dataset_name == 'MNIST DATA':
                return JsonResponse({'error': 'MNIST dataset is not available at the moment'})
            elif dataset_name == 'IRIS DATA':
                X_train, X_test, y_train, y_test, problem_type = load_iris_data()
            elif dataset_name == 'Insurance Data':
                X_train, X_test, y_train, y_test, problem_type = load_insurance_data()
            elif dataset_name == 'Red Wine Data':
                X_train, X_test, y_train, y_test, problem_type = load_red_wine_data()

            # You may want to do something with X_train, X_test, y_train, y_test here
            # Currently, just logging the selected dataset and problem type
            print(f"Selected Dataset: {dataset_name}")
            print(f"Problem Type: {problem_type}")

            return JsonResponse({'message': 'Dataset information logged successfully'})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
classification_models = {
    'Logistic Regression': LogisticRegression,
    'KNN': KNeighborsClassifier,
    'Decision Tree': DecisionTreeClassifier
}

regression_models = {
    'Linear Regression': LinearRegression,
    'Random Forest Regressor': RandomForestRegressor,
    'Ridge Regression': Ridge
}

def get_model_parameters(request, model_name):
    if model_name in default_model_params:
        parameters = default_model_params[model_name]
        return JsonResponse({'parameters': parameters})
    else:
        return JsonResponse({'error': 'Model not found'}, status=404)
    
def get_models(request, problem_type):
    if problem_type == 'classification':
        model_names = list(classification_models.keys())
    elif problem_type == 'regression':
        model_names = list(regression_models.keys())
    else:
        model_names = []

    return JsonResponse({'models': model_names})

@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            dataset_name = data.get('datasetName')
            model_name = data.get('modelName')
            split_ratio = data.get('splitRatio', 0.2)
            transform_params = data.get('transformParams', {})
            noise_level = float(transform_params.get('noiseLevel', 0))
            scaler_type = transform_params.get('scaler', None)
            provided_params = data.get('modelParams', {})  # Parameters provided in the request

            # Load and split the dataset
            if dataset_name == 'IRIS DATA':
                X_train, X_test, y_train, y_test, _ = load_iris_data(split_ratio)
                # Encode target variable
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)
                # Standardize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                # Reduce to 2D for visualization
                pca = PCA(n_components=2)
                X_train_2d = pca.fit_transform(X_train)
                X_test_2d = pca.transform(X_test)
            elif dataset_name == 'Insurance Data':
                X_train, X_test, y_train, y_test, _ = load_insurance_data(split_ratio)
                # Standardize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif dataset_name == 'Red Wine Data':
                X_train, X_test, y_train, y_test, _ = load_red_wine_data(split_ratio)
                # Standardize the data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                return JsonResponse({'error': 'Dataset not found'}, status=404)

            # Apply noise if specified
            if noise_level > 0:
                X_train = add_noise(X_train, noise_level)

            # Apply scaler if specified
            scaler = None
            if scaler_type in scalers:
                scaler_class = scalers[scaler_type]
                scaler = scaler_class()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Merge default and provided model parameters
            default_params = default_model_params.get(model_name, {})
            for key, value in provided_params.items():
                default_params[key] = value

            # Instantiate the model with merged parameters
            if model_name in classification_models:
                model_class = classification_models[model_name]
                model = model_class(**default_params)
                model.fit(X_train_2d, y_train)
                predictions = model.predict(X_test_2d)
                plot_filename = f"{model_name}_decision_boundaries.png"
                plot_decision_boundaries(X_test_2d, y_test, model, f"{model_name} Decision Boundaries (Iris)", plot_filename)
                response_data = {
                    'accuracy': accuracy_score(y_test, predictions),
                    'f1_score': f1_score(y_test, predictions, average='macro'),
                    'recall': recall_score(y_test, predictions, average='macro'),
                    'precision': precision_score(y_test, predictions, average='macro'),
                    'plot_filename': plot_filename
                }
            elif model_name in regression_models:
                model_class = regression_models[model_name]
                if model_name == 'Ridge Regression':
                    # Explicitly set solver to avoid sym_pos issue
                    model = model_class(alpha=default_params.get('alpha', 1.0), solver='auto')
                else:
                    model = model_class(**default_params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                plot_filename = f"{model_name}_regression_results.png"
                plot_regression_results(y_test, predictions, model_name, plot_filename)
                response_data = {
                    'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                    'r2_score': r2_score(y_test, predictions),
                    'plot_filename': plot_filename
                }

            return JsonResponse(response_data)  # Ensure response is always returned

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return X_noisy

def apply_scaler(X, scaler_type=None):
    print(f"Scaler Type: {scaler_type}")  # Debugging print

    if scaler_type is None or scaler_type == 'None':
        print("No scaler is being applied.")  # Debugging print
        return X

    scaler_class = scalers.get(scaler_type)
    if scaler_class:
        print(f"Applying scaler: {scaler_class}")  # Debugging print
        scaler_instance = scaler_class()
        X_scaled = scaler_instance.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    else:
        print(f"Scaler {scaler_type} not found.")  # Debugging print
        return X


@csrf_exempt
def log_model_results(request):
    if request.method == 'POST':
        try:
            # Assuming you're sending JSON with the appropriate metrics
            data = json.loads(request.body)
            # Here you could save to a database or just print
            print(f"Results: {data}")
            return JsonResponse({'message': 'Results logged successfully'})

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)

plt.rcParams['font.size'] = 14  # Change the font size as needed
plt.rcParams['text.color'] = 'white'

def plot_decision_boundaries(X, y, model, title, filename):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)

    unique_classes = np.unique(y)
    markers = ["o", "X", "s", "P", "D", "*"]
    if len(unique_classes) > len(markers):
        raise ValueError("The number of unique classes exceeds the number of available markers.")

    # Create a marker map
    marker_map = {unique_class: markers[i] for i, unique_class in enumerate(unique_classes)}

    # Customize the scatter plot with different markers
    for unique_class in unique_classes:
        class_mask = (y == unique_class)
        sns.scatterplot(x=X[class_mask, 0], y=X[class_mask, 1], 
                        marker=marker_map[unique_class], label=f"Class {unique_class}", 
                        edgecolor='k', s=100, legend=False)


    plt.title(title, color='white')
    plt.xlabel("Feature 1", color='white')
    plt.ylabel("Feature 2", color='white')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Add legend to indicate class markers
    handles = [plt.Line2D([0], [0], marker=marker_map[cls], color='w', label=f"Class {cls}",
                          markerfacecolor=sns.color_palette("bright")[i], markersize=10, markeredgecolor='k') 
               for i, cls in enumerate(unique_classes)]
    
    plt.legend(handles=handles, title="Classes", loc='upper right')

    plt.savefig('mlfrontend/public/figured.png', format='png', transparent=True)
    print("FIGGURED SAVED")
    plt.close()

def plot_regression_results(y_true, y_pred, model_name, filename):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with green points
    plt.scatter(y_true, y_pred, color='green', edgecolor='k', s=100, label='Predicted Values')
    
    # Ideal fit line in white
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'w--', lw=2, label='Ideal Fit')
    
    # Set titles and labels
    plt.title(f"{model_name} Predictions", color='white')
    plt.xlabel("True Values", color='white')
    plt.ylabel("Predictions", color='white')
    
    # Customize plot aesthetics
    plt.gca().set_facecolor('black')  # Set background color to black for contrast
    plt.legend()

    # Save the figure to the specified path
    plt.savefig('mlfrontend/public/figured.png', format='png', transparent=True)
    print("FIGGURED SAVED")
    plt.close()