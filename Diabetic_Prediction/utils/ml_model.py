import pandas as pd
import numpy as np
import warnings
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Fix threading issues
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress FutureWarnings from scikit-learn
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

class DiabetesPredictor:
    def __init__(self):
        self.ensemble_classifier = None
        self.best_rf_model = None
        self.svm_linear_model = None
        self.svm_rbf_model = None
        self.scaler = RobustScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False
        self.original_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
    def load_and_train(self):
        """Load data and train models with updated validation"""
        try:
            # Load data
            data = pd.read_csv('diabetes.csv')
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            
            # Apply feature engineering to training data
            X_engineered = self._create_features(X)
            
            # Data Augmentation using SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_engineered, y)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            
            # Define classifiers with updated parameters
            svm_linear = SVC(kernel='linear', probability=True, C=1.0, random_state=42)
            svm_rbf = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
            
            # Calibrate classifiers for better probabilities
            svm_linear_calibrated = CalibratedClassifierCV(svm_linear, method='sigmoid', cv=3)
            svm_rbf_calibrated = CalibratedClassifierCV(svm_rbf, method='sigmoid', cv=3)
            
            # Train individual SVM models for separate evaluation
            self.svm_linear_model = svm_linear_calibrated
            self.svm_rbf_model = svm_rbf_calibrated
            
            self.svm_linear_model.fit(X_train_scaled, self.y_train)
            self.svm_rbf_model.fit(X_train_scaled, self.y_train)
            
            # Add Logistic Regression for better calibration
            logistic_reg = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
            
            # Define ensemble classifier
            self.ensemble_classifier = VotingClassifier(estimators=[
                ('svm_linear', svm_linear_calibrated),
                ('svm_rbf', svm_rbf_calibrated),
                ('logistic', logistic_reg)
            ], voting='soft')
            
            # Fit ensemble classifier
            self.ensemble_classifier.fit(X_train_scaled, self.y_train)
            
            # Random Forest with improved parameters
            rf_model = RandomForestClassifier(random_state=42)
            
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            grid_search = GridSearchCV(
                estimator=rf_model, 
                param_grid=param_grid, 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            self.best_rf_model = grid_search.best_estimator_
            
            print(f"Best RF parameters: {grid_search.best_params_}")
            print(f"Best RF score: {grid_search.best_score_:.4f}")
            
            # Print individual model accuracies
            self._print_model_performance(X_test_scaled)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return False

    def _create_features(self, X):
        """Create additional features to help the model"""
        X = X.copy()
        
        # Ensure we have the basic features
        for feature in self.original_features:
            if feature not in X.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Create interaction terms and derived features
        X['BMI_Age_Interaction'] = X['BMI'] * X['Age']
        X['Glucose_BMI_Ratio'] = X['Glucose'] / (X['BMI'] + 1)  # +1 to avoid division by zero
        X['BloodPressure_Glucose_Interaction'] = X['BloodPressure'] * X['Glucose']
        
        # Create risk score combinations
        X['Metabolic_Risk'] = X['Glucose'] + X['BMI'] + X['Age']/10
        X['Blood_Risk'] = X['BloodPressure'] + X['SkinThickness']
        
        # Bin continuous variables
        X['Glucose_Bin'] = pd.cut(X['Glucose'], bins=5, labels=False).astype(int)
        X['Age_Bin'] = pd.cut(X['Age'], bins=5, labels=False).astype(int)
        
        # Fill any NaN values that might have been created
        X = X.fillna(0)
        
        return X

    def _prepare_input_data(self, input_data):
        """Prepare input data with the same feature engineering as training"""
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all original features are present
        for feature in self.original_features:
            if feature not in input_df.columns:
                raise ValueError(f"Missing feature in input: {feature}")
        
        # Apply the same feature engineering
        input_engineered = self._create_features(input_df)
        
        return input_engineered

    def _print_model_performance(self, X_test_scaled):
        """Print performance metrics of all individual models"""
        print("\n=== MODEL PERFORMANCE METRICS ===")
        
        # Ensemble metrics
        ensemble_pred = self.ensemble_classifier.predict(X_test_scaled)
        ensemble_metrics = self._calculate_metrics(self.y_test, ensemble_pred)
        print(f"Ensemble - Accuracy: {ensemble_metrics['accuracy']:.4f}, Precision: {ensemble_metrics['precision']:.4f}, Recall: {ensemble_metrics['recall']:.4f}, F1: {ensemble_metrics['f1']:.4f}")
        
        # SVM Linear metrics
        svm_linear_pred = self.svm_linear_model.predict(X_test_scaled)
        svm_linear_metrics = self._calculate_metrics(self.y_test, svm_linear_pred)
        print(f"SVM Linear - Accuracy: {svm_linear_metrics['accuracy']:.4f}, Precision: {svm_linear_metrics['precision']:.4f}, Recall: {svm_linear_metrics['recall']:.4f}, F1: {svm_linear_metrics['f1']:.4f}")
        
        # SVM RBF metrics
        svm_rbf_pred = self.svm_rbf_model.predict(X_test_scaled)
        svm_rbf_metrics = self._calculate_metrics(self.y_test, svm_rbf_pred)
        print(f"SVM RBF - Accuracy: {svm_rbf_metrics['accuracy']:.4f}, Precision: {svm_rbf_metrics['precision']:.4f}, Recall: {svm_rbf_metrics['recall']:.4f}, F1: {svm_rbf_metrics['f1']:.4f}")
        
        # RF metrics
        rf_pred = self.best_rf_model.predict(self.X_test)
        rf_metrics = self._calculate_metrics(self.y_test, rf_pred)
        print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, Precision: {rf_metrics['precision']:.4f}, Recall: {rf_metrics['recall']:.4f}, F1: {rf_metrics['f1']:.4f}")

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

    def predict(self, input_data):
        """Make predictions with individual model outputs"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        try:
            # Prepare input data with feature engineering
            input_engineered = self._prepare_input_data(input_data)
            
            # Scale the features
            input_scaled = self.scaler.transform(input_engineered)
            
            # Get predictions from all models
            ensemble_predictions = self.ensemble_classifier.predict(input_scaled)
            svm_linear_predictions = self.svm_linear_model.predict(input_scaled)
            svm_rbf_predictions = self.svm_rbf_model.predict(input_scaled)
            rf_predictions = self.best_rf_model.predict(input_engineered)
            
            # Use majority voting for final prediction
            predictions = [ensemble_predictions[0], svm_linear_predictions[0], svm_rbf_predictions[0], rf_predictions[0]]
            final_prediction = max(set(predictions), key=predictions.count)
            
            analytics = self._generate_analytics()
            
            return {
                'final_prediction': final_prediction,
                'ensemble_prediction': int(ensemble_predictions[0]),
                'svm_linear_prediction': int(svm_linear_predictions[0]),
                'svm_rbf_prediction': int(svm_rbf_predictions[0]),
                'rf_prediction': int(rf_predictions[0]),
                'analytics': analytics,
                'feature_analysis': self._analyze_prediction_features(input_engineered.iloc[0])
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {e}"}

    def _analyze_prediction_features(self, features):
        """Analyze which features are driving the prediction"""
        importance_dict = self._get_feature_importance()
        
        # Get top features for this prediction
        feature_analysis = {}
        for feature, importance in importance_dict.items():
            if feature in features:
                feature_analysis[feature] = {
                    'importance': importance,
                    'value': features[feature],
                    'contribution': importance * features[feature]
                }
        
        # Sort by contribution magnitude
        sorted_analysis = dict(sorted(
            feature_analysis.items(), 
            key=lambda x: abs(x[1]['contribution']), 
            reverse=True
        )[:5])  # Top 5 features
        
        return sorted_analysis

    def _generate_analytics(self):
        """Generate model performance analytics and visualizations"""
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Calculate performance metrics for all models
        ensemble_metrics = self._calculate_performance_metrics(self.ensemble_classifier, X_test_scaled, "Ensemble")
        svm_linear_metrics = self._calculate_performance_metrics(self.svm_linear_model, X_test_scaled, "SVM Linear")
        svm_rbf_metrics = self._calculate_performance_metrics(self.svm_rbf_model, X_test_scaled, "SVM RBF")
        rf_metrics = self._calculate_performance_metrics(self.best_rf_model, self.X_test, "Random Forest")
        
        # Generate plots
        plots = self._generate_plots(X_test_scaled)
        
        return {
            'ensemble_metrics': ensemble_metrics,
            'svm_linear_metrics': svm_linear_metrics,
            'svm_rbf_metrics': svm_rbf_metrics,
            'rf_metrics': rf_metrics,
            'plots': plots,
            'feature_importance': self._get_feature_importance()
        }

    def _calculate_performance_metrics(self, model, X_test, model_name):
        """Calculate comprehensive performance metrics for a model"""
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            metrics = self._calculate_metrics(self.y_test, y_pred)
            metrics.update({
                'confusion_matrix': cm,
                'model_name': model_name
            })
            return metrics
        return {}

    def _generate_plots(self, X_test_scaled):
        """Generate required visualization plots"""
        plots = {}
        
        # Confusion Matrices for both models
        ensemble_cm = confusion_matrix(self.y_test, self.ensemble_classifier.predict(X_test_scaled))
        plots['ensemble_cm'] = self._plot_confusion_matrix(ensemble_cm, 'Ensemble Confusion Matrix')
        
        rf_cm = confusion_matrix(self.y_test, self.best_rf_model.predict(self.X_test))
        plots['rf_cm'] = self._plot_confusion_matrix(rf_cm, 'Random Forest Confusion Matrix')
        
        # Feature Importance bar graph
        plots['feature_importance'] = self._plot_feature_importance()
        
        # Model Performance Comparison Bar Graph
        plots['performance_comparison'] = self._plot_performance_comparison(X_test_scaled)
        
        return plots

    def _plot_confusion_matrix(self, cm, title):
        """Plot confusion matrix and return base64 string"""
        plt.figure(figsize=(8, 6))
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        sns.heatmap(cm_df, annot=True, fmt='g', cmap='viridis')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted label', fontweight='bold')
        plt.ylabel('True label', fontweight='bold')
        plt.tight_layout()
        
        return self._fig_to_base64()

    def _plot_feature_importance(self):
        """Plot feature importance as bar graph and return base64 string"""
        importances = self.best_rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [self.X_train.columns[i] for i in indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(sorted_importances)), sorted_importances, color='skyblue', edgecolor='navy')
        plt.xlabel('Features', fontweight='bold')
        plt.ylabel('Importance Score', fontweight='bold')
        plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(sorted_importances)), sorted_features, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, importance in zip(bars, sorted_importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        return self._fig_to_base64()

    def _plot_performance_comparison(self, X_test_scaled):
        """Plot performance comparison bar graph for all models"""
        # Get metrics for all models
        ensemble_metrics = self._calculate_performance_metrics(self.ensemble_classifier, X_test_scaled, "Ensemble")
        svm_linear_metrics = self._calculate_performance_metrics(self.svm_linear_model, X_test_scaled, "SVM Linear")
        svm_rbf_metrics = self._calculate_performance_metrics(self.svm_rbf_model, X_test_scaled, "SVM RBF")
        rf_metrics = self._calculate_performance_metrics(self.best_rf_model, self.X_test, "Random Forest")
        
        models = ['Ensemble', 'SVM Linear', 'SVM RBF', 'Random Forest']
        accuracy = [
            ensemble_metrics.get('accuracy', 0),
            svm_linear_metrics.get('accuracy', 0),
            svm_rbf_metrics.get('accuracy', 0),
            rf_metrics.get('accuracy', 0)
        ]
        precision = [
            ensemble_metrics.get('precision', 0),
            svm_linear_metrics.get('precision', 0),
            svm_rbf_metrics.get('precision', 0),
            rf_metrics.get('precision', 0)
        ]
        recall = [
            ensemble_metrics.get('recall', 0),
            svm_linear_metrics.get('recall', 0),
            svm_rbf_metrics.get('recall', 0),
            rf_metrics.get('recall', 0)
        ]
        f1 = [
            ensemble_metrics.get('f1', 0),
            svm_linear_metrics.get('f1', 0),
            svm_rbf_metrics.get('f1', 0),
            rf_metrics.get('f1', 0)
        ]
        
        x = np.arange(len(models))
        width = 0.2
        
        plt.figure(figsize=(14, 8))
        plt.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='lightblue', edgecolor='navy')
        plt.bar(x - 0.5*width, precision, width, label='Precision', color='lightgreen', edgecolor='darkgreen')
        plt.bar(x + 0.5*width, recall, width, label='Recall', color='lightcoral', edgecolor='darkred')
        plt.bar(x + 1.5*width, f1, width, label='F1-Score', color='gold', edgecolor='orange')
        
        plt.xlabel('Models', fontweight='bold')
        plt.ylabel('Scores', fontweight='bold')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, models)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig_to_base64()

    def _get_feature_importance(self):
        """Get feature importance as dictionary"""
        importances = self.best_rf_model.feature_importances_
        return dict(zip(self.X_train.columns, importances))

    def _fig_to_base64(self):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64

# Create instance for import
predictor = DiabetesPredictor()