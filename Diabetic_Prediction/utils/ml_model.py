import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

class DiabetesPredictor:
    def __init__(self):
        self.ensemble_classifier = None
        self.best_rf_model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False
        
    def load_and_train(self):
        """Load data and train models"""
        try:
            # Load data
            data = pd.read_csv('diabetes.csv')
            X = data.drop('Outcome', axis=1)
            y = data['Outcome']
            
            # Data Augmentation using SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            
            # Define SVM classifiers
            svm_linear = SVC(kernel='linear', probability=True, C=0.01)
            svm_rbf = SVC(kernel='rbf', probability=True, C=0.01)
            svm_poly = SVC(kernel='poly', probability=True, C=0.01)
            svm_sigmoid = SVC(kernel='sigmoid', probability=True, C=0.01)
            
            # Define ensemble classifier
            self.ensemble_classifier = VotingClassifier(estimators=[
                ('svm_linear', svm_linear),
                ('svm_rbf', svm_rbf),
                ('svm_poly', svm_poly),
                ('svm_sigmoid', svm_sigmoid)], voting='soft')
            
            # Fit ensemble classifier
            self.ensemble_classifier.fit(X_train_scaled, self.y_train)
            
            # Random Forest with GridSearch
            rf_model = RandomForestClassifier(
                random_state=42, 
                min_samples_leaf=20, 
                min_samples_split=10, 
                max_features='sqrt', 
                max_depth=3, 
                n_estimators=10
            )
            
            param_grid = {
                'n_estimators': [10, 20],
                'max_depth': [3, 5],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [20, 30]
            }
            
            grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            self.best_rf_model = grid_search.best_estimator_
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error in training: {e}")
            return False
    
    def predict(self, input_data):
        """Make predictions and generate analytics"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        try:
            input_df = pd.DataFrame([input_data])
            input_scaled = self.scaler.transform(input_df)
            
            # SVM predictions
            svm_predictions = self.ensemble_classifier.predict(input_scaled)
            svm_probabilities = self.ensemble_classifier.predict_proba(input_scaled)
            
            # Random Forest predictions
            rf_predictions = self.best_rf_model.predict(input_df)
            rf_probabilities = self.best_rf_model.predict_proba(input_df)
            
            # Generate analytics
            analytics = self._generate_analytics()
            
            return {
                'svm_prediction': int(svm_predictions[0]),
                'svm_probability': float(svm_probabilities[0][1]),
                'rf_prediction': int(rf_predictions[0]),
                'rf_probability': float(rf_probabilities[0][1]),
                'analytics': analytics
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {e}"}
    
    def _generate_analytics(self):
        """Generate model performance analytics and visualizations"""
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # SVM metrics
        svm_test_pred = self.ensemble_classifier.predict(X_test_scaled)
        svm_accuracy_test = accuracy_score(self.y_test, svm_test_pred)
        svm_train_pred = self.ensemble_classifier.predict(self.scaler.transform(self.X_train))
        svm_accuracy_train = accuracy_score(self.y_train, svm_train_pred)
        
        # RF metrics
        rf_test_pred = self.best_rf_model.predict(self.X_test)
        rf_accuracy_test = accuracy_score(self.y_test, rf_test_pred)
        rf_train_pred = self.best_rf_model.predict(self.X_train)
        rf_accuracy_train = accuracy_score(self.y_train, rf_train_pred)
        
        # Generate plots
        plots = self._generate_plots(X_test_scaled, rf_test_pred)
        
        return {
            'svm_accuracy_test': svm_accuracy_test,
            'svm_accuracy_train': svm_accuracy_train,
            'rf_accuracy_test': rf_accuracy_test,
            'rf_accuracy_train': rf_accuracy_train,
            'plots': plots,
            'feature_importance': self._get_feature_importance()
        }
    
    def _generate_plots(self, X_test_scaled, rf_test_pred):
        """Generate all visualization plots"""
        plots = {}
        
        # SVM ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.ensemble_classifier.predict_proba(X_test_scaled)[:, 1])
        roc_auc = auc(fpr, tpr)
        plots['svm_roc'] = self._plot_roc_curve(fpr, tpr, roc_auc, 'SVM ROC Curve')
        
        # RF ROC Curve
        rf_probabilities = self.best_rf_model.predict_proba(self.X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, rf_probabilities)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        plots['rf_roc'] = self._plot_roc_curve(fpr_rf, tpr_rf, roc_auc_rf, 'RF ROC Curve')
        
        # Confusion Matrices
        svm_cm = confusion_matrix(self.y_test, self.ensemble_classifier.predict(X_test_scaled))
        plots['svm_cm'] = self._plot_confusion_matrix(svm_cm, 'SVM Confusion Matrix')
        
        rf_cm = confusion_matrix(self.y_test, rf_test_pred)
        plots['rf_cm'] = self._plot_confusion_matrix(rf_cm, 'RF Confusion Matrix')
        
        # Feature Importance
        plots['feature_importance'] = self._plot_feature_importance()
        
        return plots
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc, title):
        """Plot ROC curve and return base64 string"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def _plot_confusion_matrix(self, cm, title):
        """Plot confusion matrix and return base64 string"""
        plt.figure(figsize=(8, 6))
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
        plt.title(title)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def _plot_feature_importance(self):
        """Plot feature importance and return base64 string"""
        importances = self.best_rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = [self.X_train.columns[i] for i in indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_features, sorted_importances, color='green')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self._fig_to_base64()
    
    def _get_feature_importance(self):
        """Get feature importance as dictionary"""
        importances = self.best_rf_model.feature_importances_
        return dict(zip(self.X_train.columns, importances))
    
    def _fig_to_base64(self):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64

# Global instance
predictor = DiabetesPredictor()