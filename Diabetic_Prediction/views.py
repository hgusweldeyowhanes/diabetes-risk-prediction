from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views import View
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils.ml_model import predictor
from .serializers import DiabetesPredictionSerializer

def home(request):
    """Home page view"""
    return render(request, 'home.html')

class DiabetesPredictionView(View):
    template_name = 'prediction_form.html'
    result_template = 'prediction_result.html'
    
    def get(self, request):
        # For GET requests, provide field information for template
        context = {
            'field_info': {
                'pregnancies': {'min': 0, 'max': 20, 'help_text': 'Number of times pregnant'},
                'glucose': {'min': 0, 'max': 400, 'help_text': 'Plasma glucose concentration'},
                'blood_pressure': {'min': 0, 'max': 202, 'help_text': 'Diastolic blood pressure (mm Hg)'},
                'skin_thickness': {'min': 0, 'max': 200, 'help_text': 'Triceps skin fold thickness (mm)'},
                'insulin': {'min': 0, 'max': 900, 'help_text': '2-Hour serum insulin (mu U/ml)'},
                'bmi': {'min': 0.0, 'max': 70.0, 'help_text': 'Body mass index', 'step': '0.1'},
                'diabetes_pedigree_function': {'min': 0.0, 'max': 3.0, 'help_text': 'Diabetes pedigree function', 'step': '0.001'},
                'age': {'min': 1, 'max': 120, 'help_text': 'Age in years'},
            }
        }
        return render(request, self.template_name, context)
    
    def post(self, request):
        # Use serializer for validation
        serializer = DiabetesPredictionSerializer(data=request.POST)
        
        if serializer.is_valid():
            # Get validated data from serializer
            validated_data = serializer.validated_data
            
            # Convert to ML model format
            input_data = {
                'Pregnancies': validated_data['pregnancies'],
                'Glucose': validated_data['glucose'],
                'BloodPressure': validated_data['blood_pressure'],
                'SkinThickness': validated_data['skin_thickness'],
                'Insulin': validated_data['insulin'],
                'BMI': validated_data['bmi'],
                'DiabetesPedigreeFunction': validated_data['diabetes_pedigree_function'],
                'Age': validated_data['age']
            }
            
            # Load and train model if not already trained
            if not predictor.is_trained:
                success = predictor.load_and_train()
                if not success:
                    return render(request, self.template_name, {
                        'error': 'Model training failed. Please try again.',
                        'field_info': self._get_field_info()
                    })
            
            # Make prediction
            result = predictor.predict(input_data)
            
            if 'error' in result:
                return render(request, self.template_name, {
                    'error': result['error'],
                    'field_info': self._get_field_info()
                })
            
            # Prepare context for result template
            context = self._prepare_result_context(input_data, result)
            
            return render(request, self.result_template, context)
        
        else:
            # Return form with serializer errors
            return render(request, self.template_name, {
                'errors': serializer.errors,
                'field_info': self._get_field_info(),
                'submitted_data': request.POST
            })
    
    def _get_field_info(self):
        """Get field information for template"""
        return {
            'pregnancies': {'min': 0, 'max': 20, 'help_text': 'Number of times pregnant'},
            'glucose': {'min': 0, 'max': 400, 'help_text': 'Plasma glucose concentration'},
            'blood_pressure': {'min': 0, 'max': 202, 'help_text': 'Diastolic blood pressure (mm Hg)'},
            'skin_thickness': {'min': 0, 'max': 200, 'help_text': 'Triceps skin fold thickness (mm)'},
            'insulin': {'min': 0, 'max': 900, 'help_text': '2-Hour serum insulin (mu U/ml)'},
            'bmi': {'min': 0.0, 'max': 70.0, 'help_text': 'Body mass index', 'step': '0.1'},
            'diabetes_pedigree_function': {'min': 0.0, 'max': 3.0, 'help_text': 'Diabetes pedigree function', 'step': '0.001'},
            'age': {'min': 1, 'max': 120, 'help_text': 'Age in years'},
        }
    
    def _prepare_result_context(self, form_data, result):
        """Prepare all required context variables for the result template"""
        context = {
            'form_data': form_data,
            'final_prediction': 'Diabetic' if result.get('final_prediction', 0) == 1 else 'Not Diabetic',
        }
        
        # Add individual model predictions
        context.update({
            'ensemble_prediction': 'Diabetic' if result.get('ensemble_prediction', 0) == 1 else 'Not Diabetic',
            'svm_linear_prediction': 'Diabetic' if result.get('svm_linear_prediction', 0) == 1 else 'Not Diabetic',
            'svm_rbf_prediction': 'Diabetic' if result.get('svm_rbf_prediction', 0) == 1 else 'Not Diabetic',
            'rf_prediction': 'Diabetic' if result.get('rf_prediction', 0) == 1 else 'Not Diabetic',
        })
        
        # Add analytics if available
        analytics = result.get('analytics', {})
        if analytics:
            # Add performance metrics
            self._add_performance_metrics(context, analytics)
            
            # Add plots
            context['plots'] = analytics.get('plots', {})
            
            # Add feature importance
            context['feature_importance'] = analytics.get('feature_importance', {})
        
        # Add feature analysis if available
        feature_analysis = result.get('feature_analysis', {})
        if feature_analysis:
            context['feature_analysis'] = self._format_feature_analysis(feature_analysis)
        
        # Add risk level based on majority agreement
        context['risk_level'] = self._calculate_risk_level(result)
        
        return context
    
    def _add_performance_metrics(self, context, analytics):
        """Add performance metrics to context"""
        # Ensemble metrics
        ensemble_metrics = analytics.get('ensemble_metrics', {})
        context.update({
            'ensemble_accuracy': ensemble_metrics.get('accuracy', 0) * 100,
            'ensemble_precision': ensemble_metrics.get('precision', 0) * 100,
            'ensemble_recall': ensemble_metrics.get('recall', 0) * 100,
            'ensemble_f1': ensemble_metrics.get('f1', 0) * 100,
        })
        
        # SVM Linear metrics
        svm_linear_metrics = analytics.get('svm_linear_metrics', {})
        context.update({
            'svm_linear_accuracy': svm_linear_metrics.get('accuracy', 0) * 100,
            'svm_linear_precision': svm_linear_metrics.get('precision', 0) * 100,
            'svm_linear_recall': svm_linear_metrics.get('recall', 0) * 100,
            'svm_linear_f1': svm_linear_metrics.get('f1', 0) * 100,
        })
        
        # SVM RBF metrics
        svm_rbf_metrics = analytics.get('svm_rbf_metrics', {})
        context.update({
            'svm_rbf_accuracy': svm_rbf_metrics.get('accuracy', 0) * 100,
            'svm_rbf_precision': svm_rbf_metrics.get('precision', 0) * 100,
            'svm_rbf_recall': svm_rbf_metrics.get('recall', 0) * 100,
            'svm_rbf_f1': svm_rbf_metrics.get('f1', 0) * 100,
        })
        
        # Random Forest metrics
        rf_metrics = analytics.get('rf_metrics', {})
        context.update({
            'rf_accuracy': rf_metrics.get('accuracy', 0) * 100,
            'rf_precision': rf_metrics.get('precision', 0) * 100,
            'rf_recall': rf_metrics.get('recall', 0) * 100,
            'rf_f1': rf_metrics.get('f1', 0) * 100,
        })
    
    def _format_feature_analysis(self, feature_analysis):
        """Format feature analysis for template display"""
        formatted = []
        for feature, analysis in feature_analysis.items():
            formatted.append({
                'name': feature.replace('_', ' ').title(),
                'value': f"{analysis.get('value', 0):.2f}",
                'importance': f"{analysis.get('importance', 0):.4f}",
                'contribution': f"{analysis.get('contribution', 0):.4f}",
                'impact': 'Positive' if analysis.get('contribution', 0) > 0 else 'Negative'
            })
        return formatted
    
    def _calculate_risk_level(self, result):
        """Calculate risk level based on model agreement"""
        predictions = [
            result.get('ensemble_prediction', 0),
            result.get('svm_linear_prediction', 0),
            result.get('svm_rbf_prediction', 0),
            result.get('rf_prediction', 0)
        ]
        
        diabetic_count = sum(predictions)
        
        if diabetic_count >= 3:
            return 'High Risk'
        elif diabetic_count == 2:
            return 'Moderate Risk'
        elif diabetic_count == 1:
            return 'Low Risk'
        else:
            return 'Very Low Risk'
    
    def _get_confidence_style(self, confidence):
        """Return CSS class based on confidence level (kept for compatibility)"""
        if confidence >= 80:
            return 'high-confidence'
        elif confidence >= 60:
            return 'medium-confidence'
        else:
            return 'low-confidence'


# DRF API View
class DiabetesPredictionAPIView(APIView):
    """
    API endpoint for diabetes prediction using DRF serializer
    """
    
    def post(self, request):
        serializer = DiabetesPredictionSerializer(data=request.data)
        
        if serializer.is_valid():
            # Get validated data
            validated_data = serializer.validated_data
            
            # Convert to ML model format
            input_data = {
                'Pregnancies': validated_data['pregnancies'],
                'Glucose': validated_data['glucose'],
                'BloodPressure': validated_data['blood_pressure'],
                'SkinThickness': validated_data['skin_thickness'],
                'Insulin': validated_data['insulin'],
                'BMI': validated_data['bmi'],
                'DiabetesPedigreeFunction': validated_data['diabetes_pedigree_function'],
                'Age': validated_data['age']
            }
            
            # Load and train model if not already trained
            if not predictor.is_trained:
                success = predictor.load_and_train()
                if not success:
                    return Response({
                        'error': 'Model training failed'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Make prediction
            result = predictor.predict(input_data)
            
            if 'error' in result:
                return Response({
                    'error': result['error']
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Format response
            response_data = {
                'status': 'success',
                'prediction_data': result,
                'input_data': input_data
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
        
        return Response({
            'status': 'error',
            'message': 'Invalid data',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


# Legacy API views (keep for compatibility)
@csrf_exempt
@require_http_methods(["POST"])
def api_predict_diabetes(request):
    """API endpoint for predictions (legacy)"""
    try:
        # Load and train model if not already trained
        if not predictor.is_trained:
            success = predictor.load_and_train()
            if not success:
                return JsonResponse({'error': 'Model training failed'}, status=500)
        
        # Parse input data
        data = json.loads(request.body)
        
        # Use serializer for validation
        serializer = DiabetesPredictionSerializer(data=data)
        
        if not serializer.is_valid():
            return JsonResponse({
                'error': 'Validation failed',
                'details': serializer.errors
            }, status=400)
        
        validated_data = serializer.validated_data
        
        # Convert to ML model format
        input_data = {
            'Pregnancies': validated_data['pregnancies'],
            'Glucose': validated_data['glucose'],
            'BloodPressure': validated_data['blood_pressure'],
            'SkinThickness': validated_data['skin_thickness'],
            'Insulin': validated_data['insulin'],
            'BMI': validated_data['bmi'],
            'DiabetesPedigreeFunction': validated_data['diabetes_pedigree_function'],
            'Age': validated_data['age']
        }
        
        # Make prediction
        result = predictor.predict(input_data)
        
        if 'error' in result:
            return JsonResponse(result, status=500)
        
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({'status': 'healthy', 'model_trained': predictor.is_trained})