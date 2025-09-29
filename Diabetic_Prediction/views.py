from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views import View
import json
from .utils.ml_model import predictor
from .forms import DiabetesPredictionForm

def home(request):
    """Home page view"""
    return render(request, 'home.html')

class DiabetesPredictionView(View):
    template_name = 'prediction_form.html'
    result_template = 'prediction_result.html'
    
    def get(self, request):
        form = DiabetesPredictionForm()
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        form = DiabetesPredictionForm(request.POST)
        
        if form.is_valid():
            # Convert form data to ML model format
            input_data = {
                'Pregnancies': form.cleaned_data['pregnancies'],
                'Glucose': form.cleaned_data['glucose'],
                'BloodPressure': form.cleaned_data['blood_pressure'],
                'SkinThickness': form.cleaned_data['skin_thickness'],
                'Insulin': form.cleaned_data['insulin'],
                'BMI': form.cleaned_data['bmi'],
                'DiabetesPedigreeFunction': form.cleaned_data['diabetes_pedigree_function'],
                'Age': form.cleaned_data['age']
            }
            
            # Load and train model if not already trained
            if not predictor.is_trained:
                success = predictor.load_and_train()
                if not success:
                    return render(request, self.template_name, {
                        'form': form,
                        'error': 'Model training failed. Please try again.'
                    })
            
            # Make prediction
            result = predictor.predict(input_data)
            
            if 'error' in result:
                return render(request, self.template_name, {
                    'form': form,
                    'error': result['error']
                })
            
            # Prepare context for result template
            context = self._prepare_result_context(input_data, result)
            
            return render(request, self.result_template, context)
        
        return render(request, self.template_name, {'form': form})
    
    def _prepare_result_context(self, form_data, result):
        """Prepare all required context variables for the result template"""
        context = {
            'form_data': form_data,
            'svm_prediction': 'Diabetic' if result.get('svm_prediction', 0) == 1 else 'Not Diabetic',
            'rf_prediction': 'Diabetic' if result.get('rf_prediction', 0) == 1 else 'Not Diabetic',
            'svm_confidence': result.get('svm_probability', 0) * 100,
            'rf_confidence': result.get('rf_probability', 0) * 100,
        }
        
        # Add analytics if available
        analytics = result.get('analytics', {})
        if analytics:
            context.update({
                'svm_accuracy': analytics.get('svm_accuracy_test', 0) * 100,
                'rf_accuracy': analytics.get('rf_accuracy_test', 0) * 100,
                'plots': analytics.get('plots', {}),
                'feature_importance': analytics.get('feature_importance', {})
            })
        else:
            # Provide default values if analytics is missing
            context.update({
                'svm_accuracy': 0,
                'rf_accuracy': 0,
                'plots': {},
                'feature_importance': {}
            })
        
        return context

# API views (for potential mobile app or other integrations)
@csrf_exempt
@require_http_methods(["POST"])
def api_predict_diabetes(request):
    """API endpoint for predictions"""
    try:
        # Load and train model if not already trained
        if not predictor.is_trained:
            success = predictor.load_and_train()
            if not success:
                return JsonResponse({'error': 'Model training failed'}, status=500)
        
        # Parse input data
        data = json.loads(request.body)
        
        # Validate input data
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        for field in required_fields:
            if field not in data:
                return JsonResponse({'error': f'Missing field: {field}'}, status=400)
        
        # Make prediction
        result = predictor.predict(data)
        
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