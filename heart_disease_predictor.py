import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import os
from data_processing import load_and_preprocess_single_sample

def predict_heart_disease(model_path, data_dict, visualize=False, output_file=None):
    """
    Make a prediction using the trained model with risk-based thresholds.
    
    Args:
        model_path: Path to the saved model
        data_dict: Dictionary containing feature values
        visualize: Whether to generate visualization
        output_file: Path to save visualization (if visualize=True)
        
    Returns:
        Prediction result and analysis data
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'python main.py' first to train the model.")
        return None, None, None, None, None, None
        
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run 'python main.py' first to train the model.")
        return None, None, None, None, None, None
    
    # Preprocess the data
    preprocessed_data = load_and_preprocess_single_sample(data_dict)
    
    # Make prediction
    prediction = model.predict(preprocessed_data, verbose=0)
    disease_probability = prediction[0][1]  # Probability of disease (class 1)
    
    # Analyze risk factors (high risk)
    high_risk_factors = []
    if data_dict['age'] > 60:
        high_risk_factors.append(f"Age {data_dict['age']} (high risk: age > 60)")
    if data_dict['sex'] == 1:
        high_risk_factors.append("Male gender (higher risk than female)")
    if data_dict['cp'] == 3:
        high_risk_factors.append("Asymptomatic chest pain (high risk)")
    if data_dict['trestbps'] > 140:
        high_risk_factors.append(f"Elevated blood pressure: {data_dict['trestbps']} mmHg (high risk: > 140)")
    if data_dict['chol'] > 220:
        high_risk_factors.append(f"High cholesterol: {data_dict['chol']} mg/dl (high risk: > 220)")
    if data_dict['fbs'] == 1:
        high_risk_factors.append("Elevated fasting blood sugar (risk factor)")
    if data_dict['thalach'] < 150:
        high_risk_factors.append(f"Lower maximum heart rate: {data_dict['thalach']} (risk factor: < 150)")
    if data_dict['oldpeak'] > 1.5:
        high_risk_factors.append(f"Significant ST depression: {data_dict['oldpeak']} (high risk: > 1.5)")
    if data_dict['thal'] != 2:
        high_risk_factors.append("Abnormal thalassemia (risk factor)")
    if data_dict['ca'] > 0:
        high_risk_factors.append(f"Major vessels colored: {data_dict['ca']} (risk factor: > 0)")
    if data_dict['exang'] == 1:
        high_risk_factors.append("Exercise-induced angina (risk factor)")
    
    # Analyze protective factors (low risk)
    protective_factors = []
    if data_dict['age'] < 40:
        protective_factors.append(f"Young age: {data_dict['age']} (low risk: age < 40)")
    if data_dict['sex'] == 0:
        protective_factors.append("Female gender (lower risk than male)")
    if data_dict['cp'] == 0:
        protective_factors.append("No chest pain (low risk)")
    if data_dict['trestbps'] < 120:
        protective_factors.append(f"Normal/low blood pressure: {data_dict['trestbps']} mmHg (low risk: < 120)")
    if data_dict['chol'] < 200:
        protective_factors.append(f"Normal/low cholesterol: {data_dict['chol']} mg/dl (low risk: < 200)")
    if data_dict['fbs'] == 0:
        protective_factors.append("Normal fasting blood sugar (protective factor)")
    if data_dict['thalach'] > 150:
        protective_factors.append(f"Good maximum heart rate: {data_dict['thalach']} (low risk: > 150)")
    if data_dict['oldpeak'] < 0.5:
        protective_factors.append(f"Minimal/no ST depression: {data_dict['oldpeak']} (low risk: < 0.5)")
    if data_dict['thal'] == 2:
        protective_factors.append("Normal thalassemia (low risk)")
    if data_dict['ca'] == 0:
        protective_factors.append("No major vessels colored (low risk)")
    if data_dict['exang'] == 0:
        protective_factors.append("No exercise-induced angina (protective factor)")
    
    # Calculate risk scores
    high_risk_score = len(high_risk_factors)
    protective_score = len(protective_factors)
    
    # Determine patient profile
    if high_risk_score >= 5:
        profile = "VERY HIGH RISK"
    elif high_risk_score >= 3:
        profile = "HIGH RISK"
    elif protective_score >= 7:
        profile = "VERY LOW RISK"
    elif protective_score >= 5:
        profile = "LOW RISK"
    else:
        profile = "MODERATE RISK"
    
    # Determine threshold based on risk profile
    if profile == "VERY HIGH RISK":
        threshold = 0.2  # Very low threshold for very high-risk patients
    elif profile == "HIGH RISK":
        threshold = 0.3  # Lower threshold for high-risk patients
    elif profile == "VERY LOW RISK":
        threshold = 0.8  # Very high threshold for very low-risk patients
    elif profile == "LOW RISK":
        threshold = 0.7  # Higher threshold for low-risk patients
    else:
        threshold = 0.5  # Standard threshold for moderate-risk patients
    
    # Get the class and probability
    predicted_class = 1 if disease_probability > threshold else 0
    
    # Override rules for extreme cases
    if profile == "VERY HIGH RISK" and disease_probability > 0.2:
        predicted_class = 1  # Always predict disease for very high-risk patients with prob > 0.2
    elif profile == "VERY LOW RISK" and disease_probability < 0.8:
        predicted_class = 0  # Always predict no disease for very low-risk patients with prob < 0.8
    
    # Generate visualization if requested
    if visualize:
        # Analyze predictions at different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_results = []
        
        for t in thresholds:
            pred_class = 1 if disease_probability > t else 0
            result = "Disease" if pred_class == 1 else "No Disease"
            threshold_results.append((t, result))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Disease probability vs threshold
        plt.subplot(2, 1, 1)
        plt.axhline(y=disease_probability, color='r', linestyle='-', label=f'Disease Probability: {disease_probability:.4f}')
        plt.axvline(x=threshold, color='g', linestyle='--', label=f'Selected Threshold: {threshold}')
        plt.axvline(x=0.5, color='b', linestyle=':', label='Standard Threshold (0.5)')
        
        plt.xlabel('Threshold')
        plt.ylabel('Probability')
        plt.title(f'Disease Probability vs. Decision Threshold - {profile} PROFILE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Risk factors vs protective factors
        plt.subplot(2, 1, 2)
        bars = plt.bar([0, 1], [high_risk_score, protective_score], color=['red', 'green'])
        plt.xticks([0, 1], ['Risk Factors', 'Protective Factors'])
        plt.ylabel('Count')
        plt.title('Risk Assessment')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the visualization if output file is provided
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved as '{output_file}'")
        
        plt.show()
    
    return predicted_class, disease_probability, high_risk_factors, protective_factors, threshold, profile

def main():
    """
    Main function to run the heart disease prediction script.
    """
    parser = argparse.ArgumentParser(description='Predict heart disease using trained CNN model')
    parser.add_argument('--model', type=str, default='best_heart_disease_model.h5', help='Path to the trained model')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    
    # Add arguments for each feature
    parser.add_argument('--age', type=float, required=True, help='Age in years')
    parser.add_argument('--sex', type=int, required=True, help='Sex (1 = male, 0 = female)')
    parser.add_argument('--cp', type=int, required=True, help='Chest pain type (0-3)')
    parser.add_argument('--trestbps', type=float, required=True, help='Resting blood pressure (mm Hg)')
    parser.add_argument('--chol', type=float, required=True, help='Serum cholesterol (mg/dl)')
    parser.add_argument('--fbs', type=int, required=True, help='Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)')
    parser.add_argument('--restecg', type=int, required=True, help='Resting electrocardiographic results (0-2)')
    parser.add_argument('--thalach', type=float, required=True, help='Maximum heart rate achieved')
    parser.add_argument('--exang', type=int, required=True, help='Exercise induced angina (1 = yes, 0 = no)')
    parser.add_argument('--oldpeak', type=float, required=True, help='ST depression induced by exercise relative to rest')
    parser.add_argument('--slope', type=int, required=True, help='Slope of the peak exercise ST segment (0-2)')
    parser.add_argument('--ca', type=int, required=True, help='Number of major vessels colored by fluoroscopy (0-3)')
    parser.add_argument('--thal', type=int, required=True, help='Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)')
    
    args = parser.parse_args()
    
    # Create a dictionary from the arguments
    data_dict = {
        'age': args.age,
        'sex': args.sex,
        'cp': args.cp,
        'trestbps': args.trestbps,
        'chol': args.chol,
        'fbs': args.fbs,
        'restecg': args.restecg,
        'thalach': args.thalach,
        'exang': args.exang,
        'oldpeak': args.oldpeak,
        'slope': args.slope,
        'ca': args.ca,
        'thal': args.thal
    }
    
    # Print patient data
    print("\nPATIENT DATA:")
    print("=" * 60)
    for key, value in data_dict.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    predicted_class, probability, high_risk_factors, protective_factors, threshold, profile = predict_heart_disease(
        args.model, data_dict, args.visualize, args.output
    )
    
    # Print risk profile
    print(f"\nRISK PROFILE: {profile}")
    print("=" * 60)
    
    # Print risk factors
    if high_risk_factors:
        print("\nRisk Factors:")
        print("-" * 60)
        for factor in high_risk_factors:
            print(f"• {factor}")
        print(f"\nTotal risk factors: {len(high_risk_factors)}")
    
    # Print protective factors
    if protective_factors:
        print("\nProtective Factors:")
        print("-" * 60)
        for factor in protective_factors:
            print(f"• {factor}")
        print(f"\nTotal protective factors: {len(protective_factors)}")
    
    # Print result
    if predicted_class is not None:
        result = "Disease" if predicted_class == 1 else "No Disease"
        print("\nPREDICTION RESULTS:")
        print("=" * 60)
        print(f"Disease Probability: {probability:.4f}")
        print(f"Threshold Used: {threshold}")
        
        # Explain threshold choice
        if profile == "VERY HIGH RISK":
            print(f"(Very low threshold used due to very high-risk profile)")
        elif profile == "HIGH RISK":
            print(f"(Lower threshold used due to high-risk profile)")
        elif profile == "VERY LOW RISK":
            print(f"(Very high threshold used due to very low-risk profile)")
        elif profile == "LOW RISK":
            print(f"(Higher threshold used due to low-risk profile)")
        else:
            print("(Standard threshold used for moderate-risk profile)")
            
        # Explain any overrides
        if profile == "VERY HIGH RISK" and probability > 0.2 and probability <= threshold:
            print("(Prediction overridden to Disease due to very high-risk profile)")
        elif profile == "VERY LOW RISK" and probability < 0.8 and probability > threshold:
            print("(Prediction overridden to No Disease due to very low-risk profile)")
        
        print(f"\nFINAL PREDICTION: {result}")
        
        # Add appropriate clinical recommendations
        print("\nCLINICAL RECOMMENDATIONS:")
        print("-" * 60)
        
        if profile == "VERY HIGH RISK":
            print("• This patient has MULTIPLE SEVERE risk factors for heart disease.")
            print("• Medical consultation is URGENTLY recommended.")
            print("• Consider comprehensive cardiac evaluation including stress test and imaging.")
        elif profile == "HIGH RISK":
            print("• This patient has multiple risk factors for heart disease.")
            print("• Medical consultation is strongly recommended.")
            print("• Consider cardiac risk assessment and appropriate screening tests.")
        elif profile == "VERY LOW RISK":
            print("• This patient has an excellent cardiovascular health profile.")
            print("• Continue healthy lifestyle habits.")
            print("• Regular check-ups are recommended as a preventive measure.")
        elif profile == "LOW RISK":
            print("• This patient has multiple protective factors against heart disease.")
            print("• Maintain healthy lifestyle habits.")
            print("• Regular check-ups are recommended.")
        else:
            print("• This patient has a moderate risk profile for heart disease.")
            print("• Consider lifestyle modifications to reduce risk factors.")
            print("• Regular check-ups are recommended.")
    else:
        print("\nPrediction failed. Please ensure the model is trained first.")
        print("Run 'python main.py' to train the model.")

if __name__ == "__main__":
    main() 