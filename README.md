# cnn-heart-disease-predict

This project implements a Convolutional Neural Network (CNN) for heart disease detection using medical data.

## Project Structure
- `main.py`: Main script to train the model
- `model.py`: Contains the CNN model architecture and training code
- `data_processing.py`: Script for loading and preprocessing the heart disease dataset
- `heart_disease_predictor.py`: Comprehensive prediction script with risk profiling and visualization
- `test_cases.py`: Script to test the model on different patient profiles
- `requirements.txt`: Contains all required dependencies

## Setup and Installation

1. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Train the model:
```
python main.py
```

## Making Predictions

### Using the Comprehensive Predictor (Recommended)

The comprehensive predictor provides risk profiling, visualization, and clinical recommendations:

```
python heart_disease_predictor.py --age 63 --sex 1 --cp 3 --trestbps 145 --chol 233 --fbs 1 --restecg 0 --thalach 150 --exang 0 --oldpeak 2.3 --slope 0 --ca 0 --thal 1 --visualize
```

Add `--visualize` to generate visualizations and `--output filename.png` to save them.

### Testing Multiple Patient Profiles

To test the model on both high-risk and low-risk patients:

```
python test_cases.py
```

This script demonstrates how adaptive thresholds correct the predictions for different risk profiles.

### Risk Profiles

The predictor automatically categorizes patients into risk profiles:

1. **VERY HIGH RISK**: 5+ risk factors (threshold = 0.2)
2. **HIGH RISK**: 3+ risk factors (threshold = 0.3)
3. **MODERATE RISK**: Standard case (threshold = 0.5)
4. **LOW RISK**: 5+ protective factors (threshold = 0.7)
5. **VERY LOW RISK**: 7+ protective factors (threshold = 0.8)

## Dataset
This model uses the UCI Heart Disease dataset, which will be automatically downloaded when running the code. The dataset contains various features like age, sex, chest pain type, resting blood pressure, cholesterol levels, etc.

## Model Architecture
The model transforms tabular data into a format suitable for CNN processing and uses convolutional layers to detect patterns in the data that are indicative of heart disease.

## Backup
Non-essential files have been moved to the `backup` directory.
