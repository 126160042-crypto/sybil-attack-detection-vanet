# Sybil Attack Detection in VANET using Machine Learning

## Overview
This project focuses on detecting Sybil attacks in Vehicular Ad Hoc Networks (VANETs) using simulation and machine learning techniques.

A Sybil attack occurs when a malicious vehicle creates multiple fake identities, disrupting traffic flow and network trust.

---

## Key Features
- Simulation of VANET using SUMO (6x6 grid network)
- Dynamic generation of Sybil vehicles
- Real-time data collection (speed, position, ID)
- Machine learning-based detection (Random Forest)
- Optimization techniques (PSO, WOA, ACO, GWO, Bayesian)
- Visualization using graphs and confusion matrix

---

## Simulation Details
- Grid-based traffic simulation using SUMO
- Vehicles dynamically turn into Sybil nodes based on abnormal behavior

### Sybil Vehicle Behavior
- Random speeds
- Sudden position jumps
- Unusual movement patterns

- Normal vehicles → Blue  
- Sybil vehicles → Red  

---

## Dataset
Generated dataset includes:
- Vehicle ID
- Speed
- Position (X, Y)
- Acceleration
- Edge ID
- Sybil label (Yes/No)

The dataset is saved as:
- `combined_traffic_data.xlsx`

---

## Machine Learning Model

### Algorithms Used:
- Random Forest (Primary model)
- SVM
- KNN

### Best Model:
Random Forest achieved highest accuracy (~99.7%)

### Why Random Forest?
- Handles large datasets
- Less overfitting
- High accuracy and fast prediction

---

## Optimization Techniques
- Particle Swarm Optimization (PSO)
- Whale Optimization Algorithm (WOA)
- Grey Wolf Optimization (GWO)
- Ant Colony Optimization (ACO)
- Bayesian Optimization

GWO provided the best performance and accuracy.

---

## Results
- Accuracy: ~99.7%
- Fast prediction time
- Robust against noisy data

Confusion matrix and evaluation metrics are used for performance analysis.

---

## Tech Stack
- Python
- SUMO Simulator
- Scikit-learn
- Pandas, NumPy
- Matplotlib

---

## How to Run

1. Install dependencies:
pip install -r requirements.txt


2. Run simulation:

python sybil_simulation.py


3. Train model:

python train_model.py


4. Test model:

python test_model.py


---

## Project Output
The system successfully detects Sybil vehicles based on abnormal speed and movement patterns with high accuracy.

---

## Author
Sakthi Sahana V  
B.Tech ECE (CPS)
