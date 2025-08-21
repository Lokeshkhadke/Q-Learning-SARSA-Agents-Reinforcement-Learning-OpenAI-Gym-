# 🧠 Machine Learning & Reinforcement Learning Pipeline

### **A Comprehensive Implementation of Supervised and Reinforcement Learning Algorithms**

This project demonstrates a full **CRISP-DM data mining pipeline** applied to three distinct problems: binary classification, regression, and reinforcement learning. Developed as part of a Master's in Data Science, it showcases end-to-end machine learning capabilities from data understanding to model deployment.

> **🎯 Academic Context:** This work was submitted for **COMM055 – Machine Learning and Data Mining** at the University of Surrey, achieving excellence in implementing diverse ML paradigms under rigorous academic standards.

---

## ✨ Key Features & Achievements

- **Diverse ML Paradigms:** Implements supervised learning (classification & regression) and reinforcement learning in a unified framework
- **Production-Ready Pipeline:** Complete workflow from data loading → EDA → preprocessing → modeling → evaluation → deployment
- **Optimized Performance:**
  - **92.3% accuracy** on diabetes classification using Logistic Regression
  - **0.373 MSE** on forest fire area prediction using Random Forest
  - **50% success rate** on Taxi-v3 environment using Q-Learning
- **Comparative Analysis:** Includes SARSA implementation for reinforcement learning comparison
- **Model Persistence:** All trained models saved for deployment (`joblib` & `numpy`)

---

## 📊 Results Summary

| Algorithm | Dataset | Metric | Value |
| :--- | :--- | :--- | :--- |
| Logistic Regression | Diabetes | F1-Score | **0.922** |
| Random Forest | Forest Fires | MSE | **0.373** |
| Q-Learning | Taxi-v3 | Success Rate | **50.0%** |
| SARSA | Taxi-v3 | Avg Reward | -769.9 |

---

## 🛠️ Technical Implementation

### **1. Supervised Learning**

#### Diabetes Classification (`diabetes_data_upload.csv`)
- **Objective:** Binary classification (Positive/Negative diabetes diagnosis)
- **Features:** 16 medical attributes (Age, Gender, Polyuria, Polydipsia, etc.)
- **Preprocessing:** One-hot encoding for categorical variables
- **Model:** Logistic Regression with maximum iterations
- **Performance:** 92% accuracy, 93% precision, 96% recall for positive class

#### Forest Fires Regression (`forestfires.csv`)
- **Objective:** Predict burned area of forest fires
- **Features:** Temperature, relative humidity, wind speed
- **Preprocessing:** Log transformation of skewed target variable (`area`)
- **Model:** Random Forest Regressor (100 estimators)
- **Performance:** MSE 0.373, temperature identified as most important feature

### **2. Reinforcement Learning**

#### Taxi-v3 Environment (`gymnasium`)
- **Objective:** Train an agent to navigate a taxi environment (pick up and drop off passengers)
- **State space:** 500 discrete states
- **Action space:** 6 possible actions (up, down, left, right, pickup, dropoff)
- **Algorithms Implemented:**
  - **Q-Learning** (off-policy TD learning)
  - **SARSA** (on-policy TD learning) for comparison

#### Hyperparameters:
```python
EPISODES = 5000    # Training episodes
ALPHA = 0.1        # Learning rate
GAMMA = 0.99       # Discount factor
EPSILON = 1.0      # Initial exploration rate
EPSILON_MIN = 0.01 # Minimum exploration rate
EPSILON_DECAY = 0.995 # Exploration decay rate
```

#### Performance:
- **Q-Learning** achieved 50% success rate in last 100 episodes
- **SARSA** showed slower convergence with higher variance
- **Q-table** visualization provides insights into learned policy

---

## 📈 Key Insights

### Supervised Learning
- **Diabetes Classification:** Strong performance with minimal false negatives (critical for medical diagnosis)
- **Forest Fires:** Temperature is the most significant predictor of fire area (45% feature importance)
- **Model Interpretation:** Confusion matrices and feature importance provide transparent model reasoning

### Reinforcement Learning
- **Q-Learning** outperformed SARSA in the Taxi-v3 environment
- **Exploration-Exploitation** balance crucial for convergence (epsilon decay strategy)
- **Reward shaping** significantly impacts learning efficiency

---

## 🚀 How to Use

### Installation
```bash
# Install dependencies
pip install gymnasium
pip install gymnasium[toy-text]
pip install numpy pandas matplotlib seaborn tqdm scikit-learn
```

### Running the Project
```python
# Execute the full pipeline
exec(open('ml_pipeline.py').read())  # Or run cells in Jupyter notebook

# Load pre-trained models
diabetes_model = joblib.load("diabetes_model.pkl")
forestfires_model = joblib.load("forestfires_model.pkl")
q_table = np.load("taxi_q_table.npy")
```



## 🔮 Future Enhancements

1. **Advanced RL Techniques:** Implement Deep Q-Networks (DQN) for continuous state spaces
2. **Hyperparameter Optimization:** Bayesian optimization for RL parameters
3. **Explainable AI:** SHAP values for model interpretability
4. **Web Deployment:** Create interactive dashboard with Streamlit
5. **Additional Datasets:** Extend to more complex reinforcement learning environments

---

## 👨‍💻 Author

**Lokesh K** | MSc Data Science

*University of Surrey (2024-2025)*

- **LinkedIn:** [Lokesh Khadke](https://www.linkedin.com/in/lokeshkhadke)
- **Email:** lkhadke32@outlook.com


*This project demonstrates comprehensive machine learning capabilities across supervised and reinforcement learning paradigms. Open to opportunities in data science and machine learning engineering.*

---

## 📚 References

1. CRISP-DM Methodology
2. Sutton & Barto - Reinforcement Learning: An Introduction
3. Scikit-learn Documentation
4. Gymnasium/Taxi-v3 Environment

---

**⭐️ Pro Tip:** The Q-table visualization provides fascinating insights into how the agent learns optimal policies for different states. Examine the heatmap to understand the learned decision-making process!
