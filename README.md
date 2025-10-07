# 💳 Credit Approval Prediction App

This is an interactive **Streamlit** web application that predicts the **credit approval risk level** of a customer using a pre-trained **XGBoost model** (`xgb_classifier.pkl`). The app allows users to input customer financial and demographic details and returns probability scores across different credit risk categories.

---

## 🚀 Features

- **Interactive Web UI:** Built using Streamlit for intuitive use.  
- **Dynamic Inputs:** Automatically generates sidebar input fields based on model features.  
- **Real-Time Prediction:** Uses a trained XGBoost classifier to calculate credit risk probabilities.  
- **Visual Output:** Displays a probability bar chart and detailed metric breakdown.  
- **Input Transparency:** Shows all user-provided input data with human-readable labels.  

---

## 🧠 Model Details

- **Model Type:** XGBoost Classifier  
- **Serialized File:** `xgb_classifier.pkl`  
- **Input Format:** Numeric and categorical fields (encoded as integers)  
- **Output:** Probability distribution across 4 risk categories:
  - Low Risk (P1)
  - Medium Risk (P2)
  - High Risk (P3)
  - Near to Default (P4)

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit main app file
├── xgb_classifier.pkl      # Trained model file (required)
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 🧩 Column Mapping

The app includes a dictionary `COLUMN_NAMES` that maps technical model feature names to user-friendly labels.  
For example:

| Model Column | UI Display Name |
|---------------|-----------------|
| `NETMONTHLYINCOME` | Net Monthly Income |
| `Time_With_Curr_Empr` | Employment Tenure |
| `CC_Flag` | Credit Card Holder |
| `EDUCATION` | Education Level |
| `num_deliq_6_12mts` | Delinquencies in 6–12 Months |

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/UjjwalKaushik0609/credit-risk-app
   cd credit-risk-app
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model file**

   Place `xgb_classifier.pkl` in the project root directory.

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 🖥️ How It Works

1. Open the app in your browser (automatically opens at `http://localhost:8501`).  
2. Use the **sidebar** to enter customer details such as income, education, loan flags, etc.  
3. Click **“Predict Credit Approval”**.  
4. View:
   - A **bar chart** showing probability scores.
   - A **metric summary** with each risk level’s probability.
   - A **data table** of your inputs for reference.

---

## 📊 Example Output

- **Bar Chart:** Visualizes probabilities for each credit approval class.
- **Metrics:** Displays risk percentages (e.g., “Low Risk: 72.45%”).
- **Data Table:** Shows all entered customer data for transparency.

---

## 🧰 Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `pickle`
- `xgboost` *(for model loading)*

To install manually:

```bash
pip install streamlit pandas numpy xgboost
```

---

## 🧑‍💻 Author

**Developed by:** [Ujjwal Kaushik]  
**Role:** Data Scientist / Machine Learning Engineer  
**Contact:** [ujjwalkaushik0609@gmail.com]  
**LinkedIn:**[www.linkedin.com/in/ujjwal-kaushik-0b1489376]

---

## 💡 Future Enhancements

- Add data preprocessing and validation layers  
- Support for multiple model versions  
- Export results as PDF or CSV  
- Integrate model explainability using SHAP  

---

**Enjoy predicting smarter credit decisions!**
