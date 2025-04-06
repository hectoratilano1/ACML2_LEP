# 🏦 Loan Eligibility Predictor

This project uses a machine learning model to predict whether a loan application should be approved based on applicant information. It includes preprocessing, training a Logistic Regression model, and deploying a Streamlit web app.

## 📁 Project Structure

```
loan_eligibility_predictor/
├── app.py                  # Streamlit app
├── main.py                 # Run ML pipeline
├── notebooks/              # Original notebook
├── src/                    # Modular code
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── data/                   # Place credit.csv here
├── requirements.txt
└── README.md
```

## 🚀 How to Run

1. Clone the repo and navigate to the folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `credit.csv` in the `data/` folder.
4. Run the full pipeline:
   ```bash
   python main.py
   ```
5. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## ✅ Output

- Model training with accuracy report
- Streamlit app for interactive prediction