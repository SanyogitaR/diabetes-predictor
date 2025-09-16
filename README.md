
# 🧠 Diabetes Risk Predictor

A simple Flask-based web application that predicts the risk of diabetes using a neural network trained on the **Pima Indians Diabetes Dataset** (Kaggle/UCI).  

The user enters basic health parameters (like glucose, BMI, age, etc.), and the trained model predicts whether the person is at high or low risk of diabetes.  

---

## 📂 Project Structure
Diabetes-Predictor/
│
├── app.py # Flask backend application
├── model.ipynb # Jupyter notebook for training the model
├── diabetes_model.h5 # Saved trained Keras model
├── scaler.pkl # Saved Scikit-learn scaler
├── requirements.txt # Dependencies file
├── README.md # Project documentation
│
├── templates/ # HTML templates
│ └── index.html # Main web page with input form
│ └── bg.jpg # Background image
│
└── static/ # (Optional) CSS/JS files if added later



---

## 📊 Dataset

The model is trained on the **Pima Indians Diabetes dataset** from Kaggle/UCI.  
It includes health parameters such as:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function (DPF)  
- Age  

**Target variable:**  
- `Outcome` → `0 = No Diabetes`, `1 = Diabetes`

---

## 💻 Workflow

1. Train the neural network using the dataset (`model.ipynb`).  
2. Save the trained **Keras model (`diabetes_model.h5`)** and **Scaler (`scaler.pkl`)**.  
3. Load them in `app.py` to handle predictions.  
4. Serve the web interface via Flask (`index.html` inside `templates/`).  
5. User enters health values → model predicts risk → result displayed on page.  

---

## 🎨 Frontend Design

- **Background image** (`bg.jpg`) placed in `templates/` for easy access.  
- Card-style input form with grid layout.  
- Prediction result box:  
  - ✅ Green = Low Risk  
  - ❌ Red = High Risk  
- Responsive, simple, and user-friendly design.  

---

## 📌 Future Improvements

- Deploy app to **Heroku / AWS / Render**.  
- Improve UI with charts & visualizations.  
- Collect and evaluate with **real-world data**.  
- Add support for **multiple models** (Random Forest, XGBoost, etc.).  

---

## 🙌 Credits

- **Dataset:** Pima Indians Diabetes Dataset – Kaggle/UCI  
- **Frameworks:** Flask, TensorFlow/Keras, Scikit-learn  
- **Frontend:** HTML, CSS  

---
