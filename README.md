🏥 Health Risk Prediction System 🏥

A multi-task neural network that predicts risk levels (Low, Medium, High) for:

Diabetes
Heart Disease
Obesity

Includes a trained model and a Streamlit dashboard for real-time predictions.
--------------------------------------------------------------------------------------------------------------------
🚀 Features:
Predicts 3 diseases simultaneously
End-to-end pipeline (preprocessing → training → UI)
High accuracy:
Diabetes: 90%
Heart: 91%
Obesity: 95%
--------------------------------------------------------------------------------------------------------------------
📊 Dataset:
12 input features (age, BMI, BP, lifestyle, family history, etc.)
3 output labels (risk levels for each disease)
--------------------------------------------------------------------------------------------------------------------
⚙️ Preprocessing:
Continuous → Standard Scaling
Categorical → One-hot encoding
Binary → 0/1 encoding
Final input size: 22 features
--------------------------------------------------------------------------------------------------------------------
🧠 Model:
Dense layers: 256 → 128 → 64 → 32 (ReLU + BatchNorm + Dropout)
Multi-output (3 branches with Softmax)
Captures shared patterns, then specializes per disease
--------------------------------------------------------------------------------------------------------------------
🏋️ Training:
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Epochs: 100, Batch size: 32
Early Stopping + Model Checkpoint
--------------------------------------------------------------------------------------------------------------------
💻 Interface:
Built with Streamlit
User inputs → Preprocessing → Model → Risk prediction
--------------------------------------------------------------------------------------------------------------------
🛠️ Tech Stack:
Python, TensorFlow/Keras, Pandas, Scikit-learn, Streamlit
--------------------------------------------------------------------------------------------------------------------
▶️ Deployment:
Deployed project can be found on the following link:
https://healthpredictor-bbe28cdbahgtzztx4ywvqm.streamlit.app/
--------------------------------------------------------------------------------------------------------------------
📌 Future Work
Real-world datasets
More diseases
Model explainability
