import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from keras.utils import plot_model

from predict import load_manual_model 

SRC_PATH = Path(__file__).resolve().parent 
ROOT_PATH = SRC_PATH.parent
PRO_DATA_PATH = ROOT_PATH / "DATA" / "processed"
HIST_DATA_PATH = PRO_DATA_PATH / "Model_History.pkl"
PLOT_PATH = ROOT_PATH / "PLOTS"

model = load_manual_model() 
plot_model(
    model, 
    to_file=str(PLOT_PATH / 'model_architecture.png'),
    show_shapes=True, 
    show_layer_names=True,
    dpi=96 
)

with open(HIST_DATA_PATH, "rb") as f:
    history = pickle.load(f)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss Plot
ax1.plot(history['loss'], label='Training Loss', color='blue')
ax1.plot(history['val_loss'], label='Validation Loss', color='orange')
ax1.set_title('Model Loss Over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Accuracy Plot
avg_train_acc = np.mean([history['diabetes_accuracy'], history['heart_risk_accuracy'], history['obesity_accuracy']], axis=0)
avg_val_acc = np.mean([history['val_diabetes_accuracy'], history['val_heart_risk_accuracy'], history['val_obesity_accuracy']], axis=0)

ax2.plot(avg_train_acc, label='Avg Training Accuracy', color='green')
ax2.plot(avg_val_acc, label='Avg Validation Accuracy', color='red')
ax2.set_title('Average Model Accuracy Over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

fig1.tight_layout()
fig1.savefig(PLOT_PATH / "learning_curves.png", dpi=300, bbox_inches='tight')
plt.close(fig1) # Frees up memory

#EDA Distributions & Heatmap
with open(PRO_DATA_PATH / "train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
    df = train_data["X"] 

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

plt.close(fig2)

fig3, ax5 = plt.subplots(figsize=(14, 10))
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix,mask=mask, annot=True, cmap='coolwarm',annot_kws={"size": 9}, fmt=".2f", linewidths=0.5, ax=ax5)
ax5.set_title('Feature Correlation Heatmap')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

fig3.tight_layout()
fig3.savefig(PLOT_PATH / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close(fig3)


    
    