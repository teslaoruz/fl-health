# Step 8: Baseline Centralized MLP Model Training

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

### after cleanup
# Load the cleaned centralized dataset
df = pd.read_csv('../datasets/diabetes/cleaned_data.csv')

# Correct binary label: 0 = NO, 1 = <30 or >30
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 0 if x == 0 else 1)

# Drop target and leaky column(s)
df = df.drop(columns=['readmitted', 'discharge_disposition_id'])

# Prepare features and target
X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train_scaled, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=2,)

# Evaluate
y_pred_probs = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_probs > 0.7).astype(int) 

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)