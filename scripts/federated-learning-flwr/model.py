import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2

import warnings
warnings.filterwarnings("ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer")

import json
import os
# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
# Disable GPU if you don't need it
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_model(input_dim=None):
    if input_dim is None:
        input_dim = detect_input_shape()

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=RMSprop(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def detect_input_shape(csv_path="../../datasets/diabetes/processed_silos/hospital_1.csv"):
    df = pd.read_csv(csv_path)
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 0 if x == 0 else 1)
    df = df.drop(columns=['readmitted', 'discharge_disposition_id'])
    X = df.drop(columns=['readmitted_binary'])
    return X.shape[1]


def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 0 if x == 0 else 1)
    df = df.drop(columns=['readmitted', 'discharge_disposition_id'])

    X = df.drop(columns=['readmitted_binary'])
    y = df['readmitted_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train.values, X_test_scaled, y_test.values


def get_evaluate_fn():
    X_train, y_train, X_test, y_test = preprocess_data("../../datasets/diabetes/central_eval.csv")
    os.makedirs("server_logs", exist_ok=True)
    
    def evaluate(server_round, parameters_ndarrays, config):
        model = get_model(input_dim=X_test.shape[1])
        model.set_weights(parameters_ndarrays)
        
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        return loss, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1
        }

    return evaluate


def train_with_dp(model, X_train, y_train, epochs=1, batch_size=32, noise_multiplier=1.0, l2_norm_clip=1.0):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    final_loss = 0.0

    for epoch in range(epochs):
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)
                final_loss = loss.numpy()

            grads = tape.gradient(loss, model.trainable_variables)

            clipped_grads = []
            for g in grads:
                norm = tf.norm(g)
                clipped = g * tf.minimum(1.0, l2_norm_clip / (norm + 1e-6))
                clipped_grads.append(clipped)

            noisy_grads = [
                g + tf.random.normal(tf.shape(g), stddev=noise_multiplier * l2_norm_clip)
                for g in clipped_grads
            ]

            optimizer.apply_gradients(zip(noisy_grads, model.trainable_variables))

    return model, final_loss

