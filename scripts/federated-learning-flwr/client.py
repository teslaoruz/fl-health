import flwr as fl
import numpy as np
import sys
import logging
from logging.handlers import RotatingFileHandler
from model import get_model, preprocess_data, train_with_dp
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



silo_filename = sys.argv[1]
X_train, y_train, X_test, y_test = preprocess_data(silo_filename)

# --- Set up Logging ---
logger = logging.getLogger("flwr.client")
logger.setLevel(logging.INFO)
log_filename = f"client_logs/{silo_filename.split('/')[-1].replace('.csv','')}.log"
handler = RotatingFileHandler(log_filename, maxBytes=1000000, backupCount=3)
logger.addHandler(handler)


class SiloClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_model().get_weights()
        
    
    def fit(self, parameters, config):
        model = get_model()
        model.set_weights(parameters)
    
        # Store original weights for clipping
        old_weights = model.get_weights()
    
        # Train with Differential Privacy
        model, train_loss = train_with_dp(
            model,
            X_train,
            y_train,
            epochs=1,
            batch_size=32,
            noise_multiplier=1.0,
            l2_norm_clip=1.0
        )
    
        # Get updated weights after training
        new_weights = model.get_weights()
    
        # Clip weight updates
        clipped_weights = clip_weight_updates(old_weights, new_weights, clip_norm=1.0)
    
        # Evaluate using clipped weights
        model.set_weights(clipped_weights)
        loss_eval, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
        # Log training and evaluation info
        logger.info(
            f"[Client: {silo_filename}] Round {config['server_round']} - Train Loss: {train_loss:.4f} - Eval Loss: {loss_eval:.4f} - Accuracy: {accuracy:.4f}"
        )
        logger.info("[DP+Clipping] DP-SGD applied with noise_multiplier=1.0, l2_norm_clip=1.0; Weight clipping applied with norm=1.0")
    
        return clipped_weights, len(X_train), {
            "loss": float(train_loss),
            "accuracy": float(accuracy),
            "eval_loss": float(loss_eval)
        }

    
    def evaluate(self, parameters, config):
        model = get_model()
        model.set_weights(parameters)
    
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)  # Sensitivity
        f1 = f1_score(y_test, y_pred)
    
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
    
        logger.info(
            f"[Client: {silo_filename}] Round {config['server_round']} - "
            f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, "
            f"Recall: {recall:.4f}, Spec: {specificity:.4f}, F1: {f1:.4f}"
        )
    
        return loss, len(X_test), {
            "accuracy": accuracy,
            "loss": loss,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1
        }


def clip_weight_updates(old_weights, new_weights, clip_norm=1.0):
    """Clip the update (delta) between new and old weights."""
    clipped_weights = []
    for w_old, w_new in zip(old_weights, new_weights):
        delta = w_new - w_old
        norm = np.linalg.norm(delta)
        if norm > clip_norm:
            delta = delta * (clip_norm / norm)
        clipped_weights.append(w_old + delta)
    return clipped_weights



fl.client.start_client(
    server_address="[::1]:8080",
    client=SiloClient().to_client(),
)

