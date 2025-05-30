import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import project modules
import data_setup
import models
import engine
import utils

# --- CONFIGURATION ---
DATA_FILEPATH = 'training.1600000.processed.noemoticon.csv'
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 2 # Assuming binary classification (e.g., positive/negative)
N_LAYERS = 2
DROPOUT = 0.5
N_EPOCHS = 5
BATCH_SIZE = 128
MAX_LEN = 50

# --- SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = data_setup.load_and_preprocess_data(DATA_FILEPATH)
train_loader, val_loader, test_loader, vocab, label_encoder = data_setup.create_dataloaders(df, BATCH_SIZE, MAX_LEN)

# --- BASELINE LSTM MODEL ---
print("\n--- Training Baseline LSTM Model ---")
INPUT_DIM = len(vocab)
lstm_model = models.BaselineLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device)
optimizer_lstm = optim.Adam(lstm_model.parameters())
criterion = nn.CrossEntropyLoss().to(device)

history_lstm = engine.train_model(lstm_model, train_loader, val_loader, optimizer_lstm, criterion, N_EPOCHS, device)

# --- IMPROVED GRU MODEL ---
print("\n--- Training Improved GRU Model ---")
gru_model = models.ImprovedGRU(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device)
optimizer_gru = optim.Adam(gru_model.parameters())

history_gru = engine.train_model(gru_model, train_loader, val_loader, optimizer_gru, criterion, N_EPOCHS, device)

# --- EVALUATION ---
print("\n--- Evaluating Models on Test Set ---")
_, _, true_labels_lstm, pred_labels_lstm = engine.evaluate(lstm_model, test_loader, criterion, device)
_, _, true_labels_gru, pred_labels_gru = engine.evaluate(gru_model, test_loader, criterion, device)

# --- VISUALIZATIONS ---
class_names = label_encoder.classes_
utils.plot_training_curves(history_lstm, "Baseline LSTM Training")
utils.plot_confusion_matrix(true_labels_lstm, pred_labels_lstm, class_names, "LSTM Confusion Matrix")

utils.plot_training_curves(history_gru, "Improved GRU Training")
utils.plot_confusion_matrix(true_labels_gru, pred_labels_gru, class_names, "GRU Confusion Matrix")