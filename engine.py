import torch
from sklearn.metrics import accuracy_score, classification_report

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Performs a single training epoch."""
    model.train()
    epoch_loss = 0
    for text, labels in dataloader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for text, labels in dataloader:
            text, labels = text.to(device), labels.to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            
            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(dataloader), accuracy, all_labels, all_preds

def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, device):
    """Main training loop that returns training history."""
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.2f}%')
    return history