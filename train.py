import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ArithmeticTransformer import ArithmeticTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import math
from collections import defaultdict


TRAIN_FILE = '/kaggle/working/data/train.csv'
VAL_FILE = '/kaggle/working/data/val.csv'
MODEL_DIR = '/kaggle/working/models'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 0.00001
D_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
D_FF = 1024
DROPOUT = 0.0
MAX_LENGTH = 20
SAVE_EVERY_N_EPOCHS = 5


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(float)
        self.count = 0
    
    def update(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.metrics[k] += v
        self.count += 1
    
    def get_metrics(self):
        return {k: v/self.count for k, v in self.metrics.items()}

def compute_metrics(predictions, targets, tokenizer):
    """Compute various evaluation metrics efficiently."""
    metrics = {}
    
    # Convert to numpy arrays for faster computation
    pred_array = np.array(predictions)
    target_array = np.array(targets)
    
    # Exact match accuracy (vectorized)
    metrics['exact_match'] = np.mean(pred_array == target_array)
    
    # Character-level accuracy (vectorized)
    total_chars = 0
    correct_chars = 0
    
    # Process all sequences at once
    for pred, target in zip(predictions, targets):
        min_len = min(len(pred), len(target))
        if min_len > 0:
            # Compare characters up to min_len
            matches = sum(1 for i in range(min_len) if pred[i] == target[i])
            correct_chars += matches
            total_chars += min_len
    
    metrics['char_accuracy'] = correct_chars / total_chars if total_chars > 0 else 0
    
    return metrics

def compute_perplexity(loss):
    """Compute perplexity from loss."""
    return math.exp(loss)

class ArithmeticDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=20):
        # Force everything to be read as strings, so we don't get ints
        self.data = pd.read_csv(csv_file, dtype=str)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Build the input expression: "num1 op num2", e.g. "123+45"
        input_str  = row['input']
        target_str = row['target']   # already a string because of dtype=str
        
        # Now encode both
        input_ids  = self.tokenizer.encode(
            input_str,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        target_ids = self.tokenizer.encode(
            target_str,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids':  torch.tensor(input_ids,  dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

class SimpleTokenizer:
    def __init__(self):
        # Create vocabulary for digits, operators, and special tokens
        self.vocab = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '+': 3,
            '-': 4
        }
        # Add digits 0-9
        for i in range(10):
            self.vocab[str(i)] = i + 5
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text, max_length=None, padding='max_length', truncation=True):
        # Convert text to token ids
        tokens = [self.vocab['<sos>']]
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
        tokens.append(self.vocab['<eos>'])
        
        # Truncate if needed
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.vocab['<eos>']]
            
        # Pad if needed
        if padding == 'max_length' and max_length:
            padding_length = max_length - len(tokens)
            if padding_length > 0:
                tokens.extend([self.vocab['<pad>']] * padding_length)
                
        return tokens
    
    def decode(self, ids):
        return ''.join([self.reverse_vocab[id] for id in ids if id not in [self.vocab['<pad>'], self.vocab['<sos>'], self.vocab['<eos>']]])

def evaluate(model, val_loader, criterion, tokenizer, device):
    """Evaluate the model on validation data."""
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            output = model(input_ids, target_ids[:, :-1])
            
            # Calculate loss
            loss = criterion(output.reshape(-1, len(tokenizer.vocab)), target_ids[:, 1:].reshape(-1))
            
            # Generate predictions for the entire batch at once
            pred_ids = model.generate(
                input_ids,
                max_length=MAX_LENGTH,
                start_token_id=tokenizer.vocab['<sos>'],
                end_token_id=tokenizer.vocab['<eos>']
            )
            
            # Convert predictions and targets to strings efficiently
            predictions = []
            targets = []
            for pred_seq, target_seq in zip(pred_ids, target_ids):
                # Convert to numpy once per sequence
                pred_np = pred_seq.cpu().numpy()
                target_np = target_seq.cpu().numpy()
                
                # Decode sequences
                pred_str = tokenizer.decode(pred_np)
                target_str = tokenizer.decode(target_np)
                predictions.append(pred_str)
                targets.append(target_str)
            
            # Compute metrics
            batch_metrics = compute_metrics(predictions, targets, tokenizer)
            batch_metrics['loss'] = loss.item()
            batch_metrics['perplexity'] = compute_perplexity(loss.item())
            metrics_tracker.update(batch_metrics)
    
    return metrics_tracker.get_metrics()

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    vocab_size = len(tokenizer.vocab)
    
    # Create datasets and dataloaders
    print(f"Loading training data from {TRAIN_FILE}...")
    train_dataset = ArithmeticDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    print(f"Loading validation data from {VAL_FILE}...")
    val_dataset = ArithmeticDataset(VAL_FILE, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    print("Initializing model...")
    model = ArithmeticTransformer(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_seq_length=MAX_LENGTH,
        pad_token_id=tokenizer.vocab['<pad>']
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<pad>'])
    
    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Try to load latest checkpoint
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_')])
    if checkpoint_files:
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
        print(f"Loading checkpoint from {latest_checkpoint}")
        start_epoch, _ = load_checkpoint(model, optimizer, latest_checkpoint)
        start_epoch += 1
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_metrics = MetricsTracker()
        
        # Training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            output = model(input_ids, target_ids[:, :-1])
            
            # Calculate loss
            loss = criterion(output.reshape(-1, vocab_size), target_ids[:, 1:].reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_metrics = {'loss': loss.item(), 'perplexity': compute_perplexity(loss.item())}
            train_metrics.update(batch_metrics)
            train_pbar.set_postfix({'loss': train_metrics.get_metrics()['loss']})
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, tokenizer, device)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{EPOCHS}:')
        print('Training metrics:')
        for k, v in train_metrics.get_metrics().items():
            print(f'  {k}: {v:.4f}')
        print('Validation metrics:')
        for k, v in val_metrics.items():
            print(f'  {k}: {v:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_metrics, best_model_path)
            print(f'Saved best model to {best_model_path}')
        
        # Example prediction
        model.eval()
        with torch.no_grad():
            example_input = "123+456"
            input_ids = torch.tensor(tokenizer.encode(example_input)).unsqueeze(0).to(device)
            output_ids = model.generate(
                input_ids,
                max_length=MAX_LENGTH,
                start_token_id=tokenizer.vocab['<sos>'],
                end_token_id=tokenizer.vocab['<eos>']
            )
            predicted = tokenizer.decode(output_ids[0].cpu().numpy())
            print(f'  Example: {example_input} -> {predicted}')

if __name__ == '__main__':
    # Check if data files exist
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: Training file {TRAIN_FILE} not found!")
        print("Please run generate_samples.py first to create the data files.")
        exit(1)
    
    if not os.path.exists(VAL_FILE):
        print(f"Error: Validation file {VAL_FILE} not found!")
        print("Please run generate_samples.py first to create the data files.")
        exit(1)
    
    print("Starting training...")
    train()
    print("Training completed!") 