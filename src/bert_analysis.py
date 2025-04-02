from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from optimizers.adam_cpr_ws import AdamCPR_WS
from optimizers.adam_cpr_ip import AdamCPR_IP
import torch
import numpy as np

# Initialize model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load dataset
dataset = load_dataset("glue", "mnli")
train_dataset = dataset["train"].select(range(5000))  # Increased from 1000
eval_dataset = dataset["validation_matched"].select(range(1000))

# Custom collate function
def collate_fn(batch):
    inputs = tokenizer(
        [b["premise"] for b in batch],
        [b["hypothesis"] for b in batch],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs["labels"] = torch.tensor([b["label"] for b in batch])
    return inputs

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=16, collate_fn=collate_fn)

# Initialize optimizers with layer-wise learning rates
param_groups = [
    {"params": model.bert.parameters(), "lr": 3e-5},
    {"params": model.classifier.parameters(), "lr": 7e-5}
]

optimizers = {
    "AdamW": AdamW(param_groups, weight_decay=0.01),
    "AdamCPR_WS": AdamCPR_WS(
        param_groups,
        mu=0.1,
        kappa_type="w",
        s_step=200  # Earlier κ setting
    ),
    "AdamCPR_IP": AdamCPR_IP(
        param_groups,
        mu=0.05,
        kappa_type="w"
    )
}

def train(model, optimizer, train_loader, eval_loader, epochs=10):
    model.train()
    results = {
        'train_loss': [],
        'eval_loss': [],
        'lambda_history': [],
        'kappa_history': []
    }
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=len(train_loader)*epochs
    )
    
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_lambdas = []
        epoch_kappas = []
        
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            
            # Track λ and κ
            if hasattr(optimizer, 'lambdas'):
                epoch_lambdas.append(optimizer.lambdas.copy())
            if hasattr(optimizer, 'kappas'):
                epoch_kappas.append([x if x is not None else None for x in optimizer.kappas])
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}: Loss={loss.item():.4f}")
                if hasattr(optimizer, 'lambdas'):
                    print(f"λ: {[f'{x:.4f}' for x in optimizer.lambdas]}")
                if hasattr(optimizer, 'kappas'):
                    print(f"κ: {[f'{x:.4f}' if x is not None else 'None' for x in optimizer.kappas]}")
        
        # Evaluation
        model.eval()
        epoch_eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                outputs = model(**batch)
                epoch_eval_loss += outputs.loss.item()
        
        # Store results
        results['train_loss'].append(epoch_train_loss / len(train_loader))
        results['eval_loss'].append(epoch_eval_loss / len(eval_loader))
        if epoch_lambdas:
            results['lambda_history'].append(epoch_lambdas)
        if epoch_kappas:
            results['kappa_history'].append(epoch_kappas)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {results['train_loss'][-1]:.4f}")
        print(f"Eval Loss: {results['eval_loss'][-1]:.4f}")
        if epoch_lambdas:
            print(f"Avg λ: {[f'{np.mean([x[i] for x in epoch_lambdas]):.4f}' for i in range(len(epoch_lambdas[0]))]}")
    
    return results

# Run training
all_results = {}
for name, opt in optimizers.items():
    print(f"\n=== Training with {name} ===")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    all_results[name] = train(model, opt, train_loader, eval_loader, epochs=10)

# Plot results
plt.figure(figsize=(12, 8))
for name, res in all_results.items():
    plt.plot(res['train_loss'], label=f"{name} (Train)", linestyle='-')
    plt.plot(res['eval_loss'], label=f"{name} (Eval)", linestyle='--')
plt.title("Training Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("training_curves.png", dpi=300)
plt.show()

# Plot λ dynamics for CPR optimizers
for name in ["AdamCPR_WS", "AdamCPR_IP"]:
    if name in all_results and all_results[name]['lambda_history']:
        plt.figure()
        lambdas = np.array(all_results[name]['lambda_history'])  # (epochs, steps, groups)
        for g in range(lambdas.shape[2]):
            plt.plot(lambdas[:, :, g].mean(axis=1), label=f'Group {g}')
        plt.title(f"{name} λ Dynamics")
        plt.xlabel("Epoch")
        plt.ylabel("λ Value")
        plt.legend()
        plt.grid()
        plt.savefig(f"{name}_lambdas.png", dpi=300)
        plt.show()