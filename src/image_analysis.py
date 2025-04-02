import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from optimizers.adam_cpr_ws import AdamCPR_WS
from optimizers.adam_cpr_ip import AdamCPR_IP
import torchvision.models as models
from torch import nn

def main():
    # Load CIFAR-10 dataset
    # Use smaller input size (CIFAR-10 is 32x32)
    transform = transforms.Compose([
        transforms.Resize(32),  # Original CIFAR-10 size (no need for 224)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=256,  # Increased from 128 (better for M2's unified memory)
        shuffle=True,
        num_workers=4,    # Use 4-8 for M2 (avoid overloading CPU)
        pin_memory=True,  # Faster data transfer to GPU (if using MPS)
        persistent_workers=True  # Maintains workers between epochs
    )

    # Load ResNet18 without pretrained weights (overkill for CIFAR-10)
    model = models.resnet18(weights=None, num_classes=10)

    # Modify first layer for 32x32 inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Original: kernel=7, stride=2
    model.maxpool = nn.Identity()  # Remove first maxpool (redundant for small images)

    # 3. Optimizer Configuration
    optimizers = {
        "AdamW": torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01),
        "AdamCPR_WS": AdamCPR_WS(
            model.parameters(),
            lr=1e-3,
            mu=0.1,
            kappa_type="w",
            s_step=500
        ),
        "AdamCPR_IP": AdamCPR_IP(
            model.parameters(),
            lr=1e-3,
            mu=0.05,
            kappa_type="w"
        )
    }

    # 4. Training Loop
    def train(model, optimizer, trainloader, epochs=5):
        model.train()
        results = {
            'loss': [],
            'lambda': [],
            'kappa': []
        }
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1} / {epochs} ===")
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                print(f"Processing batch {i+1}")
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                # Record statistics
                running_loss += loss.item()
                
                # Print every 100 batches
                if i % 100 == 99:
                    print(f'Epoch {epoch+1}, Batch {i+1}: Loss={running_loss/100:.3f}')
                    if hasattr(optimizer, 'lambdas'):
                        print(f"λ: {optimizer.lambdas}")
                    if hasattr(optimizer, 'kappas'):
                        print(f"κ: {optimizer.kappas}")
                    running_loss = 0.0
            
            # Store epoch results
            epoch_loss = running_loss / len(trainloader)
            results['loss'].append(epoch_loss)
            
            if hasattr(optimizer, 'lambdas'):
                results['lambda'].append(optimizer.lambdas.copy())
            if hasattr(optimizer, 'kappas'):
                results['kappa'].append(optimizer.kappas.copy())
            
            print(f'Epoch {epoch+1} Finished. Avg Loss: {epoch_loss:.3f}')
        
        return results

    # 5. Run Experiments
    all_results = {}
    for name, opt in optimizers.items():
        print(f"\n=== Training with {name} ===")
        all_results[name] = train(model, opt, trainloader, epochs=5)

    # 6. Visualization
    plt.figure(figsize=(12, 5))
    for name, res in all_results.items():
        plt.plot(res['loss'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('cifar100_training.png')
    plt.show()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()