import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import dataHandling as dh

def MLP():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50  # Early stopping will stop before this
    patience = 5  # Early stopping patience
    num_classes = 10

    # Data preprocessing and normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Load CIFAR-10 dataset
    train_loader, test_loader = dh.data_loader(batch_size)

    # Define the MLP model
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, hidden_size),  # Hidden layer
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, num_classes)  # Output layer (raw scores, no activation)
            )

        def forward(self, x):
            return self.model(x)

    # Model initialization
    input_size = 32 * 32 * 3  # Flattened image size (32x32 RGB)
    hidden_size = 512  # Only one hidden layer
    model = MLP(input_size, hidden_size, num_classes)

    # Loss function (Hinge Loss)
    def multi_class_hinge_loss(outputs, labels):
        # Convert labels to one-hot encoding (Recommended by ChatGPT)
        one_hot_labels = torch.zeros_like(outputs).scatter_(1, labels.unsqueeze(1), 1)
        
        # Calculate hinge loss
        margin = 1.0
        correct_class_scores = (outputs * one_hot_labels).sum(dim=1, keepdim=True)  # Scores of the true class
        margins = torch.clamp(outputs - correct_class_scores + margin, min=0)  # Margin constraint
        margins = margins * (1 - one_hot_labels)  # Ignore the correct class in the margin calculation
        return margins.sum(dim=1).mean()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Early stopping variables
    best_loss = float('inf')
    early_stopping_counter = 0

    print("Training the multi-layered Perceptron network")
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = multi_class_hinge_loss(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        train_accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    #   --- EARLY STOPPING CODE - RECOMMENDED BY CHAT GPT ---

        # Validation step for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += multi_class_hinge_loss(outputs, labels).item()

        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered!")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # --- END OF CHATGPT GENERATED BLOCK ---

    # Evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
