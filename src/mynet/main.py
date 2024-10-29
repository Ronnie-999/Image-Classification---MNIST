# Resources that were used for documentation include pytorch.org, matplotlib.org and LLMs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from mynet.dataset import CustomMNISTDataset  
import matplotlib.pyplot as plt
from mynet.model import ThreeLayerFullyConnectedNetwork
from torchvision import transforms

def train_one_epoch(model, data_loader, criterion, optimizer, device):
  
  model.train()
  total_loss = 0

  for data, labels in data_loader:
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    predictions = model(data)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * data.size(0)

  average_loss = total_loss / len(data_loader.dataset)
  return average_loss
  

def evaluate_one_epoch(model, data_loader, criterion, device):

    model.eval()  
    total_loss_accumulated = 0.0
    correct_predictions_count = 0
    total_predictions_count = 0

    with torch.no_grad():
        for input_custom, target_custom in data_loader:
            input_custom, target_custom = input_custom.to(device), target_custom.to(device)
            
            outputs_custom = model(input_custom)  
            loss_custom = criterion(outputs_custom, target_custom)  
            batch_size_custom = input_custom.size(0)
            total_loss_accumulated += loss_custom.item() * batch_size_custom  

            predicted_custom = torch.argmax(outputs_custom, dim=1)
            correct_predictions_count += (predicted_custom == target_custom).sum().item()
            total_predictions_count += batch_size_custom

    accuracy = correct_predictions_count / total_predictions_count
    average_loss = total_loss_accumulated / len(data_loader.dataset)
    return average_loss, accuracy
  


def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None):

    training_losses = []
    evaluation_losses = []
    evaluation_accuracies = []

    for epoch_count in range(num_epochs):

        current_training_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        training_losses.append(current_training_loss)

        current_evaluation_loss, current_evaluation_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)
        evaluation_losses.append(current_evaluation_loss)
        evaluation_accuracies.append(current_evaluation_accuracy)

        if scheduler is not None:
            scheduler.step()

        print("Epoch {}/{}: Loss on Train: {:.4f}, Loss on Test: {:.4f}, Accuracy on Test: {:.4f}".format(epoch_count + 1, num_epochs, current_training_loss, current_evaluation_loss, current_evaluation_accuracy))

    return training_losses, evaluation_losses, evaluation_accuracies


def main():

    BATCH_SIZE = 50
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(CustomMNISTDataset(root="../../data/MNIST", subset="train", transformation=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(CustomMNISTDataset(root="../../data/MNIST", subset="test", transformation=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=False)

    model = ThreeLayerFullyConnectedNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    train_losses, test_losses, test_accuracies = train_and_evaluate_model(
        model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, scheduler
    )

    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()

    model.eval()
    with torch.no_grad():
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(10):
            print(f"Inference results for digit {i}:")
            found_image = False
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                for j in range(len(predicted)):
                    if predicted[j].item() == i:
                        ax = axs[i // 5, i % 5]
                        ax.imshow(inputs.cpu().numpy()[j][0], cmap='gray')
                        ax.set_title(f'Inference Result: {predicted[j].item()}, True Label: {targets[j].item()}')
                        ax.axis('off')
                        found_image = True
                        break
                if found_image:
                    break

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
