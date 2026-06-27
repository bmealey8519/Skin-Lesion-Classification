from medmnist import DermaMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

# Download and load the 28x28 version
train_dataset = DermaMNIST(split="train", download=True)
val_dataset   = DermaMNIST(split="val",   download=True)
test_dataset  = DermaMNIST(split="test",  download=True)

train_images_list = [item[0] for item in train_dataset]
train_labels_list = [item[1][0] for item in train_dataset]

val_images_list = [item[0] for item in val_dataset]
val_labels_list = [item[1][0] for item in val_dataset]

test_images_list = [item[0] for item in test_dataset]
test_labels_list = [item[1][0] for item in test_dataset]

#convert lists to NumPy arrays
X_train = np.array([np.array(img) for img in train_images_list])

y_train = np.array(train_labels_list)

X_val = np.array([np.array(img) for img in val_images_list])

y_val = np.array(val_labels_list)

X_test = np.array([np.array(img) for img in test_images_list])

y_test = np.array(test_labels_list)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


#displaying first couple training images
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    img_array = X_train[i]
    label = y_train[i]

    axes[i].imshow(img_array)

    axes[i].set_title(f"Class Label: {label}")

    axes[i].axis("off")

plt.show()

#analyzing class distribution

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

classes, counts = np.unique(y_train, return_counts=True)

total_samples = len(y_train)

print("\n Training Set Class Distribution \n")
for cls, count in zip(classes, counts):
    percentage = (count / total_samples) * 100
    print(f"Class {cls} ({class_names[cls]}): {count} images ({percentage:.1f}%)")

classes, counts = np.unique(y_val, return_counts=True)

total_samples = len(y_val)

print("\n Value Set Class Distribution \n")
for cls, count in zip(classes, counts):
    percentage = (count / total_samples) * 100
    print(f"Class {cls} ({class_names[cls]}): {count} images ({percentage:.1f}%)")


classes, counts = np.unique(y_test, return_counts=True)

total_samples = len(y_test)

print("\n Test Set Class Distribution \n")
for cls, count in zip(classes, counts):
    percentage = (count / total_samples) * 100
    print(f"Class {cls} ({class_names[cls]}): {count} images ({percentage:.1f}%)")

#normalizing pixel vals

print(f"Data type before: {X_train.dtype}")
print(f"Max pixel value before: {X_train.max()}")

X_train_norm = X_train / 255.0
X_val_norm = X_val / 255.0
X_test_norm = X_test / 255.0

print(f"\nData type after: {X_train_norm.dtype}")
print(f"Max pixel value after: {X_train_norm.max()}")

#flattening into vectors

print("\nFlattening Data\n")

X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)
X_val_flat = X_val_norm.reshape(X_val_norm.shape[0], -1)
X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], -1)

print(f"Flattened X_train shape: {X_train_flat.shape}")
print(f"Flattened X_val shape: {X_val_flat.shape}")
print(f"Flattened X_test shape: {X_test_flat.shape}")

#creating logistic regression model
print("\nCreating Logistic Regression Model\n")

model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)

start_time = time.time()

model.fit(X_train_flat, y_train)

end_time = time.time()
print(f"Training complete in {end_time - start_time:.1f} seconds.")

print("\nTesting on Validation Dataset\n")
y_val_predictions = model.predict(X_val_flat)

#calc metrics
val_accuracy = accuracy_score(y_val, y_val_predictions)
print(f"Overall Validation Accuracy: {val_accuracy * 100:.2f}%\n")

#detailed classification report
print("Detailed Classification Report:")
print(classification_report(y_val, y_val_predictions, target_names=class_names))

#Rerunning model with tuned hyperparemeter C
print("\nCreating Logistic Regression Model Again With Tuned C\n")

model_tuned = LogisticRegression(C=0.01, max_iter=5000, class_weight='balanced', random_state=42)

start_time = time.time()

model_tuned.fit(X_train_flat, y_train)

end_time = time.time()
print(f"Training complete in {end_time - start_time:.1f} seconds.")

print("\nTesting on Validation Dataset\n")
y_val_predictions_2 = model_tuned.predict(X_val_flat)

#calc metrics
val_accuracy_2 = accuracy_score(y_val, y_val_predictions_2)
print(f"Overall Validation Accuracy: {val_accuracy_2 * 100:.2f}%\n")

#detailed classification report
print("Detailed Classification Report:")
print(classification_report(y_val, y_val_predictions_2, target_names=class_names))

print("\nComparison on 'Melanoma' Recall\n")
report_1 = classification_report(y_val, y_val_predictions, target_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], output_dict=True)
report_2 = classification_report(y_val, y_val_predictions_2, target_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], output_dict=True)

print(f"Model 1 (C=1.0)  Melanoma Recall: {report_1['mel']['recall']:.2f}")
print(f"Model 2 (C=0.01) Melanoma Recall: {report_2['mel']['recall']:.2f}")

#Confusion Matrices
print("\nGenerating Confusion Matrices\n")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#plot model 1 (c=1)
cm_1 = confusion_matrix(y_val, y_val_predictions)
disp_1 = ConfusionMatrixDisplay(confusion_matrix=cm_1, display_labels=class_names)
disp_1.plot(ax=axes[0], cmap='Blues', colorbar=False, xticks_rotation=45)
axes[0].set_title('Model 1 (C=1.0) - Complex Boundaries', fontsize=14)

#plot model 2
cm_2 = confusion_matrix(y_val, y_val_predictions_2)
disp_2 = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=class_names)
disp_2.plot(ax=axes[1], cmap='Blues', colorbar=False, xticks_rotation=45)
axes[1].set_title('Model 2 (C=0.01) - Simple Boundaries', fontsize = 14)

plt.tight_layout()
plt.show()

#MLP (neural network)

class SkinNN(nn.Module):
    def __init__(self, input_size=2352, hidden_size=128, num_classes=7):
        super(SkinNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        z1 = self.fc1(x)

        #activation function using ReLu
        a1 = F.relu(z1)

        z2 = self.fc2(a1)

        return z2
    
model = SkinNN()
print(model)

#converting data to pytorch tensors

print("\nConverting Data to PyTorch Tensors\n")

X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_flat, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_flat, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print(f"X_train_tensor shape: {X_train_tensor.shape}, type: {X_train_tensor.dtype}")
print(f"y_train_tensor shape: {y_train_tensor.shape}, type: {y_train_tensor.dtype}")

#initializing loss and optimizer funcs

print("\nInitializing Loss and Optimizer Functions\n")

#using scikit-learn to calculate the exact balanced weights for the dataset
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

#convert weights into pytorch tensor
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

#inject weights into the loss function
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print("Setup complete, ready for training.")

#batching data

batch_size = 64

train_dataset_pt = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset_pt = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset_pt, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_pt, batch_size=batch_size, shuffle=False)

#training loop

epochs = 50

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    #training phase
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        #forward pass
        outputs = model(inputs)

        #calc loss
        loss = criterion(outputs, labels)

        #backward pass
        loss.backward()

        #optimizer step, update the weights
        optimizer.step()

        #track training metrics
        running_train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    #calc avg training loss and accuracy for this epoch
    epoch_train_loss = running_train_loss / total_train
    epoch_train_acc = correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    #validation pahse
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0 

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    
    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    #print an update every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.1f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.1f}%")
        
print("Training Complete")

print("\nPlotting Performance Curves\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#plot loss
ax1.plot(train_losses, label='Training Loss', color='blue')
ax1.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
ax1.set_title('Loss Over Time (Lower is Better)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cross Entropy Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

#plot accuracy
ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
ax2.plot(val_accuracies, label='Validation Accuracy', color='red', linestyle='--')
ax2.set_title('Accuracy Over Time (Higher is Better)')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.show()


#final testing of both models on test data
print("\nFinal Evaluation on Test Data\n")

#logistic regression test first
print("\nLogistc Regression Testing (Tuned C to 0.01)\n")

y_test_pred_lr = model_tuned.predict(X_test_flat)

#calc and print metrics
lr_test_acc = accuracy_score(y_test, y_test_pred_lr)
print(f"Logistic Regression Test Accuracy: {lr_test_acc * 100:.2f}%\n")
print(classification_report(y_test, y_test_pred_lr, target_names=class_names))

#NN testing
print("\nNeural Network Testing\n")

test_dataset_pt = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset_pt, batch_size=batch_size, shuffle=False)

model.eval()
correct_test = 0
total_test = 0
y_test_pred_nn = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        y_test_pred_nn.extend(predicted.cpu().numpy())

#calc and print metrics
nn_test_acc = correct_test / total_test
print(f"Neural Network Test Accuracy: {nn_test_acc * 100:.2f}%\n")
print(classification_report(y_test, y_test_pred_nn, target_names=class_names))




