from sklearn.metrics import precision_score, recall_score, f1_score
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from transformers import DeiTForImageClassificationWithTeacher, DeiTImageProcessor


### SETUP

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

class myDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    self.data.append((os.path.join(label_dir, file), int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)
        image = transform(image)
        image_np = np.array(image)
        return image, label


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.0001 # TODO Choose learning rate
batch_size = 32 # TODO Choose batch size
num_training_steps = 32 # TODO Choose number of training steps

# TODO Change to correct path, choose model
model_path = "/content/drive/MyDrive/datasets/PyTorchdeit-base-distilled-patch16-384/" 
model = DeiTForImageClassificationWithTeacher.from_pretrained(model_path)
model.to(device)

# TODO Change to correct paths
train_dataset = myDataset(root_dir="/content/drive/MyDrive/datasets/adjusteddatasets/KneeOsteoarthritisXray/train")
test_dataset = myDataset(root_dir="/content/drive/MyDrive/datasets/adjusteddatasets/KneeOsteoarthritisXray/test")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothingCrossEntropy(smoothing=0.1) # TODO Choose label smoothing

# TODO Choose Adam or SGD optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# TODO Choose dropout
model.config.hidden_dropout_prob = 0.2
model.config.attention_probs_dropout_prob = 0.2


### TRAINING

for training_step in range(num_training_steps):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)['logits']
        loss = criterion(scores, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    print(f'Cost at training step {training_step + 1} is {(sum(losses)/len(losses))}')


### TESTING

def check_accuracy(loader, model):
    print("Checking accuracy")
    num_correct = 0
    num_samples = 0
    all_labels = []
    all_preds = []

    model.eval()

    with torch.no_grad():
      for x, y in loader:
          x = x.to(device=device)
          y = y.to(device=device)

          scores = model(x)['logits']
          _, predictions = scores.max(1)
          print(f'prediction: {predictions}')
          print(f'actual: {y}')

          all_labels.extend(y.cpu().numpy())
          all_preds.extend(predictions.cpu().numpy())

          num_correct += (predictions == y).sum()
          num_samples += predictions.size(0)

    print(f'{num_correct} / {num_samples} correct')
    print(f'Accuracy: {float(num_correct)/float(num_samples)*100:.2f}')

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    model.train()

print('test set accuracy')
check_accuracy(test_loader, model)
