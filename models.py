import io
import torch
import torchvision
import numpy as np
import torch.onnx
import onnx
from torch import nn 
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
Defining several common model for testing. 
SimpleNet: MLP with 2 hidden layers.  
ConvNet: Convolutional NN with 2 convolutional layers and one FC layer. 
'''
class SimpleNet(nn.Module):
    '''Simple MLP'''
    def __init__(self, n_hidden = 100, batch_size = 200) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.fc1 = nn.Linear(in_features=784, out_features=n_hidden)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.fc3 = nn.Linear(in_features=n_hidden, out_features=10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    '''Simple Convolutional NN'''
    def __init__(self, dropout_rate=0.5):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding='same')
        img_size = 28
        self.fc1 = torch.nn.Linear(16 * img_size * img_size, 10)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = nn.functional.dropout(x, training=self.training, p=self.dropout_rate)
        img_size = 28
        x = x.view(-1, 16 * img_size * img_size)
        x = nn.functional.relu(self.fc1(x))
        return nn.functional.log_softmax(x, dim=1)

'''
The dataset that we will be using across different frameworks. 
we are using the simplest MNIST here. 
'''
class MNIST:
    def __init__(self, batch_size, splits=None, shuffle=True):
        """
        Args:
          batch_size : number of samples per batch
          splits : [train_frac, valid_frac]
          shuffle : (bool)
        """
        self.transform = torchvision.transforms.ToTensor()  
        self.batch_size = batch_size
        self.eval_batch_size = 200
        self.splits = splits
        self.shuffle = shuffle

        self._build()
      
    def _build(self):
        train_split, valid_split = self.splits
        trainset = torchvision.datasets.MNIST(
                root="data", train=True, download=True, transform=self.transform)
        num_samples = len(trainset)
        self.num_train_samples = int(train_split * num_samples)
        self.num_valid_samples = int(valid_split * num_samples)

        # create training set 
        self.train_dataset = torch.utils.data.Subset(
            trainset, range(0, self.num_train_samples))
        self.train_loader = list(iter(torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )))
        
        # create validation set
        self.valid_dataset = torch.utils.data.Subset(
            trainset, range(self.num_train_samples, num_samples))
        self.valid_loader = list(iter(torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.eval_batch_size,
            shuffle=self.shuffle,
        )))

        # create test set
        test_dataset = torchvision.datasets.MNIST(
            root="data", train=False, download=True, transform=self.transform
        )
        self.test_loader = list(iter(torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )))
        self.num_test_samples = len(test_dataset)

    def get_num_samples(self, split="train"):
        if split == "train":
            return self.num_train_samples
        elif split == "valid":
            return self.num_valid_samples
        elif split == "test":
            return self.num_test_samples

    def get_batch(self, idx, split="train"):
        if split == "train":
            return self.train_loader[idx]
        elif split == "valid":
            return self.valid_loader[idx]
        elif split == "test":
            return self.test_loader[idx]

dataset = MNIST(batch_size=200, splits=(0.9, 0.1))

'''
Training the model on plaintext data before exporting to onnx format and 
run in different PPML frameworks. 
'''
def train(model, num_epochs=10, lr=1e-3):
    all_train_losses = []
    all_val_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        train_losses = []
        model.train()
        for (data, target) in dataset.train_loader:
            # # Put the data on the same device as the model
            # data = data.to(device)
            # target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            train_losses.append(loss.item())
            train_losses = train_losses[-100:]
            optimizer.step() 
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataset.valid_loader:
                # # Put the data on the same device as the model
                # data = data.to(device)
                # target = target.to(device)
                output = model(data)
                val_loss += torch.nn.CrossEntropyLoss(reduction='sum')(output, target).item() # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        val_loss /= dataset.get_num_samples("valid")
        train_loss = np.mean(train_losses)
        print('Train Epoch: {} of {} Train Loss: {:.3f}, Val Loss: {:3f}, Val Accuracy: {:3f}'.format(
                    epoch, num_epochs, train_loss, val_loss, 100. * correct / dataset.get_num_samples("valid")))
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
    # plt.plot(all_train_losses)
    # plt.plot(all_val_losses)
    # plt.legend(['train', 'val'])
    # plt.show()
    return all_train_losses, all_val_losses

mlp_model = SimpleNet()
train_loss, val_loss = train(model=mlp_model, num_epochs=15)

# Testing for performance
# test_loss = 0
# correct = 0
# for data, target in dataset.test_loader:
#     output = model(data)
#     test_loss += torch.nn.CrossEntropyLoss(reduction='sum')(output, target).item() # sum up batch loss
#     pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#     correct += pred.eq(target.data.view_as(pred)).cpu().sum()
# test_loss /= dataset.get_num_samples("test")
# print('After training, Test Loss: {:3f}, Test Accuracy: {:3f}'.format(
#     test_loss, 100. * correct / dataset.get_num_samples("test")))

'''
set model to inference mode to prevent dropout and other training time functionalities
from interfering with inferences. 

Export the model to onnx format. 
'''
mlp_model.eval()
# Random input to trace the flow in the model 
x = torch.randn(1, 1, 28, 28, requires_grad=True)
torch_out = mlp_model(x)

# Export the model
torch.onnx.export(mlp_model,                     # model
                  x,                         # example input for tracing
                  "mlp.onnx",   # output file
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
'''
Visualize some examples to make sure that the model is running properly. 
'''
# predictions = []
# x_batch, y_batch = dataset.get_batch(1, "test")
# print(x_batch.shape)
# for i in range(50,55):
#     y_pred = model(x_batch[i]).max(1, keepdim=True)[1]
#     predictions.append(y_pred)
#     x = x_batch[i].view(28, 28)
#     plt.imshow(X=x, cmap="Greys")
#     plt.show()
# print(predictions)

'''
Checking the validity of the output models. 
'''
onnx_model = onnx.load("mlp.onnx")
onnx.checker.check_model(onnx_model)

'''
Generate testing inputs and outputs for the PPML frameworks, using the first
200 testing data points. 
'''
input_batch, output_batch = dataset.get_batch(0, 'test')
with open("input0.npy", 'xb') as f:
    np.save(f, input_batch)

with open("label0.npy", 'xb') as f:
    np.save(f, output_batch)

with open("input.npy", 'rb') as f:
    a = np.load(f, allow_pickle=True)
    a = torch.from_numpy(a)
    print(a.shape)
    prediction = mlp_model(a)
    print(prediction)

