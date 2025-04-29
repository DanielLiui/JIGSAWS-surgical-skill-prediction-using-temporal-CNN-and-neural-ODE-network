# File trains the temporal convolution network model with a neural ODE network for modelling changing dynamics
# in the data. 
# I recommend reading the main training method first (TCNWithNeuralODE_model()) for a high-level overview
# of the training steps, then inspect the functions and classes defined for further detail.

import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torchdiffeq import odeint


# Function reads the meta file in a surgical task's folder and returns a dictionary of the surgeon's skill level 
# for each kinematic file. 
# eg. {'Suturing_B001': 0, 'Suturing_B002': 1, ,,,}, where 0 = non-expert and 1 = expert
def get_label_map(surgical_task):
  label_df = pd.read_csv(f'dataset/{surgical_task}/meta_file_{surgical_task}.txt', sep="\t", header=None, engine='python')
  label_df = label_df[[0, 2]]  # keep only trial name and skill
  label_df.columns = ["trial", "skill"]

  # Convert 'E' to expert (1), everything else to non-expert (0)
  label_df["label"] = label_df["skill"].apply(lambda x: 1 if x.strip().upper() == "E" else 0)

  # Build dict: trial name -> label
  label_map = dict(zip(label_df["trial"], label_df["label"]))
  return label_map


# Function reads all kinematic files for a surgical task like suturing then returns the labels for each file,
# the matrix of tooltips kinematic data (n rows * 72 columns, each column is a tooltip dimension), and the number of features (columns)
def get_labels_kinematics_features(surgical_task):
  data_dir = f'dataset/{surgical_task}/kinematics/AllGestures'
  label_map = get_label_map(surgical_task)  # eg. {'Suturing_B001': 0, 'Suturing_B002': 1, ...}
  # print(label_map)

  all_trajectories = []
  labels = []
  scaler = StandardScaler()
  print('Reading kinematic files')
  
  for file_path in os.listdir(data_dir):
    if not '.txt' in file_path: continue
    #print(f'Reading {file_path} | ', end='')

    try:
      df = pd.read_csv(f'{data_dir}/{file_path}', sep=r"\s+", header=None)

    except:
      print(f"Skipping {file_path} since it's empty or improperly formatted")
      continue

    all_features = df.values  # numpy representation

    # if the trial name is not labeled for some reason, skip
    trial_name = os.path.splitext(file_path)[0]
    if trial_name not in label_map: 
      print(f"Skipping {file_path} since it's not in the label map")
      continue
    
    # add label
    label = label_map[trial_name]
    # normalize data
    all_features = scaler.fit_transform(all_features)
    #print(f'features shape: {all_features.shape}')

    # add trajectories (X) and labels (y)
    all_trajectories.append(torch.tensor(all_features, dtype=torch.float32))
    labels.append(label)

  # find longest sequence
  max_len = max([seq.shape[0] for seq in all_trajectories])
  n_features = all_features.shape[1]  # 76

  # pad sequences to the max length
  all_trajectories = [
    torch.cat([seq, torch.zeros(max_len - seq.shape[0], n_features)], dim=0)
    if seq.shape[0] < max_len else seq for seq in all_trajectories
  ]

  # stack and convert to tensors
  all_trajectories = torch.stack(all_trajectories)
  labels = torch.tensor(labels)

  return labels, all_trajectories, n_features


# Kinematic data generator
class KinematicDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.long)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]


# Class that defines the architecture and hyperparameters of the Neural Ordinary Differential Equations 
# function (a neural network).
class ODEF(nn.Module):

  def __init__(self, hidden_dim):
    super(ODEF, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.Tanh(),
      nn.Dropout(0.2),
      nn.Linear(hidden_dim, hidden_dim)
    )

  # Method for forward propagation
  def forward(self, t, x):
    return self.net(x)
    

# Class that defines the architecture and hyperparameters of the Neural ODE network that uses the 
# ODE function to solve ordinary differential equations in order to model changing data dynamics.
class NeuralODEBlock(nn.Module):

  def __init__(self, odefunc, time_span):
    super(NeuralODEBlock, self).__init__()
    self.odefunc = odefunc
    self.integration_time = time_span  
    #integration time has arbitrary time unit determined by the problem's context


  def forward(self, x):
    # expects x: [batch, hidden_dim]
    out = odeint(self.odefunc, x, self.integration_time, method='rk4')  
    # Use 4th order Runge-Kutta method for solving ordinary differential equations
    # training seems to stall with dopri5 method
    return out[-1]  # take the final output at time t = 1
 

# Class that defines the architecture and hyperparameters of a temporal convolution network 
class TemporalBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
    super(TemporalBlock, self).__init__()
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
    self.relu1 = nn.ReLU()
    self.net = nn.Sequential(self.conv1, self.relu1)


  def forward(self, x):
    return self.net(x)


# Class that defines a temporal convolution network (TCN) and a neural ODE to model the changing data dynamics.
# Class defines the architecture, hyperparameters, and how data is forward propagated.
class TCNWithNeuralODE(nn.Module):

  # Initializer method
  # Parameters:
  # ...
  # ode_integration_time - integration end time (time unit arbitrarily determined by the problem's context)

  def __init__(self, input_dim, hidden_dim, num_classes, ode_integration_time):
    super(TCNWithNeuralODE, self).__init__()
    self.tcn = TemporalBlock(input_dim, hidden_dim, kernel_size=3, stride=1, padding=2, dilation=1)
    self.odefunc = ODEF(hidden_dim)
    self.odeblock = NeuralODEBlock(self.odefunc, time_span=torch.tensor([0, ode_integration_time], dtype=torch.float32))
    self.fc = nn.Linear(hidden_dim, num_classes)


  # Method for forward propagation
  def forward(self, x):
    x = x.permute(0, 2, 1)    
    x = self.tcn(x)           
    x = torch.mean(x, dim=2)  
    x = self.odeblock(x)      
    return self.fc(x)


def loss_plot(train_losses, val_losses):
  plt.plot(train_losses, label="Training Loss")
  plt.plot(val_losses, label="Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training vs. Validation Loss")
  plt.legend()
  plt.show()


# Function trains a TCN model with a neural ODE attached to it.
def TCNWithNeuralODE_model(surgical_task, epochs=50, n_hidden_layers=64, integration_time=1):
  y, X, n_features = get_labels_kinematics_features(surgical_task)
  #print(f'y shape: {y.shape}  X shape: {X.shape}  n_features: {n_features}')

  # split into train/val/test sets
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
  X_train_tensor = X_train.clone().detach()
  y_train_tensor = y_train.clone().detach()
  X_val_tensor = X_val.clone().detach()
  y_val_tensor = y_val.clone().detach()
  X_test_tensor = X_test.clone().detach()
  y_test_tensor = y_test.clone().detach()

  # metrics for each epoch
  train_losses = []; val_losses = []
  train_accuracies = []; val_accuracies = []

  # variables for early train stopping
  best_loss = float('inf')
  patience = 3; patience_counter = 0

  # define model
  model = TCNWithNeuralODE(input_dim=n_features, num_classes=2, hidden_dim=n_hidden_layers, ode_integration_time=integration_time)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  
  # run forward and backward propagation
  for epoch in range(epochs): 
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # train metrics
    train_preds = outputs.argmax(dim=1)
    train_acc = (train_preds == y_train_tensor).float().mean().item()
    train_accuracies.append(train_acc)

    train_loss = loss.item()
    train_losses.append(train_loss)

    # validation metrics
    model.eval()
    with torch.no_grad():
      val_outputs = model(X_val_tensor)
      val_loss = criterion(val_outputs, y_val_tensor).item()
      val_losses.append(val_loss)

      val_preds = val_outputs.argmax(dim=1)
      val_acc = (val_preds == y_val_tensor).float().mean().item()
      val_accuracies.append(val_acc)

    print(f"[Epoch {epoch}] \nTrain Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")

    # check metrics if need to stop early
    if val_loss < best_loss - 0.0001:
      best_loss = val_loss
      patience_counter = 0

    else:
      patience_counter += 1

    if patience_counter >= patience:
      print(f"Early stopping at epoch {epoch}"); break

      
  # evaluate model
  model.eval()
  with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = test_outputs.argmax(dim=1)
    test_acc = (test_preds == y_test_tensor).float().mean().item()

  print(f"Train accuracy: {train_accuracies[-1]:.2f}")
  print(f"Validation accuracy: {val_accuracies[-1]:.2f}")
  print(f"Test accuracy: {test_acc:.2f}")
  y_pred = model(X_test_tensor).argmax(dim=1).numpy()
  print("Classification Report:\n", classification_report(y_test, y_pred))
  print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
  loss_plot(train_losses, val_losses)

  return model, train_accuracies[-1], val_accuracies[-1], test_acc


''' TEST METHODS '''

# Function averages the validation and test accuracies over 5 trained models with the same hyperparameters 
# since test set is small (~6 samples)
def best_model_test(surgical_task):
  train_accuracies = []
  val_accuracies = []
  test_accuracies = []

  for trial in range(0,5):
    tcn_ode_model, train_acc, val_acc, test_acc = TCNWithNeuralODE_model(
      surgical_task, epochs=50, n_hidden_layers=32, integration_time=0.5)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)

  av_train_acc = np.mean(train_accuracies)
  av_val_acc = np.mean(val_accuracies)
  av_test_acc = np.mean(test_accuracies)
  formatted_val_accuracies = [float(f'{x:.2f}') for x in val_accuracies]
  formatted_test_accuracies = [float(f'{x:.2f}') for x in test_accuracies]

  print(f'\nval accuracies: {formatted_val_accuracies}')
  print(f'test accuracies: {formatted_test_accuracies}')
  print(f'av train acc: {av_train_acc:.2f}, av val_acc: {av_val_acc:.2f}, av test_acc: {av_test_acc:.2f}')

