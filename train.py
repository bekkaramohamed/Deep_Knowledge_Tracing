import os
import torch
import pickle

from data_loaders.assist2009 import ASSIST2009
from dkt import DKT, dkt_train
from utils import collate_fn
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam


model_name = "DKT"
dataset_name = "ASSIST2009"

batch_size = 64
num_epochs = 100
train_ratio = 0.6
learning_rate = 1e-4
optimizer = "adam"
seq_len = 100 # sampling number of dataset.

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if not os.path.isdir("ckpts"):
    os.mkdir("ckpts")

ckpt_path = os.path.join("ckpts", model_name)
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

ckpt_path = os.path.join(ckpt_path, dataset_name)
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)
    
    
dataset = ASSIST2009(seq_len)

model = torch.nn.DataParallel(DKT(dataset.num_q, emb_size = 100, hidden_size = 100)).to(device)
train_model = dkt_train

data_size = len(dataset)
train_size = int(data_size * train_ratio) 
valid_size = int(data_size * ((1.0 - train_ratio) / 2.0))
test_size = data_size - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [train_size, valid_size, test_size], generator=torch.Generator(device=device)
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, generator=torch.Generator(device=device)
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, generator=torch.Generator(device=device)
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_fn, generator=torch.Generator(device=device)
)

if optimizer == "sgd":
    opt = SGD(model.parameters(), learning_rate, momentum=0.9)
elif optimizer == "adam":
    opt = Adam(model.parameters(), learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
opt.lr_scheduler = lr_scheduler

aucs, loss_means = \
    train_model(
        model, train_loader, valid_loader, test_loader, dataset.num_q, num_epochs, opt, ckpt_path
    )

with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
    pickle.dump(aucs, f)
with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
    pickle.dump(loss_means, f)