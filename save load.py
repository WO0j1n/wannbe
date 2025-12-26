# torch.save -> 모델의 state를 dict 형태로 직렬화하여 저장, 사람이 읽을 수 없음
# torch.save의 경우, 저장한 모델의 state를 정확하게 바인딩한 뒤 사용해야 하기에 비추천

# model.load_state_dict() ->  모델의 parameters만 저장하여

import torch
import os
import torch.nn as nn

model = ''
PATH =''
optimizer = ''

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
# Complete Model -> 해당 방식의 경우, 모델의 state를 dict 형태로 직렬화하여 저장, 사람이 읽을 수 없음
# 해당 방식의 경우, 저장한 모델의 state를 정확하게 바인딩한 뒤 사용해야 하기에 비추천
torch.save(model, PATH)

model = torch.load(PATH)
model.eval()


# STATE DICT -> 해당 방식의 경우 모든 모델의 state를 저장하는 것이 아닌 class, parameter를 저장
# 해당 방식을 추천함

torch.save(model.state_dict(), PATH)

# model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()


# SAVE CKPT & LOAD CKPT
ckpt = {
    "epoch": 100,
    "model_state_dict" : model.state_dict(),
    "optim_state_dict" : optimizer.state_dict()
}

PATH = 'CKPT.pth'
torch.save(ckpt, PATH)

ckpt_model = torch.load(PATH)
model.load_state_dict(ckpt_model['model_state'])
optimizer.load_state_dict(ckpt_model['optim_state_dict'])
epoch = ckpt_model['epoch']

model.eval()


# Best & Last CKPT
def save_ckpt(state, is_best, ckpt_dir = './ckpts', filename = 'last.pth'):
    os.makedirs(ckpt_dir, exist_ok=True)

    last_path = os.path.join(ckpt_dir, filename)
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(ckpt_dir, 'best.pth')
        torch.save(state, best_path)

def train(model, optimizer, criterion, train_loader, num_epochs = 50):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0

        for x, y in train_loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}]  Acc: {acc:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict" : model.state_dict(),
            "optim_state_dict" : optimizer.state_dict(),
            "best_acc" : best_acc
        }

        is_best = acc > best_acc

        if is_best:
            print(f">>> Best model updated: {best_acc:.4f} -> {acc:.4f}")
            best_acc = acc

        save_ckpt(ckpt, is_best)

def load_ckpt(model, optimizer, ckpt_path):

    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optim_state_dict'])

    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt.get("best_acc", 0)

    print(f"Loaded checkpoint: start at epoch {start_epoch}, best_acc={best_acc}")

    return start_epoch, best_acc
