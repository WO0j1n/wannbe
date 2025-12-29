import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent (SGD) -> 학습 과정에서 수렴이 불안정하며 local minimum 혹은 saddle point에 빠질 수 있음
optimizer_momentum = torch.optim.SGD(model.parameters(),lr = 0.01, momentum=0.9) # Stochastic Gradient Descent (SGD) + Momentum -> 위의 문제를 해결하기 위해서 이전의 기울기를 누적하여 관성을 만들어 수렴 속도를 높이고 안정성에 기여
optimizer_rmsprop = torch.optim.RMSprop(model.parameters(), lr = 0.001, alpha=0.99) # RMSProp -> 위의 기여에도 불구하고 학습의 불안정성을 해결하기 위해서 기울기에 따라 기울기가 큰 방향은 덜 움직이고 작은 방향은 더 움직이도록 학습률을 자동 조절하는 최적화
optimizer_adam = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999)) # RMSProp + Momentum, betas = (Momentum, RMSProp) -> Momentum의 장점(방향 안정성)과 RMSProp의 장점(좌표별 적응 학습률)을 동시에 활용하는 기법 -> Weight decay랑 같이 쓸 거면 AdamW를 쓰는 게 좋음
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr = 0.001, betas= (0.9, 0.99)) # Adam + Weight Decay를 분리하여 수행한 기법 -> 일반적인 Adam에서 weight decay가 최적화 과정에 영향을 미치는 문제를 해결하여 일반화 성능을 향상 


# 주로 L2 normalization의 경우, 각 optimizer의 weight_decay 파라미터를 설정하여 활용

# L1 normalization의 경우, 직접 loss function에 l1 norm을 추가해서 활용 -> 그러나 딥러닝에서 일반화 관점에서 잘 사용하지 않음.
lambda_l1 = 0.001 # L1 정규화 강도 하이퍼 파라미터
criterion = nn.CrossEntropyLoss()

outputs = model(inputs)

l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = criterion(outputs, targets) + lambda_l1 * l1_norm
