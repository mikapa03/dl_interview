import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# --- 暴力拉满参数 ---
# 增加隐藏层到 8192，增加 batch_size 到 2048
input_size, hidden_size, output_size = 1024, 8192, 10
batch_size = 2048
epochs = 20

def train_benchmark(device_name):
    device = torch.device(device_name)
    # 增加网络深度，让计算量成倍增长
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(batch_size, input_size).to(device)
    targets = torch.randint(0, output_size, (batch_size,)).to(device)

    # 强化预热：让 GPU 彻底进入状态
    for _ in range(10):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
    
    if device_name == "mps": torch.mps.synchronize()

    start_time = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
    
    if device_name == "mps": torch.mps.synchronize()
    return (time.time() - start_time) / epochs

print(f"🔥 正在进行大负载压力测试 (Hidden Size: {hidden_size})...")
cpu_time = train_benchmark("cpu")
mps_time = train_benchmark("mps")

# 绘图
labels = ['CPU (AMX Limit)', 'MPS (GPU Mode)']
times = [cpu_time, mps_time]
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, times, color=['#95a5a6', '#2ecc71']) # 换个颜色
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}s', ha='center', va='bottom')
plt.title(f'M4 Stress Test: Large MLP (Hidden: {hidden_size})')
plt.savefig('stress_result.png')
print(f"✨ 加速比: {cpu_time / mps_time:.2f} 倍")