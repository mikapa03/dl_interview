import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# 配置参数
input_size, hidden_size, output_size = 1024, 2048, 10
batch_size = 1024
epochs = 20

def train_benchmark(device_name):
    device = torch.device(device_name)
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(batch_size, input_size).to(device)
    targets = torch.randint(0, output_size, (batch_size,)).to(device)

    # 预热
    for _ in range(5):
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

# 执行测试
print("🧐 正在测试 M4 芯片性能...")
cpu_time = train_benchmark("cpu")
mps_time = train_benchmark("mps")

# --- 绘图逻辑 ---
labels = ['CPU (with AMX)', 'MPS (GPU)']
times = [cpu_time, mps_time]

plt.figure(figsize=(10, 6))
# 绘制柱状图
bars = plt.bar(labels, times, color=['#3498db', '#e74c3c'])

# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}s', ha='center', va='bottom', fontsize=12)

plt.ylabel('Average Time per Epoch (seconds)', fontsize=12)
plt.title(f'M4 Chip Training Benchmark (Batch Size: {batch_size})', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('m4_performance.png')
print("✅ 可视化结果已保存为 'm4_performance.png'")
plt.show()