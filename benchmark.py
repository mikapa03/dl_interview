import torch
import time

# 定义一个足够大的矩阵，让计算量达到亿级
SIZE = 4000

def run_test(device_name):
    device = torch.device(device_name)
    # 创建随机矩阵
    a = torch.randn(SIZE, SIZE, device=device)
    b = torch.randn(SIZE, SIZE, device=device)
    
    # 预热一下（防止第一次运行的系统开销影响结果）
    _ = torch.mm(a, b)
    if device_name == "mps": torch.mps.synchronize()
    
    # 正式计时
    start = time.time()
    for _ in range(100): # 运行100次取平均
        c = torch.mm(a, b)
    
    if device_name == "mps": torch.mps.synchronize()
    return (time.time() - start) / 10

print(f"📊 正在对比 {SIZE}x{SIZE} 矩阵乘法性能...")

# CPU 测试
cpu_time = run_test("cpu")
print(f"🐌 CPU 平均耗时: {cpu_time:.4f} 秒")

# GPU (MPS) 测试
if torch.backends.mps.is_available():
    mps_time = run_test("mps")
    print(f"🚀 MPS (GPU) 平均耗时: {mps_time:.4f} 秒")
    print(f"✨ 提升倍数: {cpu_time / mps_time:.2f} 倍")
else:
    print("❌ 未检测到 MPS 加速")