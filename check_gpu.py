import torch
import platform
import subprocess

def get_mac_gpu_info():
    """获取 Mac GPU 信息"""
    try:
        # 使用 system_profiler 命令获取 GPU 信息
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True)
        print("\n=== Mac GPU 信息 ===")
        print(result.stdout)
    except Exception as e:
        print(f"获取 GPU 信息时出错: {e}")

def get_torch_device_info():
    """获取 PyTorch 设备信息"""
    print("\n=== PyTorch 设备信息 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"MPS 是否可用: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    if torch.backends.mps.is_available():
        print("\nMPS 设备信息:")
        print(f"当前设备: {torch.device('mps')}")
        # 注意：MPS 不直接提供显存信息

def get_system_info():
    """获取系统信息"""
    print("\n=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {platform.python_version()}")
    print(f"处理器: {platform.processor()}")

if __name__ == "__main__":
    get_system_info()
    get_mac_gpu_info()
    get_torch_device_info() 