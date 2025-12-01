# 文件: src/train.py (最终修正和优化版)

from ultralytics import YOLO
import os
import shutil
from pathlib import Path # 引入 pathlib 进行跨平台路径处理

# --- 项目配置 ---
# 训练结果将保存在 runs/detect/yolov8s_baseline 中
PROJECT_DIR = 'runs/detect' 
MODEL_NAME = 'yolov8s_baseline'
WEIGHTS_PATH = 'yolov8s.pt' # 使用官方预训练权重作为起点

# 获取项目根目录 (相对于 src 目录向上两级)
PROJECT_ROOT = Path(__file__).parent.parent 


def train_baseline_model():
    """
    加载预训练的 YOLOv8s 模型，在你的雾天数据集上进行训练。
    """
    print("--- 1. 加载模型与配置 ---")
    
    model = YOLO(WEIGHTS_PATH)
    
    # 使用绝对路径定位 data.yaml
    data_yaml_path = PROJECT_ROOT / 'datasets' / 'data.yaml'
    
    # 强制检查 data.yaml 文件是否存在
    if not data_yaml_path.exists():
        # 如果找不到，抛出错误并显示实际查找的路径
        raise FileNotFoundError(f"未找到数据集配置文件: {data_yaml_path.resolve()}")
        
    print(f"--- 2. 开始在 GPU 上训练 ({MODEL_NAME}) ---")
    
    # 🚨 重要：增加 epochs，确保模型有机会保存 best.pt
    # 建议使用 50-100 epochs，这里先设置为 5 轮来快速测试
    results = model.train(
        data=str(data_yaml_path), # 将 Path 对象转换为字符串
        epochs=1,                 
        imgsz=640,
        device=0, 
        project=PROJECT_DIR,
        name=MODEL_NAME
    )
    
    print("--- 3. 训练完成，结果已保存 ---")
    
    # --- 健壮的权重移动逻辑 ---
    
    # 1. 构造 best.pt 的绝对路径
    best_weights_path = PROJECT_ROOT / PROJECT_DIR / MODEL_NAME / 'weights' / 'best.pt'
    
    # 2. 构造目标保存路径的绝对路径
    target_weights_path = PROJECT_ROOT / 'weights' / f'{MODEL_NAME}_best.pt'
    
    # 确保目标文件夹存在
    target_weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. 检查文件是否存在后再移动 (防止 FileNotFoundError)
    if best_weights_path.exists():
        shutil.move(str(best_weights_path), str(target_weights_path))
        print(f"✅ 最佳权重已移动至: {target_weights_path.resolve()}")
    else:
        # 如果文件不存在，则报告错误并指出原因
        print(f"⚠️ 警告：未在预期位置找到权重文件: {best_weights_path.resolve()}")
        print("请检查训练日志，确认训练是否成功或中途失败。")

if __name__ == '__main__':
    train_baseline_model()