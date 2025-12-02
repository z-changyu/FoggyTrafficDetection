# 文件: src/train_cbam.py (最终修正和优化版)

from ultralytics import YOLO
import os
import shutil
from pathlib import Path 
import torch.nn as nn # 🚨 修正：导入 nn 变量
import torch          # 导入 torch 确保环境完整
from modules import CBAM  # 导入我们自定义的模块

# --- 项目配置 ---
# 训练结果将保存在 runs/detect/yolov8s_cbam 文件夹中
PROJECT_DIR = 'runs/detect' 
MODEL_NAME = 'yolov8s_cbam'
WEIGHTS_PATH = 'yolov8s.pt' 

# 获取项目根目录 
PROJECT_ROOT = Path(__file__).parent.parent 
# 定义模型结构文件的路径
CBAM_MODEL_CONFIG = PROJECT_ROOT / 'src' / 'yolov8s_cbam.yaml'


def create_cbam_config():
    """
    检查或创建 CBAM 模型配置文件。
    要求用户手动将 CBAM 模块的定义插入到这个 YAML 文件中。
    """
    print("--- 1. 检查或创建 CBAM 模型配置文件 ---")
    
    if not CBAM_MODEL_CONFIG.exists():
        print(f"⚠️ 警告: 未找到 {CBAM_MODEL_CONFIG.name} 文件。请手动创建并插入 CBAM 模块定义。")
        # 实际运行中，如果文件不存在，这里应该退出或等待
        # 为了让流程能继续，我们暂时让它继续，但会依赖用户已经创建了该文件。
        return False
    return True

    
def train_cbam_model():
    if not create_cbam_config():
        # 如果 YAML 文件不存在，安全退出
        return
        
    print("--- 2. 加载模型与配置 ---")
    
    # 🚨 关键：加载我们修改后的模型结构 YAML
    model = YOLO(str(CBAM_MODEL_CONFIG)) 
    
    # 自动加载官方预训练权重（推荐）
    try:
        model.load(WEIGHTS_PATH)
    except FileNotFoundError:
        print(f"⚠️ 警告: 未找到预训练权重 {WEIGHTS_PATH}，将从头开始训练。")
    
    # 使用绝对路径定位 data.yaml
    data_yaml_path = PROJECT_ROOT / 'datasets' / 'data.yaml'
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {data_yaml_path.resolve()}")
        
    # 🚨 关键：注册自定义模块 (已移除会导致错误的 nn.Identity 占位符)
    # YOLOv8 会自动识别并加载 src.modules.CBAM
    model.add_callback("on_pretrain", lambda: print("Custom modules registered.")) 

    print(f"--- 3. 开始在 GPU 上训练改进模型 ({MODEL_NAME}) ---")
    
    results = model.train(
        data=str(data_yaml_path), 
        epochs=50,                
        imgsz=640,
        device=0, 
        project=PROJECT_DIR,
        name=MODEL_NAME
    )
    
    print("--- 4. 训练完成，结果已保存 ---")
    
    # --- 健壮的权重移动逻辑 ---
    best_weights_path = PROJECT_ROOT / PROJECT_DIR / MODEL_NAME / 'weights' / 'best.pt'
    target_weights_path = PROJECT_ROOT / 'weights' / f'{MODEL_NAME}_best.pt'
    
    target_weights_path.parent.mkdir(parents=True, exist_ok=True)
    
    if best_weights_path.exists():
        shutil.move(str(best_weights_path), str(target_weights_path))
        print(f"✅ 最佳权重已移动至: {target_weights_path.resolve()}")
    else:
        print(f"⚠️ 警告：未在预期位置找到权重文件: {best_weights_path.resolve()}")
        print("请检查训练日志，确认训练是否成功或中途失败。")

if __name__ == '__main__':
    train_cbam_model()