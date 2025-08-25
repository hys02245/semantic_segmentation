# inference.py (Unified version with auto-detection)

import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# --- 核心依賴 ---
# Transformers 用於 PyTorch 模型結構和通用的 ImageProcessor
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

# --- 可選依賴 ---
# 只有在推理 ONNX 模型時才需要 onnxruntime
try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False

# =====================================================================================
# >>>>> 1. 配置與常量 <<<<<
# =====================================================================================

# 您的類別定義 (必須與訓練時完全一致)
ID2LABEL = {
    0: "background", 1: "wall", 2: "door", 3: "roof",
    4: "floor", 5: "outside", 6: "cargo", 7: "person",
}

# 為視覺化定義一個顏色調色盤 (RGB)
COLOR_PALETTE = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
]

# =====================================================================================
# >>>>> 2. PyTorch (.pt) 專用函式 <<<<<
# =====================================================================================

def load_pt_model(model_path, device):
    """從 .pt 檔案載入 PyTorch 模型和圖像處理器。"""
    print(f"正在從 {model_path} 載入 PyTorch 模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_config_dict = checkpoint['model_config']
    config = SegformerConfig(**model_config_dict)
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    image_processor = SegformerImageProcessor.from_dict(checkpoint['processor_config'])
    print("✅ PyTorch 模型和處理器載入成功！")
    return model, image_processor

def predict_pt(model, image_processor, image, device):
    """使用 PyTorch 模型執行語義分割。"""
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
    
    upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return pred_seg.cpu().numpy()

# =====================================================================================
# >>>>> 3. ONNX (.onnx) 專用函式 <<<<<
# =====================================================================================

def load_onnx_session(model_path):
    """載入 ONNX 推理 session。"""
    if not ONNX_RUNTIME_AVAILABLE:
        raise ImportError("ONNX Runtime 未安裝。請執行 `pip install onnxruntime` 來使用 ONNX 模型。")
    print(f"正在從 {model_path} 載入 ONNX 模型...")
    session = ort.InferenceSession(model_path)
    print("✅ ONNX 模型載入成功！")
    return session

def predict_onnx(session, image_processor, image, device):
    """使用 ONNX Runtime 執行語義分割。"""
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.cpu().numpy()
    
    onnx_inputs = {'pixel_values': pixel_values}
    logits_onnx = session.run(None, onnx_inputs)[0]
    
    logits_tensor = torch.from_numpy(logits_onnx).to(device)
    upsampled_logits = F.interpolate(logits_tensor, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    return pred_seg.cpu().numpy()

# =====================================================================================
# >>>>> 4. 通用視覺化函式 <<<<<
# =====================================================================================

def draw_segmentation_map(seg_map, color_palette):
    """將分割的類別 ID 映射轉換為彩色的視覺化圖像。"""
    color_seg = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(color_palette):
        color_seg[seg_map == label, :] = color
    return Image.fromarray(color_seg)

def overlay_segmentation(image, color_mask, alpha=0.6):
    """將彩色遮罩半透明地疊加在原始圖像上。"""
    image = image.convert("RGB")
    color_mask = color_mask.convert("RGB").resize(image.size)
    return Image.blend(image, color_mask, alpha=alpha)

# =====================================================================================
# >>>>> 5. 主執行流程 <<<<<
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="使用 Segformer 模型進行語義分割推理 (自動偵測 .pt 或 .onnx)")
    parser.add_argument("--model_path", type=str, required=True, help="指向 .pt 或 .onnx 模型檔案的路徑。")
    parser.add_argument("--input_path", type=str, required=True, help="指向單張圖片或包含圖片的資料夾的路徑。")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="儲存結果的資料夾。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理設備 ('cuda' or 'cpu')。")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_path = args.model_path

    # --- 模型載入與推理函式選擇 ---
    if model_path.lower().endswith(('.pt', '.pth')):
        print("偵測到 PyTorch 模型檔案...")
        engine, image_processor = load_pt_model(model_path, device)
        predictor = predict_pt
    elif model_path.lower().endswith('.onnx'):
        print("偵測到 ONNX 模型檔案...")
        try:
            engine = load_onnx_session(model_path)
            # ONNX 模型不包含 processor 配置，我們需要從 Hub 載入一個標準的
            print("正在從 Hugging Face Hub 載入對應的圖像處理器...")
            image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            print("✅ 圖像處理器載入成功！")
            predictor = predict_onnx
        except (ImportError, Exception) as e:
            print(f"❌ 載入 ONNX 模型失敗: {e}")
            return
    else:
        print(f"錯誤: 不支援的模型檔案格式: {model_path}")
        print("請提供 .pt 或 .onnx 檔案。")
        return

    # --- 準備輸入和輸出路徑 ---
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    elif os.path.isdir(args.input_path):
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f"錯誤: 輸入路徑 {args.input_path} 不存在或無效。")
        return

    print(f"找到 {len(image_paths)} 張圖片，開始進行推理...")
    
    # --- 處理每張圖片 ---
    for image_path in tqdm(image_paths, desc="推理進度"):
        try:
            original_image = Image.open(image_path).convert("RGB")
            
            # 動態呼叫選擇的 predictor 函式
            segmentation_map = predictor(engine, image_processor, original_image, device)
            
            # 視覺化與儲存
            color_mask = draw_segmentation_map(segmentation_map, COLOR_PALETTE)
            # overlay_image = overlay_segmentation(original_image, color_mask)
            raw_mask_image = Image.fromarray(segmentation_map.astype(np.uint8))

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            raw_mask_image.save(os.path.join(args.output_dir, f"{base_filename}_raw_mask.png"))
            color_mask.save(os.path.join(args.output_dir, f"{base_filename}_color_mask.png"))
            # overlay_image.save(os.path.join(args.output_dir, f"{base_filename}_overlay.jpg"))

        except Exception as e:
            print(f"\n處理圖片 {image_path} 時發生錯誤: {e}")

    print(f"\n🎉 推理完成！所有結果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()