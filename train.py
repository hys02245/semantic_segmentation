# train.py (優化後)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, Trainer, TrainingArguments, SegformerConfig # <--- ADDED SegformerConfig
import json
import numpy as np

# =====================================================================================
# >>>>> 1. 設定您的專案配置 <<<<<
# =====================================================================================

# --- 數據集路徑 ---
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/validation"

# --- 模型配置 ---
PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# --- 您的類別定義 (!! 非常重要 !!) ---
ID2LABEL = {
    0: "background",
    1: "wall",
    2: "door",
    3: "roof",
    4: "floor",
    5: "outside",
    6: "cargo",
    7: "person",
}

# --- 訓練輸出 ---
OUTPUT_DIR = "./segformer-finetuned-custom"

# --- 模型保存格式配置 ---
SAVE_AS_PT = True          # 是否保存為 PyTorch .pt 格式
SAVE_AS_ONNX = True        # 是否保存為 ONNX 格式
SAVE_TRANSFORMERS = False   # 是否保存 Transformers 格式（原有方式）


# =====================================================================================
# >>>>> 2. 建立自訂 Dataset <<<<<
# =====================================================================================

class CustomSegmentationDataset(Dataset):
    """自訂影像分割數據集"""
    def __init__(self, root_dir, image_processor):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))
        
        self.images = [f for f in image_files if f.endswith(('.jpg', '.png'))]
        self.masks = [f for f in mask_files if f.endswith('.png')]
        
        assert len(self.images) == len(self.masks), "影像和遮罩的數量不匹配！"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx]))
        
        inputs = self.image_processor(images=image, segmentation_maps=mask, return_tensors="pt")
        
        pixel_values = inputs["pixel_values"].squeeze(0)
        labels = inputs["labels"].squeeze(0)
        
        return {"pixel_values": pixel_values, "labels": labels}

# =====================================================================================
# >>>>> 2.5. 模型保存功能 <<<<<
# =====================================================================================

def save_model_as_pt(model, image_processor, output_path, config_dict):
    """
    將模型保存為 PyTorch .pt 格式
    """
    print(f"正在保存 PyTorch .pt 格式模型到: {output_path}")
    
    # 準備保存的內容
    model_state = {
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'model_config': model.config.to_dict(),
        'processor_config': image_processor.to_dict(), # <--- MODIFIED: 確保 processor config 被儲存
        'id2label': config_dict['id2label'],
        'label2id': config_dict['label2id'],
        'num_classes': len(config_dict['id2label'])
    }
    
    torch.save(model_state, output_path)
    print(f"✅ PyTorch 模型已保存為: {output_path}")

def save_model_as_onnx(model, image_processor, output_path, config_dict):
    """
    將模型保存為 ONNX 格式
    """
    try:
        print(f"正在保存 ONNX 格式模型到: {output_path}")
        
        model.eval()
        
        # <--- MODIFIED START: 確保 dummy input 和 model 在同一個 device ---
        device = next(model.parameters()).device
        print(f"ONNX 導出: 模型位於 {device} 設備")
        
        # 創建一個與模型在相同設備上的示例輸入
        dummy_input = torch.randn(1, 3, 512, 512, device=device)
        # <--- MODIFIED END ---
        
        # 導出為 ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
                'logits': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        # 保存配置信息到同名的 JSON 文件
        config_path = output_path.replace('.onnx', '_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ ONNX 模型已保存為: {output_path}")
        print(f"✅ 配置文件已保存為: {config_path}")
        
    except Exception as e:
        print(f"❌ ONNX 導出失敗: {str(e)}")
        print("這可能是因為缺少 ONNX 相關依賴，請安裝: pip install onnx onnxruntime")

def create_standalone_model_config(model, image_processor):
    """
    創建獨立模型的配置信息
    """
    config = {
        'id2label': model.config.id2label,
        'label2id': model.config.label2id,
        'num_classes': model.config.num_labels,
        'model_type': 'segformer',
        'architecture': model.config.architectures[0] if hasattr(model.config, 'architectures') else 'SegformerForSemanticSegmentation',
        'image_size': getattr(model.config, 'image_size', 512),
        'num_channels': getattr(model.config, 'num_channels', 3),
        'num_encoder_blocks': getattr(model.config, 'num_encoder_blocks', 4),
        'depths': getattr(model.config, 'depths', [2, 2, 2, 2]),
        'sr_ratios': getattr(model.config, 'sr_ratios', [8, 4, 2, 1]),
        'hidden_sizes': getattr(model.config, 'hidden_sizes', [32, 64, 160, 256]),
        'decoder_hidden_size': getattr(model.config, 'decoder_hidden_size', 256),
        'patch_sizes': getattr(model.config, 'patch_sizes', [7, 3, 3, 3]),
        'strides': getattr(model.config, 'strides', [4, 2, 2, 2]),
        'num_attention_heads': getattr(model.config, 'num_attention_heads', [1, 2, 5, 8]),
        'mlp_ratios': getattr(model.config, 'mlp_ratios', [4, 4, 4, 4]),
        'hidden_act': getattr(model.config, 'hidden_act', 'gelu'),
        'hidden_dropout_prob': getattr(model.config, 'hidden_dropout_prob', 0.0),
        'attention_probs_dropout_prob': getattr(model.config, 'attention_probs_dropout_prob', 0.0),
        'classifier_dropout_prob': getattr(model.config, 'classifier_dropout_prob', 0.1),
        'initializer_range': getattr(model.config, 'initializer_range', 0.02),
        'drop_path_rate': getattr(model.config, 'drop_path_rate', 0.1),
        'layer_norm_eps': getattr(model.config, 'layer_norm_eps', 1e-6),
        'semantic_loss_ignore_index': getattr(model.config, 'semantic_loss_ignore_index', 255),
    }
    
    # 添加處理器相關配置
    processor_dict = image_processor.to_dict()
    config.update({f"processor_{key}": value for key, value in processor_dict.items()})
    
    return config

# =====================================================================================
# >>>>> 3. 主訓練流程 <<<<<
# =====================================================================================

def main():
    print("===== 開始設定模型 =====")
    
    label2id = {v: k for k, v in ID2LABEL.items()}
    num_classes = len(ID2LABEL)

    image_processor = SegformerImageProcessor.from_pretrained(PRETRAINED_MODEL, reduce_labels=False)
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=num_classes,
        id2label=ID2LABEL,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    print("===== 模型設定完成 =====")
    
    print("\n===== 準備數據集 =====")
    train_dataset = CustomSegmentationDataset(root_dir=TRAIN_DIR, image_processor=image_processor)
    val_dataset = CustomSegmentationDataset(root_dir=VAL_DIR, image_processor=image_processor)
    print(f"訓練集數量: {len(train_dataset)}, 驗證集數量: {len(val_dataset)}")
    print("===== 數據集準備完成 =====")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=6e-5,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\n===== ✨✨ 開始訓練！ ✨✨ =====")
    trainer.train()
    print("===== 訓練完成！ =====")

    print("\n===== 保存模型 =====")
    
    # 創建輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 創建模型配置
    model_config_for_saving = create_standalone_model_config(model, image_processor)
    
    # 1. 保存 Transformers 格式（原有方式）
    if SAVE_TRANSFORMERS:
        final_model_path = os.path.join(OUTPUT_DIR, "final")
        trainer.save_model(final_model_path)
        image_processor.save_pretrained(final_model_path)
        print(f"✅ Transformers 格式模型已儲存至: {final_model_path}")
    
    # 2. 保存為 PyTorch .pt 格式
    if SAVE_AS_PT:
        pt_model_path = os.path.join(OUTPUT_DIR, "segformer_model.pt")
        save_model_as_pt(model, image_processor, pt_model_path, model_config_for_saving)
    
    # 3. 保存為 ONNX 格式
    if SAVE_AS_ONNX:
        onnx_model_path = os.path.join(OUTPUT_DIR, "segformer_model.onnx")
        save_model_as_onnx(model, image_processor, onnx_model_path, model_config_for_saving)
    
    print(f"\n🎉 所有模型格式已保存完成！")
    print(f"輸出目錄: {OUTPUT_DIR}")
    if SAVE_TRANSFORMERS:
        print(f"  - Transformers 格式: {os.path.join(OUTPUT_DIR, 'final')}")
    if SAVE_AS_PT:
        print(f"  - PyTorch .pt 格式: {os.path.join(OUTPUT_DIR, 'segformer_model.pt')}")
    if SAVE_AS_ONNX:
        print(f"  - ONNX 格式: {os.path.join(OUTPUT_DIR, 'segformer_model.onnx')}")
        print(f"  - ONNX 配置: {os.path.join(OUTPUT_DIR, 'segformer_model_config.json')}")

# =====================================================================================
# >>>>> 4. 模型載入範例 (優化後) <<<<<
# =====================================================================================
def load_pt_model(model_path, device='cpu'):
    """
    載入 PyTorch .pt 格式的模型 (優化後，實現完全離線載入)
    
    使用範例:
    model, processor, config = load_pt_model('segformer_model.pt')
    """
    print(f"正在載入 PyTorch 模型: {model_path}")
    
    # <--- MODIFIED START: 採用完全離線的載入方式 ---
    checkpoint = torch.load(model_path, map_location=device)
    
    # 1. 從儲存的 model_config 重建模型結構
    # 這使得模型載入不再需要網路或 PRETRAINED_MODEL 字串
    model_config_dict = checkpoint['model_config']
    config = SegformerConfig(**model_config_dict)
    model = SegformerForSemanticSegmentation(config)

    # 2. 載入訓練好的權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型載入成功，設備: {device}")
    
    # 3. 從儲存的 processor_config 重建 ImageProcessor
    image_processor = SegformerImageProcessor.from_dict(checkpoint['processor_config'])
    
    print(f"✅ 圖像處理器重建成功")

    return model, image_processor, checkpoint['config']
    # <--- MODIFIED END ---

if __name__ == "__main__":
    main()