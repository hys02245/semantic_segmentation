# Semantic Segmentation with Segformer

一個基於 Hugging Face Transformers 的語義分割專案，使用 Segformer 模型進行倉庫重建場景的語義分割。

## 🚀 主要特性

- 🔥 基於 **Segformer** 架構的語義分割
- 📦 支援多種模型格式：PyTorch (.pt)、ONNX、Transformers
- 🎨 彩色分割遮罩視覺化
- 📊 訓練過程監控與評估
- 🔧 模組化設計，易於擴展和修改
- 💻 CPU/GPU 兼容推理

## 📁 專案結構

```
semantic_segmentation/
├── train.py                    # 模型訓練腳本
├── inference.py               # 推理腳本 (支援 .pt 和 .onnx)
├── requirements.txt           # 專案依賴
├── README.md                  # 專案說明文件
├── data/                      # 數據集
│   ├── train/                 # 訓練數據
│   │   ├── images/           # 訓練圖片
│   │   └── masks/            # 訓練遮罩
│   └── validation/           # 驗證數據
│       ├── images/           # 驗證圖片
│       └── masks/            # 驗證遮罩
├── segformer-finetuned-custom/  # 訓練輸出目錄
│   ├── segformer_model.pt    # PyTorch 模型
│   ├── segformer_model.onnx  # ONNX 模型
│   └── checkpoint-*/         # 訓練檢查點
└── inference_output/          # 推理結果輸出
    ├── *_raw_mask.png        # 原始分割遮罩
    ├── *_color_mask.png      # 彩色分割遮罩
    └── *_overlay.jpg         # 疊加視覺化結果
```

## 🛠️ 安裝與設置

### 1. 環境要求

- Python 3.8+
- CUDA 11.8+ (如需 GPU 加速)

### 2. 安裝依賴

```bash
# clone 專案
git clone https://github.com/hys02245/semantic_segmentation.git
cd semantic_segmentation

# 建立 conda 環境
conda create -n semantic_segmentation python=3.10 -y
conda activate semantic_segmentation

# 安裝 torch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安裝 cuda-toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# 安裝依賴
pip install -r requirements.txt
```

### 3. 數據準備

將您的訓練和驗證數據按以下結構組織：

```
data/
├── train/
│   ├── images/          # RGB 圖片 (.jpg, .png)
│   └── masks/           # 分割遮罩 (.png, 像素值對應類別 ID)
└── validation/
    ├── images/
    └── masks/
```

**重要：** 遮罩圖片中的像素值必須對應類別 ID (0-7)。

## 🏋️ 模型訓練

### 基本訓練

```bash
python train.py
```

### 訓練配置

在 `train.py` 中可以調整以下配置：

```python
# 數據路徑
TRAIN_DIR = "./data/train"
VAL_DIR = "./data/validation"

# 模型配置
PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

# 類別定義
ID2LABEL = {
    0: "background", 1: "wall", 2: "door", 3: "roof",
    4: "floor", 5: "outside", 6: "cargo", 7: "person",
}

# 輸出格式
SAVE_AS_PT = True          # 保存 PyTorch .pt 格式
SAVE_AS_ONNX = True        # 保存 ONNX 格式
SAVE_TRANSFORMERS = False  # 保存 Transformers 格式
```

### 訓練參數

```python
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
)
```

## 🔮 模型推理

本專案支援兩種模型格式的推理：

### PyTorch 模型推理

```bash
python inference.py \
    --model_path ./segformer-finetuned-custom/segformer_model.pt \
    --input_path ./test_image.jpg \
    --output_dir ./inference_output \
    --device cuda
```

### ONNX 模型推理

```bash
python inference.py \
    --model_path ./segformer-finetuned-custom/segformer_model.onnx \
    --input_path ./test_images/ \
    --output_dir ./inference_output \
    --device cpu
```

### 批量推理

```bash
# 處理整個資料夾的圖片
python inference.py \
    --model_path ./segformer-finetuned-custom/segformer_model.pt \
    --input_path ./test_images/ \
    --output_dir ./results/
```

## 📊 輸出結果

推理完成後，每張輸入圖片會生成三個輸出文件：

1. **`{filename}_raw_mask.png`** - 原始分割遮罩 (像素值為類別 ID)
2. **`{filename}_color_mask.png`** - 彩色分割遮罩 (視覺化用)
3. **`{filename}_overlay.jpg`** - 原圖與分割結果的疊加圖

## 🎨 類別顏色對應

| 類別 ID | 類別名稱 | 顏色 (RGB) |
|---------|----------|------------|
| 0 | background | (0, 0, 0) - 黑色 |
| 1 | wall | (128, 0, 0) - 深紅色 |
| 2 | door | (0, 128, 0) - 深綠色 |
| 3 | roof | (128, 128, 0) - 橄欖色 |
| 4 | floor | (0, 0, 128) - 深藍色 |
| 5 | outside | (128, 0, 128) - 紫色 |
| 6 | cargo | (0, 128, 128) - 青色 |
| 7 | person | (128, 128, 128) - 灰色 |

## 🔧 進階使用

### 載入已訓練的模型

```python
from train import load_pt_model

# 載入 PyTorch 模型
model, processor, config = load_pt_model('segformer_model.pt', device='cuda')

# 使用模型進行預測
# ... 您的推理代碼
```

### 自定義類別

要修改類別定義，請同步更新以下文件中的 `ID2LABEL` 字典：
- `train.py`
- `inference.py`

確保：
1. 類別 ID 從 0 開始連續編號
2. 遮罩數據中的像素值對應正確的類別 ID
3. 顏色調色盤有足夠的顏色定義

## 📈 性能監控

訓練過程中可使用 TensorBoard 監控：

```bash
tensorboard --logdir ./segformer-finetuned-custom
```

## 🐛 常見問題

### 1. CUDA 記憶體不足
- 減少 `per_device_train_batch_size` 和 `per_device_eval_batch_size`
- 使用 CPU 進行推理：`--device cpu`

### 2. ONNX 推理失敗
- 確保安裝了 `onnxruntime`：`pip install onnxruntime`
- 對於 GPU 推理：`pip install onnxruntime-gpu`

### 3. 圖片和遮罩數量不匹配
- 檢查 `data/train/images/` 和 `data/train/masks/` 中的文件數量
- 確保文件名稱對應（除了副檔名）

### 4. 類別預測錯誤
- 檢查遮罩圖片中的像素值是否在 0-7 範圍內
- 確認 `ID2LABEL` 定義與實際數據匹配

## 📝 許可證

本專案採用 [LICENSE](LICENSE) 許可證。

## 🤝 貢獻

歡迎提交 Issues 和 Pull Requests！

## 📧 聯絡

如有問題，請通過 GitHub Issues 聯絡。

---

**🎉 祝您使用愉快！**
