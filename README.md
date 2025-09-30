# TTS：高效能文字轉語音微調與推理框架

`TTS example` 是一個完整的端到端（End-to-End）框架，專為高品質文字轉語音（Text-to-Speech）模型的微調與部署而設計。本專案採用最先進的參數高效微調技術（QLoRA）與推理加速方案（`torch.compile`），讓使用者能在有限的硬體資源下，訓練客製化的語音模型，並實現高效能的即時生成。

## 核心特色

-   **🚀 端到端完整流程**：涵蓋從原始音訊資料處理、模型高效微調，到最終高效能推理的完整工作流程。
-   **🛠️ 參數高效微調 (QLoRA)**：透過 4-bit 量化與低秩適應（Low-Rank Adaptation），大幅降低模型訓練所需的 VRAM，讓消費級 GPU 也能微調數十億參數的大型模型。
-   **⚡ 高效能推理引擎**：整合 `torch.compile` JIT 即時編譯與 4-bit 量化載入，最大化推理速度並減少記憶體佔用，適合批次處理與即時應用。
-   **⚙️ 模組化與組態驅動**：專案的各個階段（資料、訓練、推理）皆由獨立的腳本與中央組態檔（`config.yaml`）管理，具備高度的可重複性與可擴充性。
-   **🔊 音訊品質一致性**：內建響度均勻化（Loudness Normalization）流程，確保所有訓練資料的音量標準一致，提升模型學習的穩定性。

## 專案架構與工作流程

本框架由三個核心階段組成，依序執行：

1.  **【階段一】資料準備 (`prepare_dataset.py`)**
    -   讀取本地存放的 `wavs` 音訊檔案與 `metadata.txt` 文字稿。
    -   執行音訊預處理，包括響度均勻化。
    -   使用 SNAC 神經音訊編碼器將音訊轉換為離散 Tokens。
    -   將文本與音訊 Tokens 結構化為模型所需的輸入格式。
    -   將處理完成的「訓練就緒」資料集自動上傳至 Hugging Face Hub。

2.  **【階段二】模型微調 (`train.py`)**
    -   從 `config.yaml` 讀取所有訓練參數（如基礎模型、學習率、LoRA 設定等）。
    -   從 Hugging Face Hub 下載第一階段準備好的資料集。
    -   以 QLoRA 模式對基礎模型進行高效微調。
    -   訓練完成後，自動將 LoRA 適配器與基礎模型合併。
    -   （可選）將微調並合併完成的模型推送至 Hugging Face Hub。

3.  **【階段三】高效能推理 (`inference.py`)**
    -   以 4-bit 量化模式載入微調好的模型。
    -   使用 `torch.compile` 對模型進行 JIT 編譯以達到極致性能。
    -   提供批次處理介面，可一次性生成多個句子的語音。
    -   支援音高不變的語速調整。

---

## 環境設定與安裝

在開始之前，請確保您的系統已安裝 NVIDIA 驅動、CUDA Toolkit (建議 12.1 或更高版本) 與 Python 3.10+。

**1. 複製專案庫**
```bash
git clone https://github.com/lillian-zhiyinhuang/TTS-example
cd TTS-example
```

**2. 建立虛擬環境並安裝依賴**

```bash
# 建立 Python 虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
# venv\Scripts\activate
# 啟動虛擬環境 (Linux/macOS)
source venv/bin/activate

# 安裝所有必要的套件
pip install -r requirements.txt
```

**3. 登入 Hugging Face CLI**

此步驟是上傳資料集與模型的必要前提。

```bash
huggingface-cli login
# 依照提示貼上您的 Hugging Face Access Token (具備 write 權限)
```

**4. (可選) 登入 Weights & Biases**

如果您希望在訓練時記錄實驗數據，請登入 W\&B。

```bash
wandb login
```

-----

## 使用教學

請嚴格依照以下三階段順序執行。

### 階段一：準備資料集

1.  **準備本地資料**
    請將您的音訊與文字稿依照以下結構存放：

    ```
    .
    ├── final_voice_data/
    │   ├── wavs/
    │   │   ├── audio_001.wav
    │   │   ├── audio_002.wav
    │   │   └── ...
    │   └── metadata.txt
    ├── prepare_dataset.py
    └── ...
    ```

    其中 `metadata.txt` 的格式應為：`檔案名稱|繁體中文文本|發言人名稱`

    ```
    audio_001.wav|這是一個示範音檔。|語者1
    audio_002.wav|歡迎使用本框架。|語者1
    ```

2.  **設定腳本**
    打開 `prepare_dataset.py`，修改頂部的組態設定，最重要的是 `HF_DATASET_NAME`，它將是您上傳到 Hugging Face Hub 的資料集名稱。

    ```python
    # prepare_dataset.py
    HF_DATASET_NAME = "your-hf-username/your-dataset-name"
    ```

3.  **執行腳本**

    ```bash
    python prepare_dataset.py
    ```

    執行完畢後，腳本會將處理好的資料集上傳，並顯示資料集名稱。請務必記下此名稱。

### 階段二：微調模型

1.  **設定組態檔**
    打開 `config.yaml`，這是控制訓練的核心。請至少修改以下三個地方：

    ```yaml
    # config.yaml
    # 1. 填入您在階段一上傳的資料集名稱
    TTS_dataset: "your-hf-username/your-dataset-name" 

    # 2. 為這次的訓練取一個獨一無二的名稱
    run_name: "run-my-voice"

    # 3. (可選) 設定您要上傳模型的 Hugging Face Hub 位置
    hub_repo_id: "your-hf-username/your-finetuned-model-name"
    ```

2.  **開始訓練**

    ```bash
    python train_lora.py
    ```

    訓練過程將會顯示損失（Loss）變化，並根據設定將模型檢查點儲存至本地。訓練完成後，若設定了 `hub_repo_id`，模型將被自動上傳。

### 階段三：執行推理

1.  **設定推理腳本**
    打開 `optimized_inference_engine.py`，修改主程式區塊的設定：

    ```python
    # optimized_inference_engine.py
    if __name__ == '__main__':
        # 1. 填入您在階段二微調好的模型 ID
        MODEL_ID = "your-hf-username/your-finetuned-model-name"
        TOKENIZER_ID = "your-hf-username/your-finetuned-model-name" # 通常與模型 ID 相同

        # 2. 填入您訓練時使用的發言人名稱 (必須完全一致！)
        CHOSEN_VOICE = "語者1"

        # 3. 填入您想轉換為語音的句子
        PROMPTS_TO_GENERATE = [
            "大家好，歡迎來到今天的產品發表會。",
            "這項技術將徹底改變我們的生活方式。"
        ]

        # 4. (可選) 調整語速
        SPEED_RATE = 1.0 # >1 加速, <1 減速
    ```

2.  **執行推理**

    ```bash
    python optimized_inference_engine.py
    ```

    腳本會逐一生成音訊，並將 `.wav` 檔案儲存於 `output_batch/` 資料夾中。

## 關鍵技術棧

  - **核心框架**: PyTorch
  - **模型與社群**: Hugging Face Transformers, PEFT, Datasets, Accelerate
  - **量化與效能**: BitsAndBytes
  - **音訊處理**: Torchaudio, Librosa, Pyloudnorm, PyRubberband
  - **實驗追蹤**: Weights & Biases

<!-- end list -->
