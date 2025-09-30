# ==============================================================================
# 檔案：train_lora.py
# 描述：一份完整、高效且可重複的 TTS 微調腳本。
# 採用 QLoRA 技術，在有限的硬體資源下對大型模型進行參數高效微調。
# 核心策略：
# 1. 組態驅動 (Configuration-Driven)：
# - 所有超參數、路徑和實驗名稱皆由 config.yaml 統一管理，確保實驗的可追蹤性與可重複性。
# 2. 記憶體效率 (Memory Efficiency)：
# - 透過 BitsAndBytes 函式庫實現 4-bit 量化載入 (NF4)，大幅降低模型載入時的 VRAM 佔用。
# - 啟用梯度檢查點 (Gradient Checkpointing)，以時間換空間，進一步減少訓練時的記憶體壓力。
# 3. 參數高效微調 (Parameter-Efficient Fine-Tuning, PEFT)：
# - 採用 LoRA (Low-Rank Adaptation) 技術，僅訓練一小部分「適配器」參數，而非整個模型。
# - 結合 RS-LoRA (Rank-Stabilized LoRA)，穩定訓練過程。
# 4. 訓練最佳化 (Training Optimization)：
# - 使用 paged_adamw_8bit 優化器，防止記憶體突然飆升。
# - 運用 sdpa (Scaled Dot-Product Attention) 作為 Flash Attention 2 的穩健替代方案，提升運算效率。
# 5. 自動化部署 (Automated Deployment)：
# - 訓練完成後，自動將 LoRA 適配器與基礎模型合併。
# - 可選地，將最終合併完成的模型一鍵推送至 Hugging Face Hub，方便後續推理與分享。
# ==============================================================================
import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)

def main():
    """
    主執行函式，涵蓋了從載入組態到模型上傳的完整 LoRA 微調流程。
    """
    # ==========================================================================
    # 步驟 1: 載入組態檔 (Configuration Loading)
    # 說明：
    # `config.yaml` 是本專案的「單一事實來源」(Single Source of Truth)。
    # 將所有可變參數集中管理，有助於：
    #   - 快速迭代：只需修改 YAML 檔即可發起新的實驗，無需變動程式碼。
    #   - 提升可讀性：程式碼專注於邏輯，參數設定一目了然。
    #   - 保證可重複性：同一份 config.yaml 應能產出相同的結果。
    # ==========================================================================
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("✅ 組態檔載入成功。")
    print(f"--- 實驗名稱: {config['run_name']} ---")

    # ==========================================================================
    # 步驟 2: 載入模型 (Model) 與分詞器 (Tokenizer)
    # 說明：
    # 此階段將根據組態檔的設定，從 Hugging Face Hub 下載預訓練模型與對應的分詞器。
    # 核心是應用 QLoRA 技術，在載入模型的同時進行 4-bit 量化處理。
    # ==========================================================================
    base_model_id = config["base_model_id"]
    tokenizer_id = config["tokenizer_id"]

    print(f">>> 正在載入模型: {base_model_id}")
    print(f">>> 正在載入 Tokenizer: {tokenizer_id}")

    # --- 2.1 載入分詞器 ---
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # [關鍵設定] 明確設定填充 Token (Padding Token)
    # 模型在處理批次資料時，需要將長度不一的序列填充到相同長度。
    # `pad_token_id` 告訴模型哪個 ID 是用於填充的，計算損失時應忽略此 Token。
    # 此處的 ID (128263) 必須與資料準備階段所使用的 ID 完全一致。
    tokenizer.pad_token_id = 128263
    print(f"✅ Tokenizer `pad_token_id` 已設定為 {tokenizer.pad_token_id}")

    # --- 2.2 設定 4-bit 量化組態 (BitsAndBytesConfig) ---
    # 這是 QLoRA 的核心，用於大幅降低 VRAM 需求。
    # 量化原理：將模型的權重從標準的 32-bit 浮點數壓縮到 4-bit。

    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    quantization_config = BitsAndBytesConfig(
        # 啟用 4-bit 量化
        load_in_4bit=True,
        
        # 量化類型："nf4" (Normal Float 4)
        # 一種專為常態分佈權重設計的 4-bit 資料類型，比標準 4-bit 浮點數更精確。
        bnb_4bit_quant_type="nf4",
        
        # 計算資料類型 (Compute Dtype)
        # 權重以 4-bit 儲存，但在前向/反向傳播計算時，會被「反量化」回 `bfloat16` 或 `float16` 進行運算，以維持精度與穩定性。
        # `bfloat16` 的動態範圍比 `float16` 更廣，是現代硬體 (如 Ampere 架構) 的首選。
        bnb_4bit_compute_dtype=torch.bfloat16 if is_bf16_supported else torch.float16,
        
        # 啟用雙重 量化 (Double Quantization)
        # 在對權重進行量化後，再對量化常數本身進行一次量化，可進一步節省約 0.4 bits/參數的記憶體。
        bnb_4bit_use_double_quant=True,
    )

    # --- 2.3 載入量化後的模型 ---
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config,
        
        # 注意力機制實作 (Attention Implementation)
        # `sdpa` (Scaled Dot Product Attention) 是 PyTorch 2.0 後內建的高效能注意力實作。
        # 它會根據硬體自動選擇最優的核心 (如 Flash Attention)，具備良好的相容性與效能。
        attn_implementation="sdpa",
        
        # 自動裝置映射 (Device Mapping)
        # `device_map="auto"` 會自動將模型的各個層分配到可用的硬體上 (例如：多張 GPU 或 GPU+CPU)，
        # 以便載入超出單張 GPU VRAM 上限的大型模型。
        device_map="auto"
    )

    # ==========================================================================
    # 步驟 3: 設定 LoRA (Low-Rank Adaptation)
    # 說明：
    # PEFT (參數高效微調) 的核心步驟。我們不直接修改原始模型的數十億個參數，
    # 而是凍結它們，並在指定的層 (通常是注意力層) 旁注入兩個可訓練的「低秩矩陣」(A 和 B)。
    # 訓練時，只更新這兩個小矩陣的權重。
    #
    # 數學原理：
    # 原始權重更新可表示為： $W' = W + \Delta W$
    # LoRA 的假設是 $\Delta W$ 具有低的「內在秩」(intrinsic rank)。
    # 因此，可將其分解為兩個較小的矩陣： $\Delta W \approx B \cdot A$，其中 $A \in \mathbb{R}^{r \times k}$ 且 $B \in \mathbb{R}^{d \times r}$，$r \ll \min(d, k)$。
    # 這樣，需要訓練的參數數量從 $d \times k$ 大幅減少到 $r \times (d+k)$。
    # ==========================================================================
    lora_config = LoraConfig(
        # LoRA 秩 (Rank, r)：低秩矩陣的維度。這是最重要的超參數之一。
        # r 越大，可訓練參數越多，模型的擬合能力越強，但同時也增加了過擬合的風險。
        # 常見取值為 8, 16, 32, 64。
        r=config["lora_rank"],
        
        # LoRA Alpha (α)：LoRA 輸出的縮放因子。
        # 最終的輸出由 `(B*A) * (alpha/r)` 決定。
        # 一般習慣將 alpha 設定為 r 的兩倍，以維持輸出的權重規模。
        lora_alpha=config["lora_alpha"],
        
        # LoRA Dropout：在 LoRA 層上應用的 Dropout 比率，用於防止過擬合。
        lora_dropout=config["lora_dropout"],
        
        # 目標模組 (Target Modules)：指定要應用 LoRA 的模型層。
        # 通常選擇 Transformer 中的注意力機制相關層 (查詢、鍵、值、輸出投影)。
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        
        # 偏置項 (Bias)：是否訓練偏置項。通常設為 "none"。
        bias="none",
        
        # 需額外保存的模組 (Modules to Save)：
        # 除了 LoRA 適配器外，需要一併保存與訓練的模型部分。
        # `lm_head` (語言模型頭) 和 `embed_tokens` (詞嵌入層) 通常需要與新任務對齊。
        modules_to_save=["lm_head", "embed_tokens"],
        
        # 任務類型 (Task Type)：指定為因果語言模型 (Causal LM)。
        task_type="CAUSAL_LM",
        
        # 使用 RS-LoRA (Rank-Stabilized LoRA)
        # 將縮放因子從 `alpha/r` 調整為 `alpha/sqrt(r)`，有助於在選擇較大的 alpha 時穩定訓練。
        use_rslora=True,
    )

    # 將 LoRA 組態應用到量化後的模型上
    model = get_peft_model(model, lora_config)

    # 顯示可訓練參數的數量與佔比，可以直觀地看到 LoRA 的效率。
    model.print_trainable_parameters()

    # ==========================================================================
    # 步驟 4: 載入預處理好的資料集
    # 說明：
    # 此腳本假設資料已經過預處理，並上傳至 Hugging Face Hub。
    # `load_dataset` 會從 Hub 直接串流或下載資料集。
    # ==========================================================================
    hf_dataset_name = config["TTS_dataset"]
    print(f">>> 正在從 Hugging Face Hub 載入預處理好的資料集: {hf_dataset_name}")
    train_dataset = load_dataset(hf_dataset_name, split="train")
    print(f"✅ 資料集載入完成。共 {len(train_dataset)} 筆資料。")

    # ==========================================================================
    # 步驟 5: 設定訓練參數並啟動訓練
    # 說明：
    # `TrainingArguments` 是一個涵蓋所有訓練選項的資料類別。
    # `Trainer` 則是 Hugging Face 提供的上層 API，封裝了標準的訓練迴圈。
    # ==========================================================================
    training_args = TrainingArguments(
        output_dir=f"./{config['save_folder']}",
        run_name=config["run_name"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        
        # 梯度累積 (Gradient Accumulation)
        # 效果：在執行一次反向傳播 (參數更新) 之前，累積多個批次的梯度。
        # 用途：在 VRAM 有限的情況下，模擬更大的批次大小。
        # 有效批次大小 (Effective Batch Size) = batch_size * gradient_accumulation_steps
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        
        learning_rate=float(config["learning_rate"]),
        
        # 優化器 (Optimizer)
        # `paged_adamw_8bit`：一種記憶體效率極高的 AdamW 優化器版本。
        # 它使用分頁技術 (Paging) 將優化器狀態從 GPU VRAM 卸載到 CPU RAM，
        # 防止在梯度計算時可能發生的記憶體尖峰 (Spike)，從而能用更少的 VRAM 訓練更大的模型。
        optim=config["optim"],
        
        # 學習率預熱 (Warmup)
        # 在訓練初期使用一個較小的學習率，然後逐漸增加到設定值。
        # 這有助於模型在訓練開始時保持穩定，避免因初始梯度過大而發散。
        warmup_steps=config["warmup_steps"],
        
        logging_steps=1,
        save_strategy="steps",
        save_steps=config["save_steps"],
        
        # 混合精度訓練 (Mixed-Precision Training)
        bf16=is_bf16_supported,
        fp16=not is_bf16_supported,
        
        # 梯度檢查點 (Gradient Checkpointing)
        # 效果：一種以計算換取記憶體的技術。
        # 原理：在前向傳播中，不儲存所有中間層的活化值 (Activations)，
        # 而是在反向傳播需要時重新計算它們。
        # 優點：能顯著降低 VRAM 佔用，允許使用更大的批次或訓練更大的模型。
        # 缺點：會增加約 20-30% 的訓練時間。
        gradient_checkpointing=True,
        
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("\n>>> 訓練即將開始...")
    trainer.train()
    print("✅ 訓練完成。")

    # ==========================================================================
    # 步驟 6: 儲存與合併模型
    # 說明：
    # LoRA 微調後，我們得到的是一個輕量的「適配器」(Adapter)。
    # 為了方便後續獨立部署與推理，需要將適配器的權重合併回原始的基礎模型中，
    # 生成一個標準的、完整的語言模型。
    # ==========================================================================

    # --- 6.1 儲存 LoRA 適配器 ---
    # `save_model` 會將可訓練的部分 (LoRA 權重、lm_head 等) 儲存到指定路徑。
    # 這些檔案非常小，通常只有幾十到幾百 MB。
    final_adapter_path = f"./{config['save_folder']}/final_adapter"
    trainer.save_model(final_adapter_path)
    print(f"✅ LoRA 適配器已儲存至: {final_adapter_path}")

    # --- 6.2 合併權重並卸載適配器 ---
    # `merge_and_unload` 函式執行以下操作：
    # 1. 將 LoRA 適配器 (矩陣 A 和 B) 的權重乘積加到原始模型的對應層上 ($W' = W + B \cdot A$)。
    # 2. 從模型中移除 LoRA 層，將其還原為一個標準的 Transformer 模型。
    # 3. 返回合併後的完整模型。
    print("\n>>> 正在合併 LoRA 適配器與基礎模型...")
    merged_model = model.merge_and_unload()

    # --- 6.3 儲存合併後的完整模型 ---
    # 使用 `save_pretrained` 將合併後的模型以標準 Hugging Face 格式儲存。
    # 這個模型可以直接被 `from_pretrained` 載入，無需任何 PEFT 相關的程式碼。
    merged_model_path = f"./{config['save_folder']}/final_merged_model"
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print(f"✅ 合併後的完整模型已儲存至: {merged_model_path}")

    # ==========================================================================
    # 步驟 7 (可選): 推送至 Hugging Face Hub
    # 說明：
    # 這是一個自動化步驟，方便將訓練成果分享或部署。
    # 如果在 `config.yaml` 中設定了 `hub_repo_id`，腳本會嘗試將合併後的模型
    # 推送到您的 Hugging Face 帳戶下。
    # ==========================================================================
    hub_repo_id = config.get("hub_repo_id")
    if hub_repo_id:
        print(f"\n>>> 偵測到 `hub_repo_id`，準備將合併後的模型推送至: {hub_repo_id}")
        print(">>> 請確保您已在終端機執行 `huggingface-cli login`")
        try:
            # 推送模型權重、組態檔與 Tokenizer
            merged_model.push_to_hub(hub_repo_id, private=True)
            tokenizer.push_to_hub(hub_repo_id, private=True)
            print(f"✅ 模型與 Tokenizer 已成功推送至 Hugging Face Hub!")
        except Exception as e:
            print(f"❌ 推送失敗。錯誤訊息: {e}")
    else:
        print("\n>>> 未在 config.yaml 中設定 `hub_repo_id`，跳過上傳步驟。")

    print("\n🚀 流程結束。您現在可以使用本地路徑或 Hugging Face ID 進行推理。")

if __name__ == "main":
    """
    🚀 執行說明：
    ------------------------------------------------------------------------------
    1. 確認 config.yaml 檔案已根據您的需求進行配置，特別是 run_name。
    2. 確保已安裝所有必要的函式庫 (transformers, peft, accelerate, bitsandbytes, etc.)。
    3. 如果計畫上傳模型，請先在您的終端機執行 huggingface-cli login 並輸入您的 Token。
    4. 執行此腳本： python train_lora.py

    📊 預期輸出：
    - 載入組態與模型的日誌訊息。
    - 可訓練參數的摘要報告。
    - 訓練過程的進度條與損失 (Loss) 變化。
    - 訓練完成後，模型合併與儲存的確認訊息。
    - (可選) 推送至 Hugging Face Hub 的進度與結果。
    """
    main()