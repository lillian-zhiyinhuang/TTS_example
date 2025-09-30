# ==============================================================================
# 檔案：inference.py
# 描述：一份專為高效能 TTS (文字轉語音) 設計的推理引擎。
#        此腳本將模型封裝於一個類別中，並應用多項優化技術，以實現快速、
#        低延遲的語音生成，特別適合批次處理任務。
#
# 核心策略：
# 1. 極致性能優化 (Performance Optimization)：
#    - 4-bit 量化載入：使用 `BitsAndBytes` 以 NF4 格式載入模型，大幅降低 VRAM
#      佔用，使得在消費級硬體上運行大型模型成為可能。
#    - JIT 即時編譯 (torch.compile)：利用 PyTorch 2.0+ 的 `torch.compile`
#      功能，將模型計算圖轉換為優化的底層核心，顯著減少 Python 解釋器的開銷，
#      提升生成速度。
# 2. 物件導向封裝 (Object-Oriented Encapsulation)：
#    - 將所有模型 (LLM、Tokenizer、SNAC) 和相關邏輯封裝在 `OptimizedOrpheusTTS`
#      類別中。這確保了模型僅在初始化時載入一次，避免了在處理多個請求時的
#      重複載入開銷。
# 3. 批次推理流程 (Batch Processing Workflow)：
#    - 主程式區塊採用非互動式的批次處理模式，一次性讀取所有待生成的句子，
#      並在單一引擎實例上循環處理，最大化硬體利用率並攤銷模型載入成本。
# 4. 精準的音訊後處理 (Precise Audio Post-processing)：
#    - 包含從生成 Tokens 中精確提取音訊碼流的邏輯，以及使用 `pyrubberband`
#      進行音高不變的語速調整。
# ==============================================================================

import torch
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from pathlib import Path
from typing import List, Tuple
from opencc import OpenCC
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class OptimizedOrpheusTTS:
    """
    經過性能優化的 TTS 生成器，採用 4-bit 量化和 torch.compile 加速。
    
    Attributes:
        model (torch.nn.Module): 經過量化與編譯的因果語言模型。
        tokenizer (AutoTokenizer): 文本分詞器。
        snac_model (SNAC): 用於將音訊 Tokens 解碼回波形的 SNAC 模型。
        cc (OpenCC): 用於簡繁轉換的工具。
    """
    def __init__(self, model_id: str, tokenizer_id: str, snac_id: str = "hubertsiuzdak/snac_24khz"):
        """
        初始化 TTS 引擎，包括載入、量化和編譯模型。
        
        Args:
            model_id (str): Hugging Face Hub 上的模型 ID 或本地路徑。
            tokenizer_id (str): Hugging Face Hub 上的分詞器 ID 或本地路徑。
            snac_id (str): Hugging Face Hub 上的 SNAC 模型 ID。
        """
        print(">>> 正在初始化模型與 Tokenizer (已啟用性能優化)...")
        
        # --- 1. 設定 4-bit 量化組態 ---
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f">>> 使用 4-bit 量化載入，計算精度為: {compute_dtype}")
        
        # --- 2. 載入量化後的模型 ---
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" # 自動將模型分配到可用硬體
        )
        
        # --- 3. JIT 即時編譯模型 ---
        # `torch.compile` 是 PyTorch 2.0 的核心功能，它將 Python 程式碼轉換為
        # 高效的底層程式碼，能大幅提升運算速度。
        # - mode="reduce-overhead": 減少每次操作之間的 Python 框架開銷，對小批量特別有效。
        # - fullgraph=True: 嘗試將整個模型編譯成一個單一的計算圖，如果成功，能獲得最大加速。
        print(">>> 正在使用 torch.compile() 編譯模型以獲取極致性能...")
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
            print("✅ 模型編譯成功。")
        except Exception as e:
            # 編譯可能因硬體不支援或模型架構複雜而失敗，此處作回退處理。
            print(f"⚠️ 模型編譯失敗，將使用未編譯版本 (速度較慢)。錯誤: {e}")

        # --- 4. 載入其餘輔助模型 ---
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        # SNAC 解碼在 CPU 上進行即可，因為它在 LLM 生成之後執行，計算量不大。
        self.snac_model = SNAC.from_pretrained(snac_id).cpu()
        self.cc = OpenCC('t2s.json')
        
        print("✅ 所有模型均已成功載入並優化。")

    def _prepare_prompts_for_batch(self, prompts: List[str], voice: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        準備批次化的提示詞，包括簡繁轉換、格式化和左填充。
        
        🎯 左填充 (Left Padding) 的重要性：
        對於自回歸模型 (Autoregressive Models) 的批次生成，必須使用左填充。
        這確保了每個序列的結尾 (即模型開始生成新 Token 的地方) 都是對齊的，
        讓 GPU 可以高效地並行處理所有序列的下一步預測。
        """
        simplified_prompts = [self.cc.convert(p) for p in prompts]
        formatted_prompts = [f"{voice}: {p}" for p in simplified_prompts]
        
        # 定義提示詞模板所需的特殊 Tokens
        start_token = torch.tensor([[128259]], dtype=torch.long) # <|start_of_human|>
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.long) # <|end_of_text|><|end_of_human|>
        
        # 將所有提示詞轉換為包含特殊 Tokens 的張量
        all_modified_input_ids = []
        for p in formatted_prompts:
            input_ids = self.tokenizer(p, return_tensors="pt").input_ids
            modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
            all_modified_input_ids.append(modified_ids)
            
        # 計算批次中最長序列的長度，並進行左填充
        max_length = max(ids.shape[1] for ids in all_modified_input_ids)
        all_padded_tensors, all_attention_masks = [], []
        padding_token_id = 128263 # 必須與訓練時使用的 pad_token_id 一致
        
        for ids in all_modified_input_ids:
            padding_size = max_length - ids.shape[1]
            # 在左側填充 padding_token_id
            padded_tensor = torch.cat([torch.full((1, padding_size), padding_token_id, dtype=torch.long), ids], dim=1)
            # Attention mask 同樣在左側補 0
            attention_mask = torch.cat([torch.zeros((1, padding_size), dtype=torch.long), torch.ones((1, ids.shape[1]), dtype=torch.long)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)
            
        # 將單個張量合併成一個批次
        return torch.cat(all_padded_tensors, dim=0), torch.cat(all_attention_masks, dim=0)

    def _decode_and_redistribute(self, generated_ids_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        從模型生成的 Token 序列中提取音訊 Codes，並將其重新分配回 SNAC 所需的多層格式。
        此函式是 `prepare_dataset.py` 中 `tokenise_audio` 的逆向操作。
        """
        token_start_speech, token_end_speech = 128257, 128258
        output_waveforms = []

        for row in generated_ids_batch:
            # 1. 定位 <|start_of_speech|> 和 <|end_of_speech|>，提取中間的音訊碼
            start_indices = (row == token_start_speech).nonzero(as_tuple=True)[0]
            end_indices = (row == token_end_speech).nonzero(as_tuple=True)[0]
            
            if len(start_indices) == 0 or len(end_indices) == 0:
                output_waveforms.append(torch.tensor([]))
                continue
            
            # 提取最後一組語音 Tokens
            audio_codes_flat = row[start_indices[-1] + 1 : end_indices[-1]]
            
            # 2. 清理並確保長度是 7 的倍數 (每個音訊幀由 7 個 Token 組成)
            new_length = (audio_codes_flat.size(0) // 7) * 7
            trimmed_codes = audio_codes_flat[:new_length]
            final_codes = [t.item() - 128266 for t in trimmed_codes] # 減去 ID 偏移量
            
            if not final_codes:
                output_waveforms.append(torch.tensor([]))
                continue

            # 3. 將扁平化的 Token 序列 "解交錯" (de-interleave) 回三個層次
            layer_1, layer_2, layer_3 = [], [], []
            num_frames = len(final_codes) // 7
            for i in range(num_frames):
                base = 7 * i
                layer_1.append(final_codes[base])
                layer_2.append(final_codes[base + 1] - 4096)
                layer_3.append(final_codes[base + 2] - (2 * 4096))
                layer_3.append(final_codes[base + 3] - (3 * 4096))
                layer_2.append(final_codes[base + 4] - (4 * 4096))
                layer_3.append(final_codes[base + 5] - (5 * 4096))
                layer_3.append(final_codes[base + 6] - (6 * 4096))
            
            # 4. 使用 SNAC 模型將三層 Codes 解碼回音訊波形
            codes_for_snac = [torch.tensor(layer, dtype=torch.int32).unsqueeze(0) for layer in [layer_1, layer_2, layer_3]]
            with torch.no_grad():
                audio_hat = self.snac_model.decode(codes_for_snac)
            output_waveforms.append(audio_hat)
            
        return output_waveforms

    def synthesize(self, prompt: str, voice: str, output_path: str, speed_rate: float = 1.0):
        """
        對單一提示詞進行端到端的語音合成，並儲存為 WAV 檔案。

        Args:
            prompt (str): 要合成的文本。
            voice (str): 使用的聲音角色名稱 (必須與訓練資料一致)。
            output_path (str): WAV 檔案的儲存路徑。
            speed_rate (float, optional): 語速調整比例。>1 加速，<1 減速。預設為 1.0。
        """
        print(f"\n>>> 正在合成: '{prompt}'")
        input_ids, attention_mask = self._prepare_prompts_for_batch([prompt], voice)
        input_ids, attention_mask = input_ids.to(self.model.device), attention_mask.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1200,      # 最大生成 Token 數，防止無限生成
                do_sample=True,          # 啟用採樣，增加生成多樣性
                temperature=0.4,         # 控制生成的隨機性，越低越確定
                top_p=0.65,              # 核採樣，限制採樣範圍
                repetition_penalty=1.5,  # 重複懲罰，避免生成重複詞語
                eos_token_id=128258,     # <|end_of_speech|> 作為結束符
            )
        
        # 將生成的 Tokens 解碼回音訊
        output_samples = self._decode_and_redistribute(generated_ids.to("cpu"))
        
        if output_samples and output_samples[0].numel() > 0:
            audio_numpy = output_samples[0].squeeze().numpy()
            
            # 應用語速調整
            if speed_rate != 1.0:
                print(f"    >>> 正在調整語速為: {speed_rate}x")
                audio_numpy = pyrb.time_stretch(y=audio_numpy, sr=24000, rate=speed_rate)
            
            sf.write(output_path, audio_numpy, 24000)
            print(f"✅ 音訊已成功儲存至: {output_path}")
        else:
            print("⚠️ 警告：未生成有效音訊。")


# ==============================================================================
# 步驟 4: 主程式執行區塊 (Main Execution Block)
# ==============================================================================
if __name__ == '__main__':
    # --- 組態設定 ---
    MODEL_ID = "your-hf-username/your-finetuned-model-name"
    TOKENIZER_ID = "your-hf-username/your-finetuned-model-name"
    
    # `CHOSEN_VOICE` 必須與您訓練資料 metadata.txt 中的 speaker_name 完全一致！
    CHOSEN_VOICE = "語者1"

    # 在此處定義您想一次性生成的句子列表
    PROMPTS_TO_GENERATE = [
        "大家好，歡迎來到今天的職涯分享。",
        "這款產品的設計理念是簡約與實用性的完美結合。",
        "在快速變化的市場中，我們必須保持敏捷並持續創新。",
        "祝大家有個美好的一天！"
    ]

    # 設定輸出資料夾和語速
    OUTPUT_DIR = "./output_batch"
    SPEED_RATE = 1.0 # 1.0 為原速, > 1.0 加速, < 1.0 減速

    try:
        print("--- 啟動批次處理推理任務 ---")
        # 1. 僅在程式啟動時載入和優化模型一次
        tts_engine = OptimizedOrpheusTTS(model_id=MODEL_ID, tokenizer_id=TOKENIZER_ID)
        
        # 2. 建立輸出資料夾
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        # 3. 遍歷句子列表並逐一生成音訊
        for i, prompt in enumerate(PROMPTS_TO_GENERATE):
            # 為每個檔案產生一個唯一的名稱
            output_filename = f"{OUTPUT_DIR}/batch_output_{CHOSEN_VOICE}_{i+1:03d}.wav"
            tts_engine.synthesize(prompt, CHOSEN_VOICE, output_filename, speed_rate=SPEED_RATE)

        print("\n--- ✅ 所有批次任務已完成 ---")

    except Exception as e:
        import traceback
        print("\n❌ 程式執行時發生嚴重錯誤！")
        traceback.print_exc()