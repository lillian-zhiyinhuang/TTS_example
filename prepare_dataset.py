# ==============================================================================
# 檔案：prepare_dataset.py
# 描述：一份完整、穩健且可重複的語音生成資料集預處理流程。
#        此腳本將本地的音訊-文字對資料，轉換為適用於大型語言模型訓練的格式，
#        並將最終產物上傳至 Hugging Face Hub。
#
# 核心策略：
# 1. 數據一致性 (Data Consistency)：
#    - 透過 OpenCC 將所有文字稿統一轉換為簡體中文，消除繁簡差異。
#    - 應用響度均勻化 (Loudness Normalization)，將所有音訊的音量標準化到 -23 LUFS，
#      避免模型因音量差異而產生學習偏差。
# 2. 神經音訊壓縮 (Neural Audio Compression)：
#    - 使用 SNAC (Scalable Neural Audio Codec) 模型，將連續的音訊波形轉換為離散的
#      整數序列 (Tokens)，使音訊能像文字一樣被 Transformer 模型處理。
# 3. 結構化提示詞 (Structured Prompting)：
#    - 將文字與音訊 Tokens 嵌入一個固定的對話模板中，包含各種特殊 Tokens
#      （如 <|start_of_human|>），精確地構建出模型微調時所需的輸入格式。
# 4. 記憶體效率 (Memory-Efficient Processing)：
#    - 採用 Python 生成器 (Generator) 逐筆讀取本地資料，再透過 datasets.from_generator
#      建立資料集，避免一次性將大量音訊載入記憶體。
# 5. 雲端資產化 (Cloud-Native Artifacts)：
#    - 將整個預處理流程自動化，最終產出一個版本化、可追蹤、並託管於 Hugging Face Hub
#      的「訓練就緒」(Training-Ready) 資料集。
# ==============================================================================

import os
import torch
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import torchaudio.transforms as T
from tqdm import tqdm
from opencc import OpenCC
from snac import SNAC
from datasets import Dataset, Audio
from transformers import AutoTokenizer

# ==============================================================================
# 步驟 1: 組態設定 (Global Configuration)
# 說明：
# 集中管理所有路徑、名稱與核心參數，便於修改與追蹤。
# ==============================================================================
# 本地資料路徑
METADATA_PATH = "./data/metadata.txt"
AUDIO_DIR = "./data/wavs"

# Hugging Face Hub 目標位置
# 建議每次執行時使用新的名稱或版本號，以區分不同預處理設定下的產物。
HF_DATASET_NAME = "your-hf-username/your-dataset-name"

# 文本分詞器名稱 (必須與模型訓練時使用的分詞器一致)
TOKENIZER_NAME = "canopylabs/3b-zh-pretrain-research_release"

# 響度均勻化目標 (單位: LUFS - Loudness Units Full Scale)
# LUFS 是一種衡量音訊感知響度的國際標準。
# -23.0 LUFS 是 EBU R 128 廣播標準，能提供足夠的動態範圍 (Dynamic Range)，
# 避免音訊過響導致削波 (Clipping)，或過輕導致信噪比 (SNR) 降低。
# 將所有音訊標準化到同一響度，有助於模型穩定學習。
LOUDNESS_TARGET = -23.0

# ==============================================================================
# 步驟 2: 核心函式庫 (Core Function Library)
# 說明：
# 定義資料處理流程中的所有核心轉換函式。
# ==============================================================================

def normalize_loudness(waveform: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    """
    使用 pyloudnorm 將音訊波形的響度均勻化至目標 LUFS。

    🎯 運作原理：
    --------------------------------------------------------------------------
    1.  **測量 (Measure)**：使用 `pyln.Meter` 根據 ITU-R BS.1770-4 標準計算
        輸入音訊的「積分響度」(Integrated Loudness)。
    2.  **計算增益 (Calculate Gain)**：計算 `target_lufs` 與測量響度之間的差值。
    3.  **應用增益 (Apply Gain)**：將計算出的增益應用於整個波形，使其達到目標響度。
    4.  **安全檢查 (Safety Check)**：檢查增益後的波形是否存在峰值超過 1.0 (即削波)
        的情況。如果發生削波，則進行峰值歸一化 (Peak Normalization) 以防止失真。

    Args:
        waveform (np.ndarray): 輸入的音訊波形陣列。
        sample_rate (int): 音訊的取樣率。
        target_lufs (float): 目標響度值。

    Returns:
        np.ndarray: 經過響度均勻化處理的音訊波形。
    """
    # 確保音訊格式為浮點數，這是 pyloudnorm 的要求
    if not np.issubdtype(waveform.dtype, np.floating):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max

    meter = pyln.Meter(sample_rate)

    try:
        # 測量原始音訊的積分響度
        loudness = meter.integrated_loudness(waveform)
    except ValueError: 
        # 處理靜音或過短的音訊，這些音訊無法測量響度，直接返回原樣
        return waveform

    # 使用 pyloudnorm 的內建函式，安全地將音訊標準化到目標響度
    normalized_waveform = pyln.normalize.loudness(waveform, loudness, target_lufs)

    # 處理可能因增益過大而產生的削波 (clipping)
    if np.max(np.abs(normalized_waveform)) > 1.0:
        # 進行峰值歸一化，將最大振幅縮放到 1.0
        normalized_waveform = normalized_waveform / np.max(np.abs(normalized_waveform))
        
    return normalized_waveform

def tokenise_audio(waveform: np.ndarray, orig_freq: int, snac_model: SNAC) -> list[int]:
    """
    使用 SNAC 模型將音訊波形轉換為離散的 Codec Tokens。

    🎧 運作原理：
    --------------------------------------------------------------------------
    SNAC 是一個神經音訊編解碼器，它學習如何將複雜的音訊波形「壓縮」成幾個
    平行的、離散的整數序列 (稱為 Codebooks 或 Quantizers)。
    此過程類似於文字的 "Tokenization"，將連續的訊號轉換為模型可以理解的離散單位。

    - **多流編碼 (Multi-stream Encoding)**：SNAC 輸出多個 Codec 碼流，
      每個碼流捕捉了音訊不同層次的特徵 (例如，粗糙的音高、細微的音色等)。
    - **交錯組合 (Interleaving)**：此函式遵循官方的交錯模式，將來自不同碼流的
      Codes 依序組合，形成最終的單一 Token 序列。
    - **ID 偏移 (ID Offsetting)**：為每個碼流的 Code ID 加上一個固定的偏移量，
      以確保來自不同碼流的 Token 不會重疊，它們在詞彙表中有各自獨立的範圍。

    Args:
        waveform (np.ndarray): 輸入的音訊波形。
        orig_freq (int): 原始取樣率。
        snac_model (SNAC): 預載入的 SNAC 模型。

    Returns:
        list[int]: 代表原始音訊的離散 Token ID 列表。
    """
    waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)

    # SNAC 模型要求輸入為 24kHz，若不符則進行重採樣
    if orig_freq != 24000:
        resample_transform = T.Resample(orig_freq=orig_freq, new_freq=24000)
        waveform = resample_transform(waveform)
        
    waveform = waveform.unsqueeze(0).to("cuda")

    with torch.inference_mode():
        # `snac_model.encode` 返回一個包含多個碼流的 Tuple
        codes = snac_model.encode(waveform)
        
    # 遵循官方的交錯模式組合來自不同碼流的 Codes
    all_codes = []
    # codes[0] 是第一個碼流，shape=(1, T)
    # codes[1] 是第二個碼流，shape=(1, 2T)
    # codes[2] 是第三個碼流，shape=(1, 4T)
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + 128266)
        all_codes.append(codes[1][0][2 * i].item() + 128266 + 4096)
        all_codes.append(codes[2][0][4 * i].item() + 128266 + (2 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 1].item() + 128266 + (3 * 4096))
        all_codes.append(codes[1][0][(2 * i) + 1].item() + 128266 + (4 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 2].item() + 128266 + (5 * 4096))
        all_codes.append(codes[2][0][(4 * i) + 3].item() + 128266 + (6 * 4096))
        
    return all_codes

def create_input_ids(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    將文本和音訊 Tokens 結構化為模型訓練所需的最終輸入格式。

    💬 提示詞模板 (Prompt Template)：
    --------------------------------------------------------------------------
    此函式嚴格遵循目標模型所規定的對話式輸入模板。模型的性能高度依賴於
    微調和推理時輸入格式的絕對一致。

    結構如下：
    <|start_of_human|> {speaker}: {text} <|end_of_text|> <|end_of_human|>
    <|start_of_ai|> <|start_of_speech|> {audio_codes} <|end_of_speech|> <|end_of_ai|>

    - **特殊 Tokens**：
      - `128259 <|start_of_human|>`: 人類用戶發言開始。
      - `128009 <|end_of_text|>`: 文本內容結束。
      - `128260 <|end_of_human|>`: 人類用戶發言結束。
      - `128261 <|start_of_ai|>`: AI 助理回應開始。
      - `128257 <|start_of_speech|>`: AI 的語音部分開始。
      - `128258 <|end_of_speech|>`: AI 的語音部分結束。
      - `128262 <|end_of_ai|>`: AI 助理回應結束。

    Args:
        example (dict): 包含 "text" 和 "codes_list" 的單筆資料。
        tokenizer (AutoTokenizer): 文本分詞器。

    Returns:
        dict: 增加了 "input_ids", "labels", "attention_mask" 的資料。
    """
    # 定義所有需要的特殊 Token ID
    end_of_text = 128009
    start_of_human, end_of_human = 128259, 128260
    start_of_ai, end_of_ai = 128261, 128262
    start_of_speech, end_of_speech = 128257, 128258

    # 將文本轉換為 Token IDs
    text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
    text_ids.append(end_of_text)

    # 按照模板組合所有部分
    input_ids = (
        [start_of_human] + text_ids + [end_of_human]
        + [start_of_ai] + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech] + [end_of_ai]
    )

    # 在標準的 Causal LM 訓練中，`labels` 與 `input_ids` 相同
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)

    return example

# ==============================================================================
# 步驟 3: 主執行流程 (Main Execution Pipeline)
# ==============================================================================
def main():
    print(">>> 步驟 1/7: 初始化 SNAC 音訊編碼器、文本分詞器及簡繁轉換器...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    cc = OpenCC('t2s.json') # 用於台灣正體 -> 大陸簡體
    
    def data_generator():
        """
        一個記憶體高效的 Python 生成器，逐行讀取 metadata 並產出資料。
        這避免了在處理大型資料集時一次性將所有內容載入 RAM。
        """
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading metadata"):
                line = line.strip()
                # 簡單的資料清洗，跳過空行或格式不符的行
                if not line or line.count('|') != 2: continue
                
                file_id, text, speaker_name = line.split("|", 2)
                
                # 統一文本格式：繁轉簡 -> 組合 "發言者: 文本"
                simplified_text = cc.convert(text.strip())
                formatted_text = f"{speaker_name.strip()}: {simplified_text}"
                
                audio_path = os.path.join(AUDIO_DIR, f"{file_id}")
                if os.path.exists(audio_path):
                    yield {"audio": audio_path, "text": formatted_text}

    print("\n>>> 步驟 2/7: 從本地檔案建立初步資料集...")
    # `from_generator` 會在背後迭代 `data_generator` 來建立資料集
    ds = Dataset.from_generator(data_generator)
    # `cast_column` 會懶加載 (lazy-load) 音訊檔案，並在需要時自動重採樣到 24kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=24000))

    print(f"\n>>> 步驟 3/7: 執行響度均勻化 (目標: {LOUDNESS_TARGET} LUFS)...")
    # 使用 `.map` 方法將 `normalize_loudness` 函式應用到資料集的每一筆音訊
    ds = ds.map(lambda ex: {"audio": {"array": normalize_loudness(ex["audio"]["array"], ex["audio"]["sampling_rate"], LOUDNESS_TARGET), "sampling_rate": ex["audio"]["sampling_rate"]}})

    print("\n>>> 步驟 4/7: 執行音訊 Token 化 (此步驟將使用 GPU)...")
    # 將 `tokenise_audio` 函式應用到每一筆資料
    ds = ds.map(lambda ex: {"codes_list": tokenise_audio(ex["audio"]["array"], ex["audio"]["sampling_rate"], snac_model)})

    print("\n>>> 步驟 5/7: 結構化提示詞，建立最終的 input_ids...")
    ds = ds.map(lambda ex: create_input_ids(ex, tokenizer))

    print("\n>>> 步驟 6/7: 清理不必要的暫存欄位...")
    # 最終訓練只需要 input_ids, labels, attention_mask
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
    ds = ds.remove_columns(columns_to_remove)

    print(f"\n資料集預覽 (第 1 筆資料的 Token 數量):")
    print({k: len(v) for k, v in ds[0].items()})

    print(f"\n>>> 步驟 7/7: 上傳處理好的資料集至 Hugging Face Hub...")
    try:
        # `private=True` 確保資料集僅自己可見
        ds.push_to_hub(HF_DATASET_NAME, private=True)
        print(f"✅ 資料集上傳成功！請將以下名稱填入您的 `config.yaml` 中：")
        print(f"TTS_dataset: \"{HF_DATASET_NAME}\"")
    except Exception as e:
        print(f"❌ 上傳失敗。請確保您已運行 `huggingface-cli login` 並擁有對 `{HF_DATASET_NAME}` 的寫入權限。")
        print(f"錯誤訊息: {e}")

if __name__ == "__main__":
    main()