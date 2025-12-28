import os
import argparse
import json
import librosa
import random
import csv

def parse_context_length_logic(line):
    """
    直接采用 funasr/datasets/audio_datasets/scp2jsonl.py 中的 source_len 获取逻辑
    """
    if os.path.exists(line):
        waveform, _ = librosa.load(line, sr=16000)
        sample_num = len(waveform)
        # 对应 scp2jsonl.py 第 89 行逻辑
        context_len = int(sample_num * 1000 / 16000 / 10)
        return context_len
    return 0

def convert_audiofolder_to_funasr_jsonl(data_dir, output_dir, split_ratio=0.95, target_column="target"):
    """
    将 AudioFolder 格式数据集转换为 FunASR 训练所需的 jsonl 格式。
    
    AudioFolder 格式预期:
    - data_dir 下有 metadata.csv
    - metadata.csv 第一行是 file_name,sentence (逗号分隔)
    - 音频文件路径相对于 data_dir
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")

    if not os.path.exists(metadata_path):
        print(f"错误: 找不到 metadata.csv 在 {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    data = []
    print("正在处理音频文件并计算长度，请稍候...")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        # 使用 csv 模块处理，自动处理第一行表头
        reader = csv.DictReader(f)
        for row in reader:
            # 兼容可能的列名变体
            file_name = row.get("file_name")
            transcript = row.get("sentence")
            
            if not file_name or not transcript:
                continue
                
            audio_id = os.path.splitext(file_name)[0]
            # 音频路径相对于 data_dir
            wav_path = os.path.abspath(os.path.join(data_dir, file_name))
            
            if os.path.exists(wav_path):
                # 直接使用官方 source_len 逻辑
                source_len = parse_context_length_logic(wav_path)
                
                # 对应 scp2jsonl.py 第 94 行 target_len 逻辑
                target_len = len(transcript.split()) if " " in transcript else len(transcript)
                
                entry = {
                    "key": audio_id,
                    "source": wav_path,
                    "source_len": source_len,
                    target_column: transcript,
                    f"{target_column}_len": target_len
                }
                data.append(entry)
            else:
                print(f"警告: 找不到音频文件 {wav_path}")

    # 简单随机打散数据集，确保训练集和验证集分布均匀
    random.seed(42)  # 固定种子保证结果一致
    random.shuffle(data)

    # 划分训练集和验证集
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    def save_jsonl(data_list, filename):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for entry in data_list:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"生成: {path} (共 {len(data_list)} 条)")

    save_jsonl(train_data, "train.jsonl")
    save_jsonl(val_data, "val.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AudioFolder to FunASR JSONL format converter (file_name,sentence)")
    parser.add_argument("--data_dir", type=str, required=True, help="AudioFolder 数据集根目录 (包含 metadata.csv)")
    parser.add_argument("--output_dir", type=str, default="data/list", help="输出目录")
    parser.add_argument("--split", type=float, default=0.95, help="训练集比例")
    parser.add_argument("--target_name", type=str, default="target", help="JSONL 中文本列的名称 (通常为 target 或 text)")
    
    args = parser.parse_args()
    convert_audiofolder_to_funasr_jsonl(args.data_dir, args.output_dir, args.split, args.target_name)
