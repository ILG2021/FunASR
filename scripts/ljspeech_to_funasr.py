import os
import argparse

def convert_ljspeech_to_funasr(data_dir, output_dir, split_ratio=0.9):
    """
    将 LJSpeech 格式数据集转换为 FunASR 训练所需的 wav.scp 和 text.txt。
    
    LJSpeech 格式:
    - wavs/ 文件夹包含所有音频
    - metadata.csv 格式为: ID|Transcription|Normalized Transcription
    """
    wav_dir = os.path.join(data_dir, "wavs")
    metadata_path = os.path.join(data_dir, "metadata.csv")

    if not os.path.exists(metadata_path):
        # 兼容一些变体，可能在根目录或叫 transcript.csv
        metadata_path = os.path.join(data_dir, "transcript.csv")

    if not os.path.exists(metadata_path):
        print(f"错误: 找不到 metadata.csv 或 transcript.csv 在 {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    data = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            
            audio_id = parts[0]
            # LJSpeech 通常有两列文本，取第一列或第二列，这里取 index 1 (Transcription)
            transcript = parts[1]
            wav_path = os.path.abspath(os.path.join(wav_dir, f"{audio_id}.wav"))
            
            if os.path.exists(wav_path):
                data.append((audio_id, wav_path, transcript))
            else:
                print(f"警告: 找不到音频文件 {wav_path}")

    # 简单划分训练集和验证集
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    def save_to_funasr(data_list, prefix):
        wav_scp = os.path.join(output_dir, f"{prefix}_wav.scp")
        text_txt = os.path.join(output_dir, f"{prefix}_text.txt")
        
        with open(wav_scp, "w", encoding="utf-8") as f_wav, \
             open(text_txt, "w", encoding="utf-8") as f_text:
            for audio_id, wav_path, transcript in data_list:
                f_wav.write(f"{audio_id}\t{wav_path}\n")
                f_text.write(f"{audio_id}\t{transcript}\n")
        print(f"生成: {wav_scp} 和 {text_txt} (共 {len(data_list)} 条)")

    save_to_funasr(train_data, "train")
    save_to_funasr(val_data, "val")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LJSpeech to FunASR format converter")
    parser.add_argument("--data_dir", type=str, required=True, help="LJSpeech 数据集根目录")
    parser.add_argument("--output_dir", type=str, default="data/list", help="输出目录")
    parser.add_argument("--split", type=float, default=0.95, help="训练集比例")
    
    args = parser.parse_args()
    convert_ljspeech_to_funasr(args.data_dir, args.output_dir, args.split)
