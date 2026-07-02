import os
import argparse
import json
import re
import unicodedata
import librosa
import random

# ============ 标点/文本清洗 ============

# 中英文常见标点符号（可按需增减）
_PUNCTUATION_CHARS = (
    "，。！？、；：“”‘’《》【】（）…—～·"
    ",.!?;:\"'()\\[\\]{}<>\\-~`@#$%^&*_+=|/\\\\"
)
_PUNCTUATION_PATTERN = re.compile(f"[{re.escape(_PUNCTUATION_CHARS)}]")


def clean_transcript(text: str, to_halfwidth: bool = True) -> str:
    """
    清洗训练文本：
    1. 全角转半角（可选，默认开启）
    2. 去除中英文标点符号
    3. 合并多余空白，首尾去空格
    注意：保留英文单词之间的空格（不会破坏 " " in transcript 的判断逻辑）
    """
    if to_halfwidth:
        text = "".join(
            chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E else ch
            for ch in text
        )
        text = text.replace("\u3000", " ")  # 全角空格 -> 半角空格

    text = unicodedata.normalize("NFKC", text)
    text = _PUNCTUATION_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def _find_metadata_path(data_dir):
    for name in ("metadata.csv", "transcript.csv"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_one_dataset(data_dir, target_column, clean_text, dataset_tag=None):
    """
    读取单个 LJSpeech 格式数据集目录，返回 entry 列表。
    dataset_tag: 用于给 key 加前缀，避免多数据集合并时 key 冲突。
    """
    wav_dir = os.path.join(data_dir, "wavs")
    metadata_path = _find_metadata_path(data_dir)

    if metadata_path is None:
        print(f"错误: 找不到 metadata.csv 或 transcript.csv 在 {data_dir}，跳过该数据集")
        return []

    entries = []
    skipped_missing_audio = 0
    skipped_empty_text = 0

    print(f"正在处理数据集: {data_dir} ...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue

            audio_id = parts[0]
            transcript = parts[1]

            if clean_text:
                transcript = clean_transcript(transcript)

            if not transcript:
                skipped_empty_text += 1
                continue

            # 路径逻辑
            wav_path = os.path.abspath(os.path.join(wav_dir, f"{audio_id}"))

            if not os.path.exists(wav_path) and not wav_path.endswith(".wav"):
                if os.path.exists(wav_path + ".wav"):
                    wav_path += ".wav"

            if not os.path.exists(wav_path):
                skipped_missing_audio += 1
                print(f"警告: 找不到音频文件 {wav_path}")
                continue

            # 直接使用官方 source_len 逻辑
            source_len = parse_context_length_logic(wav_path)

            # 对应 scp2jsonl.py 第 94 行 target_len 逻辑
            target_len = len(transcript.split()) if " " in transcript else len(transcript)

            key = f"{dataset_tag}_{audio_id}" if dataset_tag else audio_id

            entry = {
                "key": key,
                "source": wav_path,
                "source_len": source_len,
                target_column: transcript,
                f"{target_column}_len": target_len,
            }
            entries.append(entry)

    print(
        f"  -> 完成: {len(entries)} 条有效样本 "
        f"(跳过缺失音频 {skipped_missing_audio} 条, 跳过空文本 {skipped_empty_text} 条)"
    )
    return entries


def convert_ljspeech_to_funasr_jsonl(
    data_dirs,
    output_dir,
    split_ratio=0.9,
    target_column="target",
    clean_text=True,
    val_size=None,
):
    """
    将一个或多个 LJSpeech 格式数据集合并转换为 FunASR 训练所需的 jsonl 格式。
    并采用与官方工具一致的 source_len 和 target_len 计算方式。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_data = []
    multi = len(data_dirs) > 1

    for idx, data_dir in enumerate(data_dirs):
        # 多数据集时用目录名（或序号兜底）做 key 前缀，避免不同数据集 id 撞车
        tag = None
        if multi:
            base = os.path.basename(os.path.normpath(data_dir))
            tag = base if base else f"ds{idx}"
        entries = load_one_dataset(data_dir, target_column, clean_text, dataset_tag=tag)
        all_data.extend(entries)

    if not all_data:
        print("错误: 没有读取到任何有效数据，请检查数据目录/路径是否正确")
        return

    # 检查合并后 key 是否有重复（理论上加了前缀不会重复，做个兜底提示）
    keys = [e["key"] for e in all_data]
    dup = len(keys) - len(set(keys))
    if dup > 0:
        print(f"警告: 合并后发现 {dup} 个重复 key，请检查各数据集内部是否本身就有重复 audio_id")

    # 简单随机打散数据集，确保训练集和验证集分布均匀
    random.seed(42)  # 固定种子保证结果一致
    random.shuffle(all_data)

    # 划分训练集和验证集
    # 优先使用 val_size（验证集绝对条数），未指定时才退回按比例 split_ratio 切分
    total = len(all_data)
    if val_size is not None:
        if val_size <= 0:
            print(f"警告: val_size={val_size} 不合法，退回使用 split={split_ratio} 按比例切分")
            val_count = total - int(total * split_ratio)
        elif val_size >= total:
            print(
                f"警告: val_size={val_size} >= 总样本数 {total}，"
                f"退回使用 split={split_ratio} 按比例切分"
            )
            val_count = total - int(total * split_ratio)
        else:
            val_count = val_size
    else:
        val_count = total - int(total * split_ratio)

    train_data = all_data[:-val_count] if val_count > 0 else all_data
    val_data = all_data[-val_count:] if val_count > 0 else []

    def save_jsonl(data_list, filename):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for entry in data_list:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"生成: {path} (共 {len(data_list)} 条)")

    print(f"\n合并完成，共 {len(all_data)} 条样本（来自 {len(data_dirs)} 个数据集）")
    save_jsonl(train_data, "train.jsonl")
    save_jsonl(val_data, "val.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LJSpeech to FunASR JSONL format converter (支持多数据集合并 + 自动去标点)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        nargs="+",
        help="LJSpeech 数据集根目录，可传多个，用空格分隔，例如: "
        "--data_dir data/ds1 data/ds2 data/ds3",
    )
    parser.add_argument("--output_dir", type=str, default="data/list", help="输出目录")
    parser.add_argument(
        "--split", type=float, default=0.95, help="训练集比例（指定了 --val_size 时此参数会被忽略）"
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=None,
        help="验证集的绝对样本条数，指定后优先生效（忽略 --split 比例）。"
        "适合大数据集场景，避免按比例切分导致验证集过大、浪费训练数据，例如 --val_size 2000",
    )
    parser.add_argument(
        "--target_name", type=str, default="target", help="JSONL 中文本列的名称 (通常为 target 或 text)"
    )
    parser.add_argument(
        "--keep_punctuation",
        action="store_true",
        help="保留文本中的标点符号（默认会自动去除标点，做微调训练建议不要加这个参数）",
    )

    args = parser.parse_args()
    convert_ljspeech_to_funasr_jsonl(
        args.data_dir,
        args.output_dir,
        args.split,
        args.target_name,
        clean_text=not args.keep_punctuation,
        val_size=args.val_size,
    )