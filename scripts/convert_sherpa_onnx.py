#!/usr/bin/env python3
import os
import shutil
# Copyright (c)  2024  Xiaomi Corporation
# Author: Fangjun Kuang

from typing import Dict

import huggingface_hub
import numpy as np
import onnx
def load_tokens():
    ans = dict()
    i = 0
    with open(os.path.join(huggingface_hub.snapshot_download("JunHowie/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"), "tokens.json"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if '[' in line: continue
            if ']' in line: continue
            if '"' in line and ',' in line:
              line = line[1:-2]

            ans[i] = line.strip()
            i += 1
    print('num tokens', i)
    return ans


def write_tokens(tokens: Dict[int, str]):
    with open("output_paraformer_finetune/tokens.txt", "w", encoding="utf-8") as f:
        for idx, s in tokens.items():
            f.write(f"{s} {idx}\n")

def load_cmvn():
    neg_mean = None
    inv_stddev = None

    with open("scripts/am.mvn") as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def load_lfr_params():
    with open("output_paraformer_finetune/config.yaml", encoding="utf-8") as f:
        for line in f:
            if "lfr_m" in line:
                lfr_m = int(line.split()[-1])
            elif "lfr_n" in line:
                lfr_n = int(line.split()[-1])
                break
    lfr_window_size = lfr_m
    lfr_window_shift = lfr_n
    return lfr_window_size, lfr_window_shift


def get_vocab_size():
    if not os.path.exists("output_paraformer_finetune/tokens.txt"):
        tokens = load_tokens()
        write_tokens(tokens)
    with open("output_paraformer_finetune/tokens.txt", encoding="utf-8") as f:
        return len(f.readlines())


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.
    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)
    print(f"Updated {filename}")


def main():
    lfr_window_size, lfr_window_shift = load_lfr_params()
    neg_mean, inv_stddev = load_cmvn()
    vocab_size = get_vocab_size()

    meta_data = {
        "lfr_window_size": str(lfr_window_size),
        "lfr_window_shift": str(lfr_window_shift),
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "model_type": "paraformer",
        "version": "1",
        "model_author": "iic",
        "vocab_size": str(vocab_size),
        "description": "This is a Chinese model. It supports only Chinese",
        "comment": "JunHowie/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "git_tag": "v1.1.9",
        "url": "https://huggingface.co/JunHowie/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    }
    add_meta_data("output_paraformer_finetune/model.onnx", meta_data)
    os.makedirs("sherpa_onnx_repackaged/", exist_ok=True)
    shutil.copy("scripts/am.mvn", f"sherpa_onnx_repackaged")
    shutil.copy("output_paraformer_finetune/model.onnx", f"sherpa_onnx_repackaged")
    shutil.copy("output_paraformer_finetune/tokens.txt", f"sherpa_onnx_repackaged")


if __name__ == "__main__":
    main()
