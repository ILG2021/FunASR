# Paraformer-zh 微调指南 (LJSpeech 格式数据集)

本指南详细介绍了如何使用 LJSpeech 格式的数据集（两栏：ID|Text）对 FunASR 的 Paraformer-zh 模型进行微调。

## 0. 安装依赖
```bash
python 3.10
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -e .
```



## 1. 目录结构建议

建议将数据集和脚本组织如下：
```text
FunASR/
├── scripts/
│   └── ljspeech_to_funasr.py  # 格式转换脚本
├── data/
│   └── ljspeech/              # 原始 LJSpeech 数据
│       ├── wavs/
│       └── metadata.csv
└── output_paraformer/         # 训练结果输出
```

## 2. 数据格式转换 (一步到位生成 JSONL)

使用提供的脚本将 LJSpeech 格式直接转换为 FunASR 训练所需的 `jsonl` 格式，并自动划分训练集与验证集。

```bash
python scripts/ljspeech_to_funasr.py \
  --data_dir data/ljspeech \
  --output_dir data/list \
  --split 0.95 \
  --target_name "target"
```

执行后，将在 `data/list` 目录下直接生成：
- `train.jsonl` (包含 key, source, target)
- `val.jsonl`

此步骤替换了原有的 `scp` + `scp2jsonl` 的繁琐过程。

## 4. 启动微调 (Fine-tuning)

推荐使用 `torchrun` 进行分布式或单卡训练。以下是微调参数配置建议：

```bash
export CUDA_VISIBLE_DEVICES="0" # 根据实际情况指定 GPU ID

torchrun --nproc_per_node 1 \
-m funasr.bin.train_ds \
++model="paraformer-zh" \
++train_data_set_list="data/list/train.jsonl" \
++valid_data_set_list="data/list/val.jsonl" \
++dataset_conf.batch_size=20000 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=20 \
++train_conf.log_interval=10 \
++train_conf.validate_interval=1000 \
++train_conf.save_checkpoint_interval=1000 \
++train_conf.early_stopping_patience=3 \
++train_conf.keep_nbest_models=10 \
++optim_conf.lr=0.00002 \
++output_dir="./output_paraformer_finetune"
```

### 参数详解：
- `++model`: 可填写 "paraformer-zh"（自动下载）或本地模型目录。
- `++dataset_conf.batch_type="token"`: 动态 Batch，按 token 数量（帧数/字符数）计算。
- `++dataset_conf.batch_size=20000`: 显存消耗的关键。若发生 OOM，请调小此值（如 10000）。
- `++optim_conf.lr=0.00002`: 微调建议使用较小的学习率。
- `++train_conf.max_epoch`: 训练轮数。

## 5. 推理与验证

微调结束后，可以直接加载输出目录中的模型进行测试：

```python
from funasr import AutoModel

# 加载微调后的模型
model = AutoModel(model="./output_paraformer_finetune")

res = model.generate(input="data/ljspeech/wavs/test_sample.wav")
print(res)
```

## 6. 注意事项
- **显存保护**：如果音频包含极长样本（如超过 30s），建议在转换前进行切分，或在训练参数中增加 `++dataset_conf.max_token_length=2000` 过滤。
- **环境依赖**：确保安装了 `deepspeed` (由于脚本使用了 `train_ds.py`，即使不开启 deepspeed 也建议环境中有相关依赖库)。
