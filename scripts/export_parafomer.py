from funasr import AutoModel
from funasr.models.bicif_paraformer.model import BiCifParaformer

# 加载微调后的模型
model = AutoModel(model="./output_paraformer_finetune")

model.export(type="onnx", quantize=False)