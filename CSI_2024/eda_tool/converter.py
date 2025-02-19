import torch
from torchsummary import summary
import tensorflow as tf
import onnx
# from onnx_tf.backend import prepare
from CNN import CNN

# 사용자 정의 래퍼 함수
class CustomModel(torch.nn.Module):
  def __init__(self, model):
    super(CustomModel, self).__init__()
    self.model = model

  def forward(self, x):
    outputs = self.model(x)
    return outputs

model_name = "act"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model Load
torch_model = CNN(n_classes=3)
state_dict = torch.load(f'./{model_name}', map_location=torch.device(device), weights_only=True)
torch_model.load_state_dict(state_dict)
wrapped_model = CustomModel(torch_model)
wrapped_model = wrapped_model.to(device)

dummy_input = torch.randn(1, 166, 50).to(device) # 50, 166 으로 데이터가 들어감. (입력 텐서)
# summary(wrapped_model, input_size=(1, 166, 50)) # 모델 구조 확인
print("1️⃣ Torch Model Loaded successfully ")


# Torch to ONNX
onnx_path = f"{model_name}.onnx"
torch.onnx.export(wrapped_model, dummy_input, onnx_path, opset_version=12) 
onnx_model = onnx.load(f'./{model_name}.onnx')
onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph)) # 모델 구조 확인
print("2️⃣ ONNX model Conversion completed successfully")

# ONNX to TF
tf_rep = prepare(onnx_model)
tf_rep.export_graph(f'./{model_name}_tfmodel')
print("3️⃣ TensorFlow model conversion completed successfully.")

# TF to Tflite
converter = tf.lite.TFLiteConverter.from_saved_model(f'./{model_name}_tfmodel')
tflite_model = converter.convert()

with open(f'./{model_name}.tflite', "wb") as f:
    f.write(tflite_model)

print("4️⃣ TFLite model conversion completed successfully.")