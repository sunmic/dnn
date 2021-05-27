import torch
import urllib
from clearml import Task


task = Task.init(project_name="DNNPWr", task_name="GPUResNet50Inference")
task.execute_remotely("default")

if not torch.cuda.is_available():
    raise RuntimeError('Cannot run this cell without GPU runtime.')

gpu_model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
gpu_model.eval()
gpu_model = gpu_model.to('cuda')

# !wget -O corgi.jpg \
#     https://farm4.staticflickr.com/1301/4694470234_6f27a4f602_o.jpg

urllib.request.urlretrieve("https://farm4.staticflickr.com/1301/4694470234_6f27a4f602_o.jpg", "corgi.jpg")

from PIL import Image
img = Image.open('corgi.jpg')

batch_size = 32

from torchvision import transforms
norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
inv_norm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    norm,
])
image_tensor = preprocess(img)

gpu_input_tensor = image_tensor.unsqueeze(0)
gpu_input_batch = gpu_input_tensor.repeat(batch_size, 1, 1, 1).to('cuda')

import numpy as np
import time

times = []
for i in range(256):
    start_time = time.time()
    gpu_output_batch = gpu_model(gpu_input_batch)
    times.append((time.time() - start_time) * 1000)

print('GPU Inference times: p50={:.2f}ms p90={:.2f}ms p95={:.2f}ms'.format(
    np.percentile(times, 50), np.percentile(times, 90),
    np.percentile(times, 95)))