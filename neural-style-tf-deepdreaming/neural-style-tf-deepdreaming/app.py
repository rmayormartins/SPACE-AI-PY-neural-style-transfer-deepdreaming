import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Conf
IMAGE_SIZE = (256, 256)
style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

#  (VGG19)
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
vgg.eval()

# 
def load_image_pytorch(image, transform=None):
    if transform:
        image = transform(image).unsqueeze(0)
    return image

# 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Deep Dreaming
def deep_dream(model, image, iterations, lr):
    for i in range(iterations):
        image.requires_grad = True
        features = model(image)
        loss = features.norm()
        loss.backward()
        with torch.no_grad():
            image += lr * image.grad
            image = image.clamp(0, 1)
        image.grad = None
    return image

# 
def load_image_tensorflow(image):
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)[np.newaxis, ...] / 255.
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

# 
def apply_sharpness(image, intensity):
    kernel = np.array([[0, -intensity, 0],
                       [-intensity, 1 + 4 * intensity, -intensity],
                       [0, -intensity, 0]])
    sharp_image = cv2.filter2D(image, -1, kernel)
    return np.clip(sharp_image, 0, 255)

# 
def interpolate_images(baseline, target, alpha):
    return baseline + alpha * (target - baseline)

# + Deep Dreaming e Transferência de Estilo
def combine_images(image1, image2, weight1, style_density, content_sharpness):
    # Carregar e pré-processar as imagens
    image1_pil = Image.fromarray(image1)
    image2_pil = Image.fromarray(image2)
    
    image1_pytorch = load_image_pytorch(image1_pil, transform)
    image2_pytorch = load_image_pytorch(image2_pil, transform)
    
    # 
    weight2 = 1 - weight1
    mixed_image_pytorch = weight1 * image1_pytorch + weight2 * image2_pytorch
    
    # Deep Dreaming
    mixed_image_pytorch = deep_dream(vgg, mixed_image_pytorch, iterations=20, lr=0.01)
    
    # 
    mixed_image_pytorch = mixed_image_pytorch.squeeze(0).permute(1, 2, 0).detach().numpy()
    mixed_image_pytorch = (mixed_image_pytorch * 255).astype(np.uint8)
    
    # 
    content_image = load_image_tensorflow(mixed_image_pytorch)
    style_image = load_image_tensorflow(image2)
    
    # 
    content_image_sharp = apply_sharpness(content_image[0], intensity=content_sharpness)
    content_image_sharp = content_image_sharp[np.newaxis, ...]
    
    # Transferência de Estilo
    stylized_image = style_transfer_model(tf.constant(content_image_sharp), tf.constant(style_image))[0]
    
    # 
    stylized_image = interpolate_images(baseline=content_image[0], target=stylized_image.numpy(), alpha=style_density)
    stylized_image = np.array(stylized_image * 255, np.uint8)
    stylized_image = np.squeeze(stylized_image)
    
    return stylized_image

# 
example1 = np.array(Image.open("example1.jpg"))
example2 = np.array(Image.open("example2.jpg"))

# Gradio
interface = gr.Interface(
    fn=combine_images,
    inputs=[
        gr.Image(type="numpy", label="Imagem 1"),  
        gr.Image(type="numpy", label="Imagem 2"),    
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Peso da Imagem 1"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Densidade do Estilo"),
        gr.Slider(minimum=0, maximum=10, value=1, label="Nitidez do Conteúdo")
    ],
    outputs="image",
    title="Combinação de Imagens com Deep Dreaming e Transferência de Estilo",
    description="Ajuste os pesos e a densidade do estilo para combinar e estilizar as imagens. *v1.0 experimental 07/07/2024",
    examples=[
        ["example1.jpg", "example2.jpg", 0.5, 0.5, 1]
    ]
)

interface.launch()
