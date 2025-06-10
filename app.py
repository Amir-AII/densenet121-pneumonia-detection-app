
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import requests

model_path = "pneumonia_densenet121.h5"

# Eğer model dosyası yoksa indir
if not os.path.exists(model_path):
    print("📥 Model indiriliyor...")
    url = "https://drive.google.com/uc?id=1bnymkLz41lUlEiV5IYIxgnI1K8wkw10q"
    r = requests.get(url, allow_redirects=True)
    with open(model_path, 'wb') as f:
        f.write(r.content)
    print("✅ Model indirildi.")

# Modeli yükle
model = load_model(model_path)

# Tahmin fonksiyonu
def predict(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "🫁 Zatürre Var" if pred > 0.5 else "✅ Zatürre Yok"

# Gradio arayüzü
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Zatürre Tespit Uygulaması",
    description="Göğüs röntgeni yükleyin, DenseNet121 modeli zatürre olup olmadığını tahmin etsin."
)

app.launch()
