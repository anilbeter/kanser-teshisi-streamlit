import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# Modelin her seferinde yeniden yüklenmesini önlemek için cache'leme.
@st.cache_resource
def load_model():
    model_path = "kanser_teshis_modeli.keras"
    model = tf.keras.models.load_model(model_path)
    return model


# Modeli yükle
model = load_model()


# Görüntüyü modelin istediği formata getiren yardımcı fonksiyon
def preprocess_image(image, target_size=(50, 50)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# --- STREAMLIT ARAYÜZ KODLARI ---
st.title("Meme Kanseri Doku Görüntüsü Sınıflandırma")
st.header("Yapay Zeka Destekli Kanser Teşhis Prototipi")
st.write(
    """
Bu uygulama, bir doku görüntüsünü analiz ederek kanserli olup olmadığına dair bir tahminde bulunur. 
Aşağıdaki hazır örnekleri deneyebilir veya kendi görüntünüzü yükleyebilirsiniz.
**Not: Bu sadece bir prototiptir ve gerçek tıbbi teşhis için kullanılamaz.**
"""
)

st.markdown("---")

# --- ÖRNEK GÖRÜNTÜLER BÖLÜMÜ ---
st.subheader("Örnek Görüntülerle Deneyin")

SAMPLE_IMAGE_DIR = "sample_images"
sample_image_files = os.listdir(SAMPLE_IMAGE_DIR)

cols = st.columns(len(sample_image_files))

for i, image_file in enumerate(sample_image_files):
    with cols[i]:
        image_path = os.path.join(SAMPLE_IMAGE_DIR, image_file)
        sample_image = Image.open(image_path)
        # --- DÜZELTME 1 ---
        st.image(sample_image, caption=image_file, use_container_width=True)

        if st.button("Analiz Et", key=f"btn_{image_file}"):
            st.write(f"**{image_file}** analiz ediliyor...")
            processed_image = preprocess_image(sample_image)
            prediction = model.predict(processed_image)
            probability = prediction[0][0]
            if probability > 0.5:
                st.error(f"**Tahmin: Kanserli Doku** (Olasılık: {probability:.2%})")
            else:
                st.success(f"**Tahmin: Normal Doku** (Olasılık: {1-probability:.2%})")

st.markdown("---")

# --- DOSYA YÜKLEME BÖLÜMÜ ---
st.subheader("Veya Kendi Görüntünüzü Yükleyin")
uploaded_file = st.file_uploader(
    "Lütfen bir doku resmi yükleyin (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # --- DÜZELTME 2 ---
    st.image(image, caption="Yüklenen Görüntü", use_container_width=True)

    st.write("")
    st.write("Sınıflandırma yapılıyor...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    probability = prediction[0][0]
    if probability > 0.5:
        st.error(f"**Tahmin: Kanserli Doku** (Olasılık: {probability:.2%})")
    else:
        st.success(f"**Tahmin: Normal Doku** (Olasılık: {1-probability:.2%})")
