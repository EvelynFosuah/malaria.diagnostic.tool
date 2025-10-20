import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import json, os

st.title("Malaria Diagnostic (Demo)")
st.write("Upload a blood-smear image to get a classification. **For research/education only.**")

model_path = st.text_input("Model path", "outputs/best_model")
label_map_path = os.path.join(os.path.dirname(model_path), "label_map.json")

@st.cache_resource
def load_model(path):
    return keras.models.load_model(path)

@st.cache_data
def load_label_map(path):
    if os.path.isfile(path):
        with open(path, "r") as f:
            m = json.load(f)
        return {int(k): v for k, v in m.items()}
    return {0: "Class0", 1: "Class1"}

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Input", use_column_width=True)
    x = np.array(img) / 255.0
    x = np.expand_dims(x, 0).astype(np.float32)

    try:
        model = load_model(model_path)
        labels = load_label_map(label_map_path)
        preds = model.predict(x)
        probs = preds[0]
        top = int(np.argmax(probs))
        st.subheader(f"Prediction: {labels.get(top, str(top))}")
        st.write({labels.get(i, str(i)): float(p) for i, p in enumerate(probs)})
    except Exception as e:
        st.error(f"Error loading model or predicting: {e}")
