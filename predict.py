import argparse, json, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser(description="Run inference with a trained malaria classifier.")
    p.add_argument("--model", type=str, required=True, help="Path to SavedModel dir (e.g., outputs/best_model) or .h5 file")
    p.add_argument("--image", type=str, required=True, help="Path to image file")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--label_map", type=str, default=None, help="Path to label_map.json (optional if SavedModel contains it)")
    return p.parse_args()

def load_image(path, img_size):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def main():
    args = parse_args()
    # Load model
    model = keras.models.load_model(args.model)
    # Load label map
    label_map_path = args.label_map or os.path.join(os.path.dirname(args.model), "label_map.json")
    if os.path.isfile(label_map_path):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
    else:
        # fallback generic labels
        label_map = {0: "Class0", 1: "Class1"}

    x = load_image(args.image, args.img_size)
    preds = model.predict(x)
    probs = preds[0]
    top = int(np.argmax(probs))
    print(f"Prediction: {label_map.get(top, str(top))} (prob={probs[top]:.4f})")
    print("All class probabilities:")
    for i, p in enumerate(probs):
        print(f"  {label_map.get(i, str(i))}: {p:.4f}")

if __name__ == "__main__":
    main()
