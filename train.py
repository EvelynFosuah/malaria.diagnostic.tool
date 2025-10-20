import argparse, json, os, math
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

def parse_args():
    p = argparse.ArgumentParser(description="Train a malaria classifier with TensorFlow/Keras (MobileNetV2).")
    p.add_argument("--data_dir", type=str, default="data", help="Root data dir containing train/ and optional val/")
    p.add_argument("--img_size", type=int, default=224, help="Image size (square)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--val_split", type=float, default=0.15, help="If no val/ folder, split this fraction from train/")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def make_datasets(root, img_size, batch_size, val_split, seed):
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    if val_dir.exists():
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir, image_size=(img_size, img_size), batch_size=batch_size, seed=seed
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir, image_size=(img_size, img_size), batch_size=batch_size, seed=seed, shuffle=False
        )
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir, image_size=(img_size, img_size), batch_size=batch_size, validation_split=val_split,
            subset="training", seed=seed
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir, image_size=(img_size, img_size), batch_size=batch_size, validation_split=val_split,
            subset="validation", seed=seed, shuffle=False
        )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    # Basic performance optimizations
    train_ds = train_ds.shuffle(1000).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names

def build_model(num_classes, img_size):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    base = keras.applications.MobileNetV2(
        include_top=False, input_shape=(img_size, img_size, 3), weights="imagenet"
    )
    base.trainable = False  # freeze for initial training

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base

def compute_weights(train_ds, class_names):
    # Gather labels to compute class weights
    y = []
    for _, labels in train_ds.unbatch().take(100000):  # limit for safety
        y.append(int(labels.numpy()))
    y = np.array(y)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=y)
    return {i: float(w) for i, w in enumerate(weights)}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds, class_names = make_datasets(args.data_dir, args.img_size, args.batch_size, args.val_split, args.seed)
    class_weights = compute_weights(train_ds, class_names)

    model, base = build_model(num_classes=len(class_names), img_size=args.img_size)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_model"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        ),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    # Optional fine-tuning: unfreeze top of base model
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(2, args.epochs // 2),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2
    )

    # Save label map
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)

    # Save a separate .h5 for convenience
    model.save(os.path.join(args.output_dir, "model.h5"))

    print("Training complete. Best model at:", os.path.join(args.output_dir, "best_model"))
    print("Classes:", class_names)

if __name__ == "__main__":
    main()
