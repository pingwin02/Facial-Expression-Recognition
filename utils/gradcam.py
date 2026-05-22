import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _find_target_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            result = _find_target_layer(layer)
            if result:
                try:
                    model.get_layer(result)
                    return result
                except ValueError:
                    pass
    return None


def _find_backbone_in_model(keras_model):
    best = None
    best_params = 0
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.TimeDistributed):
            inner = layer.layer
            if isinstance(inner, tf.keras.Model):
                n_params = inner.count_params()
                if n_params > best_params:
                    best = inner
                    best_params = n_params
    if best is None:
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.TimeDistributed):
                inner = layer.layer
                if hasattr(inner, "layers"):
                    for sublayer in inner.layers:
                        if isinstance(sublayer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                            return inner
    return best


def compute_gradcam_heatmap(model, input_tensor, class_idx=None):
    backbone = _find_backbone_in_model(model)
    if backbone is None:
        print("[GradCAM] Warning: No backbone found in model")
        return None

    target_layer_name = _find_target_layer(backbone)
    if target_layer_name is None:
        print("[GradCAM] Warning: No conv layer found in backbone")
        return None

    dense_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layer = layer
            break

    if dense_layer is None:
        print("[GradCAM] Warning: No Dense layer found in full model")
        return None

    if input_tensor.ndim == 4:
        input_tensor = np.expand_dims(input_tensor, axis=0)

    input_tf = tf.cast(input_tensor, tf.float32)

    num_frames = input_tensor.shape[1]
    center_frame_idx = num_frames // 2
    center_frame = input_tf[:, center_frame_idx]

    preds = model(input_tf, training=False)
    if class_idx is None:
        if preds.shape[-1] == 1:
            class_idx = int(tf.cast(preds[0, 0] > 0.5, tf.int32).numpy())
        else:
            class_idx = int(tf.argmax(preds[0]).numpy())

    target_layer = backbone.get_layer(target_layer_name)
    grad_model = tf.keras.Model(inputs=backbone.input, outputs=[target_layer.output, backbone.output])

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = grad_model(center_frame, training=False)
        tape.watch(conv_outputs)

        if len(backbone_features.shape) == 4:
            pooled = tf.reduce_mean(backbone_features, axis=[1, 2])
        else:
            pooled = backbone_features

        class_logits = dense_layer(pooled)
        class_score = class_logits[0, class_idx]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        print("[GradCAM] Warning: gradient is None - no differentiable path")
        return None

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    heatmap = cv2.resize(cam, (224, 224))
    return heatmap


def _overlay_heatmap(image, heatmap, alpha=0.4):
    img = np.squeeze(image)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] > 3:
        img = img[:, :, :3]

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (img.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha).astype(np.uint8)
    return overlay


def save_gradcam_grid(
        model,
        frames,
        preds,
        labels,
        debugs,
        output_dir,
        model_name,
        dataset_name,
        accuracy,
        filename,
        cols=3,
):
    def _pretty_class_name(value):
        return str(value).replace("_", " ")

    class_map = {}
    for debug in debugs:
        if debug and "class_map" in debug and isinstance(debug["class_map"], dict):
            class_map.update(debug["class_map"])
    if not class_map:
        unique_vals = list(sorted(set(list(labels) + list(preds))))
        for v in unique_vals:
            class_map[v] = str(v)

    n = len(frames)
    if n == 0:
        print("No frames to generate GradCAM for.")
        return

    heatmaps = []
    for i in range(n):
        frame_input = frames[i]
        if frame_input.ndim == 3:
            frame_input = np.expand_dims(frame_input, axis=0)
        heatmap = compute_gradcam_heatmap(model, frame_input, class_idx=int(preds[i]))
        heatmaps.append(heatmap)

    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    title_parts = []
    if model_name:
        title_parts.append(str(model_name))
    if dataset_name:
        title_parts.append(str(dataset_name))
    title_parts.append("GradCAM")
    if accuracy is not None:
        title_parts.append(f"Accuracy: {accuracy * 100:.2f}%")
    title = " - ".join(title_parts)
    fig.suptitle(title, fontsize=13, y=0.995)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[a] for a in axes])

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        frame = frames[idx]
        if frame.ndim == 4:
            center_t = frame.shape[0] // 2
            frame = frame[center_t]

        heatmap = heatmaps[idx]
        if heatmap is not None:
            overlay = _overlay_heatmap(frame, heatmap)
            ax.imshow(overlay)
        else:
            img = np.squeeze(frame)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            ax.imshow(img)

        pred_name = _pretty_class_name(class_map.get(preds[idx], str(preds[idx])))
        label_name = _pretty_class_name(class_map.get(labels[idx], str(labels[idx])))

        is_correct = str(pred_name) == str(label_name)
        top_caption = f"{'✓' if is_correct else '✗'} actual: {label_name}"
        top_caption_color = "#2e7d32" if is_correct else "#c62828"

        ax.text(
            0.5,
            1.03,
            top_caption,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=top_caption_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
        )

        debug = debugs[idx] if idx < len(debugs) else None
        bottom_caption = f"predicted: {pred_name}"
        if debug and isinstance(debug, dict) and debug.get("video"):
            video_display = debug["video"]
            if len(video_display) > 20:
                video_display = video_display[:17] + "..."
            bottom_caption = f"video: {video_display}\n{bottom_caption}"

        ax.text(
            0.5,
            -0.10,
            bottom_caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
        )
        ax.axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.988))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    print(f"Saving GradCAM grid PNG to {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
