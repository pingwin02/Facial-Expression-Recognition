import os
import pickle
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "input", ".cache")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "pkl_previews")

MAX_VIDEO_NAME_LEN = 32
VIDEO_NAME_KEEP = 12


def list_cache_files(cache_dir):
    if not os.path.isdir(cache_dir):
        return []

    files = [
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.lower().endswith(".pkl") and os.path.isfile(os.path.join(cache_dir, name))
    ]
    files.sort(key=lambda path: os.path.basename(path).lower())
    return _dedupe_cache_files(files)


def _parse_cache_filename(filename):
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    match = re.match(r"^(?P<dataset>.+?)_seed(?P<seed>\d+)_v(?P<ver>\d+)(?P<rest>.*)$", stem)
    if not match:
        return {"raw": base}

    info = match.groupdict()
    rest = info.pop("rest") or ""
    for chunk in rest.split("__"):
        if not chunk:
            continue
        if "-" not in chunk:
            continue
        key, value = chunk.split("-", 1)
        info[key] = value
    return info


def _cache_key_without_class(info):
    return (
        info.get("dataset"),
        info.get("seed"),
        info.get("ver"),
        info.get("nf"),
        info.get("tr"),
        info.get("te"),
    )


def _dedupe_cache_files(files):
    preferred = {}
    order = []

    for path in files:
        info = _parse_cache_filename(path)
        if "dataset" not in info:
            order.append(("raw", path))
            continue

        key = _cache_key_without_class(info)
        cls_value = str(info.get("cls", "")).strip().lower()
        if key not in preferred:
            preferred[key] = (path, cls_value)
            order.append(("key", key))
            continue

        current_path, current_cls = preferred[key]
        if current_cls != "all" and cls_value == "all":
            preferred[key] = (path, cls_value)

    result = []
    for tag, value in order:
        if tag == "raw":
            result.append(value)
        else:
            result.append(preferred[value][0])

    return result


def _cache_row_parts(filename):
    info = _parse_cache_filename(filename)
    if "dataset" not in info:
        return [info.get("raw", os.path.basename(filename))]

    return [
        info["dataset"],
        f"v{info.get('ver', '?')}",
        f"nf={info.get('nf', '-')}",
        f"tr={info.get('tr', '-')}",
        f"te={info.get('te', '-')}",
    ]


def print_cache_files(cache_files):
    if not cache_files:
        print("No .pkl files found in the cache directory.")
        return

    rows = [
        _cache_row_parts(path)
        for path in cache_files
    ]
    max_cols = max(len(row) for row in rows)
    widths = [0] * max_cols
    for row in rows:
        if len(row) <= 1:
            continue
        for col_idx, value in enumerate(row):
            widths[col_idx] = max(widths[col_idx], len(value))

    print("Available .pkl files:")
    print("[0] all")
    for idx, row in enumerate(rows, start=1):
        if len(row) <= 1:
            print(f"[{idx}] {row[0]}")
            continue
        padded = " | ".join(value.ljust(widths[col_idx]) for col_idx, value in enumerate(row))
        print(f"[{idx}] {padded}")


def prompt_yes_no(question, default=False):
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{question} [{default_hint}]: ").strip().lower()
        if not raw:
            return bool(default)
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer with 'y' or 'n'.")


def prompt_int(question, default=0, min_value=None, max_value=None):
    while True:
        raw = input(f"{question} [default={default}]: ").strip()
        if not raw:
            return int(default)
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid integer value.")
            continue

        if min_value is not None and value < int(min_value):
            print(f"Value must be >= {int(min_value)}.")
            continue
        if max_value is not None and value > int(max_value):
            print(f"Value must be <= {int(max_value)}.")
            continue

        return value


def _to_uint_display(image):
    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Unsupported sample shape: {arr.shape}")

    if arr.shape[-1] >= 3:
        rgb = arr[:, :, :3].astype(np.float32)
    else:
        rgb = np.repeat(arr[:, :, :1].astype(np.float32), 3, axis=2)

    if np.max(rgb) > 1.5:
        rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


def _extract_mask(image):
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None

    if arr.shape[-1] >= 4:
        mask = arr[:, :, 3]
    elif arr.shape[-1] == 2:
        mask = arr[:, :, 1]
    elif arr.shape[0] >= 4 and arr.shape[-1] not in (1, 3, 4):
        mask = arr[3, :, :]
    elif arr.shape[0] == 2 and arr.shape[-1] not in (1, 2, 3, 4):
        mask = arr[1, :, :]
    else:
        return None

    mask = mask.astype(np.float32)
    if np.max(mask) > 1.0:
        max_val = float(np.max(mask))
        if max_val > 0:
            mask = mask / max_val
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _invert_label_map(label_map):
    if not isinstance(label_map, dict):
        return {}
    return {int(v): str(k) for k, v in label_map.items()}


def _labels_to_text(y_array, inv_label_map=None):
    labels = np.asarray(y_array).reshape(-1)
    inv_label_map = inv_label_map or {}
    texts = []
    for value in labels:
        try:
            label_idx = int(value)
        except (TypeError, ValueError):
            texts.append("")
            continue
        if label_idx in inv_label_map:
            texts.append(f"label={inv_label_map[label_idx]}")
        else:
            texts.append(f"label_idx={label_idx}")
    return texts


def _extract_from_dataset_cache(payload, split):
    if not (isinstance(payload, tuple) and len(payload) >= 2):
        return None

    train_part = payload[0] if isinstance(payload[0], tuple) and len(payload[0]) >= 1 else None
    val_part = payload[1] if isinstance(payload[1], tuple) and len(payload[1]) >= 1 else None
    if train_part is None and val_part is None:
        return None

    x_train = np.asarray(train_part[0]) if train_part is not None else np.array([])
    y_train = np.asarray(train_part[1]) if train_part is not None and len(train_part) >= 2 else np.array([])
    d_train = train_part[2] if train_part is not None and len(train_part) >= 3 else []
    x_val = np.asarray(val_part[0]) if val_part is not None else np.array([])
    y_val = np.asarray(val_part[1]) if val_part is not None and len(val_part) >= 2 else np.array([])
    d_val = val_part[2] if val_part is not None and len(val_part) >= 3 else []
    test_part = payload[2] if len(payload) >= 3 and isinstance(payload[2], tuple) else None
    x_test = np.asarray(test_part[0]) if test_part is not None and len(test_part) >= 1 else np.array([])
    y_test = np.asarray(test_part[1]) if test_part is not None and len(test_part) >= 2 else np.array([])
    d_test = test_part[2] if test_part is not None and len(test_part) >= 3 else []

    label_map = payload[3] if len(payload) >= 4 else {}
    inv_label_map = _invert_label_map(label_map)

    label_text_train = _labels_to_text(y_train, inv_label_map=inv_label_map)
    label_text_val = _labels_to_text(y_val, inv_label_map=inv_label_map)
    label_text_test = _labels_to_text(y_test, inv_label_map=inv_label_map)

    if split == "train":
        return x_train, list(d_train), label_text_train
    if split == "val":
        return x_val, list(d_val), label_text_val
    if split == "test":
        return x_test, list(d_test), label_text_test

    arrays = [arr for arr in (x_train, x_val, x_test) if arr.size > 0]
    if not arrays:
        return np.array([]), [], []

    x_all = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    return x_all, list(d_train) + list(d_val) + list(d_test), label_text_train + label_text_val + label_text_test


def _extract_from_checkpoint_cache(payload):
    if not isinstance(payload, dict):
        return None
    if "X" not in payload:
        return None

    x = np.asarray(payload.get("X", []))
    debugs = payload.get("debugs", [])
    y = np.asarray(payload.get("y", [])).reshape(-1)
    label_texts = _labels_to_text(y, inv_label_map={})
    return x, list(debugs), label_texts


def load_frames_and_debugs(pkl_path, split):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    is_dataset_cache = (
            isinstance(payload, tuple)
            and len(payload) >= 2
            and isinstance(payload[0], tuple)
            and isinstance(payload[1], tuple)
    )

    if split in ("train", "val", "test") and not is_dataset_cache:
        raise RuntimeError("Split previews require a dataset cache file.")

    extracted = _extract_from_dataset_cache(payload, split=split)
    if extracted is None:
        extracted = _extract_from_checkpoint_cache(payload)

    if extracted is None and isinstance(payload, np.ndarray):
        extracted = payload, [], []

    if extracted is None:
        raise RuntimeError("Unrecognized .pkl format. Expected dataset cache or checkpoint format.")

    frames, debugs, label_texts = extracted
    frames = np.asarray(frames)

    if frames.size == 0:
        raise RuntimeError("Selected .pkl file does not contain image frames to display.")

    return frames, debugs, list(label_texts)


def _safe_debug_text(debugs, idx, label_texts=None):
    label_text = None
    if label_texts is not None and idx < len(label_texts):
        candidate = str(label_texts[idx]).strip()
        if candidate:
            label_text = candidate

    debug = debugs[idx] if idx < len(debugs) and isinstance(debugs[idx], dict) else None
    if label_text is None and debug and "label_name" in debug:
        label_text = f"label={debug['label_name']}"

    lines = []
    if label_text:
        lines.append(label_text)
    if debug and "video" in debug:
        video_name = os.path.basename(str(debug["video"]))
        if video_name:
            if len(video_name) > MAX_VIDEO_NAME_LEN:
                head = video_name[:VIDEO_NAME_KEEP]
                tail = video_name[-VIDEO_NAME_KEEP:]
                video_name = f"{head}...{tail}"
            lines.append(f"video={video_name}")
    return "\n".join(lines)


def _format_tile_title(frame_idx, debug_text, max_chars=42, max_lines=2):
    if not debug_text:
        return f"frame={frame_idx}"

    lines = []
    for raw_line in str(debug_text).splitlines():
        if not raw_line:
            continue
        wrapped = textwrap.wrap(raw_line, width=max_chars) or [""]
        lines.extend(wrapped)

    if not lines:
        return f"frame={frame_idx}"

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines[-1] and not lines[-1].endswith("..."):
            lines[-1] = lines[-1][: max(0, max_chars - 3)].rstrip() + "..."

    return "\n".join(lines)


def _prepare_frames_for_display(frames, debugs, label_texts):
    arr = np.asarray(frames)

    if arr.size == 0:
        return arr, list(debugs), list(label_texts), None

    if arr.ndim == 5 and arr.shape[-1] in (1, 2, 3, 4):
        num_samples, seq_len = int(arr.shape[0]), int(arr.shape[1])
        arr = arr.reshape(num_samples * seq_len, *arr.shape[2:])

        expanded_debugs = []
        debug_list = list(debugs)
        for sample_idx in range(num_samples):
            base_debug = (
                debug_list[sample_idx]
                if sample_idx < len(debug_list) and isinstance(debug_list[sample_idx], dict)
                else {}
            )
            for seq_idx in range(seq_len):
                item = dict(base_debug)
                item["sample_idx"] = sample_idx
                item["sequence_idx"] = seq_idx
                expanded_debugs.append(item)

        expanded_labels = []
        label_list = list(label_texts)
        for sample_idx in range(num_samples):
            sample_label = label_list[sample_idx] if sample_idx < len(label_list) else ""
            for _ in range(seq_len):
                expanded_labels.append(sample_label)

        note = f"Detected sequence cache format (N,T,H,W,C) = ({num_samples},{seq_len},...). Flattened to {len(arr)} frames."
        return arr, expanded_debugs, expanded_labels, note

    if arr.ndim == 4 and arr.shape[-1] in (1, 2, 3, 4):
        return arr, list(debugs), list(label_texts), None

    if arr.ndim == 3:
        return arr[None, ...], list(debugs), list(label_texts), "Detected a single image sample; wrapped as 1 frame."

    if arr.ndim == 2:
        return (
            arr[None, ...],
            list(debugs),
            list(label_texts),
            "Detected a single grayscale sample; wrapped as 1 frame.",
        )

    raise RuntimeError(f"Unsupported frames array shape: {arr.shape}")


def _build_view_indices(total, start_index=0, max_frames=0):
    if total <= 0:
        raise RuntimeError("No frames to display.")

    start_index = int(np.clip(start_index, 0, total - 1))
    end_index = total if int(max_frames) <= 0 else min(total, start_index + int(max_frames))

    if end_index <= start_index:
        raise RuntimeError("Frame range is empty (check --start and --max-frames).")

    return list(range(start_index, end_index))


def save_frames_preview_image(
        frames,
        debugs,
        label_texts,
        title_prefix,
        output_path,
        start_index=0,
        max_frames=0,
        save_count=64,
        cols=8,
):
    total = int(len(frames))
    view_indices = _build_view_indices(total, start_index=start_index, max_frames=max_frames)

    if int(save_count) > 0 and len(view_indices) > int(save_count):
        sampled_positions = np.linspace(0, len(view_indices) - 1, int(save_count), dtype=int).tolist()
        selected_indices = [view_indices[pos] for pos in sampled_positions]
    else:
        selected_indices = view_indices

    cols = max(1, int(cols))
    rows = int(np.ceil(len(selected_indices) / float(cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows))
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    for ax_idx, frame_idx in enumerate(selected_indices):
        ax = axes[ax_idx]
        sample = frames[frame_idx]
        rgb = _to_uint_display(sample)
        ax.imshow(rgb)
        dbg = _safe_debug_text(debugs, frame_idx, label_texts=label_texts)
        subtitle = _format_tile_title(frame_idx, dbg, max_chars=42, max_lines=2)
        ax.set_title(subtitle, fontsize=6)
        ax.axis("off")

    for ax_idx in range(len(selected_indices), len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle(
        f"{title_prefix} | saved preview | shown {len(selected_indices)} of {len(view_indices)} frame(s)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path, len(selected_indices), len(view_indices)


def show_frames_interactive(
        frames, debugs, label_texts, title_prefix, start_index=0, max_frames=0
):
    total = int(len(frames))
    view_indices = _build_view_indices(total, start_index=start_index, max_frames=max_frames)
    cursor = {"pos": 0}

    first_sample = frames[view_indices[0]]
    first_rgb = _to_uint_display(first_sample)
    first_mask = _extract_mask(first_sample)
    first_overlay = first_rgb

    if first_mask is None:
        fig, ax_img = plt.subplots(1, 1, figsize=(7, 7))
        ax_mask = None
        im_img = ax_img.imshow(first_overlay)
        ax_img.axis("off")
    else:
        fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(10, 5))
        im_img = ax_img.imshow(first_overlay)
        ax_img.set_title("RGB")
        ax_img.axis("off")
        im_mask = ax_mask.imshow(first_mask, cmap="magma", vmin=0.0, vmax=1.0)
        ax_mask.set_title("Mask (channel 4)")
        ax_mask.axis("off")

    def refresh():
        global_idx = view_indices[cursor["pos"]]
        sample = frames[global_idx]
        rgb = _to_uint_display(sample)
        mask = _extract_mask(sample)
        im_img.set_data(rgb)

        text = f"{title_prefix} | frame {global_idx + 1}/{total}" f" | view {cursor['pos'] + 1}/{len(view_indices)}"
        dbg = _safe_debug_text(debugs, global_idx, label_texts=label_texts)
        if dbg:
            text = f"{text}\n{dbg}"
        fig.suptitle(text, fontsize=11)

        if ax_mask is not None:
            if mask is None:
                mask = np.zeros(rgb.shape[:2], dtype=np.float32)
            im_mask.set_data(mask)

        fig.canvas.draw_idle()

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("right", "d", "n", " "):
            cursor["pos"] = min(cursor["pos"] + 1, len(view_indices) - 1)
            refresh()
        elif key in ("left", "a", "p"):
            cursor["pos"] = max(cursor["pos"] - 1, 0)
            refresh()
        elif key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.tight_layout()
    plt.show()


def main():
    cache_dir = DEFAULT_CACHE_DIR
    cache_files = list_cache_files(cache_dir)
    print(f"Cache directory: {cache_dir}")
    print_cache_files(cache_files)

    if not cache_files:
        if not os.path.isdir(cache_dir):
            print(f"Cache directory not found: {cache_dir}")
        else:
            print(f"No .pkl files found in: {cache_dir}")
        return

    selected_paths = []
    max_index = len(cache_files)
    selection = prompt_int(
        f"Select file (0=all, 1..{max_index} for item)",
        default=0,
        min_value=0,
        max_value=max_index,
    )
    if selection == 0:
        selected_paths = list(cache_files)
        process_all = True
    else:
        selected_paths = [cache_files[selection - 1]]
        process_all = False

    start = 0
    max_frames = 0

    has_display = bool(os.environ.get("DISPLAY"))
    run_mode = "interactive" if has_display else "save"

    output_dir = DEFAULT_OUTPUT_DIR
    save_count = prompt_int("How many frames to show per preview? (0 = all)", default=64, min_value=0)
    save_cols = 8
    generate_split_previews = False
    if run_mode == "save":
        generate_split_previews = prompt_yes_no("Generate train/val/test previews?", default=False)

    if process_all and run_mode == "interactive":
        print("Processing all files in interactive mode is not supported; switching to save mode.")
        run_mode = "save"

    if process_all and max_frames == 0 and save_count > 200:
        print("Warning: processing all files with high --save-count may take longer.")

    saved_outputs = []

    for pkl_path in selected_paths:
        print(f"\nLoading: {pkl_path}")

        splits = ["all"]
        if generate_split_previews:
            splits.extend(["train", "val", "test"])

        for split in splits:
            try:
                frames, debugs, label_texts = load_frames_and_debugs(pkl_path, split=split)
            except RuntimeError as exc:
                if split == "all":
                    print(f"Skipping: {exc}")
                else:
                    print(f"Skipping split {split}: {exc}")
                continue

            frames, debugs, label_texts, preparation_note = _prepare_frames_for_display(frames, debugs, label_texts)

            if split == "all":
                print(f"Number of samples: {len(frames)}")
            else:
                print(f"Number of samples ({split}): {len(frames)}")
            if preparation_note:
                print(preparation_note)

            if run_mode == "interactive":
                print("Controls: Left/Right arrows or A/D, Space=next, Q=quit")
                title_prefix = os.path.basename(
                    pkl_path) if split == "all" else f"{os.path.basename(pkl_path)} ({split})"
                show_frames_interactive(
                    frames=frames,
                    debugs=debugs,
                    label_texts=label_texts,
                    title_prefix=title_prefix,
                    start_index=start,
                    max_frames=max_frames,
                )
                continue

            output_name = os.path.splitext(os.path.basename(pkl_path))[0]
            suffix = "" if split == "all" else f"_{split}"
            output_path = os.path.join(output_dir, f"{output_name}_preview{suffix}.png")
            title_prefix = os.path.basename(pkl_path) if split == "all" else f"{os.path.basename(pkl_path)} ({split})"
            saved_path, shown_count, available_count = save_frames_preview_image(
                frames=frames,
                debugs=debugs,
                label_texts=label_texts,
                title_prefix=title_prefix,
                output_path=output_path,
                start_index=start,
                max_frames=max_frames,
                save_count=save_count,
                cols=save_cols,
            )
            saved_outputs.append(saved_path)
            print(
                f"Saved preview image: {saved_path} "
                f"(shown {shown_count} sampled frame(s) from {available_count} selected frame(s))."
            )

    if run_mode == "save" and len(saved_outputs) > 1:
        print(f"\nSaved {len(saved_outputs)} preview file(s) to: {output_dir}")


if __name__ == "__main__":
    main()
