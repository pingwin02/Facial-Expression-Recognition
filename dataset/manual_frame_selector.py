import cv2
import hashlib
import json
import math
import numpy as np
import os
import shutil
import sys
from dataclasses import dataclass

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
except Exception:
    tk = None
    ttk = None
    Image = None
    ImageTk = None

_SELECTOR_STATE = {"skip_remaining_videos": False}


def _has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _is_interactive_stdin():
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _sanitize_name(name):
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "video"


def _read_frame_rgb(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _fit_frame_for_tile(frame_rgb, tile_width=320, tile_height=180):
    if frame_rgb is None:
        return np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

    src_h, src_w = frame_rgb.shape[:2]
    scale = min(tile_width / max(1, src_w), tile_height / max(1, src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame_rgb, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    offset_x = (tile_width - resized_w) // 2
    offset_y = (tile_height - resized_h) // 2
    canvas[offset_y: offset_y + resized_h, offset_x: offset_x + resized_w] = resized
    return canvas


def _annotate_tile(tile_rgb, candidate_idx, frame_idx):
    tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(tile_bgr, (0, 0), (tile_bgr.shape[1] - 1, tile_bgr.shape[0] - 1), (0, 180, 255), 2)
    cv2.rectangle(tile_bgr, (0, tile_bgr.shape[0] - 30), (tile_bgr.shape[1], tile_bgr.shape[0]), (0, 0, 0), -1)
    text = f"C{candidate_idx} -> F{frame_idx}"
    cv2.putText(tile_bgr, text, (10, tile_bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return tile_bgr


def _build_headless_preview(video_path, num_frames):
    preview_root = os.path.join("output", "temporal_previews", "manual_selection")
    os.makedirs(preview_root, exist_ok=True)

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
    preview_dir = os.path.join(preview_root, f"{_sanitize_name(video_stem)}_{video_hash}")
    os.makedirs(preview_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None

        candidate_count = min(total_frames, max(40, num_frames * 8))
        candidate_indices = sorted(
            {int(idx) for idx in np.linspace(0, max(0, total_frames - 1), candidate_count, dtype=int)}
        )

        page_size = 20
        columns = 4
        rows = int(math.ceil(page_size / columns))
        tile_width = 320
        tile_height = 180
        page_paths = []
        manifest = []

        for page_start in range(0, len(candidate_indices), page_size):
            page_candidates = candidate_indices[page_start: page_start + page_size]
            canvas = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)

            for local_idx, frame_idx in enumerate(page_candidates):
                global_idx = page_start + local_idx
                row = local_idx // columns
                col = local_idx % columns
                frame_rgb = _read_frame_rgb(cap, frame_idx)
                tile_rgb = _fit_frame_for_tile(frame_rgb, tile_width=tile_width, tile_height=tile_height)
                tile_bgr = _annotate_tile(tile_rgb, global_idx, frame_idx)

                y1 = row * tile_height
                y2 = y1 + tile_height
                x1 = col * tile_width
                x2 = x1 + tile_width
                canvas[y1:y2, x1:x2] = tile_bgr

                manifest.append({"candidate": global_idx, "frame": int(frame_idx)})

            page_path = os.path.join(preview_dir, f"sheet_{page_start // page_size:02d}.png")
            cv2.imwrite(page_path, canvas)
            page_paths.append(page_path)

        context = {
            "video_path": video_path,
            "preview_pages": page_paths,
            "num_frames_requested": int(num_frames),
            "candidate_mapping": manifest,
        }
        context_path = os.path.join(preview_dir, "selection_context.json")
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)

        return {
            "preview_dir": preview_dir,
            "context_path": context_path,
            "page_paths": page_paths,
            "candidate_indices": candidate_indices,
            "total_frames": total_frames,
        }
    finally:
        cap.release()


def _cleanup_headless_preview(preview):
    if not preview:
        return

    preview_dir = preview.get("preview_dir")
    if preview_dir and os.path.isdir(preview_dir):
        shutil.rmtree(preview_dir, ignore_errors=True)

    preview_root = os.path.dirname(preview_dir) if preview_dir else None
    if preview_root and os.path.isdir(preview_root) and not os.listdir(preview_root):
        try:
            os.rmdir(preview_root)
        except OSError:
            pass


def _parse_manual_response(raw_response, candidate_indices, total_frames):
    raw = str(raw_response).strip()
    if raw == "":
        return [], False

    normalized = raw.lower()
    if normalized in {"skip", "auto"}:
        return [], False
    if normalized in {"all", "auto-all", "remaining"}:
        return [], True

    apply_to_remaining = False
    selection_part = raw
    if "|" in raw:
        selection_part, suffix = [part.strip() for part in raw.split("|", 1)]
        normalized_suffix = suffix.lower()
        if normalized_suffix not in {"all", "auto-all", "remaining"}:
            raise ValueError("Unknown suffix. Use '|all' to auto-fill the remaining videos.")
        apply_to_remaining = True

    if selection_part.lower().startswith("f:"):
        values = [chunk.strip() for chunk in selection_part[2:].split(",") if chunk.strip()]
        selected = []
        for value in values:
            if not value.isdigit():
                raise ValueError(f"Invalid frame index '{value}'.")
            frame_idx = int(value)
            if frame_idx < 0 or frame_idx >= total_frames:
                raise ValueError(f"Frame index {frame_idx} is outside 0..{total_frames - 1}.")
            selected.append(frame_idx)
        return sorted(set(selected)), apply_to_remaining

    values = [chunk.strip() for chunk in selection_part.split(",") if chunk.strip()]
    selected = []
    for value in values:
        if not value.isdigit():
            raise ValueError(f"Invalid candidate index '{value}'.")
        candidate_idx = int(value)
        if candidate_idx < 0 or candidate_idx >= len(candidate_indices):
            raise ValueError(f"Candidate index {candidate_idx} is outside 0..{len(candidate_indices) - 1}.")
        selected.append(candidate_indices[candidate_idx])
    return sorted(set(selected)), apply_to_remaining


def _fill_remaining_frames(video_path, num_frames, selected_frames, auto_fill_method):
    selected_unique = sorted({int(frame_idx) for frame_idx in selected_frames})
    if len(selected_unique) >= num_frames:
        return selected_unique[:num_frames]

    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if total_frames <= 0:
        return None

    remaining = num_frames - len(selected_unique)
    available = [frame_idx for frame_idx in range(total_frames) if frame_idx not in selected_unique]
    if remaining <= 0:
        return selected_unique[:num_frames]
    if not available:
        return selected_unique if auto_fill_method == "transformer" else None

    if auto_fill_method == "transformer":
        return selected_unique

    if auto_fill_method == "random":
        auto_indices = sorted(np.random.choice(available, size=min(remaining, len(available)), replace=False).tolist())
    else:
        position_indices = np.linspace(0, max(0, len(available) - 1), remaining, dtype=int).tolist()
        auto_indices = [available[pos] for pos in position_indices]

    return sorted(selected_unique + auto_indices)[:num_frames]


@dataclass
class _HeadlessSelectionResult:
    selected_frames: list
    apply_to_remaining: bool = False


def _run_headless_selector(video_path, num_frames, auto_fill_method):
    preview = _build_headless_preview(video_path, num_frames)
    if preview is None:
        print(f"Headless selector could not prepare preview assets for {video_path}. Falling back to auto selection.")
        return _HeadlessSelectionResult(selected_frames=[])

    print("\nHeadless manual frame selection mode detected (no GUI display available).")
    print(f"Video: {video_path}")
    print(f"Preview directory: {preview['preview_dir']}")
    print(f"Selection context: {preview['context_path']}")
    print("Open the generated sheet_XX.png files in VS Code to inspect candidate frames.")
    print("Each tile uses 'C -> F': C is the candidate index on the sheet, F is the real frame number in the video.")
    print("Type candidate numbers like '0,4,9', or exact frame numbers like 'f:120,240'.")
    print("Press Enter or type 'skip' to auto-fill only this video with the configured method.")
    print("Add '|all' or type 'all' to auto-fill this and all remaining videos without further prompts.")

    if not _is_interactive_stdin():
        print("Non-interactive stdin detected. Using the configured auto-fill method for manual selection.")
        _cleanup_headless_preview(preview)
        return _HeadlessSelectionResult(selected_frames=[])

    while True:
        try:
            response = input("Manual selection> ")
        except EOFError:
            response = ""

        try:
            selected_frames, apply_to_remaining = _parse_manual_response(
                response,
                preview["candidate_indices"],
                preview["total_frames"],
            )
            finalized = _fill_remaining_frames(video_path, num_frames, selected_frames, auto_fill_method)
            _cleanup_headless_preview(preview)
            return _HeadlessSelectionResult(selected_frames=finalized or [], apply_to_remaining=apply_to_remaining)
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


class ManualFrameSelectorGUI:
    def __init__(self, video_path, num_frames_to_select):
        self.video_path = video_path
        self.num_frames_to_select = num_frames_to_select
        self.selected_indices = []
        self.result = []
        self.apply_to_remaining = False

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

    def _read_frame(self, idx):
        return _read_frame_rgb(self.cap, idx)

    def _update_display(self):
        frame = self._read_frame(self.current_frame_idx)
        if frame is None:
            return

        display_h = 400
        h, w = frame.shape[:2]
        display_w = int(w * display_h / max(1, h))
        frame_resized = cv2.resize(frame, (display_w, display_h))

        img = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=display_w, height=display_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.frame_label.config(
            text=f"Frame: {self.current_frame_idx}/{self.total_frames - 1} | "
                 f"Selected: {len(self.selected_indices)}/{self.num_frames_to_select}"
        )
        self.slider.set(self.current_frame_idx)
        self._update_selected_list()

    def _update_selected_list(self):
        self.selected_listbox.delete(0, tk.END)
        for idx in sorted(self.selected_indices):
            self.selected_listbox.insert(tk.END, f"Frame {idx}")

    def _on_slider_change(self, val):
        self.current_frame_idx = int(float(val))
        self._update_display()

    def _select_frame(self):
        if self.current_frame_idx in self.selected_indices:
            return
        if len(self.selected_indices) >= self.num_frames_to_select:
            return
        self.selected_indices.append(self.current_frame_idx)
        self._update_display()

    def _remove_frame(self):
        selection = self.selected_listbox.curselection()
        if not selection:
            return
        idx_in_list = selection[0]
        sorted_indices = sorted(self.selected_indices)
        if idx_in_list < len(sorted_indices):
            self.selected_indices.remove(sorted_indices[idx_in_list])
            self._update_display()

    def _confirm(self):
        self.result = sorted(self.selected_indices)
        self.root.destroy()

    def _auto_current(self):
        self.result = sorted(self.selected_indices)
        self.root.destroy()

    def _auto_remaining(self):
        self.result = sorted(self.selected_indices)
        self.apply_to_remaining = True
        self.root.destroy()

    def _step_forward(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self._update_display()

    def _step_backward(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._update_display()

    def run(self):
        self.root = tk.Tk()
        self.root.title(f"Manual Frame Selection - {self.video_path}")

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main_frame, width=640, height=400)
        self.canvas.pack(pady=5)

        self.frame_label = ttk.Label(main_frame, text="", font=("Arial", 11))
        self.frame_label.pack(pady=3)

        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, pady=5)

        ttk.Button(slider_frame, text="<", command=self._step_backward, width=3).pack(side=tk.LEFT)
        self.slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=max(0, self.total_frames - 1),
            orient=tk.HORIZONTAL,
            command=self._on_slider_change,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(slider_frame, text=">", command=self._step_forward, width=3).pack(side=tk.LEFT)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text="Select Frame", command=self._select_frame).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_frame).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Confirm This Video", command=self._confirm).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Auto This Video", command=self._auto_current).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Auto This + Remaining", command=self._auto_remaining).pack(side=tk.LEFT, padx=3)

        list_frame = ttk.LabelFrame(main_frame, text="Selected Frames", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.selected_listbox = tk.Listbox(list_frame, height=6)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True)

        self._update_display()
        self.root.mainloop()

        self.cap.release()
        return self.result, self.apply_to_remaining


def manual_select_frames(video_path, num_frames, auto_fill_method="uniform"):
    if _SELECTOR_STATE["skip_remaining_videos"]:
        return _fill_remaining_frames(video_path, num_frames, [], auto_fill_method)

    if tk is not None and ttk is not None and Image is not None and ImageTk is not None and _has_display():
        try:
            gui = ManualFrameSelectorGUI(video_path, num_frames)
            selected, apply_to_remaining = gui.run()
            finalized = _fill_remaining_frames(video_path, num_frames, selected, auto_fill_method)
            if apply_to_remaining:
                _SELECTOR_STATE["skip_remaining_videos"] = True
            return finalized
        except tk.TclError as exc:
            print(f"GUI manual selection unavailable ({exc}). Falling back to headless mode.")

    headless_result = _run_headless_selector(video_path, num_frames, auto_fill_method)
    if headless_result.apply_to_remaining:
        _SELECTOR_STATE["skip_remaining_videos"] = True
    return headless_result.selected_frames
