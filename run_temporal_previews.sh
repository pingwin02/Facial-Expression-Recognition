#!/bin/bash

problematic_videos=(
  "input/devemo+/devemo+_001.mp4"
  "input/devemo+/devemo+_032.mp4"
  "input/devemo+/devemo+_132.mp4"
  "input/devemo/usf7ldu1-1p94-gov0-mo78-btn8guj2g0r0-zgoda_S_PLAYER1_o3_neutral_37.655-42.247.mkv"
  "input/devemo/uv51snt7-haxx-tgb8-66r5-yfs0ov20x4vo-zgoda_F_PLAYER1_o3_neutral_0-3.044.mkv"
)

for video in "${problematic_videos[@]}"; do
  echo "Preview: ${video}"
  python misc/temporal_overlay_preview.py --video "${video}"
done

python misc/view_cache_pkl_frames.py