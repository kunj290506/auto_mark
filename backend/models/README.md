# Model Checkpoints

Place the following model files in this directory:

## Required: SAM (Segment Anything)

Download `sam_vit_b_01ec64.pth` from:
https://github.com/facebookresearch/segment-anything#model-checkpoints

Direct link:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Grounding DINO

Grounding DINO is loaded automatically from HuggingFace (`IDEA-Research/grounding-dino-base`).
No local checkpoint is needed — it downloads on first run.

## Download Commands

```bash
# SAM (required for segmentation)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Expected Files

After setup, this directory should contain:
```
models/
├── sam_vit_b_01ec64.pth  (Required - for segmentation)
└── README.md
```
