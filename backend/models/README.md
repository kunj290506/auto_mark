# Model Checkpoints

Place the following model files in this directory:

## Required: Grounding DINO

Download `groundingdino_swint_ogc.pth` from:
https://github.com/IDEA-Research/GroundingDINO/releases

Direct link:
```
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## Optional: SAM (Segment Anything)

Download `sam_vit_b_01ec64.pth` from:
https://github.com/facebookresearch/segment-anything#model-checkpoints

Direct link:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Download Commands

```bash
# Grounding DINO
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# SAM (optional)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Expected Files

After downloading, this directory should contain:
```
models/
├── groundingdino_swint_ogc.pth  (✅ Required)
├── sam_vit_b_01ec64.pth         (Optional - for segmentation)
└── README.md
```
