# Data folder

Keep the large datasets outside Git if possible.

Typical layout:

```text
 data/
 ├── raw/                 # optional local copies / symlinks
 ├── splits/              # FF++ official train.json/val.json/test.json
 ├── manifest.csv         # generated video manifest
 ├── train.csv            # generated video split
 ├── val.csv              # generated video split
 ├── test.csv             # generated video split
 ├── frame_manifest.csv   # generated face-frame manifest
 ├── faces/               # generated face crops
 └── features/            # generated CLIP/content feature caches
```

The current machine can keep the FF++ C23 data at:

```text
C:/Users/Adhiraj/Downloads/SUBHAM
```

Update `configs/base.yaml` if the location changes.
