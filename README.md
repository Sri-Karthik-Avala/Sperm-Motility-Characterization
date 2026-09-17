# Sperm Motility Characterization

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Computer Vision |
| Difficulty | Easy |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 4 |
| Last submission | 2026-08-12 |

## Problem statement

### Overview

Under a microscope a drop of semen is a field of tiny cells, and what matters clinically is not only whether a cell swims but how its swim *evolves* over the moment you watch it. This challenge asks you to read the within-clip *dynamics* of a cell's motion from a short grayscale clip and to rank cells along a five-part profile of that motion.

The target profile is deliberately not a standard motility summary. Conventional motility analysis reports whole-track averages -- one overall speed, one overall straightness across the whole path. Here, three of the five descriptors instead *contrast the early and late phases of the same clip* -- whether the cell speeds up or slows down between the two phases, and whether it changes heading between them -- and a fourth is the *direction* of travel rather than its magnitude. A routine follow-the-cell-and-average pipeline yields whole-clip means, and so reaches at most the speed and straightness axes; it misses the speed-change, turning and direction axes, which are defined by how the swim changes across the clip and cannot be read from a single whole-track average.

The hard part is therefore twofold: recovering the motion at all, and resolving it in time. You are pointed at a cell only by its position in the clip's first frame and given only the pixels, not the cell's path, so you must follow a small, low-contrast cell through a crowded field, and then split its path into an early and a late phase to read how its motion changes between them. The cells are only a few pixels across and move only a little between frames, so a model that learns to follow these cells and read their phase-to-phase dynamics does clearly better than a crude tracker.

### Task

For each query cell you are given its position `ref_cx`, `ref_cy` in the first frame of its clip and the clip's frames. Split the clip into an **earlier phase** and a **later phase**: the five descriptors are read from these phases and from the contrast between them, not from the whole clip averaged at once. You output five numbers describing the cell's motion:

- `vcl` — the cell's swimming speed in the later part of the clip, the average step length per frame there.
- `lin` — how straight the path is in the later part, the ratio of the net displacement to the path length, from wandering near 0 to perfectly straight near 1.
- `accel` — how the speed changes across the clip, the speed in the later part divided by the speed in the earlier part, above 1 if the cell speeds up.
- `turn` — how much the direction changes across the clip, the angle between the direction the cell travels in the earlier part and in the later part, from 0 for keeping course up to about 3 for a reversal.
- `vbias` — the vertical component of the cell's net direction of travel, from about minus 1 to about 1.

You are also given a fully labelled training set of clips with the true values, so you can learn to read the motion.

**Required approach.** Train a model on the provided clips, for example a learned tracker or a temporal model that follows a cell across its clip and reads its motion. Only the ranking of your estimates is scored, so you are learning to order cells by each descriptor.

### Evaluation

Submissions are scored with **MotilityScore**, higher is better, in the range 0 to 1. It is the average of five rank-agreement sub-scores, each in 0 to 1. For each of the five outputs the sub-score is the rank agreement between your estimates and the true values over the query cells, measured by Kendall's tau and clipped to the range 0 to 1:

- **Speed agreement** for `vcl`, **linearity agreement** for `lin`, **speed-change agreement** for `accel`, **turning agreement** for `turn`, and **direction agreement** for `vbias`.

**MotilityScore = the mean of the five agreements.**

Only the ordering of your estimates within each output matters, so any monotonic units are fine. This is a ranking score, not a regression error.

### Dataset

The prepared public dataset:

- `train_images.npy` — training frames, a uint8 array of shape Ntrain by 480 by 640: grayscale microscopy frames grouped into short clips of consecutive frames.
- `train_index.csv` — columns `row`, `video`, `clip`, `t`: for each training frame its source video, its clip id and its position within the clip, so you can group frames into clips and hold out whole videos when validating.
- `train_tracks.csv` — columns `clip`, `ref_cx`, `ref_cy`, `vcl`, `lin`, `accel`, `turn`, `vbias`: for each training clip, a cell seen in the clip's first frame at `ref_cx`, `ref_cy` and its five true motion descriptors.
- `test_images.npy` — test frames, a uint8 array of shape Ntest by 480 by 640.
- `test_index.csv` — columns `row`, `video`, `clip`, `t`: source video, clip id and position within the clip for each test frame.
- `mot_queries.csv` — columns `query_id`, `clip`, `ref_cx`, `ref_cy`: the query cells to characterise, each given by its `query_id` and its position in the first frame of its clip.
- `sample_submission.csv` — a valid submission in the required format.

Positions `ref_cx`, `ref_cy` are normalised to the range 0 to 1. Cells are only a few pixels across and move only a few pixels between frames, so keeping the full resolution and following a cell across the clip is what lets you read its motion.

Frames within a video are nearly identical, so hold out whole videos when you validate, using the `video` column. The test videos are entirely held out, so a validation that splits frames or clips from the same video across train and test will overestimate the test score.

### Submission

Submit a CSV with **exactly** these columns: `id`, `vcl`, `lin`, `accel`, `turn`, `vbias`.

Each row's `id` is a `query_id` taken from `mot_queries.csv`; every `query_id` must appear exactly once, and every value must be finite. Example:

```
id,vcl,lin,accel,turn,vbias
q00000,1.84,0.87,1.05,0.30,-0.42
q00001,0.63,0.31,0.80,1.90,0.75
q00002,2.55,0.94,1.20,0.10,0.05
```

Write the final submission to `./working/submission.csv`, UTF-8.

### Allowed And Prohibited

**Allowed:**

- Train any model on the provided clips: learned trackers, motion or flow estimators, recurrent or transformer temporal models, convolutional networks, and ensembling.
- Any preprocessing of the frames such as normalisation, contrast enhancement, cropping, and augmentation such as flips and rotations.
- Standard open-source deep-learning libraries such as PyTorch.

**Prohibited:**

- Do not use external datasets or any information beyond the provided files.
- Do not train on, adapt to, or fit statistics from the **test** clips, whose descriptors are not provided.
- Do not hardcode outputs or use per-id answer tables.
- Do not use external LLM APIs or any model-generated labels in your submission.
