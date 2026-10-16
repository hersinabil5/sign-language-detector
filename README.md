# ASL Word Recognition

Real-time American Sign Language word recognition from a webcam: custom hand-tracking pipeline, landmark normalization, LSTM sequence modeling, and live inference.

## Overview

Recognizes a 20-word ASL vocabulary from live video using a sequence model trained on WLASL (Word-Level ASL), the standard benchmark dataset for ASL recognition research.

**Vocabulary:** `before, cool, thin, drink, go, who, help, cousin, computer, tall, bed, thanksgiving, candy, short, accident, bowling, shirt, man, yes, basketball`

## Results

| Metric | Result |
| --- | --- |
| Validation accuracy | 55–60% across 20 classes |
| Performance vs. random baseline | 10–12x |
| Model | 2-layer LSTM, 153K parameters |
| Training data | 498 hand-tracked video sequences |

Random guessing across 20 classes lands at 5%. The gap matters more than the raw number here: with roughly 25 sequences per class, the ceiling is set by data volume rather than model capacity.

## Pipeline

```
Webcam frame
     │
     ▼
Custom HandTracker (MediaPipe Tasks Vision)
     │
     ▼
Landmark normalization — wrist-centered, scale-invariant
     │
     ▼
40-frame rolling sequence buffer
     │
     ▼
LSTM sequence classifier ──► predicted word + confidence
```

Each stage is an independent, testable component under `src/`, not a single monolithic script.

### Design decisions

**Wrist-centered, scale-invariant normalization.** Raw MediaPipe landmarks carry absolute position and apparent hand size, both of which depend on where the signer stands rather than what they sign. Translating every frame to the wrist and rescaling removes both, so the model learns hand shape and motion instead of camera framing.

**40-frame rolling buffer.** Signs are motion, not poses, so classification runs over a window rather than a single frame. The buffer advances one frame at a time, which means a prediction is available on every frame instead of only at window boundaries.

**Sequence model over frame-wise classification.** A 2-layer LSTM over the normalized landmark sequence captures the order of movement, which is what separates signs that share a hand shape but differ in direction.

## Limitations

- **One hand only.** Two-handed signs aren't supported yet, which constrains the vocabulary that can be added.
- **No face or pose tracking.** ASL grammar carries meaning in facial expression and body position; this model sees hands alone.
- **Small vocabulary.** 20 words, limited by the number of usable WLASL sequences per class.

## Tech stack

**Computer vision:** MediaPipe Tasks Vision, OpenCV
**Machine learning:** TensorFlow / Keras (LSTM), scikit-learn
**Data:** WLASL dataset, NumPy for sequence processing

## File structure

```
asl-translator/
├── app.py
├── requirements.txt
├── collect_data.py
├── explore_dataset.py
├── preprocess_videos.py
├── train_word_classifier.py
├── models/
│   ├── hand_landmarker.task
│   ├── word_classifier.h5
│   └── word_classes.json
├── src/
│   ├── hand_tracker.py         # Hand-landmark detection
│   ├── draw_utils.py           # Skeleton visualization
│   ├── landmark_utils.py       # Normalization pipeline
│   ├── dataset_collector.py    # Training data collection tooling
│   └── word_classifier.py      # LSTM inference wrapper
├── data/
├── raw_dataset/
└── tests/
    ├── test_hand_tracker.py
    └── test_word_translator.py
```

## Setup

```bash
git clone https://github.com/hersinabil5/sign-language-detector.git
cd sign-language-detector/asl-translator

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p models
curl -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

```bash
python3 tests/test_hand_tracker.py       # Live hand tracking demo
python3 tests/test_word_translator.py    # Live word recognition
```

Retrain the full pipeline:

```bash
python3 explore_dataset.py
python3 preprocess_videos.py
python3 train_word_classifier.py
```

## Roadmap

- Web interface via Flask + WebSockets for browser-based recognition
- Expanded vocabulary with additional training data
- Two-handed sign support
- Face and pose tracking for grammar-relevant signs
