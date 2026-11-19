---
title: Multimodal Human Activity Recognition (CUHK-X)
description: "Five-sensor PyTorch transformer that recognises 40 daily activities from 3D skeleton, wearable IMU, infrared, thermal and depth video, under a 100 MB model budget. Built at HKUST MINSys Lab for the CUHK-X Small Model Track. A 3-fold cross-subject ensemble scored 0.55 on the Kaggle leaderboard, up from 0.40."
coverHtml: /projects/cuhk-x-har/cover.html
keywords: [PyTorch, Python, Deep Learning, Transformers, Multimodal Sensor Fusion, Human Activity Recognition, Time-Series Classification, Computer Vision, Transfer Learning, Cross-Validation, Feature Engineering, NumPy, pandas, Matplotlib, Edge AI]
minorTags: ["HKUST MINSys Lab", "Research", Kaggle, "CUHK-X", IMU, "3D Skeleton", "Thermal Imaging", "Infrared", "Depth Video", "MobileNetV3", "R3D-18", "Frozen Backbones", "Auxiliary Loss", "Late Fusion", "Ensembling", "Label Smoothing", "AdamW", "Cosine Annealing", "Apple MPS", "Jupyter", "Subject-Grouped Splits", "Data Leakage", "Ablation Studies", "Model Recovery"]
order: 0
lastUpdated: 2026-11-18
---

Recognising what someone is doing (brushing teeth, writing, doing squats) from body-worn and room sensors is easy on people the model has already seen. The real problem is **new people**. This project was a 40-class activity recognition model for the CUHK-X Small Model Track, built during my summer at the HKUST MINSys Lab. The rules were a 100 MB model limit and no pretrained LLMs, with nine subjects to learn from.

**The data.** About 2,900 recorded sessions, each with up to five synchronised streams:

- a **3D skeleton**: 17 joints in metres, with no hand joints, so the wrist is the furthest point
- **five wearable IMUs** on the arms, chest and legs: accelerometer and gyroscope, resampled to 50 ms
- **infrared** and **thermal** video, in false colour
- **depth video**, as a colourised depth map

None of it is ordinary RGB, so every pretrained vision model sees out-of-distribution input.

**The model.** Each sensor gets its own transformer encoder, which reduces the stream to a single vector. The five vectors are concatenated and classified.

- **Skeleton:** 150 features per frame. Hip-centred joint positions, frame-to-frame velocity, and 16 bone vectors (each joint minus its parent).
- **IMU:** the 30 raw channels plus each sensor's acceleration and rotation magnitude. Magnitudes don't depend on orientation, so they still work when a sensor is strapped on at a slightly different angle.
- **Thermal, IR and depth video:** passed once through frozen MobileNetV3-Small and R3D-18 backbones and cached. A small transformer then runs over those embeddings with a frame-difference "motion" channel. Without it the branch mostly learned *who* was in the frame, not *what* they were doing: subject identity was 96.7% predictable from the raw thermal features.

The whole trained model is **2.0M parameters (7.7 MB)**, about 8% of the budget.

**What actually moved the score.**

1. **Fixing the evaluation came before improving the model.** The recording folders `user N` and `user N+15` turned out to be the same person. Splitting by folder put the same person on both sides of the split. That gave a validation accuracy of 0.72 that was really memorisation. I grouped splits by subject and switched to 3-fold cross-subject validation. After that, the gap between my local estimate and the leaderboard fell from 10–20 points to about 1.5, so local numbers could finally be trusted.
2. **Recovering a lost model.** The best early model (0.46) used preprocessing that had never been committed to git. I rebuilt its exact feature set and architecture from the checkpoint's saved normalisation statistics and weight shapes. One detail was invisible to a strict weight load: one branch used pre-norm transformer layers and the other post-norm. Getting it wrong scored 0.05. Getting it right reproduced the original.
3. **Giving every sensor its own loss.** Simply adding sensors made the model *worse* (0.428 → 0.407). The extra branches lowered training loss so quickly that early stopping kicked in before the skeleton branch had matured. I added an auxiliary classifier head per branch, then blended the fused and per-branch predictions with weights searched on validation. That turned the same five sensors into the best model. Ensembling the branches beat fusing them.
4. **Ensembling the folds.** The final submission averages the softmax probabilities of the three cross-validation models. That gave **0.565** cross-subject CV and **0.55** on Kaggle.

**Where it still fails.** Activities defined by body pose are recognised **78%** of the time, against **40%** for activities defined by the object in the hand. Walking scores 97% and sitting down 86%. Watching TV, playing games, writing and making a phone call are all under 15%. The common confusions are near-identical poses: phone call vs. eating, reading vs. turning pages. No sensor in the dataset can see what's being held. Three fusion architectures (concatenation, an attention bottleneck, and coarse-to-fine) all hit the same ceiling, so the next step is perception rather than architecture: segmenting the hands out of the depth video.

**What I'd keep doing.** Run several seeds before believing a change: two runs of one config differed by 5.6 points. Ablations that changed one thing at a time ruled out a lot: more IR frames, adversarial subject-invariance training, a two-stage classifier, and radar (empty in 55% of sessions). Each experiment went into a run log with its split and a note, and that log is what made it possible to rebuild everything in a single notebook after the lab's servers went away.

**Acknowledgements.** I would like to thank Professor Xiaomin Ouyang at HKUST for the advice throughout this project, and my teammate Alissa Shen for the help along the way.

**Stack:** Python, PyTorch (Apple MPS), NumPy, pandas, Matplotlib, Jupyter, torchvision (MobileNetV3-Small, R3D-18)

*Development note: the code was written with AI coding assistance (Claude). The competition prohibited building on pretrained LLMs, and no LLM is part of the model. Every network was trained from scratch, apart from the two frozen vision backbones, which the organisers approved.*
