# ChordHelper

This repository contains the model training pipeline and datasets developed as part of the ChordHelper application. It represents the V2 approach to intelligent music composition, focused on learning harmonic structure from real-world pop song progressions.

## Initiative
As a music composer and guitarist, I’ve grown increasingly constrained by relying on simple triad-based progressions common in pop music. I want to introduce richer harmonic movement — expanding basslines, adding color tones, and creating more expressive musical textures.

To achieve this, I aim to leverage a transformer-based language model trained on pop chord progressions. By learning both common and advanced harmonic patterns from data, the model can generate intelligent chord suggestions that support and inspire the composition process

## Goal
This repo aims to build a scalable, distributed training framework for transformer-based sequence modeling, focusing on efficient multi-GPU training and deeper understanding of modern large-model training infrastructure.

The core objective is to implement and compare different distributed data parallel (DDP) strategies — including manual gradient synchronization, asynchronous gradient reduction, and bucket-based communication — to explore how gradient flow, autograd hooks, and communication scheduling affect performance and scalability.

## Example of chord generation
![](/asset/ChordGeneration.PNG)

## Quick start

clone this repo to local, make sure you have at least 2 GPUs. I rented 2 only

- Start with default DDP (without bucket):
```sh
torchrun --nproc_per_node 2 train.py
```

- Start with DDP with Bucket:
```
torchrun --nproc_per_node 2 train.py --ddp_impl bucket
```


## Training 


### DDP without bucekt
![](./asset/DDP%20without%20bucket.PNG)

### DDP with bucekt
![](./asset/DDP%20bucket.PNG)