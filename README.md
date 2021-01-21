# Large Scale GPS Trajectory Generation Using Map based on Two-stage-GAN

Python impletement of paper Large Scale GPS Trajectory Generation Using Map based on Two-stage-GAN

<img src = "https://github.com/XingruiWang/Two-Stage-Gan-in-trajectory-generation/blob/main/Figure/pipeline.png?raw=1" style="width: 50%;"/>

### Introduction

we propose a map-based Two-Stage GAN method (TSG) to generate fine-grained and plausible large-scale trajectories. In the first stage, we first transfer GPS points data to discrete grid representation as the input for a modified deep convolutional generative adversarial network to learn the general pattern. In the second stage, inside each grid, we design an effective encoder-decoder network as the generator to extract road information from map image and then embed it into two parallel Long Short-Term Memory networks to generate GPS point sequence.

<img href = "pipeline.png"></img>

### Result

We evaluate the synthetic trajectories in terms of their similarity to real data, i.e., distribution of overall GPS coordinate, distribution of trajectory sequences length, distribution of trajectory distance, top-N visited places and road networks matching accuracy. And we compare our result with the previous benchmark.

| Model | $p_o(r)$ | $p_s(l)$ | $p_d(l) |
| ---- | ---- | ---- | ---- |
| FTS-IP | 0.413 | 0.182| 0.187|
| LSTM | 0.633 | **0.058** | 0.140|
| TSG | **0.100** | 0.139 | **0.136**|

### Installation







