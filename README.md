# CityNavAgent
Official repo for "CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory"

[![Code License](https://img.shields.io/badge/Code%20License-mit-green.svg)](CODE_LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

______________________________________________________________________

## 💡 Introduction

[**CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory**](<>)

Aerial vision-and-language navigation (VLN) — requiring drones to interpret natural language instructions and navigate complex urban environments — emerges as a critical embodied AI challenge that bridges human-robot interaction, 3D spatial reasoning, and real-world deployment.
Although existing ground VLN agents achieved notable results in indoor and outdoor settings, they struggle in aerial VLN due to the absence of predefined navigation graphs and the exponentially expanding action space in long-horizon exploration. In this work, we propose \textbf{CityNavAgent}, a large language model (LLM)-empowered agent that significantly reduces the navigation complexity for urban aerial VLN. 
Specifically, we design a hierarchical semantic planning module (HSPM) that decomposes the long-horizon task into sub-goals with different semantic levels. The agent reaches the target progressively by achieving sub-goals with different capacities of the LLM. Additionally, a global memory module storing historical trajectories into a topological graph is developed to simplify navigation for visited targets.
Extensive benchmark experiments show that our method achieves state-of-the-art performance with significant improvement. Further experiments demonstrate the effectiveness of different modules of CityNavAgent for aerial VLN in continuous city environments.

______________________________________________________________________

## AirVLN-E Dataset

The annotations of the enriched AirVLN-E dataset can be downloaded [here](https://drive.google.com/drive/folders/1gfnC64NlrFxotAq3Z5Q_-a_UOIOQ-OCD?usp=sharing). 
The simulator can be downloaded from [AirVLN](https://github.com/AirVLN/AirVLN/tree/main)

______________________________________________________________________

## Getting Started

### Prerequisites
- Python 3.8
- Conda Environment

### Installation
```bash
  # clone the repo
  git clone https://github.com/WeichenZh/CityNavAgent.git
  cd CityNavAgent-main

  # Create a virtual environment
  conda create -n citynavagent python=3.8
  conda activate citynavagent

  # install dependencies with pip
  pip install -r requirements.txt
```
The project directory structure is similar to AirVLN, which should be like this:
```
Project_dir/
├── CityNavAgent-main/
├── DATA/
│   ├── data
│   │   ├── aerialvln-s
│   │   ├── aerialvln-e
├── ENVs/
│   ├── env_1
│   ├── ...
```

### Demo
The code is coming soon

______________________________________________________________________

## 🙏 Acknowledgement

We have used code snippets from different repositories, especially from: AirVLN, LM_NAV, GroundingDino, and SAM. We would like to acknowledge and thank the authors of these repositories for their excellent work.
