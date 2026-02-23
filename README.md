# [CVPR 2026] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training

<a href="https://arxiv.org/abs/2512.13043"><img src="https://img.shields.io/badge/arXiv-2512.13043-brightgreen"></a>

## Abstract
 Multi-turn reinforcement learning (RL) for multi-modal agents built upon vision–language models (VLMs) is hampered by sparse rewards and long-horizon credit assignment. Recent methods densify the reward by querying a teacher that provides step-level feedback, e.g., Guided Thought Reinforcement (GTR) and On-Policy Distillation, but rely on costly, often privileged models as the teacher, limiting practicality and reproducibility. We introduce GTR-Turbo, a highly efficient upgrade to GTR, which matches the performance without training or querying an expensive teacher model. Specifically, GTR-Turbo merges the weights of checkpoints produced during the ongoing RLtraining, and then uses this merged model as a “free” teacher to guide the subsequent RL via supervised fine-tuning or soft logit distillation. This design removes dependence on privileged VLMs(e.g., GPT or Gemini), mitigates the “entropy collapse” observed in prior work, and keeps training stable. Across diverse visual agentic tasks, GTR-Turbo improves the accuracy of the baseline model by 10–30% while reducing wall-clock training time by 50% and compute cost by 60% relative to GTR.

## Code Structure

1. gym-cards environment for Points24 task, in accordance with RL4VLM.

2. `Turbo_P24`, code for training agent on the Points24 task.
3. `Turbo_ALF`, code for training agent on ALFWorld tasks.



<a name="getting_started"></a>

## Getting Started

Our code needs 2 GPUs to run.

### Points24

1. Setup the environment

```bash
cd <path-to-this-repo>/Turbo_P24
pip install -e ../gym-cards
pip install -r ./requirements.txt
```

2. Run the script

```bash
cd scripts
bash run_p24.sh
```

### ALFWorld

1. Setup the environment

```bash
cd <path-to-this-repo>/Turbo_ALF
conda env create -f alf_conda.yml
conda activate vrenv-alf
pip install -e ../gym-cards
pip install -r ./requirements.txt
export ALFWORLD_DATA=<storage_path>
alfworld-download
```

​	You may test the installation by running:

```bash
alfworld-play-thor
```
2. Set the ALFWORLD_DATA_DIR path and other variables in `alf-config.yaml` and `scripts/run_alf.sh`

3. Run the script

```bash
cd scripts
bash run_alf.sh
```

## Citation

If you find our work useful, please kindly cite:

```
@article{wei2025gtr,
  title={GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training},
  author={Wei, Tong and Yang, Yijun and Zhang, Changhao and Xing, Junliang and Shi, Yuanchun and Lu, Zongqing and Ye, Deheng},
  journal={arXiv preprint arXiv:2512.13043},
  year={2025}
}
```

## Acknowledgement

Our code adopts the basic environment setting and RL framework from [RL4VLM](https://github.com/RL4VLM/RL4VLM). We also refer to the code of Qwen2.5-VL in [transformer](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl).