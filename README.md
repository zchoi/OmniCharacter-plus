# OmniCharacter++: Towards Comprehensive Benchmark for Realistic Role-Playing Agents
> OmniCharacter++ evaluates the boundaries of today’s role-playing and character-aligned models on role-consistent multimodal interaction.
> It benchmarks 8 diverse topics with 31 subfields, _e.g._, negotiation, exchange, daily life, covering 10k+ characters, 118K dialogue samples, and 1M speech annotations.

[![Project Page](https://img.shields.io/badge/Project-Page-Green.svg)]()
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-orange.svg)]()
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg)](https://huggingface.co/datasets/haonanzhang/OmniCharacter-plus)

![intro](./intro.png)


## 🌟 Highlights of **OmniCharacter++**

| Dimension                    | Example Features                                            | Scale               |
| ---------------------------- | ----------------------------------------------------------- | ------------------- |
| **Multi-party Interaction**  | realistic open-world, topic-driven dialogues                | 118K+ dialogues     |
| **Character Diversity**      | games, fiction, public domains, internet culture            | 10K+ unique roles   |
| **Multi-modal Exchange**     | text–speech co-driven, emotional tones, varied styles       | 1M+ audio responses |
| **Comprehensive Evaluation** | context understanding, generation ability, human perception | 3-level pipeline    |

* **Large-scale benchmark**: first to support multi-party, multi-modal role-playing at scale
* **Expressive modalities**: natural speech synthesis with controllable emotions and speaking styles
* **Challenging setting**: state-of-the-art RPAs still struggle with realistic interactions
* **Plug-and-play evaluation**: unified scripts for automated metrics and human studies
* **Research advances**: baseline **UniCharacter** with emotion preference learning and role-contextual adaptation

## 🚀 Quick Start

```bash
# Clone the repo
git clone --recursive https://github.com/zchoi/OmniCharacter-plus
cd OmniCharacter-plus


# Create Conda env:
conda create -n omnicharacter-plus python=3.10 -y
conda activate omnicharacter-plus
pip install --upgrade pip  # enable PEP 660 support

# If you want to use UniCharacter, execute the following process
pip install -e ".[train]"
pip install -r requirements.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn" --no-build-isolation
```
## 📊 Statistics of OmniCharacter++
OmniCharacter++’s large-scale dataset spans multi-party, topic-driven conversations, expressive character role-playing, and text–speech co-driven interactions. It covers over 10K diverse characters from games, fiction, and public domains, engaging in 118K+ multi-turn dialogues with more than 1M synthesized audio responses that capture varied speaking styles and emotions. Together, these resources form a unified benchmark that comprehensively probes role consistency, contextual understanding, multimodal communication, and adaptive interaction in realistic open-world scenarios.

<div style="display: flex; align-items: flex-start; gap: 20px;">

<!-- 左侧表格 -->
<div>
<table>
<thead>
<tr>
<th>Set</th>
<th>Dialogue Type</th>
<th>#Characters</th>
<th>Avg. Turn/Conv.</th>
<th>#Samples</th>
<th>#Speech Hours</th>
</tr>
</thead>
<tbody>
<tr><td>Train</td><td>Dyadic</td><td>10,277</td><td>10.00</td><td>88,474</td><td>2867.94</td></tr>
<tr><td>Train</td><td>Multi-Character</td><td>10,277</td><td>15.05</td><td>29,543</td><td>1051.66</td></tr>
<tr><td>Test</td><td>Dyadic</td><td>10</td><td>9.89</td><td>185</td><td>6.96</td></tr>
<tr><td>Test</td><td>Multi-Character</td><td>10</td><td>16.72</td><td>334</td><td>15.20</td></tr>
</tbody>
</table>
</div>

<!-- 右侧图片 -->
<div>
<img src="./len.png" width="300"/>
</div>

</div>


![dis](./dis.png)


## 🧪 Evaluation Protocol of OmniCharacter++

OmniCharacter++ evaluates multi-modal role-playing agents from three complementary perspectives:

1. **Context Understanding** – Assess the model’s comprehension of dialogue context and character intent through role-related question answering (multi-choice) via [Circular Evaluation Strategy](https://github.com/open-compass/MMBench).
2. **Generation Ability** – Evaluate textual response generation using four metrics: `Topic Following`, `Goal Success`, `Character Consistency`, `Dialogue Coherence`.
3. **Human Perception** – Human experts rate the synthesized speech for naturalness and fidelity across six dimensions: `Fluency`, `Consistency`, `Emotional Expression`, `Clarity`, `Appropriateness`, `Immersion`.



## 📜 Citation

If you find **OmniCharacter++** useful, please cite:

```bibtex
@article{omnispatial25,
  title   = {OmniCharacter++: Towards Comprehensive Benchmark for Realistic Role-Playing Agents},
  author  = {Haonan Zhang},
  journal = {arXiv preprint arXiv:XXXX},
  year = {2025}
}
```

## 📄 License

* **Code** — MIT License
* **Data** — CC BY-NC 4.0 (non-commercial research only)  
