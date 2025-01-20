<div align="center">
<h1> FastClipStylerPro: README
</center> <br> <center> </h1>

<p align="center">
Chi Zhang, Che Wang, Jun-Zhe Wang, Chenming Tang
<br>
Computer Vision @ Peking University (2024 Fall) <br>
<br>
</div>

## **This README might be incomplete because some author(s) might fail to finish writing in time.**

## Quick Start
### Environmental Setup
Tested with `python 3.8.20` on Ubuntu 22.04.5 LTS and MacOS Sequoia 15.1.1.

```
conda create -n fastclipstyler python=3.8
pip install -r requirements.txt
conda install -c anaconda git
pip install git+https://github.com/openai/CLIP.git
```

### Datasets, Models and Others
#### ArtEmis Dataset
Download the ArtEmis Dataset from [artemisdataset](https://www.artemisdataset.org).

Rename `official_data` to `artemis` and move to `prompts`.

`prompts/artemis` should contain `artemis_dataset_release_v0.csv` and `ola_dataset_release_v0.csv`.

#### GPT-2 Model
Download GPT-2 from Huggingface🤗.
```
pip install -U huggingface_hub
huggingface-cli download --resume-download gpt2 --local-dir prompts/gpt2 --local-dir-use-symlinks False
```

#### OpenAI API Key
Create `prompts/api_key.json`. The JSON file should be like:

```
{
    "base_url": "https://api.openai.com/v1",    // OpenAI API base url
    "api_key": "sk-..." // Your OpenAI API key
}
```

### Prompt Construction
```
cd prompts
python prompt_generation.py
```

### Training Data Construction (Training Stage I)
[To be done by Chi Zhang]

### Training (Training Stage II)
[To be done by Che Wang]

### Demo System Inference

