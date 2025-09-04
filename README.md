# Trainer

This a fork of https://github.com/edenartlab/sd-lora-trainer made for use with Intel Arc GPU, meaning XPU instead of CUDA. Check out their GitHub repo for more information.

I am not personally or professionally affiliated with [**Eden** team] 

---

## Setup

Create a venv using Python<=3.12. **Important:** As of making, 2025-09-04, Python==3.13 does not have prebuilt wheels for a number of packages that are required to run this trainer. You could attempt to build them from source instead but downgrading to Python<=3.12 is by far the simpler solution. I recommend:

`pyenv install 3.12.6 && pyenv local 3.12.6` in case of warning message from pyenv run `pyenv init` and follow instructions provided by the package for your shell from there, and re-run `pyenv local 3.12.6`. You can check you current shell's python version with `python --version`

Continue as follows:

`git clone https://github.com/BioTrash/sd-lora-trainer-XPU-compatible && cd sd-lora-trainer-XPU-compatible`

Create venv with correct version of python:

`python -m venv .venv && cd .venv`

Install all dependencies using: 

`pip install -r ../requirements.txt`

From here on out you should be good to start changing the `train_configs/training_args.json`. Below you will find a description of all options possible to change in the .json file.

After changing the .json file as you wish, you can start the training via `python main.py train_configs/training_args.json` from repo's root. The results will be saved under `eden_lora_training_runs` folder.

As the `requirements.txt` uses were specific torch+xpu version, if the version ever become unreachable for whatever reason, likely because it is out of dev, run `pip install --force-reinstall torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/xpu` to manually force in the newest correct torch+xpu version.

---

## Table: parameters, types, descriptions and allowed values

| Parameter                         | Type & default                                                  | Description                                                                                                        | Allowed / example values                                                          |
| --------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `lora_training_uls`               | `str`                                                           | URL(s) or path(s) pointing to training images / dataset locations used for LoRA training.                          | Any valid URL or filesystem path (e.g. `"s3://bucket/dataset"`, `"/data/myset"`). |
| `concept_mode`                    | `Literal["face","style","object"]`                              | What kind of concept you are training for — affects masking/augmentation.                                          | `"face"`, `"style"`, `"object"`                                                   |
| `caption_prefix`                  | `str = ""`                                                      | Hardcoded caption prefix. If set, bypasses chatGPT caption injection. Use only if you know what you’re doing.      | Any string (e.g. `"photo of "`). Empty string to disable.                         |
| `prompt_modifier`                 | `str = None`                                                    | Optional extra prompt text appended to captions/prompts.                                                           | Any string or `None`.                                                             |
| `caption_model`                   | `Literal["gpt4-v","blip","florence","no_caption"] = "florence"` | Model used to generate captions for images.                                                                        | `"gpt4-v"`, `"blip"`, `"florence"`, `"no_caption"`                                |
| `caption_dropout`                 | `float = 0.1`                                                   | Fraction of training steps where the caption is replaced with an empty prompt to improve robustness.               | `0.0`–`1.0` (typical: `0.0`–`0.5`)                                                |
| `sd_model_version`                | `Literal["sdxl","sd15", None] = None`                           | Which Stable Diffusion backbone/version to target.                                                                 | `"sdxl"`, `"sd15"`, or `None` (auto/default)                                      |
| `ckpt_path`                       | `str = None`                                                    | Optional explicit checkpoint path to load a model from.                                                            | Filesystem path or `None`.                                                        |
| `pretrained_model`                | `dict = None`                                                   | Dict describing a pretrained model config (paths/urls).                                                            | `{"name": "...", "path": "..."}` or `None`.                                       |
| `seed`                            | `Union[int, None] = None`                                       | Random seed for reproducibility.                                                                                   | Integer or `None` (for nondeterministic runs).                                    |
| `resolution`                      | `int = 512`                                                     | Default square resolution used if other sizes not provided.                                                        | Typical: `256`, `512`, `768`, `1024`                                              |
| `validation_img_size`             | `Optional[Union[int, List[int]]] = None`                        | Target validation image size. If int: target\_n\_pixels\*\*0.5. If list: `[width, height]`. `None` = use defaults. | `None`, `512`, or `[width, height]` (e.g. `[640,480]`)                            |
| `train_img_size`                  | `List[int] = None`                                              | Training image size — expected as `[width, height]` (or single-value list for square).                             | e.g. `[512,512]`, `[640,480]`                                                     |
| `train_aspect_ratio`              | `float = None`                                                  | Forced aspect ratio for training crops (width/height).                                                             | e.g. `1.0`, `1.33`, `0.75`, or `None`                                             |
| `train_batch_size`                | `int = 4`                                                       | Number of images per training batch (per step before accumulation).                                                | Positive integer (typical: `1`–`64`)                                              |
| `max_train_steps`                 | `int = 300`                                                     | Maximum number of optimization steps to run.                                                                       | Positive integer                                                                  |
| `num_train_epochs`                | `int = None`                                                    | Alternative to `max_train_steps`: number of full dataset passes.                                                   | Positive integer or `None`                                                        |
| `checkpointing_steps`             | `int = 10000`                                                   | Save a checkpoint every N steps.                                                                                   | Positive integer (`0` to disable)                                                 |
| `gradient_accumulation_steps`     | `int = 1`                                                       | Number of micro-batches to accumulate before an optimizer step.                                                    | Positive integer (≥1)                                                             |
| `is_lora`                         | `bool = True`                                                   | Whether to train/apply LoRA adapters.                                                                              | `True` / `False`                                                                  |
| `unet_optimizer_type`             | `Literal["adamw","prodigy","AdamW8bit"] = "adamw"`              | Optimizer used for UNet parameters.                                                                                | `"adamw"`, `"prodigy"`, `"AdamW8bit"`                                             |
| `unet_lr_warmup_steps`            | `int = None`                                                    | Number of steps to linearly warm up the UNet LR.                                                                   | Integer or `None`                                                                 |
| `unet_lr`                         | `float = 0.0003`                                                | Base learning rate for the UNet optimizer.                                                                         | Float (typical `1e-5`–`1e-3`)                                                     |
| `prodigy_d_coef`                  | `float = 1.0`                                                   | Prodigy-specific coefficient (only used when `prodigy` optimizer chosen).                                          | Float (>=0)                                                                       |
| `unet_prodigy_growth_factor`      | `float = 1.05`                                                  | Growth factor controlling prodigy LR increase per step. Lower → slower growth.                                     | Float > 1.0 (e.g. `1.01`–`1.1`)                                                   |
| `lora_weight_decay`               | `float = 0.004`                                                 | Weight decay applied to LoRA parameters.                                                                           | Float (≥0)                                                                        |
| `ti_lr`                           | `float = 0.001`                                                 | Learning rate for token-insertion (TI) / text-embedding training.                                                  | Float (typical `1e-6`–`1e-2`)                                                     |
| `token_warmup_steps`              | `int = 0`                                                       | Number of pure-text-loss warmup steps for token embeddings.                                                        | Integer (≥0)                                                                      |
| `ti_weight_decay`                 | `float = 0.0`                                                   | Weight decay for TI / token embeddings.                                                                            | Float (≥0)                                                                        |
| `ti_optimizer`                    | `Literal["adamw","prodigy"] = "adamw"`                          | Optimizer for token-insertion/text-encoder fine-tuning.                                                            | `"adamw"`, `"prodigy"`                                                            |
| `freeze_ti_after_completion_f`    | `float = 0.7`                                                   | Freeze TI embeddings after this fraction of training has completed.                                                | `0.0`–`1.0` (fraction)                                                            |
| `freeze_unet_before_completion_f` | `float = 0.0`                                                   | Freeze UNet a certain fraction *before* training completion (useful for stage-wise training).                      |                                                                                   |



