# Trainer

This a fork of https://github.com/edenartlab/sd-lora-trainer made for use with Intel Arc GPU, meaning XPU instead of CUDA. Check out their GitHub repo for more information.

I am not personally or professionally affiliated with [**Eden** team] 

---

## Setup

Create a venv using Python<=3.12. **Important:** As of making, 2025-09-04, Python==3.13 does not have prebuilt wheels for a number of packages that are required to run this trainer. You could attempt to build them from source instead but downgrading to Python<=3.12 is by far the simpler solution. I recommend:

`pyenv install 3.12.6 && pyenv local 3.12.6` in case of warning message from pyenv run `pyenv init` and follow instructions provided by the package for your shell from there, and re-run `pyenv local 3.12.6`. You can check your current shell's python version with `python --version`

Continue as follows:

`git clone https://github.com/BioTrash/sd-lora-trainer-XPU-compatible && cd sd-lora-trainer-XPU-compatible`

Create venv with correct version of python:

`python -m venv .venv && cd .venv` remember to activate the venv via `source bin/activate` there will be several files, use the one fit for your shell, likely bash.

Install all dependencies using: 

`pip install -r ../requirements.txt`

From here on out you should be good to start changing the `train_configs/training_args.json`. Below you will find a description of all options possible to change in the .json file.

After changing the .json file as you wish, you can start the training via `python main.py train_configs/training_args.json` from repo's root. The results will be saved under `eden_lora_training_runs` folder.

As the `requirements.txt` uses specific torch+xpu version, if the version ever become unreachable for whatever reason, likely because it is out of dev, run `pip install --force-reinstall torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/xpu` to manually force in the newest correct torch+xpu version.

---

## Table: parameters, types, descriptions and allowed values

| Parameter | Type & default | Description | Allowed / example values |
|---|---:|---|---|
| `lora_training_urls` | `str` | URL(s) or path(s) pointing to training images / dataset locations used for LoRA training. | Any valid URL or filesystem path (e.g. `"s3://bucket/dataset"`, `"/data/myset"`). |
| `concept_mode` | `Literal["face","style","object"]` | What kind of concept you are training for — affects masking/augmentation. | `"face"`, `"style"`, `"object"` |
| `caption_prefix` | `str = ""` | Hardcoded caption prefix. If set, bypasses chatGPT caption injection. Use only if you know what you’re doing. | Any string (e.g. `"photo of "`). Empty string to disable. |
| `prompt_modifier` | `str = None` | Optional extra prompt text appended to captions/prompts. | Any string or `None`. |
| `caption_model` | `Literal["gpt4-v","blip","florence","no_caption"] = "florence"` | Model used to generate captions for images. | `"gpt4-v"`, `"blip"`, `"florence"`, `"no_caption"` |
| `caption_dropout` | `float = 0.1` | Fraction of training steps where the caption is replaced with an empty prompt to improve robustness. | `0.0`–`1.0` (typical: `0.0`–`0.5`) |
| `sd_model_version` | `Literal["sdxl","sd15", None] = None` | Which Stable Diffusion backbone/version to target. | `"sdxl"`, `"sd15"`, or `None` (auto/default) |
| `ckpt_path` | `str = None` | Optional explicit checkpoint path to load a model from. | Filesystem path or `None`. |
| `pretrained_model` | `dict = None` | Dict describing a pretrained model config (paths/urls). | `{"name": "...", "path": "..."}` or `None`. |
| `seed` | `Union[int, None] = None` | Random seed for reproducibility. | Integer or `None` (for nondeterministic runs). |
| `resolution` | `int = 512` | Default square resolution used if other sizes not provided. | Typical: `256`, `512`, `768`, `1024` |
| `validation_img_size` | `Optional[Union[int, List[int]]] = None` | Target validation image size. If int: target_n_pixels**0.5. If list: `[width, height]`. `None` = use defaults. | `None`, `512`, or `[width, height]` (e.g. `[640,480]`) |
| `train_img_size` | `List[int] = None` | Training image size — expected as `[width, height]` (or single-value list for square). | e.g. `[512,512]`, `[640,480]` |
| `train_aspect_ratio` | `float = None` | Forced aspect ratio for training crops (width/height). | e.g. `1.0`, `1.33`, `0.75`, or `None` |
| `train_batch_size` | `int = 4` | Number of images per training batch (per step before accumulation). | Positive integer (typical: `1`–`64`) |
| `max_train_steps` | `int = 300` | Maximum number of optimization steps to run. | Positive integer |
| `num_train_epochs` | `int = None` | Alternative to `max_train_steps`: number of full dataset passes. | Positive integer or `None` |
| `checkpointing_steps` | `int = 10000` | Save a checkpoint every N steps. | Positive integer (`0` to disable) |
| `gradient_accumulation_steps` | `int = 1` | Number of micro-batches to accumulate before an optimizer step. | Positive integer (≥1) |
| `is_lora` | `bool = True` | Whether to train/apply LoRA adapters. | `True` / `False` |
| `unet_optimizer_type` | `Literal["adamw","prodigy","AdamW8bit"] = "adamw"` | Optimizer used for UNet parameters. | `"adamw"`, `"prodigy"`, `"AdamW8bit"` |
| `unet_lr_warmup_steps` | `int = None` | Number of steps to linearly warm up the UNet LR. | Integer or `None` |
| `unet_lr` | `float = 0.0003` | Base learning rate for the UNet optimizer. | Float (typical `1e-5`–`1e-3`) |
| `prodigy_d_coef` | `float = 1.0` | Prodigy-specific coefficient (only used when `prodigy` optimizer chosen). | Float (>=0) |
| `unet_prodigy_growth_factor` | `float = 1.05` | Growth factor controlling prodigy LR increase per step. Lower → slower growth. | Float > 1.0 (e.g. `1.01`–`1.1`) |
| `lora_weight_decay` | `float = 0.004` | Weight decay applied to LoRA parameters. | Float (≥0) |
| `ti_lr` | `float = 0.001` | Learning rate for token-insertion (TI) / text-embedding training. | Float (typical `1e-6`–`1e-2`) |
| `token_warmup_steps` | `int = 0` | Number of pure-text-loss warmup steps for token embeddings. | Integer (≥0) |
| `ti_weight_decay` | `float = 0.0` | Weight decay for TI / token embeddings. | Float (≥0) |
| `ti_optimizer` | `Literal["adamw","prodigy"] = "adamw"` | Optimizer for token-insertion/text-encoder fine-tuning. | `"adamw"`, `"prodigy"` |
| `freeze_ti_after_completion_f` | `float = 0.7` | Freeze TI embeddings after this fraction of training has completed. | `0.0`–`1.0` (fraction) |
| `freeze_unet_before_completion_f` | `float = 0.0` | Freeze UNet a certain fraction *before* training completion (useful for stage-wise training). | `0.0`–`1.0` |
| `token_attention_loss_w` | `float = 3e-7` | Weight for token attention loss regularizer. | Small float (>=0) |
| `cond_reg_w` | `float = 0.0e-5` | Conditional regularization weight (for conditioning stability). | Float (>=0) |
| `tok_cond_reg_w` | `float = 0.0e-5` | Token-level conditional regularization weight. | Float (>=0) |
| `tok_cov_reg_w` | `float = 0.0` | Regularizes token covariance matrix vs pretrained tokens. | Float (>=0) |
| `l1_penalty` | `float = 0.03` | L1 penalty applied to LoRA matrices to encourage sparsity. | Float (>=0) |
| `noise_offset` | `float = 0.02` | Amount to offset noise — helps very dark/bright image training stability. | Float (can be ±) (typical small like `0.0`–`0.1`) |
| `snr_gamma` | `float = 5.0` | SNR-based weighting gamma used in loss scaling. | Float (>=0) |
| `lora_alpha_multiplier` | `float = 1.0` | Multiplier applied to LoRA alpha (scaling factor). | Float (>0) |
| `lora_rank` | `int = 16` | LoRA rank (low-rank factor dimension). Controls capacity. | Positive integer (typical: `4`–`64`) |
| `use_dora` | `bool = False` | Toggle to use DoRA (a LoRA variant) instead of standard LoRA. | `True` / `False` |
| `left_right_flip_augmentation` | `bool = True` | Whether to randomly horizontally flip images during augmentation. | `True` / `False` |
| `augment_imgs_up_to_n` | `int = 40` | Max number of augmented images to generate per concept/image. | Integer ≥0 |
| `mask_target_prompts` | `Union[None,str] = None` | Prompts that indicate masked regions to focus on (for inpainting-style training). | String or `None` (e.g. `"mask: person"`). |
| `crop_based_on_salience` | `bool = True` | Use salience maps to crop around the important part of the image. | `True` / `False` |
| `use_face_detection_instead` | `bool = False` | Use a face-detection model (instead of CLIPSeg) to generate face masks. | `True` / `False` |
| `clipseg_temperature` | `float = 0.5` | Softness/temperature parameter for CLIPSeg mask generation. | Float > 0 (e.g. `0.1`–`2.0`) |
| `n_sample_imgs` | `int = 4` | Number of sample/validation images to generate for monitoring. | Positive integer |
| `name` | `str = None` | Optional friendly name for this training job/run. | Any string or `None` |
| `output_dir` | `str = "eden_lora_training_runs"` | Directory where outputs/checkpoints/logs are written. | Filesystem path |
| `debug` | `bool = False` | Enable debug/verbose logging / extra checks. | `True` / `False` |
| `allow_tf32` | `bool = True` | Allow TF32 math on Ampere+ GPUs for speed (may affect determinism/precision). | `True` / `False` |
| `disable_ti` | `bool = False` | Disable token-insertion (TI) training entirely. | `True` / `False` |
| `skip_gpt_cleanup` | `bool = False` | Skip automatic GPT prompt cleanup steps. | `True` / `False` |
| `weight_type` | `Literal["fp16","bf16","fp32"] = "bf16"` | Precision for model weights during training/inference. | `"fp16"`, `"bf16"`, `"fp32"` |
| `n_tokens` | `int = 3` | Number of new inserted token vectors (for token-insertion). | Positive integer (matches `inserting_list_tokens` length) |
| `inserting_list_tokens` | `List[str] = ["<s0>","<s1>","<s2>"]` | The actual token strings that will be inserted/learned. | List of strings (length == `n_tokens`) |
| `token_dict` | `dict = {"TOK": "<s0><s1><s2>"}` | Mapping names to token sequences used in prompts. | Dict of `{name: token_sequence}` |
| `device` | `str = "xpu:0"` | Compute device string used by PyTorch. | e.g. `"cuda:0"`, `"cuda:1"`, `"cpu"`, `"mps:0"`, `"xpu:0"` |
| `sample_imgs_lora_scale` | `float = None` | Default LoRA scale used when sampling validation images. | Float (e.g. `0.5`, `1.0`) or `None` |
| `dataloader_num_workers` | `int = 0` | Number of worker processes for data loading. | Integer ≥0 (typical: `0`–`16`) |
| `training_attributes` | `dict = {}` | Free-form dict to store metadata/attributes about the run. | Any JSON-serializable dict |
| `aspect_ratio_bucketing` | `bool = False` | Group images into aspect-ratio buckets for efficient batching. | `True` / `False` |
| `start_time` | `float = 0.0` | Internal: epoch/timestamp when the run started. | Float (epoch seconds) |
| `job_time` | `float = 0.0` | Internal: accumulated job runtime in seconds. | Float (seconds) |
| `text_encoder_lora_optimizer` | `Union[None, Literal["adamw"]] = None` | If set (e.g. `"adamw"`), triggers training of text-encoder LoRA parameters. Otherwise text-encoder LoRA is skipped. | `None` or `"adamw"` |
| `text_encoder_lora_lr` | `float = 1.0e-5` | LR for text-encoder LoRA training. | Float (typical `1e-6`–`1e-4`) |
| `txt_encoders_lr_warmup_steps` | `int = 200` | Warmup steps for text-encoder LR. | Integer ≥0 |
| `text_encoder_lora_weight_decay` | `float = 1.0e-5` | Weight decay for text-encoder LoRA. | Float (≥0) |
| `text_encoder_lora_rank` | `int = 16` | LoRA rank for text-encoder adapters. | Positive integer (typical `4`–`64`) |

---