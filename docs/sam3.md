# Setting up and using SAM 3

**SAM 3** (Meta's Segment Anything Model 3, "Segment Anything with
Concepts") finds objects in an image from a short text prompt such as
`particle`. xanes2d-analysis can use it in **Background → Find sample**
instead of the threshold, which helps with samples that a simple
threshold separates badly (low contrast, touching particles, uneven
background).

SAM 3 is not part of xanes2d-analysis. It runs in **its own Python
environment**, which xanes2d-analysis calls when you click
**Find sample**. You set that environment up once.

## What you need

From the SAM 3 instructions (<https://github.com/facebookresearch/sam3>):

- a Linux computer with a **CUDA-capable NVIDIA GPU** and a driver for
  **CUDA 12.6 or newer** (`nvidia-smi` shows the CUDA version). The model
  has 848 million parameters; a recent GPU is strongly recommended;
- **Python 3.12 or newer** and **PyTorch 2.7 or newer** in that
  environment (the pystream environment, Python 3.10, cannot be used);
- a **Hugging Face account** with access to the SAM 3 model (the access
  request is reviewed by Meta, so do this first);
- internet access the first time, to download the model (a few GB),
  or a copy of the model file.

:::{note}
Without a GPU, xanes2d-analysis asks SAM 3 to run on the CPU. That is not
a set-up the SAM 3 authors describe; it is very slow and may not work.
Older GPUs (for example a GTX 1080 Ti) may not be supported by current
PyTorch builds.
:::

## 1. Request access to the model

1. Sign in at <https://huggingface.co> (create an account if needed).
2. Open <https://huggingface.co/facebook/sam3> and request access.
   xanes2d-analysis uses the image model, `sam3.pt` from **facebook/sam3**.
   (The newer **SAM 3.1** checkpoints, `facebook/sam3.1`, are aimed at
   faster video tracking and are not needed here.)
3. When access is granted, create an **access token** (Settings → Access
   Tokens, type *Read*).

## 2. Create the SAM 3 environment

On the GPU computer, in a terminal:

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

These are the commands from the SAM 3 instructions. Keep the `sam3`
folder: the environment uses it.

## 3. Log in to Hugging Face in that environment

```bash
conda activate sam3
hf auth login          # paste the access token
```

(With older Hugging Face tools the command is `huggingface-cli login`.)

## 4. Check that it works

Still in the `sam3` environment:

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
python -c "from sam3.model_builder import build_sam3_image_model; build_sam3_image_model(); print('SAM 3 OK')"
```

The second command downloads the model the first time (a few minutes)
and prints `SAM 3 OK`. If it reports an access error (401 / gated repo),
access was not granted yet or the login did not work.

## 5. Tell xanes2d-analysis where SAM 3 is

Find the environment's Python: with `conda activate sam3`, run
`which python` (for example `/home/<user>/miniconda3/envs/sam3/bin/python`).
Then either:

- in xanes2d-analysis choose **SAM 3** in the **Background** row, click
  the **…** next to the prompt and enter it in **SAM 3 Python**; or
- set the environment variable `XANES2D_SAM3_PYTHON` to it before starting
  xanes2d-analysis.

An environment named `sam3` in `~/miniconda3/envs`, `~/anaconda3/envs`
or `~/conda/envs` is found automatically.

```{figure} images/sam3_settings.png
:alt: SAM 3 settings dialog
:width: 75%

**SAM 3 settings**.
```

| Setting | Meaning |
|---|---|
| **SAM 3 Python** | The SAM 3 environment's Python (see above). |
| **Keep objects with score ≥** | SAM 3 gives every object it finds a score (0–1). Only objects at or above this are used (default 0.5). Lower it if objects are missed, raise it if background is included. |
| **Checkpoint (optional)** | Path to a model file (`sam3.pt`) on disk. Leave blank to use the Hugging Face download. Useful on a computer without internet: copy `sam3.pt` there and enter its path. |
| **Timeout** | How long to wait (default 600 s). The first run includes loading the model. |

## Using it

1. Open and register the scan; check the red post-edge range on the plot.
2. In the **Background** row choose **SAM 3** and type what to find in the
   prompt field (default `particle`). Short, concrete nouns work best:
   `particle`, `particles`, `cell`, `grain`, `crystal`, `fiber`.
3. Click **Find sample**. The status bar says SAM 3 is working; loading the
   model takes up to a minute or so, then the outline appears and the grey
   text says how many objects were found.
4. If nothing is found, xanes2d-analysis shows the scores SAM 3 gave: try
   another prompt or a lower **score** in the settings. If too much is
   included, raise the score.
5. Use **Subtract background** / **Hide background** as with the
   threshold (see {doc}`background`).

SAM 3 looks at the **mean post-edge image**, converted to an 8-bit grey
picture (contrast stretched between the 1st and 99th percentile). Every
click on **Find sample** starts SAM 3 again, including loading the model.

## If something goes wrong

The message shows the last lines SAM 3 printed. Common cases:

| Message | What to do |
|---|---|
| *no SAM 3 Python found* | Set **SAM 3 Python** (step 5). |
| *That Python has no sam3 package* | The Python set is not the SAM 3 environment, or step 2 did not finish. |
| *The checkpoints need Hugging Face access* / 401 / gated | Request access (step 1) and run `hf auth login` in the SAM 3 environment (step 3), or set a local **Checkpoint**. |
| CUDA / out-of-memory errors | The GPU is too small or not supported; use a computer with a recent GPU. |
| *did not finish within … s* | Raise **Timeout** (the first run downloads and loads the model). |
