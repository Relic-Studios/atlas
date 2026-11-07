# Nemotron diarization (optional)

ATLAS tells voices apart with voiceprints. In busy calls the **Nemotron diarization** plugin
(Plugins > Senses) adds NVIDIA's streaming diarizer, Nemotron-3-Diarization, which follows who
is talking moment to moment while the voiceprints keep track of who is who. It runs in its own
process on the GPU. If it's off, missing, slow or crashes, ATLAS simply
uses the voiceprints alone.

## What you need

- An NVIDIA GPU with CUDA (the same one ATLAS already uses).
- A separate Python environment with `transformers` 5.18 or newer. ATLAS keeps its own
  `transformers` version, so this one lives apart.

## Setup (Windows)

Create a separate environment; do not change ATLAS's own dependencies:

```
py -3.12 -m venv "%LOCALAPPDATA%\atlas_sidecar\venv"
"%LOCALAPPDATA%\atlas_sidecar\venv\Scripts\python.exe" -m pip install "transformers>=5.18" numpy packaging
```

Install CUDA-capable PyTorch in that environment using the command for your system from
[PyTorch's installation selector](https://pytorch.org/get-started/locally/), replacing `pip3` with
`"%LOCALAPPDATA%\atlas_sidecar\venv\Scripts\python.exe" -m pip`.

In Plugins, install **Nemotron diarization**, enter this environment's `python.exe` in Configure
(or leave it blank for the standard location), Save, then Test connection. Installing and testing
do not enable it. You can also set `ATLAS_NEMOTRON_PYTHON`; a path saved in Configure takes precedence.

Then turn on **Plugins > Senses > Nemotron diarization**. The first start downloads the model
(`nvidia/Nemotron-3-Diarization`) from Hugging Face into `%USERPROFILE%\.cache\atlas_models\hf`
(set `ATLAS_NEMOTRON_HF_HOME` to put it elsewhere). Check the model card on Hugging Face for its
license before you use it. The log shows `Nemotron side process ready` when it's working; if it
can't start, ATLAS keeps going on voiceprints and tries again 30 seconds later.
