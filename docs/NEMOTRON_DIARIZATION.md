# Nemotron diarization (optional)

ATLAS tells voices apart with voiceprints. In busy calls the **Nemotron diarization** plugin
(Plugins > Senses) adds NVIDIA's streaming diarizer, Nemotron-3-Diarization, which follows who
is talking moment to moment while the voiceprints keep track of who is who. It runs in its own
process on the GPU and uses little of it. If it's off, missing, slow or crashes, ATLAS simply
uses the voiceprints alone.

## What you need

- An NVIDIA GPU with CUDA (the same one ATLAS already uses).
- A separate Python environment with `transformers` 5.18 or newer. ATLAS keeps its own
  `transformers` version, so this one lives apart.

## Setup (Windows)

Create the environment where ATLAS looks for it, reusing the PyTorch that ATLAS's Python
already has:

```
py -3.12 -m venv --system-site-packages "%LOCALAPPDATA%\atlas_sidecar\venv"
"%LOCALAPPDATA%\atlas_sidecar\venv\Scripts\python.exe" -m pip install "transformers>=5.18"
```

To use a different Python instead, set the environment variable `ATLAS_NEMOTRON_PYTHON` to its
`python.exe`.

Then turn on **Plugins > Senses > Nemotron diarization**. The first start downloads the model
(`nvidia/Nemotron-3-Diarization`) from Hugging Face into `%USERPROFILE%\.cache\atlas_models\hf`
(set `ATLAS_NEMOTRON_HF_HOME` to put it elsewhere). Check the model card on Hugging Face for its
license before you use it. The log shows `Nemotron side process ready` when it's working; if it
can't start, ATLAS keeps going on voiceprints and tries again 30 seconds later.
