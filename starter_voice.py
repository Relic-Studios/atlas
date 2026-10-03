"""Starter voice for a fresh install with zero voices.

Qwen3-TTS base only clones from reference audio (no built-in speakers), so a brand
new public install would crash at boot. If no voice exists yet, we synthesize a
~10 s neutral reference on THIS machine with Windows' built-in speech engine
(System.Speech / SAPI) and register it as a normal user voice named "starter".
Nothing is shipped: it's generated locally, and the setup wizard lets the owner
import a real voice and delete this one.

Never runs when any voice is installed (dev build always has voices).
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

KEY = "starter"
TEXT = ("Hi, I'm the starter voice. I'm here so everything works out of the box. "
        "You can import a voice you like from the setup screen, and I'll step aside. "
        "Until then, I'll keep you company.")


def _sapi_wav(text: str, out: Path) -> bool:
    """Render text to a WAV with Windows SAPI. False on non-Windows or failure."""
    if os.name != "nt":
        return False
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.Rate=0;"
        f"$s.SetOutputToWaveFile('{out}');"
        f"$s.Speak([IO.File]::ReadAllText('{out}.txt'));"
        "$s.Dispose()"
    )
    try:
        Path(f"{out}.txt").write_text(text, encoding="utf-8")
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                       check=True, timeout=60, capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.exists() and out.stat().st_size > 10_000
    except Exception as e:  # noqa: BLE001
        logger.warning("starter voice: SAPI render failed: %s", e)
        return False
    finally:
        try:
            Path(f"{out}.txt").unlink()
        except OSError:
            pass


def _espeak_wav(text: str, out: Path) -> bool:
    """Linux/macOS: render with espeak-ng (or espeak). False if not installed."""
    import shutil
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        return False
    try:
        subprocess.run([exe, "-v", "en-us", "-s", "155", "-w", str(out), text],
                       check=True, timeout=60, capture_output=True)
        return out.exists() and out.stat().st_size > 10_000
    except Exception as e:  # noqa: BLE001
        logger.warning("starter voice: espeak render failed: %s", e)
        return False


def _render(text: str, out: Path) -> bool:
    return _sapi_wav(text, out) or _espeak_wav(text, out)


def ensure() -> str | None:
    """If no voices are installed, create and register the starter voice.
    Returns the voice key in use (existing default, or "starter"), or None."""
    import voices
    voices.refresh()
    if voices.DEFAULT_VOICE:
        return voices.DEFAULT_VOICE
    tmp = Path(tempfile.gettempdir()) / "atlas_starter_voice.wav"
    if not _render(TEXT, tmp):
        return None
    try:
        import voice_import
        voice_import.import_voice(tmp.read_bytes(), "starter.wav", KEY, transcript=TEXT)
        logger.info("🔊 No voices installed: created a local starter voice (system speech).")
    except Exception as e:  # noqa: BLE001
        logger.warning("starter voice: import failed: %s", e)
        return None
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    voices.refresh()
    return voices.DEFAULT_VOICE
