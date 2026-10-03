# llm_module.py
import re
import logging
import os
import sys
import time
import json
import uuid
import subprocess # <-- Restored usage
from typing import Generator, List, Dict, Optional, Any, Callable
from threading import Lock

# --- Library Dependencies ---
try:
    import requests
    from requests import Session # Explicit import
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("🤖⚠️ requests library not installed. Ollama backend (direct HTTP) will not function.")
    if sys.version_info >= (3, 9): Session = Any | None
    else: Session = Optional[Any]

try:
    from openai import OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    class APIError(Exception): pass
    class APITimeoutError(APIError): pass
    class RateLimitError(APIError): pass
    class APIConnectionError(APIError): pass
    logging.warning("🤖⚠️ openai library not installed. OpenAI/LMStudio backends will not function.")

# Configure logging
# Use the root logger configured by the main application if available, else basic config
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
# Check if root logger already has handlers (likely configured by main app)
if not logging.getLogger().handlers:
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout) # Default to stdout if not configured
logger = logging.getLogger(__name__) # Get logger for this module

_FIGURE_RE = re.compile(
    r"\b(?:how (?:much|many|old|tall|long|far|big)|price|cost|worth|weather|temperature|"
    r"degrees|score|scored|won|stock|market cap|rate|population|net worth|salary|"
    r"what time|when (?:is|does|did|was)|release date|record|stats?|odds|ranking|rank)\b", re.I)


def wants_figure(text: str) -> bool:
    """Is the question after a specific figure (number/date/score) rather than an opinion?"""
    return bool(_FIGURE_RE.search(text or ""))


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""

logger.setLevel(log_level) # Ensure module logger respects level

# --- Environment Variable Configuration ---
try:
    import importlib.util
    dotenv_spec = importlib.util.find_spec("dotenv")
    if dotenv_spec:
        from dotenv import load_dotenv
        load_dotenv()
        logger.debug("🤖⚙️ Loaded environment variables from .env file.")
    else:
        logger.debug("🤖⚙️ python-dotenv not installed, skipping .env load.")
except ImportError:
    logger.debug("🤖💥 Error importing dotenv, skipping .env load.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")

# --- Backend Client Creation/Check Functions ---
def _create_openai_client(api_key: Optional[str], base_url: Optional[str] = None) -> OpenAI:
    """
    Creates and configures an OpenAI API client instance.

    Handles API key logic (using a placeholder if none provided for local models)
    and optional base URL configuration. Sets default timeout and retries.

    Args:
        api_key: The OpenAI API key, or None if not required (e.g., for LMStudio).
        base_url: The base URL for the API endpoint (e.g., for LMStudio or custom deployments).

    Returns:
        An initialized OpenAI client instance.

    Raises:
        ImportError: If the 'openai' library is not installed.
        Exception: If client initialization fails for other reasons.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required for this backend but not installed.")
    try:
        effective_key = api_key if api_key else "no-key-needed"
        client_args = {
            "api_key": effective_key,
            "timeout": 30.0,
            "max_retries": 2
        }
        if base_url:
            client_args["base_url"] = base_url

        client = OpenAI(**client_args)
        logger.info(f"🤖🔌 Prepared OpenAI-compatible client (Base URL: {base_url or 'Default'}).")
        return client
    except Exception as e:
        logger.error(f"🤖💥 Failed to initialize OpenAI client: {e}")
        raise

def _check_ollama_connection(base_url: str, session: Optional[Session]) -> bool:
    """
    Performs a quick HTTP GET request to check connectivity with an Ollama server.

    Uses the provided requests Session and base URL to attempt a connection.
    Logs success or specific connection errors.

    Args:
        base_url: The base URL of the Ollama server (e.g., "http://127.0.0.1:11434").
        session: An active requests.Session object to use for the check.

    Returns:
        True if the connection check is successful (HTTP 2xx status), False otherwise.
    """
    if not REQUESTS_AVAILABLE:
        logger.warning("🤖⚠️ Cannot check Ollama connection: requests library not installed.")
        return False
    if not session:
        logger.warning("🤖⚠️ Cannot check Ollama connection: requests session not provided.")
        return False
    try:
        base_check_url = base_url.rstrip('/')
        if not base_check_url.startswith(('http://', 'https://')):
             base_check_url = 'http://' + base_check_url
        check_endpoint = f"{base_check_url}/"
        logger.debug(f"🤖🔌 Checking Ollama connection via GET to {check_endpoint}...")
        # Use a shorter timeout for the check
        response = session.get(check_endpoint, timeout=5.0)
        response.raise_for_status()
        logger.info(f"🤖🔌 Successfully connected to Ollama server via HTTP at: {base_url}")
        return True
    except requests.exceptions.ConnectionError as e:
        # Log specific connection error, but return False for caller to handle
        logger.warning(f"🤖🔌❌ Connection Error checking Ollama at {base_url}: {e}")
        return False
    except requests.exceptions.Timeout:
        logger.warning(f"🤖🔌❌ Timeout checking Ollama connection at {base_url}.")
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"🤖🔌❌ Error checking Ollama connection at {base_url}: {e}")
        return False
    except Exception as e:
        logger.error(f"🤖💥 Unexpected error during Ollama connection check: {e}")
        return False

# --- Restored _run_ollama_ps function ---
def _run_ollama_ps():
    """
    Attempts to run the 'ollama ps' command via subprocess.

    This is used as a potential fallback diagnostic/recovery step if the initial
    HTTP connection check to the Ollama server fails. It assumes the `ollama` CLI
    is installed and in the system PATH.

    Returns:
        True if the command executes successfully (exit code 0), False otherwise
        (command not found, execution error, timeout).
    """
    try:
        logger.info("🤖🩺 Attempting to run 'ollama ps' to check server status...")
        # Added timeout to prevent hanging indefinitely
        result = subprocess.run(["ollama", "ps"], check=True, capture_output=True, text=True, timeout=10.0)
        logger.info(f"🤖🩺 'ollama ps' executed successfully. Output:\n{result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("🤖💥 'ollama ps' command not found. Make sure Ollama is installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"🤖💥 'ollama ps' command failed with exit code {e.returncode}:")
        if e.stderr:
            logger.error(f"   stderr: {e.stderr.strip()}")
        if e.stdout: # Log stdout even on error, might contain info
            logger.error(f"   stdout: {e.stdout.strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("🤖💥 'ollama ps' command timed out after 10 seconds.")
        return False
    except Exception as e:
        logger.error(f"🤖💥 An unexpected error occurred while running 'ollama ps': {e}")
        return False

# --- LLM Class ---
# Routing-mode sampling. Greedy (temperature 0) made the spoken body copy
# earlier assistant turns once history filled with "Yeah, ..." replies (live
# 09:46 loop). Mild sampling + repeat penalty keeps the header reliable
# (tests/sim_repetition.py) while the body stays varied.
PARTICIPATION_SAMPLING = {
    # Greedy. sim_repetition 09:54: t=0.7 + repeat_penalty gave 3/30 INVALID
    # headers and 7/18 missed replies vs 0 and 3/18 greedy. repeat_penalty also
    # penalises the "[SPEAK to=" header tokens that recur in history.
    # 09-29 20:55: owner asked for 0.6. top_p 0.9 keeps the near-certain
    # "[SPEAK to=" / "[HOLD]" header tokens deterministic while the body varies.
    # No repeat_penalty (that is what broke headers before). Override: ATLAS_TEMP.
    "temperature": float(os.environ.get("ATLAS_TEMP", "0.6")),
    "top_p": float(os.environ.get("ATLAS_TOP_P", "0.9")),
}


class LLM:
    """
    Provides a unified interface for interacting with various LLM backends.

    Supports Ollama (via direct HTTP), OpenAI API, and LMStudio (via OpenAI-compatible API).
    Handles client initialization, streaming generation, request cancellation,
    system prompts, and basic connection management including an optional `ollama ps` check.
    """
    SUPPORTED_BACKENDS = ["ollama", "openai", "lmstudio"]

    def __init__(
        self,
        backend: str,
        model: str,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        no_think: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_executor: Optional[Callable[[str, Any], str]] = None,
        on_tool_call: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ):
        """
        Initializes the LLM interface for a specific backend and model.

        Args:
            backend: The name of the LLM backend to use (e.g., "ollama", "openai", "lmstudio").
            model: The identifier for the specific model to use within the backend.
            system_prompt: An optional system prompt to prepend to conversations.
            api_key: API key, primarily for OpenAI backend (can be omitted for others if not needed).
            base_url: Optional base URL for the backend API (overrides defaults/env vars).
            no_think: Experimental flag (currently unused in core logic, intended for future prompt modification).

        Raises:
            ValueError: If an unsupported backend is specified.
            ImportError: If required libraries for the selected backend are not installed.
        """
        logger.info(f"🤖⚙️ Initializing LLM with backend: {backend}, model: {model}, system_prompt: {system_prompt}")
        self.backend = backend.lower()
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Supported: {self.SUPPORTED_BACKENDS}")

        if self.backend == "ollama" and not REQUESTS_AVAILABLE:
             raise ImportError("requests library is required for the 'ollama' backend but not installed.")
        if self.backend in ["openai", "lmstudio"] and not OPENAI_AVAILABLE:
             raise ImportError("openai library is required for the 'openai'/'lmstudio' backends but not installed.")

        self.model = model
        self.system_prompt = system_prompt
        self._api_key = api_key
        self._base_url = base_url
        self.no_think = no_think # Not used yet, but kept for future use
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.on_tool_call = on_tool_call
        self.on_tool_result = None  # callable(name, args, result) after each executed tool

        self.client: Optional[OpenAI] = None
        self.ollama_session: Optional[Session] = None
        self._client_initialized: bool = False
        self._client_init_lock = Lock()
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._requests_lock = Lock()
        self._ollama_connection_ok: bool = False # Added explicit init

        logger.info(f"🤖⚙️ Configuring LLM instance: backend='{self.backend}', model='{self.model}'")

        self.effective_openai_key = self._api_key or OPENAI_API_KEY
        self.effective_ollama_url = self._base_url or OLLAMA_BASE_URL if self.backend == "ollama" else None
        self.effective_lmstudio_url = self._base_url or LMSTUDIO_BASE_URL if self.backend == "lmstudio" else None
        self.effective_openai_base_url = self._base_url if self.backend == "openai" and self._base_url else None

        if self.backend == "ollama" and self.effective_ollama_url:
             url = self.effective_ollama_url
             if not url.startswith(('http://', 'https://')):
                  url = 'http://' + url
             url = url.replace('/api/chat', '').replace('/api/generate', '').rstrip('/')
             self.effective_ollama_url = url
             logger.debug(f"🤖⚙️ Normalized Ollama URL: {self.effective_ollama_url}")

        if self.backend == "ollama" and REQUESTS_AVAILABLE:
            self.ollama_session = requests.Session()
            logger.info("🤖🔌 Initialized requests.Session for Ollama backend.")
        # Optional remote_llm.py swaps ollama_session for a RemoteSession; the
        # original local session is kept here for switching back / fallback.
        self._local_session = getattr(self, "ollama_session", None)
        self.remote = None

        self.system_prompt_message = None
        if self.system_prompt:
            self.system_prompt_message = {"role": "system", "content": self.system_prompt}
            logger.info(f"🤖💬 System prompt set.")

    def _ensure_local_session(self):
        if getattr(self, "_local_session", None) is None:
            self._local_session = self.ollama_session

    def _lazy_initialize_clients(self) -> bool:
        """
        Initializes backend clients or checks connections on first use (thread-safe).

        Creates the appropriate HTTP client (OpenAI SDK or requests.Session) and performs
        an initial connection check for Ollama. If the Ollama check fails, optionally
        attempts to run `ollama ps` as a fallback before retrying the connection check.

        Returns:
            True if the client is initialized and ready (or connection check passed for Ollama),
            False otherwise.
        """
        if self._client_initialized:
            if self.backend in ["openai", "lmstudio"]: return self.client is not None
            if self.backend == "ollama": return self.ollama_session is not None and self._ollama_connection_ok # Check flag
            return False

        with self._client_init_lock:
            if self._client_initialized: # Double check
                if self.backend in ["openai", "lmstudio"]: return self.client is not None
                if self.backend == "ollama": return self.ollama_session is not None and self._ollama_connection_ok
                return False

            logger.debug(f"🤖🔄 Lazy initializing/checking connection for backend: {self.backend}")
            init_ok = False
            self._ollama_connection_ok = False # Reset Ollama specific flag

            try:
                if self.backend == "openai":
                    self.client = _create_openai_client(self.effective_openai_key, base_url=self.effective_openai_base_url)
                    init_ok = self.client is not None
                elif self.backend == "lmstudio":
                    self.client = _create_openai_client(api_key="lmstudio-key", base_url=self.effective_lmstudio_url)
                    init_ok = self.client is not None
                elif self.backend == "ollama":
                    if self.ollama_session and self.effective_ollama_url:
                        # Initial direct check
                        initial_check_ok = _check_ollama_connection(self.effective_ollama_url, self.ollama_session)
                        if initial_check_ok:
                            init_ok = True
                            self._ollama_connection_ok = True
                        else:
                            # --- Restored ollama ps fallback logic ---
                            logger.warning(f"🤖🔌 Initial Ollama connection check failed for {self.effective_ollama_url}. Attempting 'ollama ps' fallback.")
                            if _run_ollama_ps():
                                # ollama ps ran, wait a bit and try connecting again
                                logger.info("🤖⏳ 'ollama ps' succeeded, waiting 3 seconds before re-checking connection...")
                                time.sleep(3)
                                second_check_ok = _check_ollama_connection(self.effective_ollama_url, self.ollama_session)
                                if second_check_ok:
                                    logger.info("🤖🔌✅ Ollama connection successful after running 'ollama ps'.")
                                    init_ok = True
                                    self._ollama_connection_ok = True
                                else:
                                    logger.error(f"🤖💥 Ollama connection check still failed after running 'ollama ps'.")
                                    init_ok = False # Explicitly set to false
                            else:
                                # ollama ps command failed or was not found
                                logger.error(f"🤖💥 'ollama ps' command failed or not found. Cannot verify/start server. Initialization failed for {self.effective_ollama_url}.")
                                init_ok = False # Explicitly set to false
                            # --- End of restored logic ---
                    else:
                        logger.error("🤖💥 Ollama session object is None or URL not set during lazy init.")
                        init_ok = False

                if init_ok:
                    logger.info(f"🤖✅ Client/Connection initialized successfully for backend: {self.backend}.")
                else:
                    logger.error(f"🤖💥 Initialization failed for backend: {self.backend}.")
            except Exception as e:
                logger.exception(f"🤖💥 Critical failure during lazy initialization for {self.backend}: {e}")
                init_ok = False
            finally:
                # Mark as initialized regardless of success/failure
                self._client_initialized = True
                # Ensure connection flag reflects reality if init failed
                if self.backend == "ollama" and not init_ok:
                    self._ollama_connection_ok = False

            return init_ok


    def cancel_generation(self, request_id: Optional[str] = None) -> bool:
        """Mark cancellation before closing transport, outside the tracking lock.

        Keep the generation's record until its owner exits: HTTP setup and tool
        rounds must never resurrect a cancelled ID. Never close the Python
        generator from a different thread.
        """
        streams = []
        with self._requests_lock:
            ids = list(self._active_requests) if request_id is None else [request_id]
            for req_id in ids:
                state = self._active_requests.get(req_id)
                if state is not None and not state["cancelled"]:
                    state["cancelled"] = True
                    streams.append(state["stream"])
                    state["stream"] = None
        for stream in streams:
            self._close_stream(stream)
        return bool(streams)

    @staticmethod
    def _close_stream(stream):
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                logger.warning("Error closing generation transport: %s", exc)

    def _is_cancelled(self, request_id: str) -> bool:
        with self._requests_lock:
            state = self._active_requests.get(request_id)
            return state is None or state["cancelled"]

    def _register_request(self, request_id: str, request_type: str, stream_obj: Optional[Any]):
        """Reserve once, then attach transports without resetting cancellation."""
        with self._requests_lock:
            state = self._active_requests.get(request_id)
            if stream_obj is None:
                if state is not None:
                    raise ValueError(f"Request ID already active: {request_id}")
                self._active_requests[request_id] = {
                    "type": request_type, "stream": None,
                    "start_time": time.time(), "cancelled": False,
                }
                return True
            if state is not None and not state["cancelled"]:
                state["stream"] = stream_obj
                return True
        self._close_stream(stream_obj)
        return False

    def _release_stream(self, request_id: str, stream):
        # Exactly one closer owns each transport: cancel or the reader finally.
        with self._requests_lock:
            state = self._active_requests.get(request_id)
            owned = state is not None and state["stream"] is stream
            if owned:
                state["stream"] = None
        if owned:
            self._close_stream(stream)

    def _finish_request(self, request_id: str):
        with self._requests_lock:
            state = self._active_requests.pop(request_id, None)
        if state is not None:
            self._close_stream(state["stream"])

    def cleanup_stale_requests(self, timeout_seconds: int = 300):
        """
        Finds and attempts to cancel requests older than the specified timeout.

        Iterates through active requests and calls `cancel_generation` for any
        request whose start time exceeds the timeout duration.

        Args:
            timeout_seconds: The maximum age in seconds before a request is considered stale.

        Returns:
            The number of stale requests for which cancellation was attempted.
        """
        stale_ids = []
        now = time.time()
        # Find stale IDs without holding lock for too long
        with self._requests_lock:
            stale_ids = [
                req_id for req_id, req_data in self._active_requests.items()
                if (now - req_data.get("start_time", 0)) > timeout_seconds
            ]

        if stale_ids:
            logger.info(f"🤖🧹 Found {len(stale_ids)} potentially stale requests (>{timeout_seconds}s). Cleaning up...")
            cleaned_count = 0
            for req_id in stale_ids:
                # cancel_generation handles locking internally and now attempts to close stream
                if self.cancel_generation(req_id):
                    cleaned_count += 1
            logger.info(f"🤖🧹 Cleaned up {cleaned_count}/{len(stale_ids)} stale requests (attempted stream close).")
            return cleaned_count
        return 0

    def prewarm(self, max_retries: int = 1) -> bool:
        """
        Attempts to "prewarm" the LLM connection and potentially load the model.

        Runs a simple, short generation task ("Respond with only the word 'OK'.")
        to trigger lazy initialization (including potential `ollama ps` check)
        and ensure the backend is responsive before actual use. Includes basic retry logic.

        Args:
            max_retries: The number of times to retry the generation task if a
                         connection/timeout error occurs (0 means one attempt total).

        Returns:
            True if the prewarm generation completed successfully (even with no content),
            False if initialization or generation failed after retries.
        """
        prompt = "Respond with only the word 'OK'."
        logger.info(f"🤖🔥 Attempting prewarm for '{self.model}' on backend '{self.backend}'...")

        # Lazy initialization now includes the 'ollama ps' logic if needed
        if not self._lazy_initialize_clients():
            logger.error("🤖🔥💥 Prewarm failed: Could not initialize backend client/connection.")
            return False

        attempts = 0
        last_error = None
        while attempts <= max_retries:
            prewarm_start_time = time.time()
            prewarm_request_id = f"prewarm-{self.backend}-{uuid.uuid4()}"
            generator = None
            full_response = ""
            token_count = 0
            first_token_time = None

            try:
                logger.info(f"🤖🔥 Prewarm Attempt {attempts + 1}/{max_retries+1} calling generate (ID: {prewarm_request_id})...")
                generator = self.generate(
                    text=prompt,
                    history=None,
                    use_system_prompt=True,
                    request_id=prewarm_request_id,
                    temperature=0.1
                )

                gen_start_time = time.time()
                # Consume the generator fully
                for token in generator:
                    if first_token_time is None:
                        first_token_time = time.time()
                        logger.info(f"🤖🔥⏱️ Prewarm TTFT: {(first_token_time - gen_start_time):.4f}s")
                    full_response += token
                    token_count += 1
                # End of loop means generator is exhausted
                gen_end_time = time.time()
                logger.info(f"🤖🔥ℹ️ Prewarm consumed {token_count} tokens in {(gen_end_time - gen_start_time):.4f}s. Full response: '{full_response}'")

                if token_count == 0 and not full_response:
                     logger.warning(f"🤖🔥⚠️ Prewarm yielded no response content, but generation finished.")
                # else: pass # If we got content, great.

                prewarm_end_time = time.time()
                logger.info(f"🤖🔥✅ Prewarm successful (generation finished naturally). Total time: {(prewarm_end_time - prewarm_start_time):.4f}s.")
                return True

            except (APIConnectionError, requests.exceptions.ConnectionError, ConnectionError, TimeoutError, APITimeoutError, requests.exceptions.Timeout) as e:
                last_error = e
                logger.warning(f"🤖🔥⚠️ Prewarm attempt {attempts + 1}/{max_retries+1} connection/timeout error during generation: {e}")
                if attempts < max_retries:
                    attempts += 1
                    wait_time = 2 * attempts
                    logger.info(f"🤖🔥🔄 Retrying prewarm generation in {wait_time}s...")
                    time.sleep(wait_time)
                    # Force re-check on next attempt via lazy_init in generate()
                    # Crucially, setting this False forces _lazy_initialize_clients to run again
                    # which will re-attempt the connection check AND the `ollama ps` fallback if needed.
                    self._client_initialized = False
                    logger.debug("🤖🔥🔄 Resetting client initialized flag to force re-check on retry.")
                    continue
                else:
                    logger.error(f"🤖🔥💥 Prewarm failed permanently after {attempts + 1} generation attempts due to connection issues.")
                    return False
            except (APIError, RateLimitError, requests.exceptions.RequestException, RuntimeError) as e:
                last_error = e
                logger.error(f"🤖🔥💥 Prewarm attempt {attempts + 1}/{max_retries+1} API/Request/Runtime error: {e}")
                if isinstance(e, ConnectionError) and "connection failed" in str(e):
                     logger.error("   (This likely indicates the initial lazy initialization failed its connection check or `ollama ps` fallback)")
                elif isinstance(e, RuntimeError) and "client failed" in str(e):
                    logger.error("   (This might indicate the initial lazy initialization failed)")
                return False # Non-connection errors are usually fatal for prewarm
            except Exception as e:
                last_error = e
                logger.exception(f"🤖🔥💥 Prewarm attempt {attempts + 1}/{max_retries+1} unexpected error.")
                return False
            finally:
                # Generate's finally block handles tracking cleanup.
                # Explicitly try closing generator here in case of error mid-stream.
                logger.debug(f"🤖🔥ℹ️ [{prewarm_request_id}] Prewarm attempt finished. generate()'s finally handles tracking cleanup.")
                if generator is not None and hasattr(generator, 'close'):
                    try:
                        generator.close()
                    except Exception as close_err:
                         logger.warning(f"🤖🔥⚠️ [{prewarm_request_id}] Error closing generator in prewarm finally: {close_err}", exc_info=False)
                generator = None # Clear local ref

            if attempts >= max_retries:
                break # Exit loop if max_retries reached without success

        logger.error(f"🤖🔥💥 Prewarm failed after exhausting retries. Last error: {last_error}")
        return False

    def generate(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        request_id: Optional[str] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]:
        """
        Generates text using the configured backend, yielding tokens as a stream.

        Handles lazy initialization (including potential `ollama ps` check), message formatting,
        backend-specific API calls, stream registration, token yielding, and resource cleanup.

        Args:
            text: The user's input prompt/text.
            history: An optional list of previous messages (dicts with "role" and "content").
            use_system_prompt: If True, prepends the configured system prompt (if any).
            request_id: An optional unique ID for this generation request. If None, one is generated.
            **kwargs: Additional backend-specific keyword arguments (e.g., temperature, top_p, stop sequences).

        Yields:
            str: Individual tokens (or small chunks of text) as they are generated by the LLM.

        Raises:
            RuntimeError: If the backend client fails to initialize.
            ConnectionError: If communication with the backend fails (initial connection or during streaming).
            ValueError: If configuration is invalid (e.g., missing Ollama URL).
            APIError: For backend-specific API errors (OpenAI/LMStudio).
            RateLimitError: For backend-specific rate limit errors (OpenAI/LMStudio).
            requests.exceptions.RequestException: For Ollama HTTP request errors.
            Exception: For other unexpected errors during the generation process.
        """
        req_id = request_id if request_id else f"{self.backend}-{uuid.uuid4()}"
        logger.info(f"🤖💬 Starting generation (Request ID: {req_id})")

        participation = kwargs.pop('participation', False)
        use_tools = kwargs.pop('use_tools', True)
        memory_context = kwargs.pop('memory_context', '')
        room_context = kwargs.pop('room_context', '')
        prefetch_tool = kwargs.pop('prefetch_tool', None)
        messages = []
        if use_system_prompt and self.system_prompt_message:
            messages.append(dict(self.system_prompt_message))
        if participation:
            from response_decision import DECISION_PROMPT
            speaker_match = re.match(r'^\s*\[(S\d+)\]', text)
            current_speaker = speaker_match[1] if speaker_match else 'user'
            protocol = (DECISION_PROMPT + '\n\nCURRENT TURN: speaker=' + current_speaker
                        + '. If speaking, the required first header is [SPEAK to='
                        + current_speaker + ']. If silent, output only [HOLD].')
            if room_context:
                protocol += '\n' + room_context
            for _k, _v in PARTICIPATION_SAMPLING.items():
                kwargs.setdefault(_k, _v)
            if messages:
                messages[0]['content'] += '\n\n' + protocol
            else:
                messages.append({'role': 'system', 'content': protocol})
        if memory_context:
            # local-model (Qwen3.5) template rejects any system message that isn't
            # the FIRST message. Merge into the lead system message instead of
            # appending a second one (that second message 500'd ollama).
            if messages:
                messages[0]['content'] += '\n\n' + memory_context
            else:
                messages.append({"role": "system", "content": memory_context})
        if history:
            # Deep-copy each history dict: generate() mutates the tail message
            # (prepends "/no_think"), and without copies that mutation leaks back
            # into SpeechPipelineManager.history, corrupting context across turns.
            messages.extend(dict(m) for m in history)

        # A held room turn leaves a USER at the tail. That is not necessarily
        # the current turn; never silently discard a new utterance based on role.
        if (not messages or messages[-1]["role"] != "user"
                or messages[-1].get('content') != text):
            added_text = text # for normal text
            if self.no_think:
                 # Qwen3-native token-level instruction. Empirically cuts the
                 # pre-content reasoning trace ~209 -> ~59 tokens (TTFT 11.5s -> 3.5s),
                 # unlike chat_template_kwargs which the model ignores.
                added_text = f"/no_think {text}"
            logger.info(f"🧠💬 llm_module.py generate adding role user to messages, content: {added_text}")
            messages.append({"role": "user", "content": added_text})
        elif self.no_think and messages[-1]["role"] == "user":
            # last message is already a user turn (history tail): prefix it in place
            messages[-1]["content"] = f"/no_think {messages[-1]['content']}"
        logger.debug(f"🤖💬 [{req_id}] Prepared messages count: {len(messages)}")

        stream_iterator = None
        stream_object_to_register = None

        self._register_request(req_id, self.backend, None)
        try:
            # Lazy initialization now includes the 'ollama ps' logic if needed
            if not self._lazy_initialize_clients():
                # Provide a clearer error if initialization failed
                if self.backend == "ollama" and not self._ollama_connection_ok:
                     raise ConnectionError(f"LLM backend '{self.backend}' connection failed. Could not connect to {self.effective_ollama_url} even after attempting 'ollama ps'. Check server status and configuration.")
                raise RuntimeError(f"LLM backend '{self.backend}' client failed to initialize.")

            if self._is_cancelled(req_id):
                return
            if self.backend == "openai":
                if self.client is None:
                    raise RuntimeError("OpenAI client not initialized (should have been caught by lazy_init).")
                payload = { "model": self.model, "messages": messages, "stream": True, **kwargs }
                logger.info(f"🤖💬 [{req_id}] Sending OpenAI request with payload:")
                logger.info(f"{json.dumps(payload, indent=2)}")
                stream_iterator = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=True, **kwargs
                )
                stream_object_to_register = stream_iterator # The Stream object itself
                if not self._register_request(req_id, "openai", stream_object_to_register):
                    return
                yield from self._yield_openai_chunks(stream_iterator, req_id)

            elif self.backend == "lmstudio":
                if self.client is None:
                    raise RuntimeError("LM Studio client not initialized (should have been caught by lazy_init).")
                if 'temperature' not in kwargs:
                    kwargs['temperature'] = 0.7
                payload = { "model": self.model, "messages": messages, "stream": True, **kwargs }
                logger.info(f"🤖💬 [{req_id}] Sending LM Studio request with payload:")
                logger.info(f"{json.dumps(payload, indent=2)}")
                stream_iterator = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=True, **kwargs
                )
                stream_object_to_register = stream_iterator # The Stream object itself
                if not self._register_request(req_id, "lmstudio", stream_object_to_register):
                    return
                yield from self._yield_openai_chunks(stream_iterator, req_id)

            elif self.backend == "ollama":
                if self.ollama_session is None:
                    raise RuntimeError("Ollama session not initialized (should have been caught by lazy_init).")
                if not self.effective_ollama_url:
                    raise ValueError("Ollama base URL not configured.")
                # Connection check (and potential ps fallback) happened in lazy_init

                ollama_api_url = f"{self.effective_ollama_url}/api/chat"
                valid_options = {"temperature", "top_k", "top_p", "num_predict", "stop", "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty"}
                options = {k: v for k, v in kwargs.items() if k in valid_options}
                if 'temperature' not in options:
                    options['temperature'] = 0.7
                # Pin context so prefill stays fast/bounded — the model default is
                # 262k, which makes cold-load TTFT balloon. History is already
                # trimmed to 16 messages, so 4096 is ample.
                options.setdefault('num_ctx', 4096)

                if use_tools and self.tools and self.tool_executor:
                    yield from self._generate_ollama_agentic(
                        ollama_api_url, messages, options, req_id, prefetch=prefetch_tool)
                else:
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                        "keep_alive": "30m",
                        "think": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "options": options
                    }
                    logger.info(f"🤖💬 [{req_id}] Sending Ollama request to {ollama_api_url} with payload:")
                    logger.info(f"{json.dumps(payload, indent=2)}")
                    # Increase read timeout significantly for generation
                    response = self.ollama_session.post(
                        ollama_api_url, json=payload, stream=True, timeout=(10.0, 600.0) # (connect_timeout, read_timeout)
                    )
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    stream_object_to_register = response # The requests.Response object
                    if not self._register_request(req_id, "ollama", stream_object_to_register):
                        return
                    yield from self._yield_ollama_chunks(response, req_id)

            else:
                # This case should technically be caught by __init__
                raise ValueError(f"Backend '{self.backend}' generation logic not implemented.")

            logger.info(f"🤖✅ Finished generating stream successfully (request_id: {req_id})")

        # Catch specific exceptions first
        except (requests.exceptions.ConnectionError, ConnectionError, APITimeoutError, requests.exceptions.Timeout) as e:
             logger.error(f"🤖💥 Connection/Timeout Error during generation for {req_id}: {e}", exc_info=False)
             # Reraise as a standard ConnectionError for consistency
             raise ConnectionError(f"Communication error during generation: {e}") from e
        except (APIError, RateLimitError, requests.exceptions.RequestException) as e: # Includes HTTPError
             logger.error(f"🤖💥 API/Request Error during generation for {req_id}: {e}", exc_info=False)
             # Reraise the original error
             raise
        except Exception as e:
            logger.error(f"🤖💥 Unexpected error in generation pipeline for {req_id}: {e}", exc_info=True) # Log traceback for unexpected
            raise # Reraise the original exception
        finally:
            self._finish_request(req_id)


    # --- Backend-Specific Chunk Yielding Helpers ---
    def _yield_openai_chunks(self, stream, request_id: str) -> Generator[str, None, None]:
        """
        Iterates over an OpenAI/LMStudio stream, yielding content chunks.

        Handles extracting content from stream chunks and checks for cancellation
        before processing each chunk. Ensures the stream is closed upon completion,
        error, or cancellation.

        Args:
            stream: The stream object returned by the OpenAI client's `create` method.
            request_id: The unique ID associated with this generation stream.

        Yields:
            str: Content chunks from the stream's delta messages.

        Raises:
            ConnectionError: If a connection error occurs during streaming, unless likely due to cancellation.
            APIError: If an API error occurs during streaming.
            Exception: For other unexpected errors during streaming.
        """
        token_count = 0
        try:
            for chunk in stream:
                # Check for cancellation *before* processing chunk
                with self._requests_lock:
                    if (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"]):
                        logger.info(f"🤖🗑️ OpenAI/LMStudio stream {request_id} cancelled or finished externally during iteration.")
                        # No need to manually close stream here; cancellation logic or finally block handles it.
                        break # Exit the loop cleanly
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        token_count += 1
                        yield content
            logger.debug(f"🤖✅ [{request_id}] Finished yielding {token_count} OpenAI/LMStudio tokens.")
        except APIConnectionError as e:
             # Often happens if the stream is closed prematurely by cancellation
             is_cancelled = False
             with self._requests_lock:
                 is_cancelled = (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"])
             if is_cancelled:
                  logger.warning(f"🤖⚠️ OpenAI/LMStudio stream connection error likely due to cancellation for {request_id}: {e}")
             else:
                  logger.error(f"🤖💥 OpenAI API connection error during streaming ({request_id}): {e}")
                  raise ConnectionError(f"OpenAI communication error during streaming: {e}") from e
        except APIError as e:
            logger.error(f"🤖💥 OpenAI API error during streaming ({request_id}): {e}")
            raise # Reraise for generate() to handle
        except Exception as e:
            # Catch other potential errors during iteration
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"])
            if is_cancelled:
                logger.warning(f"🤖⚠️ OpenAI/LMStudio stream error likely due to cancellation for {request_id}: {e}")
            else:
                logger.error(f"🤖💥 Unexpected error during OpenAI streaming ({request_id}): {e}", exc_info=True)
                raise # Reraise for generate() to handle
        finally:
            self._release_stream(request_id, stream)

    def _generate_ollama_agentic(
        self,
        ollama_api_url: str,
        messages: List[Dict[str, Any]],
        options: Dict[str, Any],
        request_id: str,
        prefetch=None,
    ) -> Generator[str, None, None]:
        """Ollama tool-calling loop: let the model call web_search and answer.

        Sends the prompt with a `tools` schema. If the model replies with
        tool_calls instead of content, we fire `on_tool_call` (so the caller can
        speak a filler like "hmm, let me look"), execute each tool via
        `tool_executor`, append the assistant tool_call message + tool results to
        the conversation, and re-request. Repeats until the model produces
        content or a bounded number of tool rounds elapse.

        Yields only final content tokens (tool plumbing is hidden from callers).
        """
        # Search budget, then one FORCED answer round without tools. Before,
        # the model could keep re-querying until the round cap and the loop
        # quit with no content -> silent turn after "lemme find it!".
        MAX_SEARCH_ROUNDS = int(os.environ.get("ATLAS_SEARCH_ROUNDS", "1"))
        working = list(messages)
        seen_queries = set()
        if prefetch:
            # Live 10-01: Ivy was asked to 'look again' and answered from her old
            # description (and once invented a game menu) without calling the tool.
            # A direct look request now ALWAYS gets fresh pixels: we run the tool
            # ourselves and hand the model the result, same shape as a real call.
            self._run_prefetch(prefetch, working, seen_queries, request_id)

        # One bonus round, offering ONLY read_page, when a search for a specific
        # figure (price, weather, score, how many...) used the budget: snippets
        # are usually link blurbs without the number (live 10-01 Pup
        # "no live numbers"). The model decides whether to open a page.
        limit = MAX_SEARCH_ROUNDS
        read_bonus = (os.environ.get("ATLAS_READ_FOLLOWUP", "1") != "0"
                      and any((t.get("function") or {}).get("name") == "read_page"
                              for t in (self.tools or []))
                      and wants_figure(_last_user_text(messages)))
        only_read = False
        _round = -1
        while _round < limit:
            _round += 1
            if self._is_cancelled(request_id):
                return
            final_round = _round == limit
            payload = {
                "model": self.model,
                "messages": working,
                "stream": True,
                "keep_alive": "30m",
                "think": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "options": options,
            }
            if not final_round:
                payload["tools"] = ([t for t in self.tools
                                     if (t.get("function") or {}).get("name") == "read_page"]
                                    if only_read else self.tools)
            response = self.ollama_session.post(
                ollama_api_url, json=payload, stream=True,
                timeout=(10.0, 600.0),
            )
            response.raise_for_status()
            if not self._register_request(request_id, "ollama", response):
                return

            tool_calls: List[Dict[str, Any]] = []
            content_parts: List[str] = []
            buffer = ""
            processed_done = False

            try:
                for chunk_bytes in response.iter_content(chunk_size=None):
                    with self._requests_lock:
                        if (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"]):
                            break  # cancelled
                    if not chunk_bytes:
                        continue
                    buffer += chunk_bytes.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if d.get("error"):
                            raise RuntimeError(f"Ollama stream error: {d['error']}")
                        m = d.get("message", {})
                        if m.get("tool_calls"):
                            tool_calls.extend(m["tool_calls"])
                        if m.get("content"):
                            content_parts.append(m["content"])
                            # Tool calls can arrive after text; never expose a
                            # speculative header/body from an intermediate round.
                        if d.get("done"):
                            processed_done = True
                            break
                    if processed_done:
                        break
            except AttributeError as e:
                # Mirrors _yield_ollama_chunks: concurrent cancel closes the
                # response mid-iteration, and urllib3 hits a None _fp in
                # read()/readline()/read_chunked(). Match the common prefix so
                # we catch every variant, not just 'read'.
                if self._is_cancelled(request_id) and "'NoneType' object has no attribute" in str(e):
                    with self._requests_lock:
                        is_cancelled = (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"])
                    logger.warning(
                        f"🤖⚠️ [{request_id}] Agentic stream closed by concurrent "
                        f"cancellation (cancelled={is_cancelled}). Stopping."
                    )
                else:
                    logger.error(f"🤖💥 [{request_id}] Unexpected AttributeError in agentic stream: {e}", exc_info=True)
                    raise
            finally:
                self._release_stream(request_id, response)

            if self._is_cancelled(request_id):
                return

            # If the model produced content, we're done.
            if content_parts and not tool_calls:
                for content in content_parts:
                    if self._is_cancelled(request_id):
                        return
                    yield content
                return
            # If no tool calls and no content, give up (avoid infinite loop).
            if not tool_calls:
                logger.warning(f"🤖❓ [{request_id}] Agentic round {_round} produced no content and no tool_calls.")
                return

            # Fire the filler callback so the caller can speak "let me look...".
            if self.on_tool_call is not None:
                try:
                    self.on_tool_call(tool_calls)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"🤖⚠️ [{request_id}] on_tool_call error: {e}")

            # Append the assistant tool_call message (as Ollama expects it back).
            working.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for tc in tool_calls
                ],
            })
            # Execute each tool and append its result.
            for tc in tool_calls:
                if self._is_cancelled(request_id):
                    return
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                qkey = (name, json.dumps(args, sort_keys=True).lower())
                if qkey in seen_queries:
                    result = "Same search as before; use the results you already have."
                else:
                    seen_queries.add(qkey)
                    try:
                        result = self.tool_executor(name, args)
                    except Exception as e:  # noqa: BLE001
                        result = f"tool error: {e}"
                        logger.warning(f"🤖⚠️ [{request_id}] tool '{name}' failed: {e}")
                    cb = getattr(self, "on_tool_result", None)
                    if cb is not None:
                        try:
                            cb(name, args, result)
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"🤖⚠️ [{request_id}] on_tool_result error: {e}")
                # Tools return text, or {"text": ..., "images": [b64]} (look_at_screen):
                # the pixels ride on the tool message so the vision model sees them.
                images = None
                if isinstance(result, dict):
                    images = result.get("images") or None
                    result = str(result.get("text") or "")
                result = str(result)
                tool_msg = {"role": "tool", "content": result}
                if images:
                    tool_msg["images"] = images
                working.append(tool_msg)
                logger.info(f"🤖🛠️ [{request_id}] tool '{name}' returned {len(result)} chars"
                            + (f" + {len(images)} image(s)." if images else "."))
            used = {(tc.get("function") or {}).get("name") for tc in tool_calls}
            if (_round + 1 == limit and read_bonus and not only_read
                    and "web_search" in used):
                read_bonus, only_read = False, True
                limit += 1
                working[-1] = dict(working[-1], content=working[-1]["content"] + (
                    "\n\n[If these results are only links/blurbs without the actual figure "
                    "they asked for, open the single best URL with read_page now. If the "
                    "figure is already here, skip that and answer with the required header.]"))
                logger.info(f"🤖📖 [{request_id}] read_page follow-up round offered.")
                continue
            only_read = False
            if _round + 1 == limit:
                # Next request is the forced-answer round (no tools offered).
                working[-1] = dict(working[-1], content=working[-1]["content"] + (
                    "\n\n[Tool budget used. Start with the required header now. If you chose "
                    "step_back, output only [HOLD]. Otherwise [SPEAK to=...] and say the answer "
                    "out loud in character, briefly, from what you found or saw; if it's still "
                    "searching or came up empty, say so honestly.]"))

        logger.warning(f"🤖⚠️ [{request_id}] Forced answer round produced no content.")

    def _run_prefetch(self, prefetch, working, seen_queries, request_id) -> None:
        """Execute one tool up front and append call + result to `working` (never raises)."""
        try:
            name, args = prefetch
            calls = [{"function": {"name": name, "arguments": dict(args or {})}}]
            if self.on_tool_call is not None:
                try:
                    self.on_tool_call(calls)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"🤖⚠️ [{request_id}] on_tool_call error: {e}")
            result = self.tool_executor(name, dict(args or {}))
            cb = getattr(self, "on_tool_result", None)
            if cb is not None:
                try:
                    cb(name, dict(args or {}), result)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"🤖⚠️ [{request_id}] on_tool_result error: {e}")
            images = None
            if isinstance(result, dict):
                images = result.get("images") or None
                result = str(result.get("text") or "")
            result = str(result)
            if images:
                result += ("\n[This screenshot was taken just now, for this turn. Describe only "
                           "what is in THIS image; anything you said about the screen before is out of date.]")
            working.append({"role": "assistant", "content": "", "tool_calls": calls})
            msg = {"role": "tool", "content": result}
            if images:
                msg["images"] = images
            working.append(msg)
            seen_queries.add((name, json.dumps(dict(args or {}), sort_keys=True).lower()))
            logger.info(f"🤖🛠️ [{request_id}] prefetched '{name}': {len(result)} chars"
                        + (f" + {len(images)} image(s)." if images else "."))
        except Exception as e:  # noqa: BLE001 - a failed prefetch must never block the turn
            logger.warning(f"🤖⚠️ [{request_id}] prefetch failed: {e}")

    def _yield_ollama_chunks(self, response: requests.Response, request_id: str) -> Generator[str, None, None]:
        """
        Iterates over an Ollama HTTP response stream, decoding JSON lines and yielding content.

        Handles reading bytes, decoding UTF-8, parsing JSON chunks, extracting message content,
        and checking for the 'done' signal. Checks for cancellation before processing each chunk.
        Ensures the response is closed upon completion, error, or cancellation.

        Args:
            response: The streaming requests.Response object from the Ollama API call.
            request_id: The unique ID associated with this generation stream.

        Yields:
            str: Content chunks from the stream's message objects.

        Raises:
            RuntimeError: If the Ollama stream returns an error message.
            ConnectionError: If a connection error occurs during streaming, unless likely due to cancellation.
            requests.exceptions.RequestException: For other request-related errors during streaming.
            Exception: For JSON decoding errors or other unexpected issues.
        """
        token_count = 0
        buffer = ""
        processed_done = False # Flag to track if 'done' message was processed
        try:
            # --- Start Change ---
            # Wrap the iteration in a try block to catch the specific AttributeError
            try:
                for chunk_bytes in response.iter_content(chunk_size=None): # None = read whatever is available
                    # Check for cancellation *before* processing chunk
                    with self._requests_lock:
                        if (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"]):
                            logger.info(f"🤖🗑️ Ollama stream {request_id} cancelled or finished externally during iteration (pre-chunk check).")
                            break # Exit the loop cleanly

                    if not chunk_bytes:
                        continue # Skip empty chunks

                    buffer += chunk_bytes.decode('utf-8')

                    # Process complete JSON objects separated by newlines in the buffer
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue # Skip empty lines

                        try:
                            chunk = json.loads(line)
                            if chunk.get('error'):
                                logger.error(f"🤖💥 Ollama stream returned error for {request_id}: {chunk['error']}")
                                raise RuntimeError(f"Ollama stream error: {chunk['error']}")
                            content = chunk.get('message', {}).get('content')
                            if content:
                                token_count += 1
                                yield content
                            if chunk.get('done'):
                                logger.debug(f"🤖✅ [{request_id}] Ollama signalled 'done'.")
                                # Ensure any remaining buffer is cleared (should be unlikely if 'done' is last)
                                buffer = ""
                                processed_done = True # Mark done as processed
                                break # Exit inner while loop on 'done'
                        except json.JSONDecodeError:
                            logger.warning(f"🤖⚠️ [{request_id}] Failed to decode JSON line: '{line[:100]}...'")
                            # Continue trying to process buffer
                        except Exception as e:
                            # Reraise other exceptions during JSON processing
                            logger.error(f"🤖💥 [{request_id}] Error processing Ollama stream chunk: {e}", exc_info=True)
                            raise # Reraise for outer try/except

                    # If 'done' was received and processed, break outer loop too
                    if processed_done:
                        break
            # Catch the specific error from the race condition
            except AttributeError as e:
                # cancellation closes the response mid-iteration; urllib3 then
                # hits a None _fp in read()/readline()/read_chunked(). The exact
                # attribute varies by where the close lands, so match the common
                # "'NoneType' object has no attribute" prefix, not a specific one.
                if self._is_cancelled(request_id) and "'NoneType' object has no attribute" in str(e):
                    with self._requests_lock:
                        is_cancelled = (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"])
                    logger.warning(
                        f"🤖⚠️ [{request_id}] Stream closed by concurrent cancellation "
                        f"(cancelled={is_cancelled}, err={e}). Stopping iteration."
                    )
                else:
                    # A genuinely different AttributeError — re-raise.
                    logger.error(f"🤖💥 [{request_id}] Unexpected AttributeError during Ollama stream iteration: {e}", exc_info=True)
                    raise
            # --- End Change ---


            # Check if loop exited due to cancellation flag (if AttributeError wasn't caught)
            if not processed_done: # Only log this if we didn't finish normally
                with self._requests_lock:
                    if (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"]):
                        logger.info(f"🤖🗑️ Ollama stream {request_id} processing stopped due to cancellation flag after loop.")

            logger.debug(f"🤖✅ [{request_id}] Finished yielding {token_count} Ollama tokens (processed_done={processed_done}).")

        except requests.exceptions.ChunkedEncodingError as e:
             # This can happen if the connection is closed prematurely (e.g., by cancellation)
             is_cancelled = False
             with self._requests_lock:
                 is_cancelled = (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"])
             if is_cancelled:
                 logger.warning(f"🤖⚠️ Ollama chunked encoding error likely due to cancellation for {request_id}: {e}")
                 # Don't raise an error if cancelled
             else:
                 logger.error(f"🤖💥 Ollama chunked encoding error during streaming ({request_id}): {e}")
                 # Reraise as ConnectionError for generate() to handle
                 raise ConnectionError(f"Ollama communication error during streaming: {e}") from e
        except requests.exceptions.RequestException as e:
            # Catch other request errors during streaming
            is_cancelled = False
            with self._requests_lock:
                is_cancelled = (request_id not in self._active_requests or self._active_requests[request_id]["cancelled"])
            if is_cancelled:
                 logger.warning(f"🤖⚠️ Ollama requests error likely due to cancellation for {request_id}: {e}")
                 # Don't raise an error if cancelled
            else:
                 logger.error(f"🤖💥 Ollama requests error during streaming ({request_id}): {e}")
                 # Reraise as ConnectionError for generate() to handle
                 raise ConnectionError(f"Ollama communication error during streaming: {e}") from e
        except Exception as e:
            # Catch the RuntimeError from 'error' field or other unexpected errors
            # Do not catch the AttributeError here if it was re-raised above
            if not isinstance(e, AttributeError):
                 logger.error(f"🤖💥 Unexpected error during Ollama streaming ({request_id}): {e}", exc_info=True)
            raise # Reraise for generate() to handle
        finally:
            self._release_stream(request_id, response)

    def measure_inference_time(
        self,
        num_tokens: int = 10,
        **kwargs: Any
    ) -> Optional[float]:
        """
        Measures the time taken to generate a target number of initial tokens.

        Uses a fixed, predefined prompt designed to elicit a somewhat predictable
        response length. Times the generation process from the moment the generator
        is obtained until the target number of tokens is yielded or generation ends.
        Ensures the backend client is initialized first.

        Args:
            num_tokens: The target number of tokens to generate before stopping measurement.
            **kwargs: Additional keyword arguments passed to the `generate` method
                      (e.g., temperature=0.1).

        Returns:
            The time taken in milliseconds to generate the actual number of tokens
            produced (up to `num_tokens`), or None if generation failed, produced 0 tokens,
            or encountered an error during initialization or generation.
        """
        if num_tokens <= 0:
            logger.warning("🤖⏱️ Cannot measure inference time for 0 or negative tokens.")
            return None

        # Ensure client is ready (handles lazy init + connection checks + ps fallback)
        if not self._lazy_initialize_clients():
            logger.error(f"🤖⏱️💥 Measurement failed: Could not initialize backend client/connection for {self.backend}.")
            return None

        # --- Define specific prompts for measurement ---
        measurement_system_prompt = "You are a precise assistant. Follow instructions exactly."
        # This text is designed to likely produce > 10 tokens across different tokenizers.
        measurement_user_prompt = "Repeat the following sequence exactly, word for word: one two three four five six seven eight nine ten eleven twelve"
        measurement_history = [
            {"role": "system", "content": measurement_system_prompt},
            {"role": "user", "content": measurement_user_prompt}
        ]
        # ---------------------------------------------

        req_id = f"measure-{self.backend}-{uuid.uuid4()}"
        logger.info(f"🤖⏱️ Measuring inference time for {num_tokens} tokens (Request ID: {req_id}). Using fixed measurement prompt.")
        logger.debug(f"🤖⏱️ [{req_id}] Measurement history: {measurement_history}")

        token_count = 0
        start_time = None
        end_time = None
        generator = None
        actual_tokens_generated = 0

        try:
            # Pass the constructed history and ensure use_system_prompt is False
            # The 'text' argument to generate is ignored when history is provided containing the user message.
            generator = self.generate(
                text="", # Text is ignored as history provides the user message
                history=measurement_history,
                use_system_prompt=False, # Explicitly disable default system prompt
                request_id=req_id,
                **kwargs # Pass any extra args like temperature
            )

            # Iterate and time
            start_time = time.time() # Start timing *after* generate() call returns generator
            for token in generator:
                if token_count == 0:
                     # Could capture TTFT here if needed: time.time() - start_time
                     pass
                token_count += 1
                # logger.debug(f"[{req_id}] Token {token_count}: '{token}'") # Optional: very verbose
                if token_count >= num_tokens:
                    end_time = time.time()
                    logger.debug(f"🤖⏱️ [{req_id}] Reached target {num_tokens} tokens.")
                    break # Stop iterating

            # If loop finished without breaking, record end time here
            if end_time is None:
                end_time = time.time()
                logger.debug(f"🤖⏱️ [{req_id}] Generation finished naturally after {token_count} tokens (may be less than requested {num_tokens}).")

            actual_tokens_generated = token_count

        except (ConnectionError, APIError, RuntimeError, Exception) as e:
            logger.error(f"🤖⏱️💥 Error during inference time measurement ({req_id}): {e}", exc_info=False)
            # Let finally block handle potential generator cleanup
            return None # Indicate failure
        finally:
            # Ensure generator resources are released if the loop was broken early
            # The generate() method's finally block handles request tracking removal AND attempts close.
            # We still explicitly try closing the generator here as a fallback.
            if generator and hasattr(generator, 'close'):
                try:
                    logger.debug(f"🤖⏱️🗑️ [{req_id}] Closing generator in measure_inference_time finally.")
                    generator.close()
                except Exception as close_err:
                    # Log but don't prevent returning time if measured
                    logger.warning(f"🤖⏱️⚠️ [{req_id}] Error closing generator in finally: {close_err}", exc_info=False)
            generator = None # Clear reference


        # --- Calculate and Return Result ---
        if start_time is None or end_time is None:
             logger.error(f"🤖⏱️💥 [{req_id}] Measurement failed: Start or end time not recorded.")
             return None

        if actual_tokens_generated == 0:
             logger.warning(f"🤖⏱️⚠️ [{req_id}] Measurement invalid: 0 tokens were generated.")
             return None

        duration_sec = end_time - start_time
        duration_ms = duration_sec * 1000

        logger.info(
            f"🤖⏱️✅ Measured ~{duration_ms:.2f} ms for {actual_tokens_generated} tokens "
            f"(target: {num_tokens}) for model '{self.model}' on backend '{self.backend}' using fixed prompt. (Request ID: {req_id})"
        )

        # Return the time taken for the actual tokens generated.
        return duration_ms


# --- Context Manager ---
class LLMGenerationContext:
    """
    A context manager for safely handling LLM generation streams.

    Ensures that the underlying generation stream is properly requested for cancellation
    (including attempting to close the network connection) when the context is exited,
    whether normally or due to an exception.
    """
    def __init__(
        self,
        llm: LLM,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        **kwargs: Any
        ):
        """
        Initializes the generation context.

        Args:
            llm: The LLM instance to use for generation.
            prompt: The user's input prompt/text.
            history: Optional list of previous messages.
            use_system_prompt: If True, uses the LLM's configured system prompt.
            **kwargs: Additional arguments to pass to the `llm.generate` method.
        """
        self.llm = llm
        self.prompt = prompt
        self.history = history
        self.use_system_prompt = use_system_prompt
        self.kwargs = kwargs
        self.generator: Optional[Generator[str, None, None]] = None
        self.request_id: str = f"ctx-{llm.backend}-{uuid.uuid4()}"
        self._entered: bool = False

    def __enter__(self) -> Generator[str, None, None]:
        """
        Enters the context, starts generation, and returns the token generator.

        Calls the LLM's `generate` method and registers the request.

        Returns:
            A generator yielding tokens from the LLM.

        Raises:
            RuntimeError: If the context is re-entered or generator creation fails.
            (Propagates exceptions from `llm.generate`).
        """
        if self._entered:
            raise RuntimeError("LLMGenerationContext cannot be re-entered")
        self._entered = True
        logger.debug(f"🤖▶️ [{self.request_id}] Entering LLMGenerationContext.")
        try:
            # Generate call now implicitly runs lazy_init (with ollama ps check restored)
            self.generator = self.llm.generate(
                self.prompt,
                self.history,
                self.use_system_prompt,
                request_id=self.request_id,
                **self.kwargs
            )
            return self.generator
        except Exception as e:
            logger.error(f"🤖💥 [{self.request_id}] Failed generator creation in context: {e}", exc_info=True)
            # Attempt to clean up if registration happened before error (tries close)
            self.llm.cancel_generation(self.request_id)
            self._entered = False
            raise # Reraise the exception

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context, ensuring the generation stream is cancelled and closed.

        Calls `llm.cancel_generation` to remove tracking and attempt stream closure.
        Also explicitly attempts to close the generator object itself as a safeguard.

        Args:
            exc_type: The type of exception that caused the context to be exited (if any).
            exc_val: The exception instance (if any).
            exc_tb: The traceback (if any).

        Returns:
            False, indicating that exceptions (if any) should not be suppressed.
        """
        logger.debug(f"🤖◀️ [{self.request_id}] Exiting LLMGenerationContext (Exc: {exc_type}).")
        # Calls the modified cancel_generation, which now attempts to close the stream
        self.llm.cancel_generation(self.request_id) # Removes tracking & attempts close

        # Explicit close attempt in __exit__ is now less critical as cancel_generation
        # and the _yield_* helpers' finally blocks also attempt closure.
        # Keep it as a final safeguard.
        if self.generator and hasattr(self.generator, 'close'):
            try:
                logger.debug(f"🤖🗑️ [{self.request_id}] Explicitly closing generator in context exit (final check).")
                self.generator.close()
            except Exception as e:
                 logger.warning(f"🤖⚠️ [{self.request_id}] Error closing generator in context exit: {e}")

        self.generator = None
        self._entered = False
        # If an exception occurred, don't suppress it
        return False



# --- Example Usage ---
if __name__ == "__main__":
    # Setup logging for the example itself
    # Use basicConfig here as it's the main script
    main_log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    main_log_level = getattr(logging, main_log_level_str, logging.INFO)
    logging.basicConfig(level=main_log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)
    main_logger = logging.getLogger(__name__) # Logger for this __main__ block
    main_logger.info("🤖🚀 --- Running LLM Module Example (With Ollama PS Check Restored) ---") # Modified title

    # --- Ollama Example ---
    ollama_llm = None
    if REQUESTS_AVAILABLE:
        try:
            # Ensure OLLAMA_MODEL env var is set or use a default
            ollama_model_env = os.getenv("OLLAMA_MODEL")
            if not ollama_model_env:
                 main_logger.warning("🤖⚠️ OLLAMA_MODEL environment variable not set. Using default 'llama3:instruct'.")
                 ollama_model_env = "llama3:instruct"

            main_logger.info(f"\n🤖⚙️ --- Initializing Ollama ({ollama_model_env}) ---")
            # Pass the model name fetched from env var
            ollama_llm = LLM(
                backend="ollama",
                model=ollama_model_env,
                system_prompt="You are concise and helpful."
            )

            # Prewarm will now trigger lazy init WITH the ps check fallback restored
            main_logger.info("🤖🔥 --- Running Ollama Prewarm (will trigger lazy init with ps check if needed) ---")
            prewarm_success = ollama_llm.prewarm(max_retries=0) # Only one attempt for prewarm after init

            if prewarm_success:
                 main_logger.info("🤖✅ Ollama Prewarm/Initialization OK.")

                 # --- Run Measurement ---
                 main_logger.info("🤖⏱️ --- Running Ollama Inference Time Measurement ---")
                 inf_time = ollama_llm.measure_inference_time(num_tokens=10, temperature=0.1)
                 if inf_time is not None:
                     main_logger.info(f"🤖⏱️ --- Measured Inference Time: {inf_time:.2f} ms ---")
                 else:
                     main_logger.warning("🤖⏱️⚠️ --- Inference Time Measurement Failed ---")

                 # --- Run Generation ---
                 main_logger.info("🤖▶️ --- Running Ollama Generation via Context (Post-Prewarm) ---")
                 try:
                     # Use the context manager
                     with LLMGenerationContext(ollama_llm, "What is the capital of France? Respond briefly.") as generator:
                         print("\nOllama Response: ", end="", flush=True)
                         response_text = ""
                         for token in generator:
                             print(token, end="", flush=True)
                             response_text += token
                         print("\n") # Newline after response
                     main_logger.info("🤖✅ Ollama generation complete.")

                     # Example of direct generate call (after context)
                     main_logger.info("🤖💬 --- Running Ollama Generation via direct call ---")
                     direct_gen = ollama_llm.generate("List three large cities in Germany.")
                     print("\nOllama Direct Response: ", end="", flush=True)
                     for token in direct_gen:
                          print(token, end="", flush=True)
                     print("\n")
                     main_logger.info("🤖✅ Ollama direct generation complete.")

                 except (ConnectionError, RuntimeError, Exception) as e:
                     # Catch specific ConnectionError raised on init/gen failure
                     if isinstance(e, ConnectionError):
                          main_logger.error(f"🤖💥 Ollama Connection Error during Generation: {e}")
                          main_logger.error("   🤖🔌 Please ensure the Ollama server is running and accessible at the configured URL.")
                     else:
                          main_logger.error(f"🤖💥 Ollama Generation Runtime/Other Error: {e}", exc_info=True)

            else:
                 main_logger.error("🤖❌ Ollama Prewarm/Initialization Failed. Could not connect or encountered error. Skipping measurement and generation tests.")

        except (ImportError, ValueError, Exception) as e:
             main_logger.error(f"🤖💥 Failed to initialize or run Ollama: {e}", exc_info=True)
    else:
        main_logger.warning("🤖⚠️ Skipping Ollama tests: 'requests' library not installed.")

    # --- Add LMStudio/OpenAI examples if needed ---
    # ... (similar structure, ensure OPENAI_AVAILABLE check)

    main_logger.info("\n" + "="*40)
    main_logger.info("🤖🏁 --- LLM Module Example Script Finished ---")