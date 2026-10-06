# speech_pipeline_manager.py
from typing import Optional, Callable
import threading
import re
import os
import logging
import time
from queue import Queue, Empty
import sys

# (Make sure real/mock imports are correct)
from audio_module import AudioProcessor
from text_similarity import TextSimilarity
from text_context import TextContext
from llm_module import LLM
from response_decision import ResponseDecision, filter_response, finish_silent_generation
from colors import Colors
from conversation_dynamics import ConversationDynamics, is_direct_address

# Web search tool (optional). If websearch.py is importable, the agent can search
# the live web when the model decides it needs current facts. The tool loop lives
# in llm_module._generate_ollama_agentic; the executor just wraps websearch.search.
try:
    import websearch as _websearch
    WEB_SEARCH_TOOLS = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": ("Search the live web. Your call: use it whenever current facts, news, prices, "
                            "scores or anything you're unsure of would make your answer better. It keeps "
                            "running in the background even if you get interrupted; the result lands on "
                            "your TASK BOARD."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query"}
                },
                "required": ["query"],
            },
        },
    }]

    def _execute_web_search(name: str, args) -> str:
        if name != "web_search":
            return f"unknown tool: {name}"
        query = args.get("query", "") if isinstance(args, dict) else str(args)
        result = _websearch.search(query)
        return result.get("text") or "No results found."
except ImportError:
    _websearch = None
    WEB_SEARCH_TOOLS = []
    _execute_web_search = None
    logging.getLogger(__name__).warning("🗣️⚠️ websearch.py not found; web search disabled.")

# Screen-look tool (native vision): same tool loop as web_search, but the
# result carries the screenshot pixels back to the model.
try:
    import screen as _screen
    SCREEN_TOOLS = [_screen.SCREEN_TOOL]
except Exception as _e:  # noqa: BLE001
    _screen = None
    SCREEN_TOOLS = []

def _fn(name, desc, props=None, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": list(required)}}}


# Tools the agent uses at its own discretion. None of them run code or touch
# files; read_page refuses local addresses (websearch.read_page).
READ_PAGE_TOOLS = [_fn(
    "read_page", "Read the text of ONE web page (a URL from your search results) when the "
    "snippets aren't enough to answer properly.",
    {"url": {"type": "string", "description": "http(s) URL from a search result"}}, ["url"])
] if WEB_SEARCH_TOOLS else []
BOARD_TOOLS = [
    _fn("note_to_self", "Write a short private note on your TASK BOARD for later: something "
        "to follow up on, a promise you made, a detail worth keeping, a prompt for your future "
        "self. You'll see open notes on later turns.",
        {"text": {"type": "string", "description": "the note, one sentence"}}, ["text"]),
    _fn("clear_note", "Remove a note from your TASK BOARD once it's done or no longer useful.",
        {"id": {"type": "integer", "description": "note number"}}, ["id"]),
    _fn("step_back", "Choose to go quiet for a few minutes when you're talking too much, "
        "the room is busy with something that isn't about you, or people just want to hang. "
        "Your name still wakes you. After calling it, output only [HOLD].",
        {"minutes": {"type": "number", "description": "0.5 to 5"}}),
]
import clock as _clock
import self_tools as _self_tools
import room_tools as _room
AGENT_TOOLS = (list(WEB_SEARCH_TOOLS) + READ_PAGE_TOOLS + list(SCREEN_TOOLS) + BOARD_TOOLS
               + [_clock.TOOL] + list(_self_tools.TOOLS) + list(_room.TOOLS))
SEARCH_WAIT_S = 9.0   # how long a reply waits for its search before handing it to the board


def _board_search(query: str) -> str:
    """Background-worker search (no LLM). Parallel backend race lives in websearch."""
    if _websearch is None:
        return "search unavailable"
    return (_websearch.search(query) or {}).get("text") or "No results found."


def _execute_tool(name: str, args):
    """Dispatch every agent tool. Returns text, or {"text", "images"} for looks."""
    if name == "look_at_screen" and _screen is not None:
        reason = args.get("reason", "") if isinstance(args, dict) else ""
        return _screen.capture(reason)
    if name == "web_search" and _execute_web_search is not None:
        return _execute_web_search(name, args)
    if name == "check_date_time":
        tz = args.get("timezone", "") if isinstance(args, dict) else ""
        return _clock.check(str(tz or ""))
    return f"unknown tool: {name}"

# (Logging setup)
logger = logging.getLogger(__name__)

# Rolling history window: keep the last N messages (≈N/2 turns). Bounds the LLM
# prefill time so TTFT stays flat on long calls instead of growing with history.
HISTORY_MAX_MESSAGES = 16

# Long-call memory: when history grows past this many messages, the oldest
# are asynchronously compacted into a rolling summary so the agent keeps
# call-level context without unbounded prefill. Runs off the hot path.
HISTORY_SUMMARY_THRESHOLD = 24
HISTORY_KEEP_RECENT = 10
SUMMARY_INTERVAL_S = 15.0
# Diarization-backed, token-budgeted conversation log (conversation_log.py).
# ATLAS_CONVO_LOG=0 falls back to the raw 16-message history.
CONVO_LOG_ENABLED = os.environ.get('ATLAS_CONVO_LOG', '1') != '0'
VIBE_ENABLED = os.environ.get('ATLAS_VIBE', '1') != '0'  # banter/roast steering (floor.py)
LOOKUP_TTL_S = 15 * 60
LOOKUP_SNIPPET_CHARS = 550
ENABLE_MEMORY = False  # mem0 stripped — slot contention + chromadb bugs out the pipeline

# Agents come from agent_registry (dev pack + user-created). No names are
# hard-coded here, so a public build with zero agents still boots: it gets one
# placeholder agent ("setup needed") until the owner creates a real one.
import agent_registry as _registry

PLACEHOLDER_ID = "agent"
PERSONA_FILES: dict = {}
PERSONAS: dict = {}
PERSONA_META: dict = {}
DEFAULT_PERSONA = PLACEHOLDER_ID


def _rebuild_personas() -> None:
    """(Re)load every agent's prompt + display meta from the registry, in place."""
    global DEFAULT_PERSONA
    PERSONA_FILES.clear(); PERSONAS.clear(); PERSONA_META.clear()
    for _aid in _registry.agent_ids():
        _txt = _registry.prompt_text(_aid)
        if not _txt:
            continue
        PERSONAS[_aid] = _txt
        PERSONA_FILES[_aid] = _registry.agent(_aid).get("prompt_file") or ""
        PERSONA_META[_aid] = _registry.meta(_aid)
        logger.info(f"🗣️📄 Persona '{_aid}' loaded ({len(_txt)} chars).")
    if not PERSONAS:
        PERSONAS[PLACEHOLDER_ID] = _registry.GENERIC_PROMPT
        PERSONA_META[PLACEHOLDER_ID] = {"role": "no agents yet", "accent": "#64748b",
                                        "name": "Setup", "placeholder": True}
        logger.warning("🗣️📄 No agents installed. Using the setup placeholder.")
    d = _registry.default_agent()
    try:  # the agent the user created/picked in setup wins in the public build
        import user_settings as _us
        if _us.is_public() and _us.load().get("agent") in PERSONAS:
            d = _us.load()["agent"]
    except Exception:  # noqa: BLE001
        pass
    DEFAULT_PERSONA = d if d in PERSONAS else next(iter(PERSONAS))


_rebuild_personas()
system_prompt = PERSONAS[DEFAULT_PERSONA]


def display_name(pid: str) -> str:
    return PERSONA_META.get(pid, {}).get("name") or _registry.display_name(pid)


USE_ORPHEUS_UNCENSORED = False

orpheus_prompt_addon_normal = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", and " <gasp> ".

Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()

orpheus_prompt_addon_uncensored = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <moans> ", " <panting> ", " <grunting> ", " <gagging sounds> ", " <chokeing> ", " <kissing noises> ", " <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", " <gasp> ".
Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()

orpheus_prompt_addon = orpheus_prompt_addon_uncensored if USE_ORPHEUS_UNCENSORED else orpheus_prompt_addon_normal


class PipelineRequest:
    """
    Represents a request to be processed by the SpeechPipelineManager's request queue.

    Holds information about the action to perform (e.g., 'prepare', 'abort'),
    associated data (e.g., text input), and a timestamp for potential de-duplication.
    """
    def __init__(self, action: str, data: Optional[any] = None):
        """
        Initializes a PipelineRequest instance.

        Args:
            action: The type of action requested (e.g., "prepare", "abort").
            data: Optional data associated with the action (e.g., input text for "prepare").
        """
        self.action = action
        self.data = data
        self.timestamp = time.time()

class RunningGeneration:
    """
    Holds the state and resources for a single, ongoing text-to-speech generation process.

    This includes the generation ID, input text, the LLM generator object, flags indicating
    the status of LLM and TTS stages (quick and final), threading events for synchronization,
    queues for audio chunks, and text buffers for partial/complete answers.
    """
    def __init__(self, id: int):
        """
        Initializes a RunningGeneration state object.

        Args:
            id: A unique identifier for this generation attempt.
        """
        self.id: int = id # Store the generation ID
        self.text: Optional[str] = None
        self.timestamp = time.time()

        self.llm_generator = None
        self.decision = ResponseDecision()
        self.response_held = False
        self.llm_finished: bool = False
        self.llm_finished_event = threading.Event()
        self.llm_aborted: bool = False

        self.quick_answer: str = ""
        self.quick_answer_provided: bool = False
        self.quick_answer_first_chunk_ready: bool = False
        self.quick_answer_overhang: str = "" # This is the part of the text that was not used in the context
        self.tts_quick_started: bool = False

        self.tts_quick_allowed_event = threading.Event()
        self.audio_chunks = Queue()
        self.audio_quick_finished: bool = False
        self.audio_quick_aborted: bool = False
        self.tts_quick_finished_event = threading.Event()

        self.abortion_started: bool = False

        self.tts_final_finished_event = threading.Event()
        self.tts_final_started: bool = False
        self.audio_final_aborted: bool = False
        self.audio_final_finished: bool = False
        self.final_answer: str = ""

        self.completed: bool = False


class _AgentField:
    """Attribute that lives on the ACTIVE agent's runtime (agent_runtime.py)."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(obj._runtime(), self.name)

    def __set__(self, obj, value):
        setattr(obj._runtime(), self.name, value)


class SpeechPipelineManager:
    history = _AgentField()
    lookups = _AgentField()
    floor = _AgentField()
    threads = _AgentField()
    bait = _AgentField()
    convo = _AgentField()
    cutoff = _AgentField()

    def _runtime(self):
        """Active agent runtime; created lazily so __new__-built test stubs work."""
        reg = self.__dict__.get("agents")
        if reg is None:
            from agent_runtime import AgentRegistry
            cur = self.__dict__.get("current_persona") or DEFAULT_PERSONA
            reg = self.agents = AgentRegistry(
                prompt_for=lambda pid: PERSONAS.get(pid, ""), name_for=display_name,
                meta_for=lambda pid: PERSONA_META.get(pid, {}))
            reg.activate(cur)
        return reg.active

    @property
    def agent(self):
        """The active AgentRuntime (profile + everything this agent knows)."""
        return self._runtime()

    """
    Orchestrates the text-to-speech pipeline, managing LLM and TTS workers.

    This class handles incoming text requests, manages the lifecycle of a generation
    (including LLM inference, TTS synthesis for both quick and final parts),
    facilitates aborting ongoing generations, manages conversation history,
    and coordinates worker threads using queues and events.
    """
    def __init__(
            self,
            tts_engine: str = "kokoro",
            llm_provider: str = "ollama",
            llm_model: str = "hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M",
            no_think: bool = False,
            orpheus_model: str = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf",
        ):
        """
        Initializes the SpeechPipelineManager.

        Sets up configuration, instantiates dependencies (AudioProcessor, LLM, etc.),
        loads system prompts, initializes state variables (queues, events, flags),
        measures initial inference latencies, and starts the background worker threads.

        Args:
            tts_engine: The TTS engine to use (e.g., "kokoro", "orpheus").
            llm_provider: The LLM backend provider (e.g., "ollama").
            llm_model: The specific LLM model identifier.
            no_think: If True, removes specific thinking tags from LLM output.
            orpheus_model: Path or identifier for the Orpheus TTS model, if used.
        """
        self.tts_engine = tts_engine
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.no_think = no_think
        self.orpheus_model = orpheus_model

        from identity import note as _id_note
        try:
            import response_decision as _rd_life; _rd_life.set_agent(DEFAULT_PERSONA)
        except Exception:
            pass
        self.system_prompt = f"Your name is {display_name(DEFAULT_PERSONA)}. {_id_note(DEFAULT_PERSONA, display_name(DEFAULT_PERSONA))} " + system_prompt
        self.current_persona = DEFAULT_PERSONA
        if tts_engine == "orpheus":
            self.system_prompt += f"\n{orpheus_prompt_addon}"

        # --- Instance Dependencies ---
        self.audio = AudioProcessor(
            engine=self.tts_engine,
            orpheus_model=self.orpheus_model
        )
        self.audio.on_first_audio_chunk_synthesize = self.on_first_audio_chunk_synthesize
        self.text_similarity = TextSimilarity(focus='end', n_words=5)
        self.text_context = TextContext()
        self.generation_counter: int = 0
        self.abort_lock = threading.Lock()
        self.llm = LLM(
            backend=self.llm_provider, # Or your backend
            model=self.llm_model,
            system_prompt=self.system_prompt,
            no_think=no_think,
            tools=AGENT_TOOLS or None,
            tool_executor=_execute_tool if AGENT_TOOLS else None,
        )
        # Search backchannels: pre-rendered "hold on, let me look" clips in the
        # active voice, played the moment the model actually calls web_search.
        from backchannels import Backchannels
        self.backchannels = Backchannels()
        self.on_backchannel = None  # set by server: callable(int16_24k_bytes)
        self._backchannel_gen = None
        self.llm.on_tool_call = self._on_tool_call
        # Lookup memory: search results + the backchannel he said, kept in the
        # prompt for a while so "did you find it?" has something to refer to.
        self.llm.on_tool_result = self._on_tool_result
        # Task board (tasks.py): searches outlive interrupted turns; notes to self.
        from tasks import Boards
        self.boards = Boards(root="memory_db", searcher=_board_search)
        if AGENT_TOOLS:
            self.llm.tool_executor = self._run_tool
        # Plugins (plugins.py): the owner switches abilities on/off; the model is only
        # offered tools whose plugin is on, and _run_tool refuses the rest.
        import plugins as _plugins
        _plugins.apply_all()
        self._refresh_plugin_tools()
        _plugins.on_change(self._refresh_plugin_tools)
        # A fresh install may have no model yet (setup wizard pulls it, or the
        # user picked a cloud provider): boot anyway, never crash on prewarm.
        samples = []
        try:
            self.llm.prewarm()
            # Best-of-3 WARM measurement. A single sample right after load can read
            # 4-5s (cold KV/graph setup) and that number becomes the turn-end
            # silence floor -> every turn waited ~5s. Take the fastest warm run.
            samples = [t for t in (self.llm.measure_inference_time() for _ in range(3)) if t]
        except Exception as e:  # noqa: BLE001
            logger.warning("🗣️⚠️ local LLM not ready at boot (%s): continuing; setup/cloud can supply one", e)
        self.llm_inference_time = min(samples) if samples else 500.0
        logger.info(f"🗣️⏱️ LLM warm inference samples (ms): {[round(t) for t in samples]} -> using {self.llm_inference_time:.0f}")
        logger.debug(f"🗣️🧠🕒 LLM inference time: {self.llm_inference_time:.2f}ms")

        # --- State ---
        # Per-agent state (history, lookups, floor, threads, bait, convo) lives in
        # one AgentRuntime per persona. See agent_runtime.py.
        from agent_runtime import AgentRegistry
        self.agents = AgentRegistry(
            prompt_for=lambda pid: PERSONAS.get(pid, ""), name_for=display_name,
            meta_for=lambda pid: PERSONA_META.get(pid, {}))
        self.agents.activate(self.current_persona)
        self.requests_queue = Queue()
        self.running_generation: Optional[RunningGeneration] = None
        # Speak / stay-silent gate (zero-latency: pure arithmetic on live signals).
        self.dynamics = ConversationDynamics(agent_names=(self.current_persona,))
        # Multi-party floor: who the agent is talking with right now (floor.py).
        self.vibe_enabled = VIBE_ENABLED
        # Voice tag -> human name. Survives persona switches (same people in the call).
        from people import NameBook
        self.people = NameBook(agent_names=tuple(PERSONAS))
        try:
            import name_hearing
            name_hearing.set_names([display_name(getattr(self, 'current_persona', '') or DEFAULT_PERSONA)])
            name_hearing.set_protect(lambda: [p.name for p in list(self.people.people.values()) if p.name])
        except Exception as e:  # noqa: BLE001
            logger.warning('name_hearing init failed: %s', e)
        # Per-agent semantic hypergraph memory (Hebbian strengthening + decay).
        # Embeds on CPU in a background writer; the reply path only reads, with a time budget.
        self.hgmem = None
        try:
            import hypergraph_memory as _hg
            if _hg.ENABLED:
                self.hgmem = _hg.MemoryService()
                def _warm(svc=self.hgmem):
                    try:
                        import call_memory as _cm
                        import agent_memory as _am
                        _am.ensure_all(list(PERSONAS))   # every agent gets its memory folder
                        _hg.shared_embedder().encode(["warm up"])
                        for p in list(PERSONAS):
                            svc.sync_approved(p, [it["text"] for it in _cm.approved(p)])
                        logger.info("hypergraph memory ready")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("hypergraph warmup failed: %s", e)
                threading.Thread(target=_warm, name="HypergraphWarm", daemon=True).start()
        except Exception as e:  # noqa: BLE001
            logger.warning("hypergraph memory disabled: %s", e)
        self.convo_enabled = CONVO_LOG_ENABLED

        # --- Threading Events ---
        self.shutdown_event = threading.Event()
        self.generator_ready_event = threading.Event()
        self.llm_answer_ready_event = threading.Event()
        self.stop_everything_event = threading.Event()
        self.stop_llm_request_event = threading.Event()
        self.stop_llm_finished_event = threading.Event()
        self.stop_tts_quick_request_event = threading.Event()
        self.stop_tts_quick_finished_event = threading.Event()
        self.stop_tts_final_request_event = threading.Event()
        self.stop_tts_final_finished_event = threading.Event()
        self.abort_completed_event = threading.Event()
        self.abort_block_event = threading.Event()
        self.abort_block_event.set()
        self.check_abort_lock = threading.Lock()

        # --- State Flags ---
        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        self.previous_request = None

        # --- Worker Threads ---
        self.request_processing_thread = threading.Thread(target=self._request_processing_worker, name="RequestProcessingThread", daemon=True)
        self.llm_inference_thread = threading.Thread(target=self._llm_inference_worker, name="LLMProcessingThread", daemon=True)
        self.tts_quick_inference_thread = threading.Thread(target=self._tts_quick_inference_worker, name="TTSQuickProcessingThread", daemon=True)
        self.tts_final_inference_thread = threading.Thread(target=self._tts_final_inference_worker, name="TTSFinalProcessingThread", daemon=True)

        self.request_processing_thread.start()
        self.llm_inference_thread.start()
        self.tts_quick_inference_thread.start()
        self.tts_final_inference_thread.start()

        # Rolling-summary worker: compacts old history off the hot path.
        self.summary_lock = threading.Lock()
        self.summary_thread = threading.Thread(target=self._summarize_worker, name="SummaryWorker", daemon=True)
        self.summary_thread.start()

        # Situation-memory worker: extracts + persists situation facts off the hot path.
        self.memories = {}
        self.memory_queue = Queue()
        self.memory_lock = threading.Lock()
        self.memory_worker_thread = threading.Thread(target=self._memory_worker, name="MemoryWorker", daemon=True)
        self.memory_worker_thread.start()

        self.on_partial_assistant_text: Optional[Callable[[str], None]] = None

        self.full_output_pipeline_latency = self.llm_inference_time + self.audio.tts_inference_time
        logger.info(f"🗣️⏱️ Full output pipeline latency: {self.full_output_pipeline_latency:.2f}ms (LLM: {self.llm_inference_time:.2f}ms, TTS: {self.audio.tts_inference_time:.2f}ms)")

        logger.info("🗣️🚀 SpeechPipelineManager initialized and workers started.")

    def is_valid_gen(self) -> bool:
        """
        Checks if there is a currently running generation that has not started aborting.

        Returns:
            True if `running_generation` exists and its `abortion_started` flag is False,
            False otherwise.
        """
        return self.running_generation is not None and not self.running_generation.abortion_started

    def _request_processing_worker(self):
        """
        Worker thread target that processes requests from the `requests_queue`.

        Continuously monitors the queue. When a request arrives, it drains the queue
        to process only the most recent one, preventing processing of stale requests.
        It waits for any ongoing abort operation to complete (`abort_block_event`)
        before processing the next request. Handles 'prepare' actions by calling
        `process_prepare_generation`. Runs until `shutdown_event` is set.
        """
        logger.info("🗣️🚀 Request Processor: Starting...")
        while not self.shutdown_event.is_set():
            try:
                # Get the most recent request by emptying the queue first
                request = self.requests_queue.get(block=True, timeout=1)

                if self.previous_request:
                    # Simple timestamp-based deduplication for identical consecutive requests
                    if self.previous_request.data == request.data and isinstance(request.data, str):
                        if request.timestamp - self.previous_request.timestamp < 2:
                            logger.info(f"🗣️🗑️ Request Processor: Skipping duplicate request - {request.action}")
                            continue

                # Drain the queue to get the most recent request
                while not self.requests_queue.empty():
                    skipped_request = self.requests_queue.get(False)  # Non-blocking get
                    logger.debug(f"🗣️🗑️ Request Processor: Skipping older request - {skipped_request.action}")
                    request = skipped_request # Keep the last one we retrieved
                
                self.abort_block_event.wait() # Wait if an abort is in progress
                logger.debug(f"🗣️🔄 Request Processor: Processing most recent request - {request.action}")
                
                if request.action == "prepare":
                    self.process_prepare_generation(request.data)
                    self.previous_request = request
                elif request.action == "finish":
                     # Note: 'finish' action currently has no specific handling logic here.
                     logger.info(f"🗣️🤷 Request Processor: Received 'finish' action (currently no-op).")
                     self.previous_request = request # Still update previous_request
                else:
                    logger.warning(f"🗣️❓ Request Processor: Unknown action '{request.action}'")

            except Empty:
                continue
            except Exception as e:
                logger.exception(f"🗣️💥 Request Processor: Error: {e}")
        logger.info("🗣️🏁 Request Processor: Shutting down.")

    def on_first_audio_chunk_synthesize(self):
        """
        Callback method invoked by AudioProcessor when the first TTS audio chunk is ready.

        Sets the `quick_answer_first_chunk_ready` flag on the current `running_generation`
        if one exists. This flag might be used for fine-grained timing or state checks.
        """
        logger.info("🗣️🎶 First audio chunk synthesized. Setting TTS quick allowed event.")
        if self.running_generation:
            self.running_generation.quick_answer_first_chunk_ready = True

    def preprocess_chunk(self, chunk: str) -> str:
        """
        Preprocesses a text chunk before sending it to the TTS engine.

        Replaces specific characters (em-dashes, quotes, ellipsis) with simpler equivalents
        to potentially improve TTS pronunciation or compatibility.

        Args:
            chunk: The input text chunk.

        Returns:
            The preprocessed text chunk.
        """
        return chunk.replace("—", "-").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("…", "...")

    def clean_quick_answer(self, text: str) -> str:
        """
        Removes specific leading patterns (like '<think>', newlines, spaces) from text.

        Intended for cleaning the initial output of the LLM, especially when `no_think`
        is enabled, to remove processing tags before TTS.

        Args:
            text: The input text.

        Returns:
            The text with specified leading patterns removed.
        """
        patterns_to_remove = ["<think>", "</think>", "\n", " "]
        previous_text = None
        current_text = text
        
        while previous_text != current_text:
            previous_text = current_text
            
            # Remove all patterns from the beginning of the string
            for pattern in patterns_to_remove:
                while current_text.startswith(pattern):
                    current_text = current_text[len(pattern):]
        
        return current_text

    def _llm_inference_worker(self):
        """
        Worker thread target that handles LLM inference for a generation.

        Waits for `generator_ready_event`. Once signaled, it iterates through the
        LLM generator provided in `running_generation`. It accumulates the generated
        text, optionally cleans it (`no_think`), checks for a natural sentence boundary
        to define the `quick_answer` using `TextContext`. If a quick answer is found,
        it signals `llm_answer_ready_event`. Handles stop requests (`stop_llm_request_event`)
        and signals completion/abortion via `stop_llm_finished_event` and internal flags.
        Runs until `shutdown_event` is set.
        """
        logger.info("🗣️🧠 LLM Worker: Starting...")
        while not self.shutdown_event.is_set():
            
            ready = self.generator_ready_event.wait(timeout=1.0)
            if not ready:
                continue

            # Check if aborted *while waiting* before clearing the ready event
            if self.stop_llm_request_event.is_set():
                logger.info("🗣️🧠❌ LLM Worker: Abort detected while waiting for generator_ready_event.")
                self.stop_llm_request_event.clear()
                self.stop_llm_finished_event.set()
                self.llm_generation_active = False
                continue # Go back to waiting

            self.generator_ready_event.clear()
            self.stop_everything_event.clear() # Assuming a new generation clears global stop
            current_gen = self.running_generation

            if not current_gen or not current_gen.llm_generator:
                logger.warning("🗣️🧠❓ LLM Worker: No valid generation or generator found after event.")
                self.llm_generation_active = False
                continue # Go back to waiting

            gen_id = current_gen.id
            logger.info(f"🗣️🧠🔄 [Gen {gen_id}] LLM Worker: Processing generation...")

            # Set state for active generation
            self.llm_generation_active = True
            self.stop_llm_finished_event.clear()
            start_time = time.time()
            token_count = 0

            try:
                for chunk in current_gen.llm_generator:
                    # Check for stop *before* processing the chunk
                    if self.stop_llm_request_event.is_set():
                        logger.info(f"🗣️🧠❌ [Gen {gen_id}] LLM Worker: Stop request detected during iteration.")
                        self.stop_llm_request_event.clear()
                        current_gen.llm_aborted = True
                        break # Exit the generator loop

                    chunk = self.preprocess_chunk(chunk)
                    token_count += 1
                    current_gen.quick_answer += chunk
                    if self.no_think:
                        current_gen.quick_answer = self.clean_quick_answer(current_gen.quick_answer)

                    if token_count == 1:
                        logger.info(f"🗣️🧠⏱️ [Gen {gen_id}] LLM Worker: TTFT: {(time.time() - start_time):.4f}s")

                    # Check for quick answer boundary only if not already provided
                    if not current_gen.quick_answer_provided:
                        context, overhang = self.text_context.get_context(current_gen.quick_answer)
                        if context:
                            logger.info(f"🗣️🧠✔️ [Gen {gen_id}] LLM Worker:  {Colors.apply('QUICK ANSWER FOUND:').magenta} {context}, overhang: {overhang}")
                            current_gen.quick_answer = context
                            if self.on_partial_assistant_text:
                                self.on_partial_assistant_text(current_gen.quick_answer)
                            current_gen.quick_answer_overhang = overhang
                            current_gen.quick_answer_provided = True
                            self.llm_answer_ready_event.set() # Signal TTS quick worker
                            break
                            # Do NOT break here, continue iterating to finish the full LLM response


                # Loop finished naturally or broke due to stop request
                logger.info(f"🗣️🧠🏁 [Gen {gen_id}] LLM Worker: Generator loop finished%s" % (" (Aborted)" if current_gen.llm_aborted else ""))

                # If loop finished naturally and no quick answer was ever found (e.g., short response)
                # Set the whole thing as the quick answer.
                if not current_gen.llm_aborted and finish_silent_generation(current_gen):
                    logger.info("Model silent turn completed: %s", current_gen.decision.action)
                    for lk in getattr(self, "lookups", ()):
                        if lk.get("gen") == gen_id:
                            lk["answered"] = False
                elif not current_gen.llm_aborted and not current_gen.quick_answer_provided:
                    logger.info(f"🗣️🧠✔️ [Gen {gen_id}] LLM Worker: No context boundary found, using full response as quick answer.")
                    # quick_answer already contains the full text
                    current_gen.quick_answer_provided = True # Mark as provided
                    if self.on_partial_assistant_text:
                        self.on_partial_assistant_text(current_gen.quick_answer)
                    self.llm_answer_ready_event.set() # Signal TTS quick worker

            except Exception as e:
                logger.exception(f"🗣️🧠💥 [Gen {gen_id}] LLM Worker: Error during generation: {e}")
                current_gen.llm_aborted = True # Mark as aborted on error
            finally:
                # Clean up state regardless of how the loop/try block exited
                self.llm_generation_active = False
                self.stop_llm_finished_event.set() # Signal that this worker's processing attempt is done

                if current_gen.llm_aborted:
                    # If LLM was aborted, ensure TTS (both quick and final) is also stopped
                    logger.info(f"🗣️🧠❌ [Gen {gen_id}] LLM Aborted, requesting TTS quick/final stop.")
                    self.stop_tts_quick_request_event.set()
                    self.stop_tts_final_request_event.set()
                    # Wake up TTS quick worker if it's waiting
                    self.llm_answer_ready_event.set()

                logger.info(f"🗣️🧠🏁 [Gen {gen_id}] LLM Worker: Finished processing cycle.")

                current_gen.llm_finished = True
                current_gen.llm_finished_event.set()

    @staticmethod
    def _same_utterance(a: str, b: str) -> bool:
        """True if b is the SAME utterance as a (whisper still refining/extending),
        not a NEW utterance (interruption). Word-set overlap, so prepends and
        word corrections don't read as a brand-new turn."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return False
        return len(wa & wb) >= 0.7 * min(len(wa), len(wb))

    @staticmethod
    def _adds_content(a: str, b: str) -> bool:
        """True if b carries real new content beyond draft a (demo take 8, 10-05:
        a draft built on 'Okay, Wren. Settle it.' was kept when the line went on
        'Co-op or competitive? There are four of us.', so she answered the
        half-sentence). Punctuation-insensitive; 3+ new words is a new clause."""
        tok = lambda t: [w for w in re.findall(r"[a-z0-9']+", t.lower()) if w]
        wa = set(tok(a))
        new = [w for w in tok(b) if w not in wa]
        return len(new) >= 3

    @staticmethod
    def _trim_transcript(txt: str, max_words: int = 60) -> str:
        """Cap the transcript fed to the LLM to the most recent words.

        In a Discord call the STT returns a whole multi-sentence paragraph as
        one turn, which blows up the LLM prefill and slams the pipeline. Feed
        only the tail — the agent responds to what was just said, not a 200-word
        wall. Preserves a leading speaker label like [S1].
        """
        if not txt:
            return txt
        body = txt.strip()
        label = ''
        if body.startswith('['):
            end = body.find(']')
            if end != -1:
                label, body = body[:end + 1], body[end + 1:].lstrip()
        words = body.split()
        if len(words) <= max_words:
            return txt
        tail = ' '.join(words[-max_words:])
        return f'{label} {tail}'.strip() if label else tail

    def _llm_context(self, txt: str):
        """(history, room_context) for this turn. Never raises."""
        self._ctx_text = txt
        room = self._room_context(txt)
        if not self.convo_enabled:
            history = [dict(m) for m in self.history]
            return self._finish_context(history, room)
        try:
            history, in_view = self.convo.messages(txt)
            m = re.match(r'^\s*\[(S\d+)\]', txt or '')
            state = self.convo.state_note(m[1] if m else None, in_view)
            room = (room + "\n" + state).strip() if state else room
            room = self._with_people(room, m[1] if m else None, in_view)
            act = self.threads.active
            if act is not None and act.text == txt:
                tnote = self.threads.context_note(act)
                room = (room + "\n" + tnote).strip() if tnote else room
            return self._finish_context(history, room)
        except Exception as e:  # noqa: BLE001 - context must never block a turn
            logger.warning("conversation log render failed, using raw history: %s", e)
            history = [dict(m) for m in self.history]
            return self._finish_context(history, room)

    # ------------------------------------------------------------ task board
    def board(self):
        return self.boards.get(self.current_persona)

    def _attach_board(self, txt: str, gen_id) -> bool:
        """Link finished-but-untold searches to the generation that can tell them.

        A delivery cue owns its task; a normal turn from the asker owns theirs
        (the TASK BOARD note tells the model to bring it up). Returns True for cues.
        """
        try:
            from tasks import CUE_RE
            b = self.board()
            if _room.ROOM_CUE_RE.search(txt or ""):
                return True
            mc = CUE_RE.search(txt or "")
            if mc:
                b.attach_gen(int(mc[1]), gen_id)
                return True
            m = re.match(r'^\s*\[(S\d+)\]', txt or "")
            if m:
                for t in b.snapshot():
                    if t["kind"] == "search" and t["status"] == "done" and t["asker"] == m[1]:
                        b.attach_gen(t["id"], gen_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("task board attach failed: %s", e)
        return False

    def _refresh_plugin_tools(self):
        import plugins as _plugins
        self.llm.tools = _plugins.filter_tools(AGENT_TOOLS)
        logger.info("🧩 agent tools: %s", ", ".join(t['function']['name'] for t in self.llm.tools) or "none")

    def _run_tool(self, name, args):
        """Instance tool executor: board-backed tools, then the module ones."""
        args = args if isinstance(args, dict) else {}
        import plugins as _plugins
        if name in _plugins.tool_names_disabled():
            return "That ability is switched off by the owner right now. Answer without it."
        gen = self.running_generation
        m = re.match(r'^\s*\[(S\d+)\]', getattr(gen, "text", "") or "")
        asker = m[1] if m else ""
        try:
            if name == "web_search" and _websearch is not None:
                b = self.board()
                q = str(args.get("query", "") or "").strip()
                if not q:  # never run an empty search; use what they actually asked
                    q = re.sub(r'^\s*\[S\d+\]\s*', '', getattr(gen, "text", "") or "")[:160]
                    logger.warning("web_search called with no query; using the request text: %r", q)
                said = re.sub(r'^\s*\[S\d+\]\s*', '', getattr(gen, "text", "") or "")
                q2 = _websearch.freshen_query(q, said)
                if q2 != q:
                    logger.info("📋 stale year in query freshened: %r -> %r", q, q2)
                    q = q2
                t = b.start_search(q, asker, getattr(gen, "id", None))

                def cancelled():
                    return (gen is None or self.running_generation is not gen
                            or getattr(gen, "abortion_started", False) or self.shutdown_event.is_set())
                res = b.wait(t, SEARCH_WAIT_S, cancelled)
                if res is None:
                    return (f"Still searching in the background (task #{t['id']}). It keeps running "
                            "even if you get interrupted and will land on your TASK BOARD with a cue "
                            "to share it. For now just say briefly that you're still digging.")
                return res
            if name == "read_page" and _websearch is not None:
                r = _websearch.read_page(str(args.get("url", "")))
                return (__import__("untrusted").wrap(f"{r.get('title', '')}: {r['text']}", "web page") if r.get("ok")
                        else f"Couldn't read that page ({r.get('text')}). Answer from the search results.")
            if name == "note_to_self":
                t = self.board().add_note(str(args.get("text", "")), asker)
                return (f"Saved as note #{t['id']} on your TASK BOARD. Now continue: header, then "
                        "your reply (or [HOLD]).")
            if name == "clear_note":
                ok = self.board().clear(int(args.get("id") or 0))
                return "Cleared." if ok else "No such note."
            if name in _room.NAMES:
                who = ""
                try:
                    who = self.people.name_of(asker) or ""
                except Exception:  # noqa: BLE001
                    pass
                _ag = getattr(self, "agent", None)
                _anames = tuple(getattr(getattr(_ag, "profile", None), "names", ()) or ()) \
                    or (getattr(self, "current_persona", "") or "",)
                out = _room.execute(name, args, asker, who, settings=_plugins.settings_of,
                                    name_of=getattr(getattr(self, "people", None), "name_of", None),
                                    agent_names=_anames,
                                    agent=getattr(self, "current_persona", "") or "")
                logger.info("🎲 %s(%s) -> %s", name, ", ".join(f"{k}={str(v)[:40]!r}" for k, v in args.items()), out[:120])
                return out
            if name in _self_tools.NAMES:
                out = _self_tools.execute(name, args, getattr(self, "current_persona", "") or "")
                logger.info("🗣️🔎 %s %s(%s) -> %d chars", self.current_persona, name,
                            ", ".join(f"{k}={str(v)[:40]!r}" for k, v in args.items()), len(out))
                return out
            if name == "step_back":
                mins = self.floor.step_back(args.get("minutes") or 2.0)
                logger.info(f"🗣️🤫 {self.current_persona} chose to step back for {mins:g} min")
                return (f"You're stepping back for about {mins:g} min; your name still wakes you. "
                        "Output only [HOLD] now.")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"🗣️🛠️ tool {name} failed: {e}")
            return f"tool error: {e}"
        return _execute_tool(name, args)

    def _look_prefetch(self, txt: str, is_cue: bool = False):
        """('look_at_screen', args) when a line aimed at this agent asks it to look.

        Owner Eyes switch still gates everything; side chatter between other people
        ('what do you see in her') never triggers a capture."""
        try:
            import capability as _cap
            if is_cue or _screen is None or not _screen.status().get('enabled'):
                return None
            if not _cap.wants_look(txt):
                return None
            from floor import names_agent
            m = re.match(r'^\s*\[(S\d+)\]', txt or '')
            speaker = m[1] if m else None
            agent = getattr(self, 'agent', None)
            names = tuple(getattr(getattr(agent, 'profile', None), 'names', ()) or ()) \
                or (getattr(self, 'current_persona', '') or '',)
            floor = getattr(self, 'floor', None)
            partner = floor.active_partner() if floor is not None else None
            if not (names_agent(txt, names) or speaker is None or (partner and speaker == partner)):
                return None
            return ('look_at_screen', {'reason': f"{speaker or 'someone'} asked me to look at the screen"})
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.warning('look prefetch check failed: %s', e)
            return None

    def _code_prefetch(self, txt: str, is_cue: bool = False):
        """('read_own_code', args) when a line aimed at this agent asks it to read its code."""
        try:
            import self_tools as _st
            if is_cue:
                return None
            pf = _st.code_prefetch(txt)
            if pf is None:
                return None
            from floor import names_agent
            m = re.match(r'^\s*\[(S\d+)\]', txt or '')
            speaker = m[1] if m else None
            agent = getattr(self, 'agent', None)
            names = tuple(getattr(getattr(agent, 'profile', None), 'names', ()) or ()) \
                or (getattr(self, 'current_persona', '') or '',)
            floor = getattr(self, 'floor', None)
            partner = floor.active_partner() if floor is not None else None
            if not (names_agent(txt, names) or speaker is None or (partner and speaker == partner)):
                return None
            return pf
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.warning('code prefetch check failed: %s', e)
            return None

    def _room_prefetch(self, txt: str, is_cue: bool = False):
        """(tool, args) for a clear dice/coin/poll/timer/weather request aimed at this agent.

        Live sim 10-05: models invented rolls, flips and 'timer set' without calling
        the tool, so clear requests run the real tool first. A vote counts from anyone
        while a poll is open (that's the point of a room poll)."""
        try:
            if is_cue:
                return None
            pf = _room.intent(txt)
            if pf is None:
                return None
            if pf[0] == "cast_vote":
                return pf
            from floor import names_agent
            m = re.match(r'^\s*\[(S\d+)\]', txt or '')
            speaker = m[1] if m else None
            agent = getattr(self, 'agent', None)
            names = tuple(getattr(getattr(agent, 'profile', None), 'names', ()) or ()) \
                or (getattr(self, 'current_persona', '') or '',)
            floor = getattr(self, 'floor', None)
            partner = floor.active_partner() if floor is not None else None
            if not (names_agent(txt, names) or speaker is None or (partner and speaker == partner)):
                return None
            return pf
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.warning('room prefetch check failed: %s', e)
            return None

    def _lookup_note(self) -> str:
        """Recent web lookups for follow-ups (never raises)."""
        try:
            import time as _t
            now = _t.time()
            recent = [lk for lk in self.lookups if now - lk["ts"] < LOOKUP_TTL_S][-2:]
            if not recent:
                return ""
            lines = ["Lookups you did recently (use them for follow-ups like 'did you find it?'; don't pretend you didn't look):"]
            for lk in recent:
                mins = max(1, int((now - lk["ts"]) // 60))
                if lk.get("kind") == "screen":
                    why = f' (looking for: {lk["query"]})' if lk.get("query") else ""
                    lines.append(f"- ~{mins} min ago you took a screenshot of the owner's screen{why}. "
                                 "That image is gone and the screen has probably changed. Never describe "
                                 "the screen now from memory; if asked about it, look again.")
                    continue
                who = f" for {lk['asker']}" if lk.get("asker") else ""
                said = f' You said "{lk["said"]}" while searching.' if lk.get("said") else ""
                unanswered = (" You never actually told them the answer - if they follow up, "
                              "give it now.") if lk.get("answered") is False else ""
                lines.append(f'- ~{mins} min ago{who}: searched "{lk["query"]}".{said}{unanswered} '
                             f'Results: {lk["result"]}')
            return chr(10).join(lines)
        except Exception as e:  # noqa: BLE001
            logger.warning("lookup note failed: %s", e)
            return ""

    def _on_tool_result(self, name, args, result) -> None:
        """Remember what a search returned (called from the LLM tool loop)."""
        try:
            import time as _t
            if name == "look_at_screen":
                if isinstance(result, dict) and result.get("images"):
                    reason = args.get("reason", "") if isinstance(args, dict) else ""
                    gen = self.running_generation
                    self.lookups.append({"ts": _t.time(), "gen": getattr(gen, "id", None), "kind": "screen",
                                         "asker": "", "query": reason[:120], "result": "",
                                         "said": getattr(self, "_last_backchannel_text", ""),
                                         "answered": True})
                    self.lookups = self.lookups[-6:]
                return
            if name != "web_search" or getattr(self, "boards", None) is not None:
                return  # searches are remembered on the TASK BOARD
            gen = self.running_generation
            gid = getattr(gen, "id", None)
            query = args.get("query", "") if isinstance(args, dict) else str(args)
            m = re.match(r'^\s*\[(S\d+)\]', getattr(gen, "text", "") or "")
            snippet = " ".join(str(result).split())[:LOOKUP_SNIPPET_CHARS]
            same = [lk for lk in self.lookups if lk.get("gen") == gid and gid is not None]
            if same:  # follow-up query in the same turn: merge
                lk = same[-1]
                lk["query"] = (lk["query"] + " / " + query)[:160]
                lk["result"] = (lk["result"] + " | " + snippet)[:LOOKUP_SNIPPET_CHARS * 2]
            else:
                self.lookups.append({"ts": _t.time(), "gen": gid, "asker": m[1] if m else "",
                                     "query": query[:120], "result": snippet,
                                     "said": getattr(self, "_last_backchannel_text", ""),
                                     "answered": True})
                self.lookups = self.lookups[-6:]
        except Exception as e:  # noqa: BLE001
            logger.warning("lookup record failed: %s", e)

    def _with_people(self, room: str, current, in_view=()) -> str:
        """Append who's-who (names for voice tags + name-ask cue). Never raises."""
        try:
            book = getattr(self, "people", None)
            if book is None:
                return room
            labels = list(in_view or ()) + list(getattr(self.convo, "roster", {}) or {})
            note = book.who_note(current, labels, roster=getattr(self.convo, "roster", None))
            return (room + "\n" + note).strip() if note else room
        except Exception as e:  # noqa: BLE001
            logger.warning("people note failed: %s", e)
            return room

    def _guard_lines(self, txt: str):
        """Lines the reply must not recite: dictations still in force (sticky) and the
        room's recent lines (verbatim echo of what people said). Shared by the live path
        and the sims so gating can't drift. Never raises."""
        forbid, echo = [], []
        try:
            if getattr(self, '_dictations', None) is None:
                from loopbait import DictationMemory
                self._dictations = DictationMemory()
            forbid = self._dictations.note(txt)
        except Exception as e:  # noqa: BLE001
            logger.warning('dictation check failed: %s', e)
        try:
            from collections import deque
            if getattr(self, '_recent_heard', None) is None:
                self._recent_heard = deque(maxlen=8)
            body = re.sub(r"^\s*\[[^\]]*\]\s*", "", txt or "").strip()
            if body:
                self._recent_heard.append(body)
            echo = list(self._recent_heard)
        except Exception as e:  # noqa: BLE001
            logger.warning('echo guard failed: %s', e)
        return forbid, echo

    def _check_bait(self, txt: str):
        """Loop-bait verdict for this turn (never raises)."""
        try:
            v = self.bait.check(txt, self.history)
            return v if v.kind else None
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.warning("loop-bait check failed: %s", e)
            return None

    def _finish_context(self, history, room: str):
        """Collapse repeated agent lines (breaks the catchphrase loop), then add the anti-repeat note."""
        look = self._lookup_note()
        if look:
            room = (room + chr(10) + look).strip()
        try:
            cut = self.cutoff
            if cut is not None:
                if cut.live():
                    room = (room + chr(10) + cut.note()).strip()
                else:
                    self.cutoff = None
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.warning("cutoff note failed: %s", e)
        searched = False
        try:
            b = self.board()
            bnote = b.note()
            if bnote:
                room = (room + chr(10) + bnote).strip()
            searched = any(t["kind"] == "search" for t in b.snapshot())
        except Exception as e:  # noqa: BLE001
            logger.warning("task board note failed: %s", e)
        try:
            from backchannels import search_note
            nudge = search_note(getattr(self, "_ctx_text", ""), bool(look) or searched)
            if nudge:
                room = (room + chr(10) + nudge).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("search note failed: %s", e)
        try:
            import capability as _cap
            import plugins as _plugins
            _off = {p for p in ('web_search', 'eyes', 'clock', 'notes', 'step_back', 'self_check')
                    if not _plugins.is_enabled(p)}
            eyes_on = bool(_screen is not None and _screen.status().get("enabled")) and 'eyes' not in _off
            ctx = getattr(self, "_ctx_text", "")
            room = (room + chr(10) + _cap.abilities_note(eyes_on, bool(WEB_SEARCH_TOOLS) and 'web_search' not in _off,
                                                      getattr(self, 'current_persona', '') or '', _off)).strip()
            try:
                pn = _self_tools.context_note(getattr(self, "current_persona", "") or "")
                if pn:
                    room = (room + chr(10) + pn).strip()
                cn = _self_tools.change_note(ctx)
                if cn:
                    room = (room + chr(10) + cn).strip()
            except Exception as e:  # noqa: BLE001
                logger.warning("prediction note failed: %s", e)
            try:
                import languages as _langs
                ln = _langs.ROOM.note()
                if ln:
                    room = (room + chr(10) + ln).strip()
            except Exception as e:  # noqa: BLE001
                logger.warning("language note failed: %s", e)
            try:
                import call_memory as _cmem
                memn = _cmem.memory_note(getattr(self, "current_persona", ""))
                if memn:
                    room = (room + chr(10) + memn).strip()
                hg = getattr(self, "hgmem", None)
                if hg is not None and ctx:
                    hgn = hg.recall_note(getattr(self, "current_persona", ""), ctx,
                                         exclude={i["text"] for i in _cmem.approved(
                                             getattr(self, "current_persona", ""))})
                    if hgn:
                        room = (room + chr(10) + hgn).strip()
            except Exception as e:  # noqa: BLE001
                logger.warning("memory note failed: %s", e)
            selfn = _cap.self_note(ctx)
            if selfn:
                room = (room + chr(10) + selfn).strip()
            lnote = _cap.look_note(ctx, eyes_on)
            if lnote:
                room = (room + chr(10) + lnote).strip()
            import moment as _moment
            mnote = _moment.note()
            if mnote:
                room = (room + chr(10) + mnote).strip()
            import beliefs as _beliefs
            bnote_ = _beliefs.belief_note(getattr(self, "current_persona", ""), ctx)
            if bnote_:
                room = (room + chr(10) + bnote_).strip()
            rnote = "" if mnote else _cap.request_note(ctx, getattr(getattr(self, "agent", None), "profile", None)
                                      and self.agent.profile.names or ())
            if rnote:
                room = (room + chr(10) + rnote).strip()
            try:
                from voice_cues import cue_note as _cue_note
                cnote = _cue_note(getattr(getattr(self, "audio", None), "current_voice", "") or "", ctx)
            except Exception:  # noqa: BLE001
                cnote = ""
            if cnote:
                room = (room + chr(10) + cnote).strip()
            if os.environ.get("ATLAS_SCRAP_NOTE", "1") != "0":  # facts note: complements the tree
                snote = _cap.scrap_note(ctx, history, getattr(getattr(self, "agent", None), "profile", None)
                                        and self.agent.profile.names or ())
                if snote:
                    room = (room + chr(10) + snote).strip()
            history = _cap.scrub_refusals(history, eyes_on, bool(WEB_SEARCH_TOOLS))
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.warning("capability note failed: %s", e)
        try:
            # Data, not rules: clean the agent's own past lines instead of adding
            # "don't say X" notes (naming a phrase primes it; notes stack up).
            from floor import collapse_openers, drop_stub_replies, thin_short_replies, clip_own_replies, strip_filler_openers
            from echo_reply import drop_parrots
            history = strip_filler_openers(clip_own_replies(thin_short_replies(collapse_openers(drop_parrots(drop_stub_replies(history))))))
            from identity import scrub_history
            history = scrub_history(history, getattr(self, "current_persona", ""))
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.warning("collapse_repeats failed: %s", e)
        return history, room

    @staticmethod
    def _with_anti_repeat(history, room: str) -> str:
        """Append the anti-repetition note (never raises)."""
        try:
            from floor import anti_repeat_note
            note = anti_repeat_note(history)
        except Exception as e:  # noqa: BLE001
            logger.warning("anti-repeat note failed: %s", e)
            note = ""
        return (room + "\n" + note).strip() if note else room

    def _room_context(self, txt: str) -> str:
        from floor import steering_note
        m = re.match(r'^\s*\[(S\d+)\]', txt or '')
        try:
            prof = self.agent.profile
            note = steering_note(txt, m[1] if m else None, self.floor,
                                 prof.names, vibe=self.vibe_enabled, profile=prof)
        except Exception as e:  # noqa: BLE001 - steering must never block a turn
            logger.warning("room context failed: %s", e)
            note = ""
        # Plugin notes stand on their own: a steering failure must not silently drop them.
        if True:
            try:
                import plugins as _plugins
                if _plugins.is_enabled("floor_referee") and \
                        (_plugins.settings_of("floor_referee") or {}).get("invite_quiet", "on") == "on":
                    import floor_referee as _fr
                    qn = _fr.quiet_note(m[1] if m else None,
                                        getattr(getattr(self, "people", None), "name_of", None))
                    if qn:
                        note = (note + "\n" + qn).strip() if note else qn
            except Exception as e:  # noqa: BLE001
                logger.debug("floor referee note failed: %s", e)
            try:
                import plugins as _plugins
                if _plugins.is_enabled("teach"):
                    import teach as _teach
                    tn = _teach.note(getattr(self, "current_persona", "") or "", txt)
                    if tn:
                        note = (note + "\n" + tn).strip() if note else tn
            except Exception as e:  # noqa: BLE001
                logger.debug("teach note failed: %s", e)
            try:
                import plugins as _plugins
                if _plugins.is_enabled("soul_reflection"):
                    import soul_reflection as _soul
                    sn = _soul.note(getattr(self, "current_persona", "") or "")
                    if sn:
                        note = (note + "\n" + sn).strip() if note else sn
            except Exception as e:  # noqa: BLE001
                logger.debug("soul reflection note failed: %s", e)
        return note or ""

    def check_abort(self, txt: str, wait_for_finish: bool = True, abort_reason: str = "unknown") -> bool:
        """
        Checks if the current generation should be aborted based on new input text.

        Compares the provided text (`txt`) with the text of the `running_generation`.
        If a generation is running and not already aborting:
        1. If `txt` is very similar (>= 0.95 similarity) to the running generation's
           input text, it ignores the new request and returns False.
        2. If `txt` is different, it initiates an abort of the current generation
           by calling the public `abort_generation` method.

        If `wait_for_finish` is True, this method waits for the abortion process
        initiated by `abort_generation` to complete before returning.

        If a generation is already in the process of aborting when this method is called,
        it will wait (if `wait_for_finish` is True) for that ongoing abort to finish.

        Args:
            txt: The new text input to check against the current generation's input.
            wait_for_finish: Whether to block until the initiated/ongoing abort completes.
            abort_reason: A string describing why the abort check is being performed.

        Returns:
            True if an abortion was processed (either newly initiated or waited for),
            False if no active generation was found or the new text was too similar.
        """
        with self.check_abort_lock:
            if self.running_generation:
                current_gen_id_str = f"Gen {self.running_generation.id}"
                logger.info(f"🗣️🛑❓ {current_gen_id_str} Abort check requested (reason: {abort_reason})")

                if self.running_generation.abortion_started:
                    logger.info(f"🗣️🛑⏳ {current_gen_id_str} Active generation is already aborting, waiting to finish (if requested).")

                    # Only wait if wait_for_finish is True
                    if wait_for_finish:
                        start_time = time.time()
                        # Wait using the abort_completed_event for better synchronization
                        completed = self.abort_completed_event.wait(timeout=5.0) # Use the event from abort_generation

                        if not completed:
                             logger.error(f"🗣️🛑💥💥 {current_gen_id_str} Timeout waiting for ongoing abortion to complete. State inconsistency possible!")
                             # Force clear it just in case, though this indicates a deeper issue.
                             self.running_generation = None
                        elif self.running_generation is not None:
                            logger.error(f"🗣️🛑💥💥 {current_gen_id_str} Abortion completed event set, but running_generation still exists. State inconsistency likely!")
                            # Force clear it.
                            self.running_generation = None
                        else:
                            logger.info(f"🗣️🛑✅ {current_gen_id_str} Ongoing abortion finished.")
                    else:
                        logger.info(f"🗣️🛑🏃 {current_gen_id_str} Not waiting for ongoing abortion as wait_for_finish=False")

                    return True # An abort was processed (waited for)
                else:
                    # No abortion in progress, check similarity
                    logger.info(f"🗣️🛑🤔 {current_gen_id_str} Found active generation, checking text similarity.")
                    run_text = self.running_generation.text or ""
                    if run_text and txt and self._same_utterance(run_text, txt) and (
                            self.running_generation.tts_quick_started
                            or not self._adds_content(run_text, txt)):
                        # Already speaking: keep it (cutting it off orphaned audio, demo
                        # 10-04). Not speaking yet and the line grew a new clause: redraft.
                        logger.info(f"🗣️🛑🙅 {current_gen_id_str} Text refines the same utterance (word overlap). Ignoring.")
                        return False
                    try:
                         # Ensure running_generation.text is not None before comparison
                        if self.running_generation.text is None:
                            logger.warning(f"🗣️🛑❓ {current_gen_id_str} Running generation text is None, cannot compare similarity. Assuming different.")
                            similarity = 0.0
                        else:
                            similarity = self.text_similarity.calculate_similarity(self.running_generation.text, txt)
                    except Exception as e:
                        logger.warning(f"🗣️🛑💥 {current_gen_id_str} Error calculating similarity: {e}. Assuming different.")
                        similarity = 0.0 # Assume different on error

                    if similarity >= 0.95:
                        logger.info(f"🗣️🛑🙅 {current_gen_id_str} Text ('{txt[:30]}...') too similar ({similarity:.2f}) to current '{self.running_generation.text[:30] if self.running_generation.text else 'None'}...'. Ignoring.")
                        return False # No abort needed

                    # Texts are different enough, initiate abort
                    logger.info(f"🗣️🛑🚀 {current_gen_id_str} Text ('{txt[:30]}...') different enough ({similarity:.2f}) from '{self.running_generation.text[:30] if self.running_generation.text else 'None'}...'. Requesting synchronous abort.")
                    start_time = time.time()
                    # Call the synchronous public abort method - THIS IS KEY
                    self.abort_generation(wait_for_completion=wait_for_finish, timeout=7.0, reason=f"check_abort found different text ({abort_reason})")

                    if wait_for_finish:
                         # Check state *after* waiting for the abort call
                        if self.running_generation is not None:
                            logger.error(f"🗣️🛑💥💥 {current_gen_id_str} !!! Abort call completed but running_generation is still not None. State inconsistency likely!")
                            # Force clear it.
                            self.running_generation = None
                        else:
                            logger.info(f"🗣️🛑✅ {current_gen_id_str} Synchronous abort completed in {time.time() - start_time:.2f}s.")

                    return True # An abort was processed (initiated)
            else:
                logger.info("🗣️🛑🤷 No active generation found during abort check.")
                return False # No active generation to abort

    def _tts_quick_inference_worker(self):
        """
        Worker thread target that handles TTS synthesis for the 'quick answer'.

        Waits for `llm_answer_ready_event`. Once signaled, it checks if the generation
        is valid and has a `quick_answer`. It then waits for the `tts_quick_allowed_event`
        (intended for potential rate limiting or timing control, currently seems unused).
        If allowed, it calls `audio.synthesize` with the `quick_answer`, feeding audio
        chunks into the `audio_chunks` queue. Handles stop requests
        (`stop_tts_quick_request_event`) and signals completion/abortion via
        `stop_tts_quick_finished_event` and internal flags. Runs until `shutdown_event` is set.
        """
        logger.info("🗣️👄🚀 Quick TTS Worker: Starting...")
        while not self.shutdown_event.is_set():
            ready = self.llm_answer_ready_event.wait(timeout=1.0)
            if not ready:
                continue

            # Check if aborted *while waiting* before clearing the ready event
            if self.stop_tts_quick_request_event.is_set():
                logger.info("🗣️👄❌ Quick TTS Worker: Abort detected while waiting for llm_answer_ready_event.")
                self.stop_tts_quick_request_event.clear()
                self.stop_tts_quick_finished_event.set()
                self.tts_quick_generation_active = False
                continue # Go back to waiting

            self.llm_answer_ready_event.clear() # Clear the event now that we're processing
            current_gen = self.running_generation

            if not current_gen or not current_gen.quick_answer:
                logger.warning("🗣️👄❓ Quick TTS Worker: No valid generation or quick answer found after event.")
                self.tts_quick_generation_active = False
                continue # Go back to waiting

            # Double-check if this generation was aborted *just* before we got here
            if current_gen.audio_quick_aborted or current_gen.abortion_started:
                logger.info(f"🗣️👄❌ [Gen {current_gen.id}] Quick TTS Worker: Generation already marked as aborted. Skipping.")
                continue

            gen_id = current_gen.id
            logger.info(f"🗣️👄🔄 [Gen {gen_id}] Quick TTS Worker: Processing TTS for quick answer...")

            # Set state for active generation
            self.tts_quick_generation_active = True
            self.stop_tts_quick_finished_event.clear()
            current_gen.tts_quick_finished_event.clear() # Reset TTS finish marker for this attempt
            current_gen.tts_quick_started = True

            # --- tts_quick_allowed_event Wait Logic ---
            # This event seems intended for external control/timing, but isn't set anywhere
            # in the current code. Added a timeout and logging for clarity. If it's meant
            # to be used, something needs to .set() it externally.
            allowed_to_speak = False
            start_wait_time = time.time()
            wait_timeout = 5.0 # Example timeout
            logger.debug(f"🗣️👄⏳ [Gen {gen_id}] Quick TTS Worker: Waiting for tts_quick_allowed_event (timeout: {wait_timeout}s)...")
            # TODO: Determine if this event is actually used/needed. If not, remove the wait.
            # If it IS needed, ensure something sets it. Currently, it might always timeout.
            # For now, we'll proceed even if it times out, assuming it's optional or not yet implemented.
            # allowed_to_speak = current_gen.tts_quick_allowed_event.wait(timeout=wait_timeout)
            allowed_to_speak = True # Temporarily bypass wait for testing/if event is unused.
            # if not allowed_to_speak:
            #    logger.warning(f"🗣️👄⏱️ [Gen {gen_id}] Quick TTS Worker: Timed out waiting for tts_quick_allowed_event after {time.time() - start_wait_time:.2f}s. Proceeding anyway.")
            # else:
            #    logger.debug(f"🗣️👄✔️ [Gen {gen_id}] Quick TTS Worker: tts_quick_allowed_event received or bypassed.")
            # --- End tts_quick_allowed_event Wait Logic ---


            try:
                # Check again for aborts right before synthesis call
                if self.stop_tts_quick_request_event.is_set() or current_gen.abortion_started:
                     logger.info(f"🗣️👄❌ [Gen {gen_id}] Quick TTS Worker: Aborting TTS synthesis due to stop request or abortion flag.")
                     current_gen.audio_quick_aborted = True
                else:
                    logger.info(f"🗣️👄🎶 [Gen {gen_id}] Quick TTS Worker: Synthesizing: '{current_gen.quick_answer[:50]}...'")
                    completed = self.audio.synthesize(
                        current_gen.quick_answer,
                        current_gen.audio_chunks,
                        self.stop_tts_quick_request_event # Pass the event for the synthesizer to check
                    )

                    if not completed:
                        # Synthesis was stopped by the stop_tts_quick_request_event
                        logger.info(f"🗣️👄❌ [Gen {gen_id}] Quick TTS Worker: Synthesis stopped via event.")
                        current_gen.audio_quick_aborted = True
                    else:
                        logger.info(f"🗣️👄✅ [Gen {gen_id}] Quick TTS Worker: Synthesis completed successfully.")


            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] Quick TTS Worker: Error during synthesis: {e}")
                current_gen.audio_quick_aborted = True # Mark as aborted on error
            finally:
                # Clean up state regardless of how the try block exited
                self.tts_quick_generation_active = False
                self.stop_tts_quick_finished_event.set() # Signal that this worker's processing attempt is done
                logger.info(f"🗣️👄🏁 [Gen {gen_id}] Quick TTS Worker: Finished processing cycle.")

                # Check if synthesis completed naturally or was stopped/aborted
                if current_gen.audio_quick_aborted or self.stop_tts_quick_request_event.is_set():
                    logger.info(f"🗣️👄❌ [Gen {gen_id}] Quick TTS Marked as Aborted/Incomplete.")
                    self.stop_tts_quick_request_event.clear() # Clear the request if it was set
                    current_gen.audio_quick_aborted = True # Ensure flag is set
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] Quick TTS Finished Successfully.")
                    current_gen.tts_quick_finished_event.set() # Signal natural completion

                current_gen.audio_quick_finished = True # Mark quick audio phase as done (even if aborted)

    def _tts_final_inference_worker(self):
        """
        Worker thread target that handles TTS synthesis for the 'final' part of the answer.

        Continuously checks the `running_generation`. It waits until the 'quick' TTS
        phase (`tts_quick_started` and `audio_quick_finished`) is complete and was not
        aborted (`audio_quick_aborted`). It also requires that a `quick_answer` was
        actually identified (`quick_answer_provided`).

        If conditions are met, it sets flags (`tts_final_started`), defines an inner
        generator (`get_generator`) that yields the `quick_answer_overhang` followed
        by the remaining chunks from the `llm_generator`. It then calls
        `audio.synthesize_generator` with this generator, feeding audio chunks into the
        *same* `audio_chunks` queue used by the quick worker. Handles stop requests
        (`stop_tts_final_request_event`) and signals completion/abortion via
        `stop_tts_final_finished_event` and internal flags. Runs until `shutdown_event` is set.
        """
        logger.info("🗣️👄🚀 Final TTS Worker: Starting...")
        while not self.shutdown_event.is_set():
            current_gen = self.running_generation
            time.sleep(0.01) # Prevent tight spinning when idle

            # --- Wait for prerequisites ---
            if not current_gen: continue # No active generation
            if current_gen.tts_final_started: continue # Final TTS already running for this gen
            if not current_gen.tts_quick_started: continue # Quick TTS hasn't even started
            if not current_gen.audio_quick_finished: continue # Quick TTS hasn't finished (successfully or aborted)

            gen_id = current_gen.id # Get ID once prerequisites seem met

            # --- Check conditions to *start* final TTS ---
            if current_gen.audio_quick_aborted:
                #logger.debug(f"🗣️👄🙅 [Gen {gen_id}] Final TTS Worker: Quick TTS was aborted, skipping final TTS.")
                continue
            if not current_gen.quick_answer_provided:
                 logger.debug(f"🗣️👄🙅 [Gen {gen_id}] Final TTS Worker: Quick answer boundary was not found, skipping final TTS (quick TTS handled everything).")
                 continue
            if current_gen.abortion_started:
                 logger.debug(f"🗣️👄🙅 [Gen {gen_id}] Final TTS Worker: Generation is aborting, skipping final TTS.")
                 continue

            # --- Conditions met, start final TTS ---
            logger.info(f"🗣️👄🔄 [Gen {gen_id}] Final TTS Worker: Processing final TTS...")

            def get_generator():
                """Yields remaining text chunks for final TTS synthesis."""
                # Yield overhang first
                if current_gen.quick_answer_overhang:
                    preprocessed_overhang = self.preprocess_chunk(current_gen.quick_answer_overhang)
                    logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Gen: Yielding overhang: '{preprocessed_overhang[:50]}...'")
                    current_gen.final_answer += preprocessed_overhang # Add preprocessed version
                    if self.on_partial_assistant_text:
                         logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Worker on_partial_assistant_text: Sending overhang.")
                         try:
                            self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                         except Exception as cb_e:
                             logger.warning(f"🗣️💥 Callback error in on_partial_assistant_text (overhang): {cb_e}")
                    yield preprocessed_overhang

                # Yield remaining chunks from LLM generator
                logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Gen: Yielding remaining LLM chunks...")
                try:
                    for chunk in current_gen.llm_generator:
                         # Check for stop *before* processing chunk
                         if self.stop_tts_final_request_event.is_set():
                             logger.info(f"🗣️👄❌ [Gen {gen_id}] Final TTS Gen: Stop request detected during LLM iteration.")
                             current_gen.audio_final_aborted = True
                             break # Stop yielding

                         preprocessed_chunk = self.preprocess_chunk(chunk)
                         current_gen.final_answer += preprocessed_chunk
                         if self.on_partial_assistant_text:
                             # logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Worker on_partial_assistant_text: Sending final chunk: {preprocessed_chunk[:30]}")
                            try:
                                 self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                            except Exception as cb_e:
                                 logger.warning(f"🗣️💥 Callback error in on_partial_assistant_text (final chunk): {cb_e}")

                         yield preprocessed_chunk
                    logger.debug(f"🗣️👄< [Gen {gen_id}] Final TTS Gen: Finished iterating LLM chunks.")
                except Exception as gen_e:
                     logger.exception(f"🗣️👄💥 [Gen {gen_id}] Final TTS Gen: Error iterating LLM generator: {gen_e}")
                     current_gen.audio_final_aborted = True # Mark as aborted on error

            # Set state for active generation
            self.tts_final_generation_active = True
            self.stop_tts_final_finished_event.clear()
            current_gen.tts_final_started = True
            current_gen.tts_final_finished_event.clear() # Reset TTS finish marker

            try:
                logger.info(f"🗣️👄🎶 [Gen {gen_id}] Final TTS Worker: Synthesizing remaining text...")
                completed = self.audio.synthesize_generator(
                    get_generator(),
                    current_gen.audio_chunks,
                    self.stop_tts_final_request_event # Pass the event for the synthesizer to check
                )

                if not completed:
                     logger.info(f"🗣️👄❌ [Gen {gen_id}] Final TTS Worker: Synthesis stopped via event.")
                     current_gen.audio_final_aborted = True
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] Final TTS Worker: Synthesis completed successfully.")


            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] Final TTS Worker: Error during synthesis: {e}")
                current_gen.audio_final_aborted = True # Mark as aborted on error
            finally:
                # Clean up state regardless of how the try block exited
                self.tts_final_generation_active = False
                self.stop_tts_final_finished_event.set() # Signal that this worker's processing attempt is done
                # logger.info(f"🗣️👄🏁 [Gen {gen_id}] Final TTS Worker: Finished processing cycle. Final answer accumulated: '{current_gen.final_answer[:50]}...'")
                logger.info(f"🗣️👄🏁 [Gen {gen_id}] Final TTS Worker: Finished processing cycle.")


                # Check if synthesis completed naturally or was stopped
                if current_gen.audio_final_aborted or self.stop_tts_final_request_event.is_set():
                    logger.info(f"🗣️👄❌ [Gen {gen_id}] Final TTS Marked as Aborted/Incomplete.")
                    self.stop_tts_final_request_event.clear() # Clear the request if it was set
                    current_gen.audio_final_aborted = True # Ensure flag is set
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] Final TTS Finished Successfully.")
                    current_gen.tts_final_finished_event.set() # Signal natural completion

                current_gen.audio_final_finished = True # Mark final audio phase as done (even if aborted)


    # --- Processing Methods ---

    def process_prepare_generation(self, txt: str):
        """
        Handles the 'prepare' action: initiates a new text-to-speech generation.

        1. Calls `check_abort` to potentially stop and clean up any existing generation
           if the new input `txt` is significantly different. Waits for the abort to finish.
        2. Increments the `generation_counter`.
        3. Resets state flags and events relevant to starting a new generation.
        4. Creates a new `RunningGeneration` instance with the new ID and input text.
        5. Calls `llm.generate` to get the LLM response generator.
        6. Stores the generator in `running_generation.llm_generator`.
        7. Sets `generator_ready_event` to signal the LLM worker thread to start processing.
        8. Cleans up `running_generation` if LLM generator creation fails.

        Args:
            txt: The user input text for the new generation.
        """
        # Repeated potential-sentence callbacks must not replace a generator
        # still owned by either worker (check_abort returns False for duplicates).
        current = self.running_generation
        if current is not None and not current.abortion_started and current.text == txt:
            logger.debug("Ignoring duplicate partial for active generation")
            return

        # --- Abort existing generation if necessary ---
        id_in_spec = self.generation_counter + 1 # Prospective ID for logging
        aborted = self.check_abort(txt, wait_for_finish=True, abort_reason=f"process_prepare_generation for new id {id_in_spec}")

        # check_abort declined (same utterance refined): the running generation stays
        # the owner. Replacing it here orphaned a reply that was already speaking --
        # its audio played to nobody and the turn ended "empty" (demo 10-04, Riley intro).
        keep = self.running_generation
        if not aborted and keep is not None and not keep.abortion_started:
            logger.info(f"🗣️✨🙅 [Gen {keep.id}] Keeping running generation for refined text: '{txt[:50]}...'")
            keep.text = txt
            return

        # --- No live generation remains (none existed, or it was aborted) ---
        self.generation_counter += 1
        new_gen_id = self.generation_counter
        logger.info(f"🗣️✨🔄 [Gen {new_gen_id}] Preparing new generation for: '{txt[:50]}...'")

        # Reset flags and events (mostly redundant after sync abort, but safe)
        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        self.llm_answer_ready_event.clear()
        self.generator_ready_event.clear()
        self.stop_llm_request_event.clear()
        self.stop_llm_finished_event.clear()
        self.stop_tts_quick_request_event.clear()
        self.stop_tts_quick_finished_event.clear()
        self.stop_tts_final_request_event.clear()
        self.stop_tts_final_finished_event.clear()
        self.abort_completed_event.clear()
        self.abort_block_event.set() # Ensure block is released if check_abort didn't run/clear it

        # --- Create new generation object ---
        # Local ref: a barge-in abort can set self.running_generation = None while we
        # build this one (demo 10-04: AttributeError 'NoneType' .decision -> pipeline error).
        gen = RunningGeneration(id=new_gen_id)
        self.running_generation = gen
        gen.text = txt
        from response_decision import expected_target_for
        gen.decision.expected_target = expected_target_for(txt)
        try:
            from floor import recent_fillers, recent_templates, recent_content
            gen.decision.recent_fillers = (
                recent_fillers(self.history) | recent_templates(self.history)
                | recent_content(self.history))
        except Exception as e:  # noqa: BLE001
            logger.warning("recent_fillers failed: %s", e)
        gen.decision.namebook = getattr(self, "people", None)
        gen.decision.persona = getattr(self, "current_persona", "")
        try:
            from echo_reply import strip_label
            prev = next((m.get("content", "") for m in reversed(self.history)
                         if m.get("role") == "user" and m.get("content") != txt), "")
            gen.decision.heard = (strip_label(prev) + " " + strip_label(txt)).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("parrot heard failed: %s", e)
        is_cue = self._attach_board(txt, new_gen_id)

        try:
            logger.info(f"🗣️🧠🚀 [Gen {new_gen_id}] Calling LLM generate...")
            bait = None if is_cue else self._check_bait(txt)
            if bait is not None and bait.action == 'hold':
                logger.info(f"🗣️🪤 [Gen {new_gen_id}] Loop bait continued after roast ({bait.kind}: {bait.line[:50]!r}) -> HOLD, no LLM call")
                gen.llm_generator = filter_response(iter(['[HOLD]']), gen.decision)
                self.generator_ready_event.set()
                return
            history, room = self._llm_context(txt)
            gen.decision.forbid, gen.decision.echo_lines = self._guard_lines(txt)
            if bait is not None and bait.note:
                logger.info(f"🗣️🪤 [Gen {new_gen_id}] Loop bait ({bait.kind} x{bait.count}: {bait.line[:50]!r}) -> roast + pivot")
                room = (room + chr(10) + bait.note).strip()
            # TODO: Update history management if needed
            # self.history.append({"role": "user", "content": txt}) # Example history update
            mem_ctx = self._memory_context(txt)
            look = (self._look_prefetch(txt, is_cue) or self._code_prefetch(txt, is_cue)
                    or self._room_prefetch(txt, is_cue))
            try:   # a switched-off plugin's tool is never prefetched (plugins.py)
                import plugins as _plugins
                if look and look[0] in _plugins.tool_names_disabled():
                    look = None
            except Exception:  # noqa: BLE001
                pass

            def _gen(text_in, _h=history, _r=room, _m=mem_ctx, _l=look):
                return self.llm.generate(
                    text=text_in,
                    history=_h,
                    use_system_prompt=True,
                    participation=True,
                    memory_context=_m,
                    room_context=_r,
                    prefetch_tool=_l,
                )
            trimmed = self._trim_transcript(txt)
            try:
                from response_decision import directly_named, retry_named_hold, NAMED_NUDGE
                names = tuple(getattr(getattr(self.agent, "profile", None), "names", ()) or ())
                named = bool(names) and not is_cue and directly_named(txt, names)
                who = gen.decision.expected_target or "them"
            except Exception as e:  # noqa: BLE001
                logger.warning("named-hold setup failed: %s", e)
                named, who = False, "them"
                retry_named_hold = None
            raw_stream = _gen(trimmed)
            if retry_named_hold is not None:
                raw_stream = retry_named_hold(
                    raw_stream,
                    lambda: _gen(trimmed + chr(10) + NAMED_NUDGE.format(who=who)),
                    named,
                    cancelled=lambda g=gen: bool(g.abortion_started or g.llm_aborted
                                                 or self.stop_llm_request_event.is_set()))
            gen.llm_generator = filter_response(
                raw_stream,
                gen.decision,
            )
            logger.info(f"🗣️🧠✔️ [Gen {new_gen_id}] LLM generator created. Setting generator ready event.")
            self.generator_ready_event.set() # Signal LLM worker
        except Exception as e:
            logger.exception(f"🗣️🧠💥 [Gen {new_gen_id}] Failed to create LLM generator: {e}")
            if self.running_generation is gen:
                self.running_generation = None # Clean up if generator creation failed


    def process_abort_generation(self):
        """
        Handles the core logic of aborting the current generation.

        Synchronized using `abort_lock`. If a `running_generation` exists:
        1. Sets the `abortion_started` flag on the generation.
        2. Blocks new requests by clearing `abort_block_event`.
        3. Sets stop request events (`stop_llm_request_event`, `stop_tts_quick_request_event`,
           `stop_tts_final_request_event`) for active worker threads.
        4. Wakes up workers that might be waiting on start events (`generator_ready_event`,
           `llm_answer_ready_event`) so they can see the stop request.
        5. Waits (with timeouts) for each worker to acknowledge the stop by setting their
           respective `stop_..._finished_event`.
        6. Calls external cancellation methods if available (e.g., `llm.cancel_generation`).
        7. Attempts to close the LLM generator stream.
        8. Clears the `running_generation` reference.
        9. Clears stale start events (`generator_ready_event`, `llm_answer_ready_event`).
        10. Signals completion by setting `abort_completed_event`.
        11. Releases the block on new requests by setting `abort_block_event`.
        """
        # This method assumes it's called within the public abort_generation or internally
        with self.abort_lock:
            current_gen_obj = self.running_generation # Store ref before potential clear
            current_gen_id_str = f"Gen {current_gen_obj.id}" if current_gen_obj else "Gen None"

            if current_gen_obj is None or current_gen_obj.abortion_started:
                if current_gen_obj is None:
                    logger.info(f"🗣️🛑🤷 {current_gen_id_str} No active generation found to abort.")
                else:
                    logger.info(f"🗣️🛑⏳ {current_gen_id_str} Abortion already in progress.")
                # Ensure events are managed correctly even if called redundantly
                self.abort_completed_event.set() # Signal completion if nothing to do/already done
                self.abort_block_event.set() # Ensure block is released
                return

            # --- Start Abort Process ---
            logger.info(f"🗣️🛑🚀 {current_gen_id_str} Abortion process starting...")
            current_gen_obj.abortion_started = True # Mark immediately
            _dec = getattr(current_gen_obj, "decision", None)
            if _dec is not None:
                _dec.cancelled = True
            self.abort_block_event.clear() # Block new requests *before* waiting
            self.abort_completed_event.clear() # Clear completion flag at start
            self.stop_everything_event.set() # General signal (might be unused by workers)
            aborted_something = False
            # Interrupt transport BEFORE waiting for workers. The final TTS
            # worker also owns/consumes the LLM stream after the first sentence.
            self.stop_llm_request_event.set()
            self.stop_tts_quick_request_event.set()
            self.stop_tts_final_request_event.set()
            if hasattr(self.llm, 'cancel_generation'):
                try:
                    self.llm.cancel_generation()
                except Exception as cancel_e:
                    logger.warning("LLM transport cancel failed: %s", cancel_e)


            # --- Abort LLM ---
            # Check if LLM is potentially active (running OR waiting to start)
            # Need to check generator_ready_event too, as it might be waiting there.
            is_llm_potentially_active = self.llm_generation_active or self.generator_ready_event.is_set()
            if is_llm_potentially_active:
                logger.info(f"🗣️🛑🧠❌ {current_gen_id_str} - Stopping LLM...")
                self.stop_llm_request_event.set()
                self.generator_ready_event.set() # Wake up LLM worker if it's waiting
                stopped = self.stop_llm_finished_event.wait(timeout=5.0) # Wait for LLM worker
                if stopped:
                    logger.info(f"🗣️🛑🧠👍 {current_gen_id_str} LLM stopped confirmation received.")
                    self.stop_llm_finished_event.clear() # Reset for next time
                else:
                    logger.warning(f"🗣️🛑🧠⏱️ {current_gen_id_str} Timeout waiting for LLM stop confirmation.")
                # Transport was already cancelled before worker waits.
                self.llm_generation_active = False # Ensure flag is off
                aborted_something = True
            else:
                logger.info(f"🗣️🛑🧠📴 {current_gen_id_str} LLM appears inactive, no stop needed.")
            self.stop_llm_request_event.clear() # Ensure stop request is clear

            # --- Abort Quick TTS ---
            # Check if TTS Quick is potentially active (running OR waiting to start)
            is_tts_quick_potentially_active = self.tts_quick_generation_active or self.llm_answer_ready_event.is_set()
            if is_tts_quick_potentially_active:
                logger.info(f"🗣️🛑👄❌ {current_gen_id_str} Stopping Quick TTS...")
                self.stop_tts_quick_request_event.set()
                self.llm_answer_ready_event.set() # Wake up TTS worker if it's waiting
                stopped = self.stop_tts_quick_finished_event.wait(timeout=5.0) # Wait for TTS worker
                if stopped:
                    logger.info(f"🗣️🛑👄👍 {current_gen_id_str} Quick TTS stopped confirmation received.")
                    self.stop_tts_quick_finished_event.clear() # Reset
                else:
                    logger.warning(f"🗣️🛑👄⏱️ {current_gen_id_str} Timeout waiting for Quick TTS stop confirmation.")
                self.tts_quick_generation_active = False # Ensure flag is off
                aborted_something = True
            else:
                logger.info(f"🗣️🛑👄📴 {current_gen_id_str} Quick TTS appears inactive, no stop needed.")
            self.stop_tts_quick_request_event.clear() # Ensure stop request is clear

            # --- Abort Final TTS ---
            # Check if TTS Final is potentially active (just running, doesn't wait on an event like others)
            is_tts_final_potentially_active = self.tts_final_generation_active
            if is_tts_final_potentially_active:
                logger.info(f"🗣️🛑👄❌ {current_gen_id_str} Stopping Final TTS...")
                self.stop_tts_final_request_event.set()
                # No event to .set() here to wake it up, it polls state
                stopped = self.stop_tts_final_finished_event.wait(timeout=5.0) # Wait for TTS worker
                if stopped:
                    logger.info(f"🗣️🛑👄👍 {current_gen_id_str} Final TTS stopped confirmation received.")
                    self.stop_tts_final_finished_event.clear() # Reset
                else:
                    logger.warning(f"🗣️🛑👄⏱️ {current_gen_id_str} Timeout waiting for Final TTS stop confirmation.")
                self.tts_final_generation_active = False # Ensure flag is off
                aborted_something = True
            else:
                logger.info(f"🗣️🛑👄📴 {current_gen_id_str} Final TTS appears inactive, no stop needed.")
            self.stop_tts_final_request_event.clear() # Ensure stop request is clear

            # --- Stop Audio Playback (if AudioProcessor handles it) ---
            # Assuming AudioProcessor might have playback control that needs stopping
            if hasattr(self.audio, 'stop_playback'):
                logger.info(f"🗣️🛑🔊 {current_gen_id_str} Requesting audio playback stop.")
                try:
                    self.audio.stop_playback() # Or similar method
                except Exception as audio_e:
                    logger.warning(f"🗣️🛑🔊💥 {current_gen_id_str} Error stopping audio playback: {audio_e}")


            # --- Clear the running generation object and close generator ---
            # Re-check self.running_generation in case it changed *during* the waits above
            # Use the initially captured current_gen_obj for closing the generator if needed
            if self.running_generation is not None and self.running_generation.id == current_gen_obj.id:
                logger.info(f"🗣️🛑🧹 {current_gen_id_str} Clearing running generation object.")
                # NOTE: do NOT call llm_generator.close() here. The LLM worker
                # thread is still iterating the generator at this point, so a
                # concurrent .close() raises "generator already executing".
                # The stream is already cancelled via llm.cancel_generation()
                # (closes the underlying response), and the worker's own finally
                # block tears the generator down. .close() here is redundant.
                self.running_generation = None # Clear the reference
            elif self.running_generation is not None and self.running_generation.id != current_gen_obj.id:
                 logger.warning(f"🗣️🛑❓ {current_gen_id_str} Mismatch: self.running_generation changed during abort (now Gen {self.running_generation.id}). Clearing current ref.")
                 self.running_generation = None # Clear the unexpected new one too? Or just log? Clearing seems safer.
            elif aborted_something:
                logger.info(f"🗣️🛑🤷 {current_gen_id_str} Worker(s) aborted but running_generation was already None.")
            else:
                logger.info(f"🗣️🛑🤷 {current_gen_id_str} Nothing seemed active to abort, running_generation is None.")


            # --- Final Cleanup of Trigger Events ---
            # Ensure workers don't accidentally pick up stale signals if they restart quickly
            self.generator_ready_event.clear()
            self.llm_answer_ready_event.clear()

            # --- Signal Completion ---
            logger.info(f"🗣️🛑✅ {current_gen_id_str} Abort processing complete. Setting completion event and releasing block.")
            self.abort_completed_event.set() # Signal that the abort process is fully done
            self.abort_block_event.set() # Release the block for the request processor

    # --- Public Methods ---

    def prepare_generation(self, txt: str):
        """
        Public method to request the preparation of a new speech generation.

        Queues a 'prepare' action with the provided text onto the `requests_queue`
        for the request processing worker thread.

        Args:
            txt: The user input text to be synthesized.
        """
        logger.info(f"🗣️📥 Queueing 'prepare' request for: '{txt[:50]}...'")
        self.requests_queue.put(PipelineRequest("prepare", txt))

    def should_speak(self, txt: str, diarizer_participants: Optional[list] = None) -> bool:
        """Speak/not-speak gate — call BEFORE prepare_generation.

        Pure arithmetic on live signals (name address, silence windows, floor
        balance, diarizer participant count). Zero added latency: it's
        microseconds, and returning False *skips* the LLM round-trip entirely
        (faster than speaking). Returns True if the agent should respond.
        """
        decision = self.dynamics.should_speak(
            txt,
            recent_history=self.history,
            diarizer_participants=diarizer_participants,
        )
        if not decision.should_speak:
            logger.info(
                f"🗣️🤫 Holding (not speaking): appetite={decision.appetite:.2f} "
                f"thresh={decision.threshold:.2f} | {decision.reason_str()}"
            )
        else:
            logger.info(
                f"🗣️🔊 Speaking: appetite={decision.appetite:.2f} "
                f"thresh={decision.threshold:.2f} | {decision.reason_str()}"
            )
        return decision.should_speak

    def finish_generation(self):
        """
        Public method to signal the end of user input or interaction.

        Queues a 'finish' action onto the `requests_queue`.
        Note: Currently, the request worker acknowledges this action but doesn't
        trigger specific pipeline behavior based on it. It might be used for
        future features like finalizing history or state.
        """
        logger.info(f"🗣️📥 Queueing 'finish' request")
        self.requests_queue.put(PipelineRequest("finish"))

    def abort_generation(self, wait_for_completion: bool = False, timeout: float = 7.0, reason: str = ""):
        """
        Public method to initiate the abortion of the current speech generation.

        Calls the internal `process_abort_generation` method to handle the actual
        stopping of workers and cleanup. Optionally waits for the abortion to fully
        complete.

        Args:
            wait_for_completion: If True, blocks until the abort process finishes
                                 (signaled by `abort_completed_event`).
            timeout: Maximum time in seconds to wait if `wait_for_completion` is True.
            reason: A string describing why the abort was requested (for logging).
        """
        if self.shutdown_event.is_set():
            logger.warning("🗣️🔌 Shutdown in progress, ignoring abort request.")
            return

        gen_id_str = f"Gen {self.running_generation.id}" if self.running_generation else "Gen None"
        logger.info(f"🗣️🛑🚀 Requesting 'abort' (wait={wait_for_completion}, reason='{reason}') for {gen_id_str}")

        # Call the internal synchronous processor
        self.process_abort_generation()

        # Optionally wait for completion
        if wait_for_completion:
            logger.info(f"🗣️🛑⏳ Waiting for abort completion (timeout={timeout}s)...")
            completed = self.abort_completed_event.wait(timeout=timeout)
            if completed:
                logger.info(f"🗣️🛑✅ Abort completion confirmed.")
            else:
                logger.warning(f"🗣️🛑⏱️ Timeout waiting for abort completion event.")
            # Ensure block is released after waiting, even on timeout
            self.abort_block_event.set()


    def trim_history(self):
        """Drop the oldest messages beyond HISTORY_MAX_MESSAGES.

        Called after every history append so the LLM prefill stays bounded.
        Keeps the most recent turns (which matter for conversational continuity)
        and discards stale ones.
        """
        if len(self.history) > HISTORY_MAX_MESSAGES:
            dropped = len(self.history) - HISTORY_MAX_MESSAGES
            self.history = self.history[dropped:]
            logger.info(f"🗣️🧹 Trimmed {dropped} old history messages (now {len(self.history)}).")

    def _summarize_worker(self) -> None:
        """Periodically compact long history into a rolling summary."""
        while not self.shutdown_event.is_set():
            self.shutdown_event.wait(timeout=SUMMARY_INTERVAL_S)
            if self.shutdown_event.is_set():
                break
            try:
                self.compact_history()
            except Exception as e:  # noqa: BLE001
                logger.warning("🗣️🧠⚠️ summary worker error: %s", e)

    def compact_history(self) -> bool:
        """Summarize the oldest messages when history outgrows its window.

        Runs only when no generation is active (avoids concurrent Ollama calls
        and the shared cancellation state). Oldest messages are replaced by a
        single system note carrying the gist; the most recent stay verbatim.
        """
        with self.summary_lock:
            if len(self.history) <= HISTORY_SUMMARY_THRESHOLD:
                return False
            if self.llm_generation_active or self.running_generation is not None:
                return False
            split = len(self.history) - HISTORY_KEEP_RECENT
            old = self.history[:split]
            recent = self.history[split:]
            summary = self._summarize_messages(old)
            if not summary:
                return False
            self.history = [{"role": "system", "content": f"[conversation so far] {summary}"}] + recent
            logger.info(f"🗣️🧠🧹 Compacted {split} messages into a rolling summary (now {len(self.history)}).")
            return True

    def _summarize_messages(self, messages) -> str:
        """One-shot resident-LLM summarization (no protocol header, no thinking)."""
        if not messages or self.llm is None:
            return ""
        text = "\n".join(f"{m.get('role','?')}: {m.get('content','')}" for m in messages)
        prompt = ("Summarize this conversation excerpt into 2-4 sentences capturing the key facts, "
                  "decisions, people involved, and open questions. Be concrete; no editorializing.\n\n"
                  f"CONVERSATION:\n{text}")
        try:
            chunks = list(self.llm.generate(prompt, use_system_prompt=False, use_tools=False))
            return "".join(chunks).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("🗣️🧠⚠️ summarization failed: %s", e)
            return ""

    def remember(self, text: str) -> None:
        """Queue a user turn for async situation-fact extraction (non-blocking)."""
        if ENABLE_MEMORY and text and text.strip():
            try:
                self.memory_queue.put((self.current_persona, text.strip()))
            except Exception as e:  # noqa: BLE001
                logger.warning("🗣️🧠⚠️ remember() enqueue failed: %s", e)

    def _memory_worker(self) -> None:
        """Extract + persist situation facts only when the pipeline is idle."""
        while not self.shutdown_event.is_set():
            try:
                persona, text = self.memory_queue.get(timeout=1.0)
            except Empty:
                continue
            if self.llm_generation_active or self.running_generation is not None:
                # never compete with a live generation for the model; requeue
                self.memory_queue.put((persona, text))
                self.shutdown_event.wait(timeout=2.0)
                continue
            mem = self._memory_for(persona)
            if mem:
                mem.add_situation(text)

    def _memory_for(self, persona: str):
        with self.memory_lock:
            if persona not in self.memories:
                try:
                    from memory_module import SituationMemory
                    self.memories[persona] = SituationMemory(persona, store_dir="memory_db")
                except Exception as e:  # noqa: BLE001
                    logger.warning("🗣️🧠⚠️ memory init failed for '%s': %s", persona, e)
                    self.memories[persona] = None
            return self.memories[persona]

    def _memory_context(self, text: str) -> str:
        if not ENABLE_MEMORY:
            return ""
        mem = self._memory_for(self.current_persona)
        if not mem:
            return ""
        try:
            return mem.context_block(text, limit=4)
        except Exception as e:  # noqa: BLE001
            logger.warning("🗣️🧠⚠️ memory recall failed: %s", e)
            return ""

    def set_persona(self, name: str) -> bool:
        """Swap the active persona (system prompt) and reset conversation state.

        Each persona has its own identity, so switching must also clear history
        to avoid cross-persona bleed (e.g. Max's slang in Ivy's replies).
        The LLM's cached system prompt is updated in place so no restart is needed.
        """
        name = (name or "").strip().lower()
        if name not in PERSONAS:
            logger.warning(f"🗣️❓ Unknown persona '{name}'. Available: {list(PERSONAS)}.")
            return False
        self.current_persona = name
        from identity import note as _id_note
        try:
            import response_decision as _rd_life; _rd_life.set_agent(name)
        except Exception:
            pass
        self.system_prompt = f"Your name is {display_name(name)}. {_id_note(name, display_name(name))} " + PERSONAS[name]
        try:
            import name_hearing
            name_hearing.set_names([display_name(name)])
        except Exception as e:  # noqa: BLE001
            logger.warning('name_hearing set failed: %s', e)
        self.dynamics.agent_names = (display_name(name).lower(),)
        # Update the LLM's cached system prompt message in place.
        if self.llm is not None:
            self.llm.system_prompt = self.system_prompt
            self.llm.system_prompt_message = {"role": "system", "content": self.system_prompt}
        # Clear history so the new identity starts fresh (also aborts any live gen).
        self.abort_generation(wait_for_completion=True, timeout=7.0, reason=f"set_persona:{name}")
        # Each agent owns its own runtime: the previous agent keeps its memory,
        # this one resumes its own (fresh if stale). Nothing is shared except
        # people's names (same humans) and the per-persona task boards.
        self._runtime()  # ensure registry exists
        rt = self.agents.activate(name)
        self.dynamics.agent_names = rt.profile.names
        # Agents created after startup (e.g. with +) must count as agent names in the
        # name book too, or 'Hi Wren. I'm Riley.' is misjudged (demo take 10).
        try:
            self.people.agent_names |= {display_name(name).lower(), str(name).lower(),
                                        *(str(n).lower() for n in (rt.profile.names or ()))}
        except Exception as e:  # noqa: BLE001
            logger.warning('people.agent_names update failed: %s', e)
        # Swap to the persona's assigned voice (cache-backed hot-swap).
        from voices import PERSONA_VOICES, DEFAULT_VOICE
        _voice = PERSONA_VOICES.get(name, DEFAULT_VOICE)
        if _voice:
            self.audio.set_voice(_voice)
        logger.info(f"🗣️🎭 Switched persona to '{name}'.")
        try:
            from call_recorder import REC
            REC.event("persona", persona=name)
        except Exception:  # noqa: BLE001
            pass
        return True

    def set_voice(self, name: str) -> bool:
        """Independent voice override (hot-swap the TTS voice without changing persona)."""
        ok = self.audio.set_voice(name)
        if ok:
            logger.info(f"🗣️🔊 Voice override set to '{name}'.")
        return ok

    def _on_tool_call(self, tool_calls) -> None:
        """Fire ONE search backchannel per generation (never enters history)."""
        try:
            names = {(tc.get("function") or {}).get("name") for tc in tool_calls or []}
            looking = "look_at_screen" in names and "web_search" not in names
            if not names & {"web_search", "look_at_screen"}:
                return
            gen = self.running_generation
            gid = getattr(gen, "id", None)
            if gid is not None and gid == self._backchannel_gen:
                return
            self._backchannel_gen = gid
            self._last_backchannel_text = ""
            cb = self.on_backchannel
            clip = (self.backchannels.pick(self.audio.current_voice, kind="look" if looking else "search")
                    if cb else None)
            if clip:
                text, pcm = clip
                logger.info(f"🗣️⏳ {'look' if looking else 'search'} backchannel ({self.audio.current_voice}): {text}")
                self._last_backchannel_text = text
                cb(pcm)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"🗣️⏳ backchannel failed: {e}")

    def refresh_registry(self) -> None:
        """Re-read agents/voices after a create/delete/import (no restart)."""
        import voices as _v
        _registry.reload()
        _v.refresh()
        _rebuild_personas()
        if self.current_persona not in PERSONAS:
            self.current_persona = DEFAULT_PERSONA

    def register_agent(self, rec: dict, prompt: str) -> None:
        """Make a freshly created agent selectable without a restart."""
        self.refresh_registry()
        if rec["id"] not in PERSONAS:  # registry file not written yet (tests)
            PERSONAS[rec["id"]] = prompt.strip()
            PERSONA_META[rec["id"]] = {"role": rec.get("role", ""), "accent": rec.get("accent", "#22d3ee"),
                                       "name": rec.get("name"), "custom": True}
            import voices as _v
            if rec.get("voice"):
                _v.PERSONA_VOICES[rec["id"]] = rec["voice"]
        if "agents" in self.__dict__:
            self.agents.forget(rec["id"])  # pick up interests/talkativeness fresh
        logger.info(f"🗣️🆕 Agent '{rec['id']}' registered (voice {rec.get('voice')}).")

    def remove_agent(self, pid: str) -> bool:
        if not PERSONA_META.get(pid, {}).get("custom") or pid == self.current_persona:
            return False
        import voices as _v
        PERSONAS.pop(pid, None); PERSONA_FILES.pop(pid, None); PERSONA_META.pop(pid, None)
        _v.PERSONA_VOICES.pop(pid, None)
        if "agents" in self.__dict__:
            self.agents.forget(pid)
        return True

    def persona_voice(self, name: str) -> str:
        from voices import PERSONA_VOICES, DEFAULT_VOICE
        return PERSONA_VOICES.get(name, DEFAULT_VOICE)

    def personas_info(self) -> dict:
        """Snapshot of personas + voices + current selection for the client UI."""
        from voices import PERSONA_VOICES, DEFAULT_VOICE, voice_names
        personas = [
            {
                "id": name,
                "name": display_name(name),
                "custom": bool(PERSONA_META.get(name, {}).get("custom")),
                "voice": PERSONA_VOICES.get(name, DEFAULT_VOICE),
                "role": PERSONA_META.get(name, {}).get("role", ""),
                "accent": PERSONA_META.get(name, {}).get("accent", "#22d3ee"),
                "placeholder": bool(PERSONA_META.get(name, {}).get("placeholder")),
            }
            for name in PERSONAS
        ]
        return {
            "personas": personas,
            "voices": voice_names(),
            "current": {
                "persona": self.current_persona,
                "voice": self.audio.current_voice,
            },
        }

    def reset(self):
        """
        Resets the pipeline state completely.

        Aborts any currently running generation (waiting for completion) and
        clears the conversation history.
        """
        logger.info("🗣️🔄 Resetting pipeline state...")
        self.abort_generation(wait_for_completion=True, timeout=7.0, reason="reset") # Ensure clean slate
        self.agent.reset_conversation()
        self.agent.reset_session()
        logger.info("🗣️🧹 History cleared. Reset complete.")

    def shutdown(self):
        """
        Initiates a graceful shutdown of the pipeline manager and worker threads.

        1. Sets the `shutdown_event`.
        2. Attempts a final abort of any running generation.
        3. Signals all relevant events to unblock any waiting worker threads.
        4. Joins each worker thread with a timeout, logging warnings if they fail to exit.
        """
        logger.info("🗣️🔌 Initiating shutdown...")
        self.shutdown_event.set()

        # Try a final synchronous abort to ensure clean state before join
        logger.info("🗣️🔌🛑 Attempting final abort before joining threads...")
        self.abort_generation(wait_for_completion=True, timeout=3.0, reason="shutdown")

        # Wake up threads that might be waiting on events so they can check shutdown_event
        logger.info("🗣️🔌🔔 Signaling events to wake up any waiting threads...")
        self.generator_ready_event.set()
        self.llm_answer_ready_event.set()
        # Also signal 'finished' and 'completion' events
        self.stop_llm_finished_event.set()
        self.stop_tts_quick_finished_event.set()
        self.stop_tts_final_finished_event.set()
        self.abort_completed_event.set()
        self.abort_block_event.set() # Ensure request processor isn't blocked

        # Join threads
        threads_to_join = [
            (self.request_processing_thread, "Request Processor"),
            (self.llm_inference_thread, "LLM Worker"),
            (self.tts_quick_inference_thread, "Quick TTS Worker"),
            (self.tts_final_inference_thread, "Final TTS Worker"),
            (self.summary_thread, "Summary Worker"),
            (self.memory_worker_thread, "Memory Worker"),
        ]

        for thread, name in threads_to_join:
             if thread.is_alive():
                 logger.info(f"🗣️🔌⏳ Joining {name}...")
                 thread.join(timeout=5.0)
                 if thread.is_alive():
                     logger.warning(f"🗣️🔌⏱️ {name} thread did not join cleanly.")
             else:
                  logger.info(f"🗣️🔌👍 {name} thread already finished.")


        logger.info("🗣️🔌✅ Shutdown complete.")