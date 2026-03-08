#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prosper Health scheduling voice agent.

A real-time voice bot that collects patient name and date of birth,
looks up the patient in Healthie, then books an appointment — all
over a live phone-style WebRTC call.

Required services:
- ElevenLabs (realtime STT + TTS)
- OpenAI (LLM + lightweight mid-wait responder)
- Playwright / Healthie credentials (see healthie.py)

Run the bot::

    uv run bot.py

Then open http://localhost:7860 and click Connect.
"""

import asyncio
import os
from datetime import date

from dateutil.parser import parse as parse_date
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger.info("Loading pipeline components...")
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import TranscriptionUserTurnStopStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

logger.info("All components loaded")

load_dotenv(override=True)

import healthie

SYSTEM_PROMPT = """You are a warm, friendly scheduling assistant for Prosper Health clinic. Your job is to book appointments for patients over the phone.

Follow these steps in order:

1. Ask for the patient's full name.
2. Ask for their date of birth.
3. Call find_patient with the name and date of birth. You MUST call this function — never skip it or assume a result.
   - If not found, gently ask the patient to spell their last name (speech recognition sometimes splits names) and call find_patient again. Only give up after 2 failed attempts.
4. Once found, say exactly once: "Got it, I have your record!" — never repeat this confirmation even if you see it in the conversation history.
5. Ask for their preferred appointment date.
   - If the patient gives a date in the past, politely point it out and ask for a future date.
6. Ask for their preferred time.
   - If the patient gives a vague time (e.g. "morning", "afternoon"), ask for a specific time.
7. Confirm: "Just to confirm — [date] at [time], does that work for you?"
8. Call create_appointment. You MUST call this function — never say the appointment is booked without calling it first.
   - If successful, confirm warmly and wish them well.
   - If it fails, apologize and suggest trying a different date or time.

Speech recognition sometimes merges multiple utterances into one. For example: "Yes. Hello? 3:00 PM sounds good." — treat the LAST clear date or time mentioned as their final answer. If they confirm ("yes", "sounds good", "perfect") AND give a new time in the same turn, use the new time and skip re-confirming — just proceed to book it.

If a patient says something that sounds like a greeting (e.g. "hello", "hi", "hey") in the middle of their response, ignore it — they are just checking you're still there. Focus on the actual content of what they said.

If the patient asks an off-topic question (e.g. clinic hours, directions, cost), answer it briefly and honestly — do not make up specific details you don't know — then continue exactly where you left off in the scheduling flow. Do not re-confirm things you already confirmed.

If a single patient turn contains both a question AND scheduling information (e.g. "It's May 17th, 2001. Are you open weekends?"), answer the question briefly first, then continue the flow (e.g. call the function, ask the next question). Do not silently skip the question.

Sometimes you will see in the conversation history that a question was already answered while the system was processing (you'll see a user question followed immediately by a short assistant reply, before a tool result). In that case, do not repeat or re-answer those questions. Just acknowledge naturally if needed (e.g. "And as I mentioned, ...") and move the flow forward.

If the patient seems to want to end the call without booking, wish them well and let them go — don't push.

Style:
- This is a phone call — speak naturally, use short sentences.
- Use warm filler phrases like "Sure!", "Of course!", "Great!" to sound human.
- Never use bullet points, lists, or markdown.
- Today is {today}."""


def _make_tools() -> ToolsSchema:
    find_patient_fn = FunctionSchema(
        name="find_patient",
        description="Search for a patient in the Healthie EHR system by their full name and date of birth. Call this after collecting both pieces of information from the patient.",
        properties={
            "name": {
                "type": "string",
                "description": "The patient's full name, e.g. 'John Doe'.",
            },
            "date_of_birth": {
                "type": "string",
                "description": "The patient's date of birth, e.g. '1990-01-15' or 'January 15, 1990'.",
            },
        },
        required=["name", "date_of_birth"],
    )

    create_appointment_fn = FunctionSchema(
        name="create_appointment",
        description="Create an appointment in Healthie for a patient. Call this after you have a confirmed patient_id and the patient has provided their preferred date and time.",
        properties={
            "patient_id": {
                "type": "string",
                "description": "The unique patient ID returned by find_patient.",
            },
            "date": {
                "type": "string",
                "description": "The appointment date, e.g. 'March 10, 2026'.",
            },
            "time": {
                "type": "string",
                "description": "The appointment time in 12-hour format, e.g. '2:00 PM' or '10:30 AM'.",
            },
        },
        required=["patient_id", "date", "time"],
    )

    return ToolsSchema(standard_tools=[find_patient_fn, create_appointment_fn])


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    elevenlabs_key = os.environ["ELEVENLABS_API_KEY"]
    stt = ElevenLabsRealtimeSTTService(
        api_key=elevenlabs_key,
        params=ElevenLabsRealtimeSTTService.InputParams(
            # MANUAL mode: ElevenLabs commits when Pipecat's Silero VAD sends
            # VADUserStoppedSpeakingFrame. We set a 0.8s silence threshold so
            # short responses ("Yeah.", "Yes.") commit quickly without merging
            # into subsequent noise, while still capturing full sentences.
            commit_strategy=CommitStrategy.MANUAL,
            vad_silence_threshold_secs=0.8,
        ),
    )
    tts = ElevenLabsTTSService(
        api_key=elevenlabs_key,
        voice_id="SAz9YHcvj6GT2YYXdXww",
    )

    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])

    # --- Context and pipeline (built first so task is available to handlers) ---

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().strftime("%B %-d, %Y"))}]
    tools = _make_tools()
    context = LLMContext(messages, tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3()),
                    TranscriptionUserTurnStopStrategy(),
                ]
            ),
        ),
    )

    rtvi = RTVIProcessor()

    # Mutable container shared by the interceptor class and function handler closures.
    # q:                asyncio.Queue for mid-wait transcript routing (set while a function call is active)
    # pending:          raw transcripts received during a function call (used to suppress downstream flow)
    # pending_exchanges: (user_text, assistant_reply) pairs from the side-channel responder, injected
    #                   into the main LLM context as already-resolved exchanges before result_callback
    #                   fires — prevents the post-tool LLM from re-answering questions already handled.
    _state: dict = {"q": None, "pending": [], "pending_exchanges": []}

    class TranscriptInterceptor(FrameProcessor):
        """Passes all frames downstream unchanged, but also feeds committed
        TranscriptionFrames into _state['q'] when a Playwright op is active."""
        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            q = _state["q"]
            if (
                q is not None
                and isinstance(frame, TranscriptionFrame)
                and direction == FrameDirection.DOWNSTREAM
                and frame.text.strip()
            ):
                text = frame.text.strip()
                await q.put(text)
                # Stage for manual context injection after the tool call returns.
                # Drop the frame here so user_aggregator doesn't also add it —
                # we'll inject it ourselves via context.add_message before
                # result_callback fires, preventing a duplicate user turn.
                _state["pending"].append(text)
                return  # do NOT push downstream; we own this turn now
            await self.push_frame(frame, direction)

    interceptor = TranscriptInterceptor()

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            interceptor,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # --- Mid-wait conversational responsiveness ---
    #
    # While a Playwright function call is in-flight, the pipeline LLM is blocked by
    # OpenAI's tool call protocol. TranscriptInterceptor (defined above, placed between
    # STT and user_aggregator in the pipeline) feeds committed transcripts into
    # _state["q"] when a Playwright op is active. A responder coroutine reads from the
    # queue and makes a direct gpt-4o-mini call to generate a short reply via TTS —
    # without touching the main LLM context or message history.

    _openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def _respond_while_waiting(q: asyncio.Queue, filler: str, stop_event: asyncio.Event):
        """Speak filler immediately, then reply naturally to each patient turn.

        Each (user_text, reply) pair is appended to _state['pending_exchanges'] so the
        function handler can inject both sides into the main LLM context as an already-
        resolved exchange. This prevents the post-tool LLM call from re-answering
        questions the side-channel already handled.
        """
        await task.queue_frames([TTSSpeakFrame(filler)])
        while not stop_event.is_set():
            try:
                user_text = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            logger.info(f"[mid-wait] user: {user_text!r}")
            try:
                resp = await _openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            "You are a warm phone scheduling assistant for Prosper Health clinic. "
                            "You are currently in the middle of a system lookup (either finding the patient's record "
                            "or booking their appointment) — you cannot take any booking actions right now. "
                            "Answer the patient's question as best you can in 1-2 short sentences. "
                            "If you genuinely don't know the answer (e.g. specific clinic hours, specialist availability), "
                            "say so briefly and honestly. "
                            "Do not mention that a lookup is happening or use technical language. "
                            "Sound natural and conversational."
                        )},
                        {"role": "user", "content": user_text},
                    ],
                    max_tokens=80,
                )
                reply = resp.choices[0].message.content.strip()
                logger.info(f"[mid-wait] reply: {reply!r}")
                await task.queue_frames([TTSSpeakFrame(reply)])
                # Record the exchange so the handler can replay it into context.
                _state["pending_exchanges"].append((user_text, reply))
            except Exception as e:
                logger.warning(f"[mid-wait] error: {e}")

    # How long to keep the intercept queue open after the Playwright call finishes,
    # to catch transcripts that commit late (ElevenLabs STT has a ~1-1.5s commit delay).
    _INTERCEPT_GRACE_SECS = 2.0

    async def _wait_for_late_transcripts():
        """Hold _state['q'] open for a short grace window so TranscriptInterceptor
        can still stage late-committing STT transcripts into pending_exchanges.
        The side-channel responder is already stopped; we just need the interceptor
        to keep dropping frames out of the normal aggregator path."""
        logger.info("[intercept] grace period started")
        await asyncio.sleep(_INTERCEPT_GRACE_SECS)
        _state["q"] = None
        logger.info("[intercept] grace period ended, queue closed")


    async def handle_find_patient(params: FunctionCallParams):
        name = params.arguments.get("name", "").strip()
        dob = params.arguments.get("date_of_birth", "").strip()
        logger.info(f"[find_patient] name={name!r} dob={dob!r}")

        if not name or not dob:
            await params.result_callback({"error": "Missing name or date of birth."})
            return

        _state["q"] = asyncio.Queue()
        stop_event = asyncio.Event()
        responder = asyncio.ensure_future(
            _respond_while_waiting(_state["q"], "One moment while I pull up your records.", stop_event)
        )
        try:
            result = await healthie.find_patient(name, dob)
        except Exception as e:
            logger.error(f"[find_patient] unexpected error: {e}")
            result = None
        finally:
            stop_event.set()
            responder.cancel()
            # Keep queue open during grace period so late STT commits are still intercepted.
            await _wait_for_late_transcripts()

        # Inject any questions the patient asked while we were looking them up as
        # already-resolved user+assistant pairs. The side-channel already spoke the
        # reply, so the post-tool LLM just needs to see it as settled history — not
        # as an open question to answer again.
        # Any remaining in pending (late commits during grace period, no side-channel
        # reply) get a synthetic assistant line so the main LLM doesn't re-answer.
        exchanges = _state["pending_exchanges"]
        pending = _state["pending"]
        _state["pending"] = []
        _state["pending_exchanges"] = []
        for user_text, assistant_reply in exchanges:
            context.add_message({"role": "user", "content": user_text})
            context.add_message({"role": "assistant", "content": assistant_reply})
            logger.info(f"[find_patient] injecting exchange into context: user={user_text!r} assistant={assistant_reply!r}")
        for user_text in pending:
            context.add_message({"role": "user", "content": user_text})
            context.add_message({"role": "assistant", "content": "I acknowledged that briefly while pulling up your record."})
            logger.info(f"[find_patient] injecting late pending into context: user={user_text!r}")

        logger.info(f"[find_patient] result={result!r}")
        if result is None:
            await params.result_callback({"error": "Patient not found. Please verify the name and date of birth."})
        else:
            await params.result_callback(result)

    async def handle_create_appointment(params: FunctionCallParams):
        patient_id = params.arguments.get("patient_id", "").strip()
        appt_date = params.arguments.get("date", "").strip()
        appt_time = params.arguments.get("time", "").strip()
        logger.info(f"[create_appointment] patient_id={patient_id!r} date={appt_date!r} time={appt_time!r}")

        if not patient_id or not appt_date or not appt_time:
            await params.result_callback({"error": "Missing required appointment details."})
            return

        # Guard against past dates before hitting Healthie
        try:
            appt_dt = parse_date(appt_date, default=date.today().replace(day=1))
            if appt_dt.date() < date.today():
                await params.result_callback({"error": f"The date {appt_date} is in the past. Please provide a future date."})
                return
        except Exception:
            pass  # If we can't parse the date, let Healthie handle it

        _state["q"] = asyncio.Queue()
        stop_event = asyncio.Event()
        responder = asyncio.ensure_future(
            _respond_while_waiting(_state["q"], "Perfect, I'm booking that for you now — just a moment.", stop_event)
        )
        try:
            result = await healthie.create_appointment(patient_id, appt_date, appt_time)
        except Exception as e:
            logger.error(f"[create_appointment] unexpected error: {e}")
            result = None
        finally:
            stop_event.set()
            responder.cancel()
            # Keep queue open during grace period so late STT commits are still intercepted.
            await _wait_for_late_transcripts()

        # Inject any questions asked during the booking wait as already-resolved pairs.
        # Late commits (grace period only) get a synthetic assistant line.
        exchanges = _state["pending_exchanges"]
        pending = _state["pending"]
        _state["pending"] = []
        _state["pending_exchanges"] = []
        for user_text, assistant_reply in exchanges:
            context.add_message({"role": "user", "content": user_text})
            context.add_message({"role": "assistant", "content": assistant_reply})
            logger.info(f"[create_appointment] injecting exchange into context: user={user_text!r} assistant={assistant_reply!r}")
        for user_text in pending:
            context.add_message({"role": "user", "content": user_text})
            context.add_message({"role": "assistant", "content": "I acknowledged that while booking."})
            logger.info(f"[create_appointment] injecting late pending into context: user={user_text!r}")

        logger.info(f"[create_appointment] result={result!r}")
        if result is None:
            await params.result_callback({"error": "Failed to create appointment. The slot may be unavailable — please try a different date or time."})
        else:
            await params.result_callback(result)

    llm.register_function("find_patient", handle_find_patient, cancel_on_interruption=False)
    llm.register_function("create_appointment", handle_create_appointment, cancel_on_interruption=False)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({
            "role": "system",
            "content": "Say hello and briefly introduce yourself as a scheduling assistant from Prosper Health clinic. Then ask for the patient's full name.",
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    # Pre-warm the Healthie browser session so the first find_patient call
    # doesn't pay the cold-start login cost (~20s) during a live call.
    logger.info("Pre-warming Healthie session...")
    try:
        await healthie.login_to_healthie()
        logger.info("Healthie session ready.")
    except Exception as e:
        logger.warning(f"Healthie pre-warm failed (will retry on first call): {e}")

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                confidence=0.8,   # filter fan/background noise
                start_secs=0.3,   # require 0.3s of real speech to start a turn
                stop_secs=0.6,    # 0.6s silence ends the VAD turn
                min_volume=0.75,  # fan noise typically below this
            )),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
