"""ElevenLabs Conversational AI — native voice agent capability for Oflo.

Gives every BaseAgentV2 a full real-time voice channel:
  • Microphone input → ElevenLabs STT → agent logic → ElevenLabs TTS → speakers
  • All Oflo tools are auto-registered with ElevenLabs ClientTools
  • Transcripts flow back to the agent's conversation history and memory
  • Callbacks expose latency, corrections, and alignment data for telemetry

Docs: https://elevenlabs.io/docs/eleven-agents/libraries/python

Usage::

    from oflo_agent_protocol.voice import VoiceAgent

    agent = VoiceAgent(
        name="Sales Rep",
        system_prompt="You are a helpful sales representative.",
        elevenlabs_agent_id="your-agent-id",  # create in ElevenLabs dashboard
        voice_id="Rachel",                     # optional TTS voice for non-voice contexts
    )

    @agent.tool(description="Look up product pricing")
    async def get_pricing(product: str) -> dict:
        return {"product": product, "price": 99.0, "currency": "USD"}

    # Start a real-time voice session (blocks until conversation ends)
    await agent.start_voice_session()

    # Or use as a text agent with TTS output
    audio_bytes = await agent.speak("Hello! How can I help you today?")
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from oflo_agent_protocol.core.agent import BaseAgentV2
from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.types import AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class VoiceSessionStats:
    """Statistics collected during a voice session."""
    conversation_id: Optional[str] = None
    user_transcripts: List[str] = field(default_factory=list)
    agent_responses: List[str] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    corrections: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / max(len(self.latencies_ms), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": len(self.agent_responses),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "corrections": self.corrections,
        }


class VoiceAgent(BaseAgentV2):
    """
    BaseAgentV2 with a native ElevenLabs voice channel.

    The ElevenLabs agent handles STT → LLM → TTS in real-time.
    This class bridges the tool registry and conversation history
    between the two systems, keeping Oflo as the source of truth.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "You are a helpful voice assistant.",
        elevenlabs_agent_id: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        voice_id: str = "Rachel",
        requires_auth: bool = True,
        tts_model: str = "eleven_turbo_v2_5",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, system_prompt=system_prompt, **kwargs)
        self._el_agent_id = elevenlabs_agent_id or os.getenv("ELEVENLABS_AGENT_ID", "")
        self._el_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self._voice_id = voice_id
        self._requires_auth = requires_auth
        self._tts_model = tts_model
        self._session_stats: Optional[VoiceSessionStats] = None
        self._el_client: Optional[Any] = None
        self._conversation: Optional[Any] = None

    def _get_el_client(self) -> Any:
        if self._el_client is None:
            try:
                from elevenlabs import ElevenLabs
                self._el_client = ElevenLabs(api_key=self._el_api_key or None)
            except ImportError:
                raise ImportError(
                    "elevenlabs package required. Install with: pip install elevenlabs"
                )
        return self._el_client

    def _build_client_tools(self) -> Any:
        """Register all Oflo tools with ElevenLabs ClientTools."""
        from elevenlabs.conversational_ai.conversation import ClientTools

        cl = ClientTools()
        for td in self._tools.values():
            is_async = asyncio.iscoroutinefunction(td.handler)
            cl.register(td.name, td.handler, is_async=is_async)
            logger.debug("Registered voice tool: %s (async=%s)", td.name, is_async)
        return cl

    # ------------------------------------------------------------------
    # Voice session lifecycle
    # ------------------------------------------------------------------

    async def start_voice_session(
        self,
        on_user_transcript: Optional[Callable[[str], None]] = None,
        on_agent_response: Optional[Callable[[str], None]] = None,
        on_session_end: Optional[Callable[[VoiceSessionStats], None]] = None,
    ) -> VoiceSessionStats:
        """
        Start a real-time voice conversation.

        Blocks until the user ends the session (Ctrl+C or conversation ends).
        Returns session statistics when done.
        """
        try:
            from elevenlabs.conversational_ai.conversation import Conversation
            from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
        except ImportError:
            raise ImportError(
                "elevenlabs[pyaudio] required for voice sessions. "
                "Install with: pip install 'elevenlabs[pyaudio]'"
            )

        if not self._el_agent_id:
            raise ValueError(
                "elevenlabs_agent_id is required. Create an agent at elevenlabs.io "
                "and set ELEVENLABS_AGENT_ID env var."
            )

        stats = VoiceSessionStats()
        self._session_stats = stats
        client = self._get_el_client()
        client_tools = self._build_client_tools() if self._tools else None

        def _on_user_transcript(transcript: str) -> None:
            stats.user_transcripts.append(transcript)
            logger.info("[Voice] User: %s", transcript)
            # Inject into text conversation history for audit
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: self._history.append(CanonicalMessage.user(transcript))
            )
            if on_user_transcript:
                on_user_transcript(transcript)

        def _on_agent_response(response: str) -> None:
            stats.agent_responses.append(response)
            logger.info("[Voice] Agent: %s", response[:100])
            asyncio.get_event_loop().call_soon_threadsafe(
                lambda: self._history.append(CanonicalMessage.assistant(response))
            )
            if on_agent_response:
                on_agent_response(response)

        def _on_correction(original: str, corrected: str) -> None:
            stats.corrections += 1
            logger.debug("[Voice] Correction: '%s' → '%s'", original[:50], corrected[:50])

        def _on_latency(latency_ms: int) -> None:
            stats.latencies_ms.append(float(latency_ms))

        self._status = AgentStatus.WORKING
        logger.info("Starting voice session for agent '%s'", self.name)

        try:
            conversation = Conversation(
                client=client,
                agent_id=self._el_agent_id,
                requires_auth=self._requires_auth,
                audio_interface=DefaultAudioInterface(),
                callback_agent_response=_on_agent_response,
                callback_agent_response_correction=_on_correction,
                callback_user_transcript=_on_user_transcript,
                callback_latency_measurement=_on_latency,
                client_tools=client_tools,
            )
            self._conversation = conversation

            # Handle Ctrl+C gracefully
            def _handle_sigint(*_: Any) -> None:
                logger.info("Voice session interrupted — ending...")
                conversation.end_session()

            signal.signal(signal.SIGINT, _handle_sigint)

            conversation.start_session()
            conv_id = conversation.wait_for_session_end()
            stats.conversation_id = str(conv_id) if conv_id else None

        finally:
            self._status = AgentStatus.ACTIVE
            self._conversation = None

        if on_session_end:
            on_session_end(stats)

        logger.info(
            "Voice session ended: %d turns, %.0f ms avg latency",
            len(stats.agent_responses),
            stats.avg_latency_ms,
        )
        return stats

    def end_voice_session(self) -> None:
        """Programmatically end an ongoing voice session."""
        if self._conversation:
            self._conversation.end_session()

    # ------------------------------------------------------------------
    # Text-to-Speech (for text contexts)
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> bytes:
        """
        Synthesise text to audio bytes using ElevenLabs TTS.
        Returns raw MP3 audio — pipe to a player or save to file.
        """
        client = self._get_el_client()
        try:
            audio_stream = client.text_to_speech.convert(
                voice_id=self._voice_id,
                text=text,
                model_id=self._tts_model,
                output_format="mp3_44100_128",
            )
            chunks = [c for c in audio_stream]
            return b"".join(chunks)
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc)
            raise

    async def speak_and_save(self, text: str, path: str) -> str:
        """Synthesise text and save to a file. Returns the file path."""
        audio = await self.speak(text)
        with open(path, "wb") as fh:
            fh.write(audio)
        logger.info("Saved TTS audio to %s (%d bytes)", path, len(audio))
        return path

    # ------------------------------------------------------------------
    # Streaming TTS
    # ------------------------------------------------------------------

    async def speak_stream(self, text: str):
        """Async generator that yields audio chunks for streaming playback."""
        client = self._get_el_client()
        try:
            stream = client.text_to_speech.convert_as_stream(
                voice_id=self._voice_id,
                text=text,
                model_id=self._tts_model,
            )
            for chunk in stream:
                yield chunk
        except Exception as exc:
            logger.error("TTS stream failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Convenience: voice-aware chat (returns text + optionally speaks)
    # ------------------------------------------------------------------

    async def voice_chat(self, user_message: str, play_audio: bool = False) -> str:
        """
        Text chat with optional TTS output.
        Useful when you want the agent to speak its reply without a full voice session.
        """
        reply = await self.chat(user_message)
        if play_audio:
            await self._play_audio(await self.speak(reply))
        return reply

    @staticmethod
    async def _play_audio(audio_bytes: bytes) -> None:
        """Play audio bytes through system speakers using pyaudio."""
        try:
            import io
            import pyaudio
            import wave
            # Minimal MP3 → PCM via pydub if available
            try:
                from pydub import AudioSegment
                from pydub.playback import play as pydub_play
                seg = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                pydub_play(seg)
            except ImportError:
                # Fallback: write to tmp file and open with system player
                import tempfile, subprocess
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(audio_bytes)
                    tmp = f.name
                subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", tmp])
        except Exception as exc:
            logger.warning("Audio playback failed: %s", exc)

    def session_stats(self) -> Optional[VoiceSessionStats]:
        return self._session_stats
