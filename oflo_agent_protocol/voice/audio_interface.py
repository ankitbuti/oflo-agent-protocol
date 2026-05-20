"""Custom audio interfaces for ElevenLabs Conversational AI.

Provides alternatives to DefaultAudioInterface for:
  • Byte-buffer audio (testing, recording, piping)
  • File-based input (pre-recorded audio files)
  • WebSocket audio (browser / remote clients)
  • Silent / null audio (integration testing)

All interfaces implement the ElevenLabs audio interface protocol:
  - start() / stop()
  - input() → generator of audio chunks
  - output(chunk: bytes)
"""
from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
from typing import Any, Callable, Generator, Iterator, Optional

logger = logging.getLogger(__name__)


class BufferedAudioInterface:
    """
    In-memory audio interface — reads input from a byte buffer
    and collects output audio into a buffer.

    Useful for:
      • Automated testing without hardware
      • Processing pre-recorded audio files
      • Piping audio from WebSocket or other source

    Usage::

        interface = BufferedAudioInterface()
        interface.feed_input(audio_bytes)          # feed PCM audio
        output = interface.get_output_bytes()      # get TTS output
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 4096,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._input_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._output_buffer = io.BytesIO()
        self._running = False

    def feed_input(self, audio_bytes: bytes) -> None:
        """Feed PCM audio bytes as microphone input."""
        for i in range(0, len(audio_bytes), self._chunk_size):
            self._input_queue.put(audio_bytes[i : i + self._chunk_size])

    def signal_end_of_input(self) -> None:
        """Signal that no more input will be fed."""
        self._input_queue.put(None)

    def get_output_bytes(self) -> bytes:
        """Get all captured TTS output audio."""
        return self._output_buffer.getvalue()

    def clear_output(self) -> None:
        self._output_buffer = io.BytesIO()

    # ElevenLabs audio interface protocol
    def start(self) -> None:
        self._running = True
        logger.debug("BufferedAudioInterface started")

    def stop(self) -> None:
        self._running = False
        self.signal_end_of_input()

    def input(self) -> Generator[bytes, None, None]:
        """Yield audio chunks from the input buffer."""
        while self._running:
            chunk = self._input_queue.get(timeout=30)
            if chunk is None:
                break
            yield chunk

    def output(self, chunk: bytes) -> None:
        """Receive TTS audio output."""
        self._output_buffer.write(chunk)


class WebSocketAudioInterface:
    """
    WebSocket audio interface — bridges browser/remote audio to ElevenLabs.

    Usage::

        interface = WebSocketAudioInterface()

        # In your WebSocket handler:
        async def on_audio(ws, data):
            interface.feed_audio(data)

        # ElevenLabs outputs arrive via callback:
        interface.on_output = lambda chunk: ws.send_bytes(chunk)
    """

    def __init__(
        self,
        on_output: Optional[Callable[[bytes], None]] = None,
        sample_rate: int = 16000,
    ) -> None:
        self._on_output = on_output
        self._sample_rate = sample_rate
        self._input_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=100)
        self._running = False

    def feed_audio(self, chunk: bytes) -> None:
        """Called by WebSocket handler when audio arrives from client."""
        if self._running and not self._input_queue.full():
            self._input_queue.put_nowait(chunk)

    def set_output_handler(self, handler: Callable[[bytes], None]) -> None:
        self._on_output = handler

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._input_queue.put(None)

    def input(self) -> Generator[bytes, None, None]:
        while self._running:
            try:
                chunk = self._input_queue.get(timeout=0.1)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                continue

    def output(self, chunk: bytes) -> None:
        if self._on_output:
            self._on_output(chunk)


class FileAudioInterface:
    """
    Play a pre-recorded audio file as microphone input.
    Collects TTS output to a file.

    Usage::

        interface = FileAudioInterface(
            input_path="question.wav",
            output_path="response.mp3",
        )
    """

    def __init__(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> None:
        self._input_path = input_path
        self._output_path = output_path
        self._output_fh = None

    def start(self) -> None:
        if self._output_path:
            self._output_fh = open(self._output_path, "wb")

    def stop(self) -> None:
        if self._output_fh:
            self._output_fh.close()
            self._output_fh = None

    def input(self) -> Generator[bytes, None, None]:
        if not self._input_path:
            return
        chunk_size = 4096
        with open(self._input_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def output(self, chunk: bytes) -> None:
        if self._output_fh:
            self._output_fh.write(chunk)


class NullAudioInterface:
    """
    Silent audio interface for integration testing.
    Consumes input silently and discards output.
    """

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def input(self) -> Generator[bytes, None, None]:
        # Yield 1 second of silence at 16kHz, 16-bit
        yield b"\x00" * 32000
        return

    def output(self, chunk: bytes) -> None:
        pass
