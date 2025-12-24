"""Text-to-speech functionality for greeting recognized persons."""
from __future__ import annotations

import pyttsx3


class Speaker:
    """Text-to-speech speaker for greeting recognized persons."""

    def __init__(self) -> None:
        """Initialize the TTS engine."""
        self._engine = pyttsx3.init()

    def say_hello(self, name: str) -> None:
        """Speak a greeting to a person.

        Args:
            name: Name of the person to greet.
        """
        self._engine.say("Hello")
        self._engine.say(str(name))
        self._engine.runAndWait()
