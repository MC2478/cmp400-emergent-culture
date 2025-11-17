# src/model/llm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    # If llama-cpp-python is not installed yet, I handle that
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore


@dataclass
class LLMConfig:
    model_path: str = "models/llama-3-8b-instruct-q4_k_m.gguf"
    n_ctx: int = 2048
    n_gpu_layers: int = 50
    temperature: float = 0.1
    max_tokens: int = 128


class LLMDecisionClient:
    def __init__(self, config: Optional[LLMConfig] = None, enabled: bool = False):
        self.config = config or LLMConfig()
        self.enabled = enabled
        self._llm: Optional[Llama] = None

        if self.enabled:
            if Llama is None:
                raise RuntimeError(
                    "llama-cpp-python is not installed but LLMDecisionClient is enabled."
                )
            # I load the local GGUF model once here.
            self._llm = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=False,
            )

    def compose_prompt(self, state: Dict[str, Any]) -> str:
        return (
            "You are the ruler of the territory 'East'.\n"
            "Decide ONE action based on the current state.\n"
            "Valid actions: 'gather', 'consume', 'wait'.\n"
            "Respond ONLY with JSON in this format:\n"
            '{"action": <action>, "target": "None", "reason": <short reason>}.\n\n'
            f"Current state: {state}\n"
            "JSON:"
        )

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self._llm is None:
            raise RuntimeError("LLMDecisionClient is disabled.")

        prompt = self.compose_prompt(state)

        output = self._llm(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=["\n\n"],
        )
        text = output["choices"][0]["text"].strip()

        import json

        try:
            decision = json.loads(text)
        except json.JSONDecodeError:
            # I try a basic cleanup before giving up.
            text_fixed = text.replace("'", '"')
            brace_idx = text_fixed.rfind("}")
            if brace_idx != -1:
                text_fixed = text_fixed[: brace_idx + 1]
            decision = json.loads(text_fixed)

        return {
            "action": decision.get("action", "wait"),
            "target": decision.get("target", "None"),
            "reason": decision.get("reason", "no reason provided"),
        }
