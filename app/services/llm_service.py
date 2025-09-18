import httpx
import json
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class LMStudioService:
    def __init__(self):
        self.base_url = f"{settings.lm_studio_url}/v1"
        self.model_name = settings.lm_studio_model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def check_server_status(self) -> bool:
        """Check if LM Studio server is running"""
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LM Studio server check failed: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_response(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            max_tokens: int = 1500
    ) -> Dict[str, Any]:
        """Generate a response via the LM Studio API"""

        if not await self.check_server_status():
            raise Exception("LM Studio server is not running! Please start it in LM Studio app.")

        # Konwertuj messages na pojedynczy prompt
        prompt = self._messages_to_prompt(messages)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            logger.info(f"Sending request to LM Studio: prompt length {len(prompt)}")
            response = await self.client.post(
                f"{self.base_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()

            return {
                "content": result["choices"][0]["text"].strip(),
                "usage": result.get("usage", {}),
                "model": result.get("model", self.model_name)
            }

        except httpx.HTTPError as e:
            logger.error(f"LM Studio API HTTP error: {e}")
            raise Exception(f"LM Studio API error: {e}")
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise Exception(f"Unexpected API response format: {e}")

    async def generate_streaming_response(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            max_tokens: int = 1500
    ) -> AsyncGenerator[str, None]:
        """Generator for streaming responses"""

        if not await self.check_server_status():
            yield "ERROR: LM Studio server is not running!"
            return

        # Konwertuj messages na prompt
        prompt = self._messages_to_prompt(messages)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        try:
            async with self.client.stream(
                    "POST",
                    f"{self.base_url}/completions",
                    json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0].get("text", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"ERROR: {str(e)}"

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Konwertuje messages na pojedynczy prompt"""
        prompt_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt_parts.append(f"System: {content}\n\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n\n")

        prompt_parts.append("Assistant:")
        return "".join(prompt_parts)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()