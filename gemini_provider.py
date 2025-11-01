"""
google_genai_provider.py

Unified Google Gen AI provider with PostHog analytics, supporting both
Gemini Developer API and Vertex AI (express mode) backends.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

from google import genai
from google.genai import types
from posthog import Posthog
from posthog.ai.gemini import Client as PosthogGeminiClient

from .classes import BaseLLMProvider, LLMResponse, ModelRegistry, RateLimiter, Usage
from .config import LLMConfig

logger = logging.getLogger(__name__)

Message = Dict[str, Any]


class ProviderError(Exception):
    """Raised when the Google Gen AI provider encounters an unrecoverable error."""


class ConfigurationError(ProviderError):
    """Raised when provider configuration is invalid."""


class GoogleGenAIProvider(BaseLLMProvider):
    """Google Gen AI provider (Gemini + Vertex) with PostHog instrumentation."""

    _BACKEND_PREFIXES = {"vertex": "vertex", "vertexai": "vertex", "googleai": "googleai", "google": "googleai"}
    _DEFAULT_BACKEND = "googleai"
    _IMAGEN_PREFIXES = ("imagen-",)

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        super().__init__()
        self.config = config or LLMConfig()
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.clients: Dict[str, PosthogGeminiClient] = {}
        self.posthog = self._init_posthog()
        self.supports_caching = True
        self._current_generation: Optional[Iterable[Any]] = None
        self.last_usage: Optional[Usage] = None
        self.last_response: Optional[str] = None
        self.last_citations: Optional[Any] = None

    # Public API

    async def generate_parallel(self, requests: List[List[Message]], model: str, **kwargs: Any) -> List[LLMResponse]:
        backend, model_id = self._resolve_backend(model)
        rate_limiter = self._get_rate_limiter(model)
        tasks = [
            self._generate_with_rate_limit(rate_limiter, req, model=model_id, backend=backend, **kwargs)
            for req in requests
        ]
        return await asyncio.gather(*tasks)

    async def generate_async(self, messages: List[Message], model: str, stream: bool = False, **kwargs: Any) -> Union[LLMResponse, Generator[str, None, None]]:
        return await self.generate(messages, model=model, stream=stream, **kwargs)

    async def generate(self, messages: List[Message], model: str, stream: bool = False, **kwargs: Any) -> Union[LLMResponse, Generator[str, None, None]]:
        backend, model_id = self._resolve_backend(model)
        client = self._get_or_create_client(backend)
        rate_limiter = self._get_rate_limiter(model)
        kwargs['_original_model_name'] = model

        if stream:
            return await self._stream_with_rate_limit(rate_limiter, messages, model=model_id, backend=backend, client=client, **kwargs)
        return await self._generate_with_rate_limit(rate_limiter, messages, model=model_id, backend=backend, client=client, **kwargs)

    def stop_generation(self) -> None:
        if self._current_generation and (close := getattr(self._current_generation, "close", None)):
            try:
                close()
            except Exception as exc:
                logger.warning("Error closing generation stream: %s", exc)
        self._current_generation = None

    # Core generation

    async def _generate_with_rate_limit(self, rate_limiter: RateLimiter, messages: List[Message], model: str, backend: Optional[str] = None, client: Optional[PosthogGeminiClient] = None, **kwargs: Any) -> LLMResponse:
        await rate_limiter.wait_for_capacity()
        backend = backend or self._resolve_backend(model)[0]
        client = client or self._get_or_create_client(backend)
        
        if self._is_imagen_model(model):
            return await self._generate_image(messages, model, backend, client, **kwargs)
        return await self._generate_text(messages, model, backend, client, **kwargs)

    async def _stream_with_rate_limit(self, rate_limiter: RateLimiter, messages: List[Message], model: str, backend: Optional[str] = None, client: Optional[PosthogGeminiClient] = None, **kwargs: Any) -> Generator[str, None, None]:
        await rate_limiter.wait_for_capacity()
        backend = backend or self._resolve_backend(model)[0]
        client = client or self._get_or_create_client(backend)
        return self._stream_text(messages, model, backend, client, **kwargs)

    async def _generate_text(self, messages: List[Message], model: str, backend: str, client: PosthogGeminiClient, **kwargs: Any) -> LLMResponse:
        original_model_name = kwargs.pop("_original_model_name", None)
        contents, system_instruction, tool_state = self._convert_messages(messages)
        config, posthog_kwargs, request_kwargs = self._build_request_kwargs(kwargs, system_instruction, tool_state)

        start_time = time.perf_counter()
        response = await asyncio.to_thread(client.models.generate_content, model=model, contents=contents, config=config, **posthog_kwargs, **request_kwargs)
        latency = time.perf_counter() - start_time

        self._raise_on_safety_block(response)
        text, audio_data, citations = self._extract_content(response)
        usage = self._extract_usage(response)

        self.last_response, self.last_usage, self.last_citations = text, usage, citations

        return LLMResponse(
            content=text,
            model_name=original_model_name or model,
            usage=usage,
            latency=latency,
            audio_data=audio_data,
            citations=citations,
        )

    def _stream_text(self, messages: List[Message], model: str, backend: str, client: PosthogGeminiClient, **kwargs: Any) -> Generator[str, None, None]:
        kwargs.pop("_original_model_name", None)
        contents, system_instruction, tool_state = self._convert_messages(messages)
        config, posthog_kwargs, request_kwargs = self._build_request_kwargs(kwargs, system_instruction, tool_state)

        stream = client.models.generate_content_stream(model=model, contents=contents, config=config, **posthog_kwargs, **request_kwargs)
        self._current_generation = stream

        aggregated_text, last_usage, citations = [], None, None

        try:
            for chunk in stream:
                if chunk.text:
                    aggregated_text.append(chunk.text)
                    yield chunk.text
                if chunk_usage := self._extract_usage(chunk, default=None):
                    last_usage = chunk_usage
                citations = citations or self._extract_citations(chunk)
        finally:
            self._current_generation = None

        self.last_response = "".join(aggregated_text) if aggregated_text else None
        self.last_usage = last_usage
        self.last_citations = citations

    async def _generate_image(self, messages: List[Message], model: str, backend: str, client: PosthogGeminiClient, **kwargs: Any) -> LLMResponse:
        original_model_name = kwargs.pop("_original_model_name", None)
        if not (prompt := self._extract_prompt_for_image(messages, kwargs)):
            raise ProviderError("Unable to resolve prompt for image generation.")

        config = self._build_image_config(kwargs)
        start_time = time.perf_counter()
        response = await asyncio.to_thread(client.models.generate_images, model=model, prompt=prompt, config=config)
        latency = time.perf_counter() - start_time

        if not response.generated_images:
            raise ProviderError("Image generation returned no results.")

        image_bytes = response.generated_images[0].image.image_bytes
        content_b64 = base64.b64encode(image_bytes).decode("utf-8")
        usage = self._extract_usage(response)

        return LLMResponse(content=content_b64, model_name=original_model_name or model, usage=usage, latency=latency)

    # Message conversion

    def _convert_messages(self, messages: List[Message]) -> Tuple[List[types.Content], Optional[Union[str, types.Content]], Dict[str, Any]]:
        contents, system_texts, tool_names_by_id = [], [], {}

        for message in messages:
            role = message.get("role")
            if role == "system":
                if extracted := self._render_text_parts(message.get("content")):
                    system_texts.append(extracted)
            elif role == "user":
                if parts := self._convert_content_parts(message.get("content")):
                    contents.append(types.Content(role="user", parts=parts))
            elif role in {"assistant", "model"}:
                parts = list(self._convert_content_parts(message.get("content")))
                for tool_call in message.get("tool_calls", []):
                    name = tool_call.get("function", {}).get("name") or tool_call.get("id")
                    args = self._safe_json_load(tool_call.get("function", {}).get("arguments", "{}"))
                    parts.append(types.Part.from_function_call(name=name, args=args))
                    if tool_call.get("id"):
                        tool_names_by_id[tool_call["id"]] = name
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                tool_name = tool_names_by_id.get(message.get("tool_call_id")) or message.get("name") or "tool_response"
                response_payload = self._parse_tool_response(message.get("content"))
                contents.append(types.Content(role="tool", parts=[types.Part.from_function_response(name=tool_name, response=response_payload)]))
            else:
                if fallback_parts := self._convert_content_parts(message.get("content")):
                    contents.append(types.Content(role=role, parts=fallback_parts))

        return contents, "\n\n".join(system_texts) if system_texts else None, {"tool_names_by_id": tool_names_by_id}

    def _convert_content_parts(self, content: Any) -> List[types.Part]:
        if content is None:
            return []
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]
        if isinstance(content, list):
            parts = []
            for item in content:
                item_type = item.get("type")
                if item_type in {"text", "input_text"}:
                    parts.append(types.Part.from_text(text=item.get("text", "")))
                elif item_type in {"image_url", "input_image"}:
                    if data_url := (item.get("image_url") or item.get("input_image") or {}).get("url") or (item.get("image_url") or item.get("input_image") or {}).get("data"):
                        parts.append(self._part_from_image_reference(data_url))
                elif item_type in {"input_audio", "audio"}:
                    audio_info = item.get("input_audio") or item.get("audio") or {}
                    if data := audio_info.get("data"):
                        mime = audio_info.get("mime_type") or audio_info.get("format") or "audio/wav"
                        parts.append(self._part_from_base64_blob(data, mime))
                elif item_type == "file_path":
                    if file_url := item.get("file_path"):
                        parts.append(types.Part.from_uri(file_uri=file_url, mime_type="application/octet-stream"))
            return parts
        if isinstance(content, dict) and "text" in content:
            return [types.Part.from_text(text=content["text"])]
        return [types.Part.from_text(text=str(content))]

    def _part_from_image_reference(self, reference: str) -> types.Part:
        if reference.startswith("data:"):
            mime, _, payload = reference.partition(",")
            mime = mime.split(";")[0][5:] or "image/png"
            return self._part_from_base64_blob(payload, mime)
        mime, _ = mimetypes.guess_type(reference)
        return types.Part.from_uri(file_uri=reference, mime_type=mime or "image/*")

    def _part_from_base64_blob(self, data_b64: str, mime_type: str) -> types.Part:
        try:
            raw = base64.b64decode(data_b64)
        except Exception as exc:
            raise ProviderError(f"Invalid base64 payload: {exc}") from exc
        return types.Part.from_bytes(data=raw, mime_type=mime_type)

    def _render_text_parts(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)

    def _parse_tool_response(self, content: Any) -> Any:
        if isinstance(content, str):
            return parsed if (parsed := self._safe_json_load(content, default=None)) is not None else {"output": content}
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
            combined = "\n".join(filter(None, texts))
            return parsed if (parsed := self._safe_json_load(combined, default=None)) is not None else {"output": combined}
        return content

    # Request building

    def _build_request_kwargs(self, kwargs: Dict[str, Any], system_instruction: Optional[Union[str, types.Content]], tool_state: Dict[str, Any]) -> Tuple[types.GenerateContentConfig, Dict[str, Any], Dict[str, Any]]:
        kwargs = dict(kwargs)
        config_kwargs, posthog_kwargs, request_kwargs = {}, {}, {}

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Extract config parameters
        for key in ["temperature", "top_p", "top_k", "max_output_tokens", "presence_penalty", "frequency_penalty", "candidate_count", "seed"]:
            if key in kwargs:
                config_kwargs[key] = kwargs.pop(key)

        # Handle thinking configuration
        effort_map = {"none": 0, "low": 512, "medium": 2048, "high": 8192}
        reasoning = kwargs.pop("reasoning", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        
        if isinstance(reasoning, dict) and "thinking_budget" in reasoning:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=reasoning["thinking_budget"])
        elif reasoning_effort:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=effort_map.get(reasoning_effort.lower()))
        elif thinking_config := kwargs.pop("thinking_config", None):
            config_kwargs["thinking_config"] = thinking_config

        if "stop" in kwargs:
            stop_sequences = kwargs.pop("stop")
            config_kwargs["stop_sequences"] = [stop_sequences] if isinstance(stop_sequences, str) else stop_sequences

        if response_format := kwargs.pop("response_format", None):
            if response_format.get("type") == "json_object":
                config_kwargs["response_mime_type"] = "application/json"
            elif response_format.get("type") == "json_schema":
                if schema := response_format.get("json_schema", {}).get("schema"):
                    config_kwargs["response_schema"] = schema

        if modalities := kwargs.pop("modality", None):
            config_kwargs["response_modalities"] = [modalities] if isinstance(modalities, str) else modalities

        if audio_params := kwargs.pop("audio", None):
            if speech_config := self._build_speech_config(audio_params):
                config_kwargs["speech_config"] = speech_config

        if tools := kwargs.pop("tools", None):
            config_kwargs["tools"] = self._convert_tools(tools)

        if tool_choice := kwargs.pop("tool_choice", None):
            config_kwargs["tool_config"] = self._build_tool_config(tool_choice)

        if safety_settings := kwargs.pop("safety_settings", None):
            config_kwargs["safety_settings"] = safety_settings

        # Extract PostHog parameters
        for key in list(kwargs.keys()):
            if key.startswith("posthog_"):
                posthog_kwargs[key] = kwargs.pop(key)

        if posthog_dict := kwargs.pop("posthog", None):
            if isinstance(posthog_dict, dict):
                for key, value in posthog_dict.items():
                    posthog_kwargs[f"posthog_{key}"] = value

        request_kwargs.update(kwargs)
        return types.GenerateContentConfig(**config_kwargs), posthog_kwargs, request_kwargs

    def _build_tool_config(self, tool_choice: Any) -> Optional[types.ToolConfig]:
        if isinstance(tool_choice, str):
            choice = tool_choice.lower()
            if choice == "none":
                return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))
            if choice == "auto":
                return None
        if isinstance(tool_choice, dict):
            mode = tool_choice.get("type")
            if mode == "function":
                if name := tool_choice.get("function", {}).get("name"):
                    return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=[name]))
            if mode == "none":
                return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))
            if mode == "auto":
                return None
        return None

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[types.Tool]:
        declarations = [
            types.FunctionDeclaration(name=fn.get("name"), description=fn.get("description"), parameters=fn.get("parameters"))
            for tool in tools if tool.get("type") == "function" and (fn := tool.get("function") or {})
        ]
        return [types.Tool(function_declarations=declarations)] if declarations else []

    def _build_speech_config(self, audio_params: Any) -> Optional[types.SpeechConfig]:
        if isinstance(audio_params, types.SpeechConfig):
            return audio_params
        if isinstance(audio_params, dict):
            voice_name = audio_params.pop("voice_name", None)
            language_code = audio_params.get("language_code")
            try:
                if voice_name:
                    voice_config = types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name))
                    return types.SpeechConfig(voice_config=voice_config, language_code=language_code, **audio_params)
                return types.SpeechConfig(**audio_params)
            except Exception as exc:
                logger.warning("Invalid speech configuration: %s", exc)
        return None

    def _build_image_config(self, kwargs: Dict[str, Any]) -> Optional[types.GenerateImagesConfig]:
        known_keys = {"image_size", "aspect_ratio", "number_of_images", "guidance_scale", "negative_prompt", "seed", "safety_filter_level"}
        config_params = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in known_keys}
        return types.GenerateImagesConfig(**config_params) if config_params else None

    # Content extraction

    def _extract_content(self, response: Any) -> Tuple[str, Optional[str], Optional[Any]]:
        text_segments, audio_b64 = [], None
        citations = self._extract_citations(response)

        for candidate in getattr(response, "candidates", []):
            if not (content := getattr(candidate, "content", None)):
                continue

            parts_iterable = []
            if hasattr(content, "parts"):
                parts_iterable = getattr(content, "parts") or []
            elif isinstance(content, list):
                for item in content:
                    if hasattr(item, "parts") and getattr(item, "parts"):
                        parts_iterable.extend(getattr(item, "parts"))
                    elif isinstance(item, dict):
                        parts_iterable.extend(item.get("parts", []))
            elif isinstance(content, dict):
                parts_iterable = content.get("parts", [])

            for part in parts_iterable:
                if getattr(part, "text", None):
                    text_segments.append(part.text)
                if (inline := getattr(part, "inline_data", None)) and inline.mime_type.startswith("audio/"):
                    audio_b64 = base64.b64encode(inline.data).decode("utf-8")

        if hasattr(response, "text") and response.text:
            text_segments.append(response.text)

        return "\n".join(filter(None, text_segments)), audio_b64, citations

    def _extract_citations(self, response: Any) -> Optional[Any]:
        for candidate in getattr(response, "candidates", []):
            if metadata := getattr(candidate, "citation_metadata", None):
                return metadata
        return None

    def _extract_usage(self, response: Any, default: Optional[Usage] = None) -> Usage:
        if not (usage_meta := getattr(response, "usage_metadata", None)):
            return default or Usage(input_tokens=0, output_tokens=0, cached_tokens=0)
        return Usage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            cached_tokens=getattr(usage_meta, "cached_content_token_count", 0) or 0,
        )

    def _extract_prompt_for_image(self, messages: List[Message], kwargs: Dict[str, Any]) -> Optional[str]:
        if prompt := kwargs.pop("prompt", None):
            return prompt
        for message in reversed(messages):
            if message.get("role") == "user":
                if isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(message.get("content"), list):
                    text_parts = [item.get("text") for item in message["content"] if isinstance(item, dict) and item.get("type") in {"text", "input_text"}]
                    if text_parts:
                        return "\n".join(text_parts)
        return None

    # Backend & clients

    def _resolve_backend(self, model: str) -> Tuple[str, str]:
        if match := re.match(r"^(?P<prefix>[a-zA-Z]+)[/:.](?P<model>.+)$", model):
            prefix = match.group("prefix").lower()
            if not (normalized := self._BACKEND_PREFIXES.get(prefix)):
                raise ConfigurationError(f"Unknown backend prefix '{prefix}' in model '{model}'.")
            return normalized, match.group("model")
        return self._DEFAULT_BACKEND, model

    def _get_or_create_client(self, backend: str) -> PosthogGeminiClient:
        if backend in self.clients:
            return self.clients[backend]

        client_kwargs: Dict[str, Any] = {}
        if backend == "vertex":
            if not (api_key := getattr(self.config, "vertex_api_key", None) or os.getenv("VERTEX_API_KEY")):
                raise ConfigurationError("Missing Vertex API key.")
            client_kwargs.update({"vertexai": True, "api_key": api_key})
            if project := getattr(self.config, "vertex_project", None) or os.getenv("GOOGLE_CLOUD_PROJECT"):
                client_kwargs["project"] = project
            if location := getattr(self.config, "vertex_location", None) or os.getenv("GOOGLE_CLOUD_LOCATION"):
                client_kwargs["location"] = location
        else:
            if not (api_key := getattr(self.config, "gemini_api_key", None) or os.getenv("GOOGLE_API_KEY")):
                raise ConfigurationError("Missing Gemini Developer API key.")
            client_kwargs["api_key"] = api_key

        if http_options := getattr(self.config, "gemini_http_options", None):
            client_kwargs["http_options"] = http_options

        client = PosthogGeminiClient(posthog_client=self.posthog, **client_kwargs)
        self.clients[backend] = client
        return client

    def _init_posthog(self) -> Posthog:
        return Posthog(
            project_api_key=os.getenv("POSTHOG_API_KEY", "phc_1uBDKATKfxK7ougGiL9F9hnCgeXJvc4k6TMP2oekfnK"),
            host=os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
        )

    def _get_rate_limiter(self, model_name: str) -> RateLimiter:
        if model_name not in self.rate_limiters:
            model_config = ModelRegistry.get_config(model_name)
            self.rate_limiters[model_name] = RateLimiter(model_config=model_config, model_name=model_name)
        return self.rate_limiters[model_name]

    # Helpers

    def _is_imagen_model(self, model: str) -> bool:
        return any(model.startswith(prefix) for prefix in self._IMAGEN_PREFIXES)

    def _raise_on_safety_block(self, response: Any) -> None:
        if (feedback := getattr(response, "prompt_feedback", None)) and getattr(feedback, "block_reason", None):
            raise ProviderError(f"Request blocked by safety filters: {feedback.block_reason}")

    @staticmethod
    def _safe_json_load(payload: Union[str, bytes, None], default: Optional[Any] = None) -> Any:
        if not payload:
            return default
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="ignore")
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return default
