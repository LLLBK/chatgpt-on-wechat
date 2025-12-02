"""
Google Gemini bot

@author zhayujie
@Date 2023/12/15
"""
# encoding:utf-8

import base64
import mimetypes
import os
import uuid
from typing import List, Optional, Tuple

import httpx
from bot.bot import Bot
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.session_manager import SessionManager
from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from common.expired_dict import ExpiredDict
from common.log import logger
from common.safety import sensitive_filter
from config import conf, get_appdata_dir
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from google.genai.types import HarmBlockThreshold, HarmCategory


class GoogleGeminiBot(Bot):
    DEFAULT_TEXT_MODEL = "gemini-3-pro-preview"
    DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
    DEFAULT_GUARD_MODEL = "gemini-2.5-flash"

    def __init__(self):
        super().__init__()
        self.api_key = conf().get("gemini_api_key")
        http_options = self._build_http_options()
        self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        self.sessions = SessionManager(ChatGPTSession, model=conf().get("model") or self.DEFAULT_TEXT_MODEL)
        self.model = self._normalize_model(conf().get("model"), self.DEFAULT_TEXT_MODEL)
        self.fallback_model = self._normalize_model(conf().get("gemini_fallback_model"), self.DEFAULT_TEXT_MODEL)
        image_model_conf = conf().get("gemini_image_model") or conf().get("text_to_image")
        self.image_model = self._normalize_model(image_model_conf, self.DEFAULT_IMAGE_MODEL)
        self.fallback_image_model = self._normalize_model(conf().get("gemini_image_fallback_model"), self.DEFAULT_IMAGE_MODEL)
        self.proxy = (conf().get("proxy") or "").strip()
        self.attachment_ttl = conf().get("gemini_attachment_cache_seconds", 900)
        self.attachment_max_size = conf().get("gemini_attachment_max_size", 10 * 1024 * 1024)
        self.pending_attachments = ExpiredDict(self.attachment_ttl) if self.attachment_ttl else {}
        self.google_search_enabled = conf().get("gemini_enable_google_search", False)
        search_tool_conf = (conf().get("gemini_google_search_tool") or "auto").strip().lower()
        if search_tool_conf == "google_search_retrieval":
            search_tool_conf = "google_search"
        if search_tool_conf not in ("auto", "google_search"):
            search_tool_conf = "auto"
        if search_tool_conf == "auto":
            search_tool_conf = "google_search"
        self.google_search_tool = search_tool_conf
        mode_conf = (conf().get("gemini_google_search_mode") or "MODE_DYNAMIC").upper()
        self.google_search_mode = mode_conf
        try:
            self.google_search_threshold = float(conf().get("gemini_google_search_dynamic_threshold", 0.7))
        except (TypeError, ValueError):
            self.google_search_threshold = 0.7
        self.google_search_show_citations = conf().get("gemini_google_search_show_citations", True)
        self.google_search_tools = self._build_google_search_tools()
        if self.google_search_tools:
            logger.info(f"[Gemini] Google Search tool enabled: {self.google_search_tool}")
        self.china_guard_enabled = conf().get("china_politics_guard_enabled", True)
        self.china_guard_model = self._normalize_model(conf().get("china_politics_guard_model"), self.DEFAULT_GUARD_MODEL)
        self.china_guard_prompt = conf().get("china_politics_guard_prompt") or (
            "You are a strict content safety classifier. Determine whether the following user request discusses contemporary Chinese politics, including current government, political figures, and political events in modern China. Answer ONLY with YES or NO.\n\nUser request:\n\"\"\"{query}\"\"\"\n\nDoes this request involve contemporary Chinese politics?"
        )

    def reply(self, query, context: Context = None) -> Reply:
        session_id = context["session_id"]
        normalized_query = (query or "").strip()
        try:
            if self._should_clear(normalized_query):
                self.sessions.clear_session(session_id)
                self._clear_attachments(session_id)
                return Reply(ReplyType.INFO, "已清空对话，开始新的会话。")

            if context.type in [ContextType.IMAGE, ContextType.FILE, ContextType.VOICE, ContextType.VIDEO]:
                stored = self._cache_attachments(session_id, context.get("attachments") or [])
                if not stored:
                    return Reply(ReplyType.ERROR, "未收到可用的附件，请重试。")
                return Reply(ReplyType.INFO, f"已收到附件 {len(stored)} 个，请继续输入你的问题。")

            if context.type == ContextType.IMAGE_CREATE or self._is_image_request(normalized_query):
                attachments = self._pop_attachments(session_id)
                return self._handle_image_generation(normalized_query, attachments, session_id)

            if context.type != ContextType.TEXT:
                logger.warning(f"[Gemini] Unsupported message type, type={context.type}")
                return Reply(ReplyType.ERROR, "当前消息类型暂不支持")

            logger.info(f"[Gemini] query={normalized_query}")
            if self.china_guard_enabled and self._violates_china_politics_policy(normalized_query):
                self._clear_attachments(session_id)
                return Reply(ReplyType.ERROR, "抱歉，该请求无法处理")

            attachments = self._pop_attachments(session_id)
            session = self.sessions.session_query(normalized_query, session_id)
            gemini_messages, system_instruction = self._convert_to_gemini_messages(self.filter_messages(session.messages), attachments)
            logger.debug(f"[Gemini] messages={gemini_messages}")
            generation_config = self._build_generation_config(system_instruction=system_instruction)
            logger.info(
                f"[Gemini] generation config: tools={generation_config.tools}, "
                f"auto_fc={generation_config.automatic_function_calling}, tool_config={getattr(generation_config, 'tool_config', None)}"
            )
            response = self._generate_with_tools(self.model, gemini_messages, generation_config)
            if self.google_search_enabled:
                # 仅记录是否返回了grounding信息，供排查是否触发了搜索工具
                self._has_grounding(response)
                self._log_tool_invocations(response)

            reply_text = self._extract_text_response(response)
            if reply_text:
                reply_text, citation_items = self._append_grounding_citations(response, reply_text)
                ok, filtered_text = sensitive_filter.filter(reply_text)
                if not ok:
                    reply_text = filtered_text
                logger.info(f"[Gemini] reply={reply_text}")
                self.sessions.session_reply(reply_text, session_id)
                reply_obj = Reply(ReplyType.TEXT, reply_text)
                if citation_items:
                    references_text = self._format_citation_message(citation_items)
                    if references_text:
                        reply_obj.add_extra_text(references_text)
                return reply_obj

            logger.warning("[Gemini] No valid response generated. Checking safety ratings.")
            if hasattr(response, "candidates") and response.candidates:
                for rating in response.candidates[0].safety_ratings:
                    logger.warning(f"Safety rating: {rating.category} - {rating.probability}")
            error_message = "No valid response generated due to safety constraints."
            self.sessions.session_reply(error_message, session_id)
            return Reply(ReplyType.ERROR, error_message)
        except Exception as e:
            logger.error(f"[Gemini] Error generating response: {str(e)}", exc_info=True)
            error_message = "Failed to invoke [Gemini] api!"
            self.sessions.session_reply(error_message, session_id)
            return Reply(ReplyType.ERROR, error_message)

    def _generate_with_tools(self, model_name, messages, generation_config):
        try:
            return self.client.models.generate_content(
                model=model_name,
                contents=messages,
                config=generation_config,
            )
        except (genai_errors.ClientError, ValueError) as e:
            if self._is_model_not_found(e) and self.fallback_model and self.fallback_model != model_name:
                logger.warning(f"[Gemini] model {model_name} unavailable, fallback to {self.fallback_model}")
                return self.client.models.generate_content(
                    model=self.fallback_model,
                    contents=messages,
                    config=generation_config,
                )
            raise

    def _build_generation_config(self, include_tools: bool = True, response_mime_type: str = None, system_instruction: str = None, enable_thinking: bool = True, force_tool_call: bool = False):
        config = genai_types.GenerateContentConfig(
            temperature=conf().get("temperature"),
            top_p=conf().get("top_p"),
            presence_penalty=conf().get("presence_penalty"),
            frequency_penalty=conf().get("frequency_penalty"),
            max_output_tokens=conf().get("gemini_max_output_tokens"),
        )
        config.safety_settings = self._build_safety_settings()
        if response_mime_type:
            config.response_mime_type = response_mime_type
        if system_instruction:
            config.system_instruction = system_instruction
        thinking_level = (conf().get("gemini_thinking_level") or "").strip().upper()
        if enable_thinking and thinking_level:
            try:
                level_enum = genai_types.ThinkingLevel[thinking_level]
                config.thinking_config = genai_types.ThinkingConfig(thinking_level=level_enum)
            except KeyError:
                logger.warning(f"[Gemini] Unsupported thinking level: {thinking_level}")
        max_tool_calls = conf().get("gemini_max_tool_calls")
        if include_tools:
            try:
                max_calls = int(max_tool_calls) if max_tool_calls is not None else None
            except (TypeError, ValueError):
                max_calls = None
            if max_calls and max_calls > 0:
                config.automatic_function_calling = genai_types.AutomaticFunctionCallingConfig(
                    disable=False,
                    maximum_remote_calls=max_calls,
                )
            else:
                config.automatic_function_calling = genai_types.AutomaticFunctionCallingConfig(disable=False)
            if force_tool_call or self.google_search_enabled:
                try:
                    config.tool_config = genai_types.ToolConfig(
                        function_calling_config=genai_types.FunctionCallingConfig(
                            mode=genai_types.FunctionCallingConfigMode.ANY,
                        )
                    )
                except Exception as e:
                    logger.debug(f"[Gemini] tool config setup failed: {e}")
        if include_tools and self.google_search_tools:
            config.tools = self.google_search_tools
        return config

    def _build_safety_settings(self):
        return [
            genai_types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            genai_types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            genai_types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            genai_types.SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    def _convert_to_gemini_messages(self, messages: list, attachments: list = None) -> Tuple[List[genai_types.Content], Optional[str]]:
        attachments = attachments or []
        res: List[genai_types.Content] = []
        system_instruction = None
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "system":
                system_instruction = msg.get("content") or system_instruction
                continue
            if role == "assistant":
                mapped_role = "model"
            elif role == "user":
                mapped_role = "user"
            else:
                continue
            parts: List[genai_types.Part] = [genai_types.Part.from_text(text=msg.get("content") or "")]
            if idx == len(messages) - 1 and mapped_role == "user" and attachments:
                attachment_parts = []
                for att in attachments:
                    part = self._attachment_to_part(att)
                    if part:
                        attachment_parts.append(part)
                if attachment_parts:
                    parts = attachment_parts + parts
            res.append(genai_types.Content(role=mapped_role, parts=parts))
        return res, system_instruction

    def _build_google_search_tools(self):
        if not self.google_search_enabled:
            return None
        # SDK unified google_search tool; keep default config and let the model decide to call.
        return [genai_types.Tool(google_search=genai_types.GoogleSearch())]

    def _has_grounding(self, response) -> bool:
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return False
            metadata = self._maybe_get(candidates[0], "grounding_metadata", "groundingMetadata")
            if not metadata:
                return False
            supports = self._maybe_get(metadata, "grounding_supports", "groundingSupports") or []
            chunks = self._maybe_get(metadata, "grounding_chunks", "groundingChunks") or []
            has_grounding = bool(supports) and bool(chunks)
            if has_grounding:
                logger.info(f"[Gemini] Grounding found: supports={len(supports)}, chunks={len(chunks)}")
            else:
                logger.info("[Gemini] No grounding metadata in response.")
            return has_grounding
        except Exception as e:
            logger.debug(f"[Gemini] grounding detection failed: {e}")
            return False

    def _log_tool_invocations(self, response):
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                logger.info("[Gemini] tool call check: no candidates")
                return
            content = getattr(candidates[0], "content", None)
            if not content or not getattr(content, "parts", None):
                logger.info("[Gemini] tool call check: no content parts")
                return
            tool_calls = []
            for part in content.parts:
                fc = getattr(part, "function_call", None) or (part.get("functionCall") if isinstance(part, dict) else None)
                if fc:
                    name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
                    tool_calls.append(name or "unknown")
            if tool_calls:
                logger.info(f"[Gemini] tool calls in response: {tool_calls}")
            else:
                logger.info("[Gemini] tool call check: no function_call parts")
        except Exception as e:
            logger.debug(f"[Gemini] tool invocation logging failed: {e}")

    def _append_grounding_citations(self, response, text: str) -> Tuple[str, List[dict]]:
        if not text or not self.google_search_enabled or not self.google_search_show_citations:
            return text, []
        try:
            candidates = getattr(response, "candidates", None)
            if not candidates:
                return text, []
            candidate = candidates[0]
            metadata = self._maybe_get(candidate, "grounding_metadata", "groundingMetadata")
            if not metadata:
                return text, []
            supports = self._maybe_get(metadata, "grounding_supports", "groundingSupports") or []
            chunks = self._maybe_get(metadata, "grounding_chunks", "groundingChunks") or []
            if not supports or not chunks:
                return text, []
            sorted_supports = sorted(
                list(supports),
                key=lambda s: self._maybe_get(self._maybe_get(s, "segment"), "end_index", "endIndex") or 0,
                reverse=True,
            )
            text_with_citations = text
            chunks_list = list(chunks)
            chunk_entries = self._build_citation_entries(chunks_list)
            if not chunk_entries:
                return text, []
            ordered_citations: List[dict] = []

            def ensure_entry(idx: int):
                entry = chunk_entries.get(idx)
                if not entry:
                    return None
                if "number" not in entry:
                    entry["number"] = len(ordered_citations) + 1
                    ordered_citations.append(entry)
                return entry

            for support in sorted_supports:
                segment = self._maybe_get(support, "segment")
                end_index = self._maybe_get(segment, "end_index", "endIndex")
                if end_index is None:
                    continue
                chunk_indices = self._maybe_get(support, "grounding_chunk_indices", "groundingChunkIndices") or []
                citation_links = []
                for idx in chunk_indices:
                    if not isinstance(idx, int) or idx < 0 or idx >= len(chunks_list):
                        continue
                    entry = ensure_entry(idx)
                    if entry:
                        citation_links.append(f"[{entry['number']}]")
                if not citation_links:
                    continue
                end_index = max(0, min(int(end_index), len(text_with_citations)))
                citation_string = ", ".join(citation_links)
                text_with_citations = (
                    text_with_citations[:end_index] + citation_string + text_with_citations[end_index:]
                )
            return text_with_citations, ordered_citations
        except Exception as e:
            logger.warning(f"[Gemini] append citations failed: {e}")
            return text, []

    def _build_citation_entries(self, chunks_list: List) -> dict:
        entries = {}
        for idx, chunk in enumerate(chunks_list):
            retrieved = self._maybe_get(chunk, "retrieved_context", "retrievedContext") or {}
            web_info = self._maybe_get(chunk, "web") or {}
            url = self._maybe_get(retrieved, "uri") or self._maybe_get(web_info, "uri")
            title = (
                self._maybe_get(retrieved, "title")
                or self._maybe_get(web_info, "title")
                or self._maybe_get(web_info, "domain")
            )
            resolved_url = self._resolve_citation_url(url)
            if not resolved_url:
                continue
            entries[idx] = {
                "url": resolved_url,
                "title": title or resolved_url,
            }
        return entries

    def _format_citation_message(self, citations: List[dict]) -> Optional[str]:
        if not citations:
            return None
        lines = ["参考资料："]
        for entry in citations:
            url = entry.get("url")
            title = entry.get("title") or url
            number = entry.get("number")
            if not url or not number:
                continue
            lines.append(f"{number}. [{title}]({url})")
        if len(lines) == 1:
            return None
        return "\n".join(lines)

    def _resolve_citation_url(self, url: Optional[str]) -> Optional[str]:
        # 官方文档说明 redirect 链接无法本地解析，这里直接返回原始 redirect 链接。
        return url

    @staticmethod
    def _maybe_get(obj, *keys):
        if obj is None:
            return None
        for key in keys:
            if not key:
                continue
            if isinstance(obj, dict):
                value = obj.get(key)
            else:
                value = getattr(obj, key, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def filter_messages(messages: list):
        res = []
        turn = "user"
        if not messages:
            return res
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            role = message.get("role")
            if role == "system":
                res.insert(0, message)
                continue
            if role != turn:
                continue
            res.insert(0, message)
            if turn == "user":
                turn = "assistant"
            elif turn == "assistant":
                turn = "user"
        return res

    def _attachment_to_part(self, att: dict):
        path = att.get("path")
        if not path or not os.path.exists(path):
            return None
        mime_type = att.get("mime_type") or mimetypes.guess_type(path)[0] or "application/octet-stream"
        try:
            with open(path, "rb") as f:
                data = f.read()
            return genai_types.Part.from_bytes(data=data, mime_type=mime_type)
        except Exception as e:
            logger.warning(f"[Gemini] read attachment failed: {e}")
            return None

    def _cache_attachments(self, session_id, attachments: list):
        if not attachments:
            return []
        cached = []
        holder = self.pending_attachments
        existing = holder.get(session_id, [])
        for att in attachments:
            path = att.get("path")
            if not path or not os.path.exists(path):
                continue
            try:
                size = os.path.getsize(path)
                if self.attachment_max_size and size > self.attachment_max_size:
                    logger.warning(f"[Gemini] attachment too large, skip: {path}")
                    continue
                att_copy = {
                    "path": path,
                    "mime_type": att.get("mime_type"),
                    "file_name": att.get("file_name"),
                    "type": att.get("type"),
                    "size": size,
                }
                cached.append(att_copy)
            except Exception as e:
                logger.warning(f"[Gemini] cache attachment failed: {e}")
                continue
        if cached:
            holder[session_id] = existing + cached
        return cached

    def _pop_attachments(self, session_id):
        holder = self.pending_attachments
        attachments = holder.get(session_id, []) or []
        if session_id in holder:
            del holder[session_id]
        return attachments

    def _clear_attachments(self, session_id):
        if session_id in self.pending_attachments:
            del self.pending_attachments[session_id]

    def _should_clear(self, query: str) -> bool:
        clear_commands = [c.strip() for c in conf().get("clear_memory_commands", []) if c]
        return query in clear_commands

    def _is_image_request(self, query: str) -> bool:
        if not query:
            return False
        keywords = ["生成图片", "画", "出一张图", "生成一张图", "画个", "绘制", "draw", "image", "picture", "photo", "sketch"]
        return any(kw in query.lower() if kw.isascii() else kw in query for kw in keywords)

    def _handle_image_generation(self, prompt: str, attachments: list, session_id):
        prompt_text = prompt or "请根据描述生成一张图。"
        try:
            parts: List[genai_types.Part] = []
            for att in attachments or []:
                part = self._attachment_to_part(att)
                if part:
                    parts.append(part)
            parts.append(genai_types.Part.from_text(text=prompt_text))
            generation_config = self._build_generation_config(include_tools=False, enable_thinking=False)
            contents = [genai_types.Content(role="user", parts=parts)]
            response = self._generate_image_response(contents, generation_config)
            image_paths = self._extract_image_paths(response)
            if image_paths:
                return Reply(ReplyType.IMAGE, image_paths[0])
            logger.warning("[Gemini] image generation succeeded but no images returned.")
            return Reply(ReplyType.ERROR, "图片生成失败，未获取到结果")
        except Exception as e:
            logger.error(f"[Gemini] image generation request failed: {e}", exc_info=True)
            return Reply(ReplyType.ERROR, "调用图片生成接口失败，请稍后再试")

    def _generate_image_response(self, contents: List[genai_types.Content], generation_config: genai_types.GenerateContentConfig):
        try:
            return self.client.models.generate_content(
                model=self.image_model,
                contents=contents,
                config=generation_config,
            )
        except (genai_errors.ClientError, ValueError) as e:
            if self._is_model_not_found(e) and self.fallback_image_model and self.fallback_image_model != self.image_model:
                logger.warning(f"[Gemini] image model {self.image_model} unavailable, fallback to {self.fallback_image_model}")
                return self.client.models.generate_content(
                    model=self.fallback_image_model,
                    contents=contents,
                    config=generation_config,
                )
            raise

    def _violates_china_politics_policy(self, query: str) -> bool:
        if not query or not self.china_guard_model:
            return False
        guard_prompt = self.china_guard_prompt.format(query=query)
        try:
            generation_config = self._build_generation_config(include_tools=False, enable_thinking=False)
            response = self.client.models.generate_content(
                model=self.china_guard_model,
                contents=[genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=guard_prompt)])],
                config=generation_config,
            )
            verdict = self._extract_text_response(response) or ""
            verdict = verdict.strip().lower()
            if verdict.startswith("yes"):
                logger.warning("[Gemini] request blocked by China politics policy")
                return True
            if verdict.startswith("no"):
                return False
            logger.warning(f"[Gemini] guard indeterminate verdict: {verdict}")
            return True
        except Exception as e:
            logger.error(f"[Gemini] guard check failed: {e}", exc_info=True)
            if isinstance(e, (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout)):
                return False
            return False

    def _extract_image_paths(self, response):
        image_paths = []
        generated_images = getattr(response, "generated_images", None)
        if generated_images:
            for item in generated_images:
                image_obj = getattr(item, "image", None)
                if not image_obj:
                    continue
                binary = getattr(image_obj, "image_bytes", None)
                if not binary:
                    continue
                mime_type = getattr(image_obj, "mime_type", None) or "image/png"
                path = self._save_image_bytes(binary, mime_type)
                if path:
                    image_paths.append(path)
        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", []) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None) or (part.get("inline_data") if isinstance(part, dict) else None)
                if not inline_data:
                    continue
                data = getattr(inline_data, "data", None) or inline_data.get("data")
                if not data:
                    continue
                if isinstance(data, str):
                    try:
                        binary = base64.b64decode(data)
                    except Exception as e:
                        logger.warning(f"[Gemini] decode image failed: {e}")
                        continue
                else:
                    binary = data
                mime_type = getattr(inline_data, "mime_type", None) or inline_data.get("mime_type") or "image/png"
                path = self._save_image_bytes(binary, mime_type)
                if path:
                    image_paths.append(path)
        return image_paths

    def _save_image_bytes(self, data: bytes, mime_type: str):
        ext = self._mime_to_ext(mime_type)
        directory = os.path.join(get_appdata_dir(), "generated_images")
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{uuid.uuid4().hex}.{ext}")
        try:
            with open(file_path, "wb") as f:
                f.write(data)
            return file_path
        except Exception as e:
            logger.error(f"[Gemini] save image failed: {e}")
            return None

    def _mime_to_ext(self, mime_type: str):
        mapping = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
        return mapping.get(mime_type, "png")

    def _normalize_model(self, model_name: Optional[str], default_value: str) -> str:
        if not model_name:
            return default_value
        name = str(model_name).strip()
        if name.lower() == "gemini":
            return "gemini-3.0-pro"
        return name

    def _is_model_not_found(self, error: Exception) -> bool:
        msg = str(error).lower()
        if isinstance(error, genai_errors.ClientError) and getattr(error, "status_code", None) == 404:
            return True
        return "not found" in msg or "not_found" in msg

    def _extract_text_response(self, response) -> Optional[str]:
        if hasattr(response, "text") and response.text:
            return response.text
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if content and getattr(content, "parts", None):
            first_part = content.parts[0]
            if hasattr(first_part, "text"):
                return first_part.text
            if isinstance(first_part, dict):
                return first_part.get("text")
        return None

    def _build_http_options(self) -> Optional[genai_types.HttpOptions]:
        proxy = (conf().get("proxy") or "").strip()
        timeout_value = conf().get("gemini_http_timeout")
        retry_attempts = conf().get("gemini_http_retry_attempts")
        trust_env_conf = conf().get("gemini_http_trust_env", True)
        if not proxy and not timeout_value and not retry_attempts:
            return None
        client_args = {}
        if timeout_value:
            client_args["timeout"] = timeout_value
        if proxy:
            client_args["proxies"] = {"http": proxy, "https": proxy}
            client_args["trust_env"] = False
        else:
            client_args["trust_env"] = bool(trust_env_conf)
        retry_options = None
        try:
            attempts = int(retry_attempts) if retry_attempts is not None else None
            if attempts and attempts > 0:
                retry_options = genai_types.HttpRetryOptions(
                    attempts=attempts,
                    max_delay=5.0,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                )
        except (TypeError, ValueError):
            retry_options = None
        try:
            options_kwargs = {}
            if client_args:
                options_kwargs["client_args"] = client_args
            if timeout_value:
                options_kwargs["timeout"] = timeout_value
            if retry_options:
                options_kwargs["retry_options"] = retry_options
            if not options_kwargs:
                return None
            return genai_types.HttpOptions(**options_kwargs)
        except Exception as e:
            logger.warning(f"[Gemini] http options setup failed: {e}")
            return None
