"""
Google gemini bot

@author zhayujie
@Date 2023/12/15
"""
# encoding:utf-8

import base64
import os
import mimetypes
import uuid

from bot.bot import Bot
import google.generativeai as genai
from bot.session_manager import SessionManager
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf, get_appdata_dir
from common.expired_dict import ExpiredDict
from common.safety import sensitive_filter
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.baidu.baidu_wenxin_session import BaiduWenxinSession
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# OpenAI对话模型API (可用)
class GoogleGeminiBot(Bot):

    def __init__(self):
        super().__init__()
        self.api_key = conf().get("gemini_api_key")
        # 复用chatGPT的token计算方式
        self.sessions = SessionManager(ChatGPTSession, model=conf().get("model") or "gpt-3.5-turbo")
        self.model = conf().get("model") or "gemini-3.0-pro"
        if self.model == "gemini":
            self.model = "gemini-3.0-pro"
        self.attachment_ttl = conf().get("gemini_attachment_cache_seconds", 900)
        self.attachment_max_size = conf().get("gemini_attachment_max_size", 10 * 1024 * 1024)
        self.pending_attachments = ExpiredDict(self.attachment_ttl) if self.attachment_ttl else {}
        self.nano_banana_model = conf().get("nano_banana_model") or "nano-banana"
        self.google_search_enabled = conf().get("gemini_enable_google_search", False)
        search_tool_conf = (conf().get("gemini_google_search_tool") or "auto").strip().lower()
        if search_tool_conf not in ("auto", "google_search", "google_search_retrieval"):
            search_tool_conf = "auto"
        if search_tool_conf == "auto":
            if "1.5" in self.model:
                search_tool_conf = "google_search_retrieval"
            else:
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
        self.china_guard_model = conf().get("china_politics_guard_model", "gemini-2.5-flash")
        self.china_guard_prompt = conf().get("china_politics_guard_prompt") or (
            "You are a strict content safety classifier. Determine whether the following user request discusses contemporary Chinese politics, including current government, political figures, and political events in modern China. Answer ONLY with YES or NO.\n\nUser request:\n\"\"\"{query}\"\"\"\n\nDoes this request involve contemporary Chinese politics?"
        )

    def reply(self, query, context: Context = None) -> Reply:
        try:
            session_id = context["session_id"]
            normalized_query = (query or "").strip()
            if self._should_clear(normalized_query):
                self.sessions.clear_session(session_id)
                if session_id in self.pending_attachments:
                    del self.pending_attachments[session_id]
                return Reply(ReplyType.INFO, "已清空对话，开始新的会话。")

            if context.type in [ContextType.IMAGE, ContextType.FILE, ContextType.VOICE, ContextType.VIDEO]:
                stored = self._cache_attachments(session_id, context.get("attachments") or [])
                if not stored:
                    return Reply(ReplyType.ERROR, "未收到可用的附件，请重试。")
                return Reply(ReplyType.INFO, f"已收到附件 {len(stored)} 个，请继续输入你的问题。")

            # 图片生成分支
            if context.type == ContextType.IMAGE_CREATE or self._is_image_request(normalized_query):
                attachments = self._pop_attachments(session_id)
                return self._handle_image_generation(normalized_query, attachments, session_id)

            if context.type != ContextType.TEXT:
                logger.warn(f"[Gemini] Unsupported message type, type={context.type}")
                return Reply(ReplyType.TEXT, None)

            logger.info(f"[Gemini] query={normalized_query}")
            # Configure API client upfront for guard/model calls
            genai.configure(api_key=self.api_key)
            if self.china_guard_enabled and self._violates_china_politics_policy(normalized_query):
                return Reply(ReplyType.ERROR, "抱歉，该请求无法处理。")
            attachments = self._pop_attachments(session_id)
            session = self.sessions.session_query(normalized_query, session_id)
            gemini_messages = self._convert_to_gemini_messages(self.filter_messages(session.messages), attachments)
            logger.debug(f"[Gemini] messages={gemini_messages}")
            model = genai.GenerativeModel(self.model)

            # 添加安全设置
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # 生成回复，包含安全设置
            request_args = {
                "safety_settings": safety_settings
            }
            if self.google_search_tools:
                request_args["tools"] = self.google_search_tools
            response = model.generate_content(
                gemini_messages,
                **request_args
            )
            if response.candidates and response.candidates[0].content:
                reply_text = getattr(response, "text", None) or response.candidates[0].content.parts[0].text
                reply_text = self._append_grounding_citations(response, reply_text)
                ok, filtered_text = sensitive_filter.filter(reply_text)
                if not ok:
                    reply_text = filtered_text
                logger.info(f"[Gemini] reply={reply_text}")
                self.sessions.session_reply(reply_text, session_id)
                return Reply(ReplyType.TEXT, reply_text)
            else:
                # 没有有效响应内容，可能内容被屏蔽，输出安全评分
                logger.warning("[Gemini] No valid response generated. Checking safety ratings.")
                if hasattr(response, 'candidates') and response.candidates:
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

    def _convert_to_gemini_messages(self, messages: list, attachments: list = None):
        """
        将历史消息转换为Gemini内容结构，如果有附件，挂载到当前用户消息上
        """
        attachments = attachments or []
        res = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                role = "user"
            elif msg.get("role") == "assistant":
                role = "model"
            elif msg.get("role") == "system":
                role = "user"
            else:
                continue
            parts = [{"text": msg.get("content")}]
            # 将附件挂载到最新的user消息上
            if idx == len(messages) - 1 and role == "user" and attachments:
                attachment_parts = []
                for att in attachments:
                    part = self._attachment_to_part(att)
                    if part:
                        attachment_parts.append(part)
                parts = attachment_parts + parts
            res.append({
                "role": role,
                "parts": parts
            })
        return res

    def _build_google_search_tools(self):
        if not self.google_search_enabled:
            return None
        if self.google_search_tool == "google_search_retrieval":
            dynamic_config = {"mode": self.google_search_mode or "MODE_DYNAMIC"}
            if dynamic_config["mode"] == "MODE_DYNAMIC" and self.google_search_threshold is not None:
                threshold = max(0.0, min(1.0, float(self.google_search_threshold)))
                dynamic_config["dynamic_threshold"] = threshold
            return [{
                "google_search_retrieval": {
                    "dynamic_retrieval_config": dynamic_config
                }
            }]
        return [{"google_search": {}}]

    def _append_grounding_citations(self, response, text: str) -> str:
        if not text or not self.google_search_enabled or not self.google_search_show_citations:
            return text
        try:
            candidates = getattr(response, "candidates", None)
            if not candidates:
                return text
            candidate = candidates[0]
            metadata = self._maybe_get(candidate, "grounding_metadata", "groundingMetadata")
            if not metadata:
                return text
            supports = self._maybe_get(metadata, "grounding_supports", "groundingSupports") or []
            chunks = self._maybe_get(metadata, "grounding_chunks", "groundingChunks") or []
            if not supports or not chunks:
                return text
            sorted_supports = sorted(
                list(supports),
                key=lambda s: self._maybe_get(self._maybe_get(s, "segment"), "end_index", "endIndex") or 0,
                reverse=True
            )
            text_with_citations = text
            chunks_list = list(chunks)
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
                    chunk = chunks_list[idx]
                    web_info = self._maybe_get(chunk, "web")
                    uri = self._maybe_get(web_info, "uri")
                    if uri:
                        citation_links.append(f"[{idx + 1}]({uri})")
                if not citation_links:
                    continue
                end_index = max(0, min(int(end_index), len(text_with_citations)))
                citation_string = ", ".join(citation_links)
                text_with_citations = text_with_citations[:end_index] + citation_string + text_with_citations[end_index:]
            return text_with_citations
        except Exception as e:
            logger.warning(f"[Gemini] append citations failed: {e}")
            return text

    @staticmethod
    def _maybe_get(obj, *keys):
        if obj is None:
            return None
        for key in keys:
            if not key:
                continue
            value = None
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
            return {"inline_data": {"mime_type": mime_type, "data": data}}
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

    def _should_clear(self, query: str) -> bool:
        clear_commands = [c.strip() for c in conf().get("clear_memory_commands", []) if c]
        return query in clear_commands

    def _is_image_request(self, query: str) -> bool:
        if not query:
            return False
        keywords = ["生成图片", "画一张", "出一张图", "生成一张图", "画个", "绘制"]
        return any(kw in query for kw in keywords)

    def _handle_image_generation(self, prompt: str, attachments: list, session_id):
        prompt_text = prompt or "请根据描述生成一张图片"
        try:
            model = genai.GenerativeModel(self.nano_banana_model)
            parts = []
            for att in attachments or []:
                part = self._attachment_to_part(att)
                if part:
                    parts.append(part)
            parts.append({"text": prompt_text})
            response = model.generate_content(
                parts,
                generation_config={"response_mime_type": "image/png"},
            )
            image_paths = self._extract_image_paths(response)
            if image_paths:
                return Reply(ReplyType.IMAGE, image_paths[0])
            logger.warning("[Gemini] image generation succeeded but no images returned.")
            return Reply(ReplyType.ERROR, "图片生成失败，未获取到结果。")
        except Exception as e:
            logger.error(f"[Gemini] Nano Banana request failed: {e}", exc_info=True)
            return Reply(ReplyType.ERROR, "调用图片生成接口失败，请稍后再试。")

    def _violates_china_politics_policy(self, query: str) -> bool:
        if not query or not self.china_guard_model:
            return False
        guard_prompt = self.china_guard_prompt.format(query=query)
        try:
            guard_model = genai.GenerativeModel(self.china_guard_model)
            response = guard_model.generate_content(guard_prompt)
            verdict = ""
            if hasattr(response, "text") and response.text:
                verdict = response.text
            elif getattr(response, "candidates", None):
                candidate = response.candidates[0]
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    verdict = content.parts[0].text
            verdict = (verdict or "").strip().lower()
            if verdict.startswith("yes"):
                logger.warning("[Gemini] request blocked by China politics policy")
                return True
            if verdict.startswith("no"):
                return False
            # 未能解析明确答案时默认拦截
            logger.warning(f"[Gemini] guard indeterminate verdict: {verdict}")
            return True
        except Exception as e:
            logger.error(f"[Gemini] guard check failed: {e}", exc_info=True)
            return True

    def _extract_image_paths(self, response):
        image_paths = []
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
        mapping = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp"
        }
        return mapping.get(mime_type, "png")
