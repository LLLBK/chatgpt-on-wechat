"""
Google gemini bot

@author zhayujie
@Date 2023/12/15
"""
# encoding:utf-8

import os
import mimetypes

import requests
from bot.bot import Bot
import google.generativeai as genai
from bot.session_manager import SessionManager
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf
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
        self.nano_banana_key = conf().get("nano_banana_api_key")
        self.nano_banana_base = conf().get("nano_banana_api_base") or "https://api.nano-banana.com/v1"
        self.nano_banana_model = conf().get("nano_banana_model") or "nano-banana"

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
            attachments = self._pop_attachments(session_id)
            session = self.sessions.session_query(normalized_query, session_id)
            gemini_messages = self._convert_to_gemini_messages(self.filter_messages(session.messages), attachments)
            logger.debug(f"[Gemini] messages={gemini_messages}")
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)

            # 添加安全设置
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # 生成回复，包含安全设置
            response = model.generate_content(
                gemini_messages,
                safety_settings=safety_settings
            )
            if response.candidates and response.candidates[0].content:
                reply_text = response.candidates[0].content.parts[0].text
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
        if not self.nano_banana_key:
            return Reply(ReplyType.ERROR, "未配置 Nano Banana API Key，无法生成图片。")
        files = []
        for att in attachments or []:
            path = att.get("path")
            if not path or not os.path.exists(path):
                continue
            mime = att.get("mime_type") or mimetypes.guess_type(path)[0] or "application/octet-stream"
            try:
                files.append(
                    ("images", (att.get("file_name") or os.path.basename(path), open(path, "rb"), mime))
                )
            except Exception as e:
                logger.warning(f"[Gemini] attach reference image failed: {e}")
                continue
        headers = {"Authorization": f"Bearer {self.nano_banana_key}"}
        data = {
            "prompt": prompt or "请根据描述生成一张图片",
            "model": self.nano_banana_model,
        }
        try:
            resp = requests.post(
                f"{self.nano_banana_base.rstrip('/')}/images/generations",
                headers=headers,
                data=data,
                files=files if files else None,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            url = None
            if isinstance(result, dict):
                data_field = result.get("data")
                if isinstance(data_field, list) and data_field:
                    url = data_field[0].get("url")
                url = url or result.get("url")
            if url:
                return Reply(ReplyType.IMAGE_URL, url)
            return Reply(ReplyType.ERROR, "图片生成失败，未获取到结果。")
        except Exception as e:
            logger.error(f"[Gemini] Nano Banana request failed: {e}", exc_info=True)
            return Reply(ReplyType.ERROR, "调用图片生成接口失败，请稍后再试。")
