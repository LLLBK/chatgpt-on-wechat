# encoding:utf-8

"""
wechat channel message
"""

import os

from bridge.context import ContextType
from channel.chat_message import ChatMessage
from common.log import logger
from config import get_appdata_dir
from wcferry import WxMsg


class WechatfMessage(ChatMessage):
    """
    微信消息封装类
    """

    def __init__(self, channel, wcf_msg: WxMsg, is_group=False):
        """
        初始化消息对象
        :param wcf_msg: wcferry消息对象
        :param is_group: 是否是群消息
        """
        super().__init__(wcf_msg)
        self.msg_id = wcf_msg.id
        self.create_time = wcf_msg.ts  # 使用消息时间戳
        self.is_group = is_group or wcf_msg._is_group
        self.wxid = channel.wxid
        self.name = channel.name
        self.file_name = None
        self.media_path = None
        self._media_dir = os.path.join(get_appdata_dir(), "wcf_attachments")
        os.makedirs(self._media_dir, exist_ok=True)

        # 解析消息类型
        msg_type = getattr(wcf_msg, "type", None)
        if wcf_msg.is_text():
            self.ctype = ContextType.TEXT
            self.content = wcf_msg.content
        elif hasattr(wcf_msg, "is_image") and wcf_msg.is_image() or msg_type == 3:
            self.ctype = ContextType.IMAGE
            self.content = self._build_media_path(wcf_msg, "jpg")
            self._prepare_fn = self._build_prepare_fn(channel, wcf_msg, self.content)
        elif hasattr(wcf_msg, "is_voice") and wcf_msg.is_voice() or msg_type == 34:
            self.ctype = ContextType.VOICE
            self.content = self._build_media_path(wcf_msg, "amr")
            self._prepare_fn = self._build_prepare_fn(channel, wcf_msg, self.content)
        elif hasattr(wcf_msg, "is_video") and wcf_msg.is_video() or msg_type == 43:
            self.ctype = ContextType.VIDEO
            self.content = self._build_media_path(wcf_msg, "mp4")
            self._prepare_fn = self._build_prepare_fn(channel, wcf_msg, self.content)
        elif hasattr(wcf_msg, "is_file") and wcf_msg.is_file() or msg_type == 49:
            self.ctype = ContextType.FILE
            ext = self._guess_ext(wcf_msg)
            self.content = self._build_media_path(wcf_msg, ext or "bin")
            self._prepare_fn = self._build_prepare_fn(channel, wcf_msg, self.content)
        else:
            raise NotImplementedError(f"Unsupported message type: {getattr(wcf_msg, 'type', 'unknown')}")

        # 设置发送者和接收者信息
        self.from_user_id = self.wxid if wcf_msg.sender == self.wxid else wcf_msg.sender
        self.from_user_nickname = self.name if wcf_msg.sender == self.wxid else channel.contact_cache.get_name_by_wxid(wcf_msg.sender)
        self.to_user_id = self.wxid
        self.to_user_nickname = self.name
        self.other_user_id = wcf_msg.sender
        self.other_user_nickname = channel.contact_cache.get_name_by_wxid(wcf_msg.sender)

        # 群消息特殊处理
        if self.is_group:
            self.other_user_id = wcf_msg.roomid
            self.other_user_nickname = channel.contact_cache.get_name_by_wxid(wcf_msg.roomid)
            self.actual_user_id = wcf_msg.sender
            self.actual_user_nickname = channel.wcf.get_alias_in_chatroom(wcf_msg.sender, wcf_msg.roomid)
            if not self.actual_user_nickname:  # 群聊获取不到企微号成员昵称，这里尝试从联系人缓存去获取
                self.actual_user_nickname = channel.contact_cache.get_name_by_wxid(wcf_msg.sender)
            self.room_id = wcf_msg.roomid
            self.is_at = wcf_msg.is_at(self.wxid)  # 是否被@当前登录用户

        # 判断是否是自己发送的消息
        self.my_msg = wcf_msg.from_self()

    def _build_media_path(self, wcf_msg: WxMsg, default_ext: str) -> str:
        file_name = getattr(wcf_msg, "file_name", None) or f"{wcf_msg.id}.{default_ext}"
        self.file_name = file_name
        return os.path.join(self._media_dir, file_name)

    def _guess_ext(self, wcf_msg: WxMsg):
        file_name = getattr(wcf_msg, "file_name", None)
        if file_name and "." in file_name:
            return file_name.split(".")[-1]
        extra = getattr(wcf_msg, "extra", None)
        if extra and "." in extra:
            return extra.split(".")[-1]
        return None

    def _build_prepare_fn(self, channel, wcf_msg: WxMsg, target_path: str):
        def _prepare():
            self._download_media(channel, wcf_msg, target_path)
        return _prepare

    def _download_media(self, channel, wcf_msg: WxMsg, target_path: str):
        downloader = None
        for attr in ["get_msg_file", "download_file", "get_file"]:
            if hasattr(channel.wcf, attr):
                downloader = getattr(channel.wcf, attr)
                break
        if downloader is None:
            logger.warning("[WCF] downloader not found, skip media fetch")
            return
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        # 尝试多种调用签名，兼容不同版本wcferry
        download_dir = os.path.dirname(target_path)
        try:
            downloader(wcf_msg.id, target_path)
        except TypeError:
            try:
                downloader(wcf_msg.id, getattr(wcf_msg, "extra", ""), download_dir)
            except TypeError:
                try:
                    downloader(wcf_msg.id, download_dir)
                except Exception as e:
                    logger.error(f"[WCF] download media failed: {e}")
        # 兜底：如果目标路径不存在，尝试在目录下找到文件
        if not os.path.exists(target_path):
            candidate = self._find_downloaded_file(download_dir)
            if candidate:
                self.content = candidate

    def _find_downloaded_file(self, download_dir: str):
        if not os.path.isdir(download_dir):
            return None
        try:
            files = sorted(
                [os.path.join(download_dir, f) for f in os.listdir(download_dir)],
                key=os.path.getmtime,
                reverse=True,
            )
            for f in files:
                if os.path.isfile(f):
                    return f
        except Exception as e:
            logger.debug(f"[WCF] find downloaded file failed: {e}")
        return None
