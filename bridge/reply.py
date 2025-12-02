# encoding:utf-8

from enum import Enum


class ReplyType(Enum):
    TEXT = 1  # 文本
    VOICE = 2  # 音频文件
    IMAGE = 3  # 图片文件
    IMAGE_URL = 4  # 图片URL
    VIDEO_URL = 5  # 视频URL
    FILE = 6  # 文件
    CARD = 7  # 微信名片，仅支持ntchat
    INVITE_ROOM = 8  # 邀请好友进群
    INFO = 9
    ERROR = 10
    TEXT_ = 11  # 强制文本
    VIDEO = 12
    MINIAPP = 13  # 小程序

    def __str__(self):
        return self.name


class Reply:
    def __init__(self, type: ReplyType = None, content=None):
        self.type = type
        self.content = content
        self.extra_replies = []

    def __str__(self):
        return "Reply(type={}, content={})".format(self.type, self.content)

    def add_extra_reply(self, reply: "Reply"):
        if isinstance(reply, Reply):
            self.extra_replies.append(reply)

    def add_extra_text(self, text: str):
        if text:
            self.extra_replies.append(Reply(ReplyType.TEXT, text))
