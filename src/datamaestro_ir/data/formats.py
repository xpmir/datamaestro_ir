from functools import cached_property
from typing import Any, Tuple
from attrs import define
from .base import TextItem, SimpleTextItem


@define
class DocumentWithTitle(TextItem):
    """Web document with title and body"""

    title: str
    body: str

    @cached_property
    def text(self):
        return f"{self.title} {self.body}"


@define
class MsMarcoDocument(TextItem):
    url: str
    title: str
    body: str

    @cached_property
    def text(self):
        return self.body


@define
class TitleDocument(TextItem):
    body: str
    title: str

    @cached_property
    def text(self):
        return f"{self.title} {self.body}"


@define
class TitleUrlDocument(TitleDocument):
    url: str


@define
class WapoDocument(TextItem):
    url: str
    title: str
    author: str
    published_date: int
    kicker: str
    body: str
    body_paras_html: Tuple[str, ...]
    body_media: Tuple[Any, ...]

    @cached_property
    def text(self):
        return f"{self.title} {self.body_paras_html}"


@define()
class TrecTopic(SimpleTextItem):
    description: str
    narrative: str
