from functools import cached_property
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
class TitleDocument(TextItem):
    body: str
    title: str

    @cached_property
    def text(self):
        return f"{self.title} {self.body}"


@define()
class TrecTopic(SimpleTextItem):
    description: str
    narrative: str
