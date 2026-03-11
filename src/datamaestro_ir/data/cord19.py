from csv import DictReader
from typing import Iterator

from experimaestro import documentation
from datamaestro.data import File
from datamaestro_ir.data import Documents, IDTextRecord, Topics
from datamaestro_ir.data.formats import (
    DocumentWithTitle,
    TrecTopic,
)
from datamaestro.data.csv import Generic as GenericCSV
import xml.etree.ElementTree as ET


class Topics(Topics, File):
    """XML format used in Adhoc topics"""

    def iter(self) -> Iterator[IDTextRecord]:
        """Returns an iterator over topics"""
        tree = ET.parse(self.path)
        for topic in tree.findall("topic"):
            yield {
                "id": topic.get("number"),
                "text_item": TrecTopic(
                    topic.find("query").text,
                    question=topic.find("question").text,
                    narrative=topic.find("narrative").text,
                ),
            }


class Documents(Documents, GenericCSV):
    @documentation
    def iter(self) -> Iterator[IDTextRecord]:
        """Returns an iterator over adhoc documents"""
        with self.path.open("r") as fp:
            for row in DictReader(fp):
                yield {
                    "id": row["cord_uid"],
                    "text_item": DocumentWithTitle(row["abstract"], row["title"]),
                }
