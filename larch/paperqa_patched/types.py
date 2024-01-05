from typing import Any, Optional

from paperqa.types import Doc

DocKey = Any


class DocWithMetadata(Doc):
    doc_type: Optional[str] = None
