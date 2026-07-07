"""本地小说编辑器（独立子应用，依赖根目录 LocalHarness 核心）。"""

from .app import NovelWriterWindow, main

__all__ = ["NovelWriterWindow", "main"]
