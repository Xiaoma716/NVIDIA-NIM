"""
监控面板：从静态文件加载 HTML 页面
"""

from pathlib import Path

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_DASHBOARD_FILE = _STATIC_DIR / "dashboard.html"


def get_dashboard_html() -> str:
    """从静态文件读取 Dashboard HTML 内容"""
    if not _DASHBOARD_FILE.exists():
        return "<html><body><h1>Dashboard HTML file not found</h1></body></html>"
    return _DASHBOARD_FILE.read_text(encoding="utf-8")


DASHBOARD_HTML = get_dashboard_html()
