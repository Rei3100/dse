import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_metrics_tab_instantiates():
    """MetricsTab が Qt アプリ内でクラッシュなしにインスタンス化できること。"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from DSRE import MetricsTab
    tab = MetricsTab()
    assert tab is not None


def test_metrics_tab_refresh_no_crash():
    """DB が空でも refresh() がクラッシュしないこと。"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from DSRE import MetricsTab
    tab = MetricsTab()
    tab.refresh()  # should not raise
