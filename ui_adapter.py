"""UI adapter that abstracts message/dialog calls so the code can run
standalone or inside QGIS.

Set `iface` to the QGIS interface when running as a plugin to route messages
via the QGIS message bar; otherwise it falls back to QMessageBox and QFileDialog.
"""
from typing import Optional, Tuple

iface = None

try:
    from PyQt5 import QtWidgets
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
except Exception:
    QtWidgets = None
    QFileDialog = None
    QMessageBox = None


def info(msg: str, title: Optional[str] = None, parent=None):
    if iface is not None:
        iface.messageBar().pushMessage(title or 'Info', msg, level=0)
        return
    if QMessageBox is not None:
        QMessageBox.information(parent, title or 'Info', msg)
    else:
        print((title or 'Info') + ': ' + msg)


def warning(msg: str, title: Optional[str] = None, parent=None):
    if iface is not None:
        iface.messageBar().pushMessage(title or 'Warning', msg, level=1)
        return
    if QMessageBox is not None:
        QMessageBox.warning(parent, title or 'Warning', msg)
    else:
        print('Warning: ' + msg)


def critical(msg: str, title: Optional[str] = None, parent=None):
    if iface is not None:
        iface.messageBar().pushMessage(title or 'Error', msg, level=3)
        return
    if QMessageBox is not None:
        QMessageBox.critical(parent, title or 'Error', msg)
    else:
        print('Error: ' + msg)


def get_open_filename(parent=None, caption='Open', filter='All Files (*)') -> Tuple[str, str]:
    if QFileDialog is not None:
        # QFileDialog.getOpenFileName signature: (parent, caption, directory, filter)
        return QFileDialog.getOpenFileName(parent, caption, '', filter)
    return ('', '')


def get_save_filename(parent=None, caption='Save', filter='All Files (*)') -> Tuple[str, str]:
    if QFileDialog is not None:
        # QFileDialog.getSaveFileName signature: (parent, caption, directory, filter)
        return QFileDialog.getSaveFileName(parent, caption, '', filter)
    return ('', '')
