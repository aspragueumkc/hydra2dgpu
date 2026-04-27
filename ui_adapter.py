"""UI adapter for routing dialogs and messages through QGIS/PyQGIS APIs."""
from typing import Optional, Tuple

iface = None

try:
    from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
    from qgis.core import Qgis
except Exception:
    QFileDialog = None
    QMessageBox = None
    Qgis = None


def _message_level(level_name: str):
    if Qgis is None:
        return 0
    return getattr(Qgis.MessageLevel, level_name, getattr(Qgis, level_name, 0))


def _default_parent(parent=None):
    if parent is not None:
        return parent
    if iface is not None:
        try:
            return iface.mainWindow()
        except Exception:
            return None
    return None


def info(msg: str, title: Optional[str] = None, parent=None):
    if iface is not None:
        iface.messageBar().pushMessage(title or 'Info', msg, level=_message_level('Info'))
        return
    if QMessageBox is not None:
        QMessageBox.information(_default_parent(parent), title or 'Info', msg)
    else:
        print((title or 'Info') + ': ' + msg)


def warning(msg: str, title: Optional[str] = None, parent=None):
    if iface is not None:
        iface.messageBar().pushMessage(title or 'Warning', msg, level=_message_level('Warning'))
        return
    if QMessageBox is not None:
        QMessageBox.warning(_default_parent(parent), title or 'Warning', msg)
    else:
        print('Warning: ' + msg)


def critical(msg: str, title: Optional[str] = None, parent=None):
    if iface is not None:
        iface.messageBar().pushMessage(title or 'Error', msg, level=_message_level('Critical'))
        return
    if QMessageBox is not None:
        QMessageBox.critical(_default_parent(parent), title or 'Error', msg)
    else:
        print('Error: ' + msg)


def get_open_filename(parent=None, caption='Open', filter='All Files (*)') -> Tuple[str, str]:
    if QFileDialog is not None:
        # QFileDialog.getOpenFileName signature: (parent, caption, directory, filter)
        return QFileDialog.getOpenFileName(_default_parent(parent), caption, '', filter)
    return ('', '')


def get_save_filename(parent=None, caption='Save', filter='All Files (*)') -> Tuple[str, str]:
    if QFileDialog is not None:
        # QFileDialog.getSaveFileName signature: (parent, caption, directory, filter)
        return QFileDialog.getSaveFileName(_default_parent(parent), caption, '', filter)
    return ('', '')
