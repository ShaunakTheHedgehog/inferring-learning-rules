"""Miscellaneous tools that don't fit the other categories."""

import sys
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from qtpy.QtWidgets import QApplication, QMainWindow, QWidget

P = ParamSpec('P')
Q = TypeVar('Q')
R = TypeVar('R', bound=QWidget)


def get_app() -> QApplication:
    """
    Get the current QApplication instance.

    This function retrieves the existing QApplication instance. If no such instance
    exists or the instance is not of type QApplication (e.g., it's a QCoreApplication),
    a RuntimeError is raised.

    Returns
    -------
    QApplication
        The currently running QApplication instance.

    Raises
    ------
    RuntimeError
        If there is no running QApplication instance or if it is not a QApplication.
    """
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        raise RuntimeError('No QApplication instance is currently running.')
    return app


def get_or_create_app(argv: list[str] | None = None) -> QApplication:
    """
    Return the existing QApplication instance or create a new one.

    This helper checks for an existing QApplication instance; if none exists, it creates
    a new instance using the provided command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments to pass to QApplication. If `None`, `sys.argv` is used.

    Returns
    -------
    QApplication
        The existing or newly created QApplication instance.
    """
    try:
        return get_app()
    except RuntimeError:
        return QApplication(argv or sys.argv)


def require_qt(function: Callable[P, R]) -> Callable[P, R]:
    """
    Specify that a function requires a running Qt application.

    Use this decorator to wrap functions that depend on a QApplication being active. If
    no QApplication is running at the time the function is called, a RuntimeError is
    raised.

    Parameters
    ----------
    function : Callable
        The function that requires a Qt application.

    Returns
    -------
    Callable
        The wrapped function with Qt application requirement enforcement.

    Raises
    ------
    RuntimeError
        If no QApplication instance is running when the function is called.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            get_app()
        except RuntimeError as e:
            raise RuntimeError(
                f"'{function.__name__}' requires a running QApplication."
            ) from e
        return function(*args, **kwargs)

    return wrapped


def run_as_qt_app(function: Callable[P, R]) -> Callable[P, int]:
    """
    Run a function as a Qt application.

    This decorator wraps a function that returns a QWidget (or subclass) to be used as
    the main widget of a Qt application. It ensures a QApplication instance is created
    (or retrieved if already existing), calls the function to get the widget, and then
    starts the Qt event loop.

    Parameters
    ----------
    function : Callable[..., QWidget]
        A function that returns a QWidget instance to be shown as the main widget of the
        Qt app.

    Returns
    -------
    Callable[..., int]
        A wrapped function that runs the Qt application event loop and returns its exit
        code.
    """

    @wraps(function)
    def wrapper(*args, **kwargs) -> int:
        app = get_or_create_app()
        widget: QWidget = function(*args, **kwargs)  # noqa: F841
        widget.show()
        return app.exec_()

    return wrapper


@require_qt
def get_main_window() -> QMainWindow:
    """
    Get the main QMainWindow instance of the running Qt application.

    This function searches all top-level widgets in the current QApplication instance
    and returns the first one that is an instance of QMainWindow.

    Returns
    -------
    QMainWindow
        The first top-level widget that is a QMainWindow.

    Raises
    ------
    RuntimeError
        If no QApplication is running or no QMainWindow is found.
    """
    app = get_app()
    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow):
            return widget
    raise RuntimeError('No QMainWindow instance found among top-level widgets.')
