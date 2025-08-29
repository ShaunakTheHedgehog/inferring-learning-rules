"""Graphical user interface components."""

import webbrowser
from enum import IntEnum
from pathlib import Path
from shutil import _ntuple_diskusage, disk_usage
from typing import Any

from qtpy.QtCore import (
    Property,
    QAbstractItemModel,
    QEvent,
    QModelIndex,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    QSizeF,
    Qt,
    QThreadPool,
    QUrl,
    Signal,
    Slot,
)
from qtpy.QtGui import (
    QColor,
    QIcon,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
)
from qtpy.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from iblqt import resources  # noqa: F401
from iblqt.core import QAlyx, RestrictedWebEnginePage, Worker
from iblutil.util import format_bytes


class CheckBoxDelegate(QStyledItemDelegate):
    """
    A custom delegate for rendering checkboxes in a QTableView or similar widget.

    This delegate allows for the display and interaction with boolean data as checkboxes.
    """

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        """
        Paints the checkbox in the view.

        Parameters
        ----------
        painter : QPainter
            The painter used to draw the checkbox.
        option : QStyleOptionButton
            The style option containing the information needed for painting.
        index : QModelIndex
            The index of the item in the model.
        """
        super().paint(painter, option, index)
        control = QStyleOptionButton()
        control.rect = QRect(option.rect.topLeft(), QCheckBox().sizeHint())
        control.rect.moveCenter(option.rect.center())
        control.state = QStyle.State_On if index.data() is True else QStyle.State_Off
        QApplication.style().drawControl(
            QStyle.ControlElement.CE_CheckBox, control, painter
        )

    def displayText(self, value: Any, locale: Any) -> str:
        """
        Return an empty string to hide the text representation of the data.

        Parameters
        ----------
        value : Any
            The value to be displayed (not used).
        locale : Any
            The locale to be used (not used).

        Returns
        -------
        str
            An empty string.
        """
        return ''

    def editorEvent(
        self,
        event: QEvent,
        model: QAbstractItemModel,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> bool:
        """
        Handle user interaction with the checkbox.

        Parameters
        ----------
        event : QEvent
            The event that occurred (e.g., mouse click).
        model : QAbstractItemModel
            The model associated with the view.
        option : QStyleOptionViewItem
            The style option containing the information needed for handling the event.
        index : QModelIndex
            The index of the item in the model.

        Returns
        -------
        bool
            True if the event was handled, False otherwise.
        """
        if isinstance(event, QMouseEvent) and event.type() == QEvent.MouseButtonRelease:
            checkbox_rect = QRect(option.rect.topLeft(), QCheckBox().sizeHint())
            checkbox_rect.moveCenter(option.rect.center())
            if checkbox_rect.contains(event.pos()):
                model.setData(index, not model.data(index))
                event.accept()
                return True
        return super().editorEvent(event, model, option, index)


class ColoredButton(QPushButton):
    """A QPushButton that can change color."""

    def __init__(
        self,
        text: str,
        color: QColor | None = None,
        parent: QWidget | None = None,
        **kwargs,
    ):
        """
        Initialize the ColoredButton.

        Parameters
        ----------
        text : str
            The text to be displayed.
        color : QColor, optional
            The color to use for the button.
        parent : QWidget
            The parent widget.
        **kwargs : dict
            Arbitrary keyword arguments (passed to QPushButton).
        """
        super().__init__(text, parent, **kwargs)
        self._original_color = self.palette().color(QPalette.Button)
        if color is not None:
            self.setColor(color)

    def setColor(self, color: QColor | None) -> None:
        """
        Set the color of the button.

        Parameters
        ----------
        color : QColor, optional
            The new color of the button. If None, the color will be reset to it's
            default color.
        """
        palette = self.palette()
        palette.setColor(QPalette.Button, color or self._original_color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.update()

    def color(self) -> QColor:
        """
        Return the color of the button.

        Returns
        -------
        QColor
            The color of the button.
        """
        return self.palette().color(QPalette.Button)


class StatefulButton(ColoredButton):
    """A QPushButton that maintains an active/inactive state."""

    clickedWhileActive = Signal()  # type: Signal
    """Emitted when the button is clicked while it is in the active state."""

    clickedWhileInactive = Signal()  # type: Signal
    """Emitted when the button is clicked while it is in the inactive state."""

    stateChanged = Signal(bool)  # type: Signal
    """Emitted when the button's state has changed. The signal carries the new state
    (True for active, False for inactive)."""

    def __init__(
        self,
        textActive: str | None = None,
        textInactive: str | None = None,
        active: bool = False,
        parent: QWidget | None = None,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the StatefulButton with the specified active state.

        Parameters
        ----------
        textActive : str, optional
            The text shown on the button when in active state.
        textInactive : str, optional
            The text shown on the button when in inactive state.
        active : bool, optional
            Initial state of the button (default is False).
        parent : QWidget
            The parent widget.
        **kwargs : dict
            Arbitrary keyword arguments (passed to ColoredButton).
        """
        self._isActive = active
        self._textActive = textActive or ''
        self._textInactive = textInactive or ''
        super().__init__(
            text=self._textActive if active else self._textInactive,
            color=None,
            parent=parent,
            **kwargs,
        )

        self.clicked.connect(self._onClick)
        self.stateChanged.connect(self._onStateChange)

    def getActive(self) -> bool:
        """Get the active state of the button.

        Returns
        -------
        bool
            True if the button is active, False otherwise.
        """
        return self._isActive

    @Slot(bool)
    def setActive(self, value: bool):
        """Set the active state of the button.

        Emits `stateChanged` if the state has changed.

        Parameters
        ----------
        value : bool
            The new active state of the button.
        """
        if self._isActive != value:
            self._isActive = value
            self.stateChanged.emit(self._isActive)

    active = Property(bool, fget=getActive, fset=setActive, notify=stateChanged)  # type: Property
    """The active state of the button."""

    def getTextActive(self) -> str:
        """Get the text shown on the button when in active state.

        Returns
        -------
        str
            The text shown during active state.
        """
        return self._textActive

    def setTextActive(self, text: str):
        """Set the text shown on the button when in active state.

        Parameters
        ----------
        text : str
            The text to be shown during active state.
        """
        self._textActive = text
        if self.active:
            self.setText(self._textActive)

    textActive = Property(str, fget=getTextActive, fset=setTextActive)
    """The text shown on the button during active state."""

    def getTextInactive(self) -> str:
        """Get the text shown on the button during inactive state.

        Returns
        -------
        str
            The text shown during inactive state.
        """
        return self._textInactive

    def setTextInactive(self, text: str):
        """Set the text shown on the button during inactive state.

        Parameters
        ----------
        text : str
            The text to be shown during inactive state.
        """
        self._textInactive = text
        if not self.active:
            self.setText(self._textInactive)

    textInactive = Property(str, fget=getTextInactive, fset=setTextInactive)
    """The text shown on the button during inactive state."""

    @Slot()
    def _onClick(self):
        """Handle the button click event.

        Emits `clickedWhileActive` if the button is active,
        otherwise emits `clickedWhileInactive`.
        """
        if self._isActive:
            self.clickedWhileActive.emit()
        else:
            self.clickedWhileInactive.emit()

    @Slot(bool)
    def _onStateChange(self, state: bool):
        """Handle the state change event."""
        if state is True:
            self.setText(self._textActive)
        if state is False:
            self.setText(self._textInactive)


class UseTokenCache(IntEnum):
    """Enumeration that defines the strategy for caching the login token."""

    NEVER = -1
    """Indicates that the login token should never be stored."""

    ASK = 0
    """Indicates that the user should be prompted whether to store the login token."""

    ALWAYS = 1
    """Indicates that the login token should always be stored."""


class AlyxUserEdit(QLineEdit):
    """A specialized :class:`QLineEdit` for logging in to Alyx.

    A one-line text editor for entering a username. The widget handles login
    actions, including displaying the login status. A :class:`AlyxLoginDialog`
    is triggered when no authentication token is available.
    """

    def __init__(
        self, alyx: QAlyx, parent: QWidget, cache: UseTokenCache = UseTokenCache.ASK
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        alyx : QAlyx
            The alyx instance.
        parent : QWidget
            The parent widget.
        cache : UseTokenCache
            Strategy for handling the token cache. Defaults to UseTokenCache.ASK.
        """
        super().__init__(parent)
        self.alyx = alyx
        self._cache = cache
        self._checkIcon = QIcon(':/icon/check')

        self.setPlaceholderText('Username')
        self.returnPressed.connect(self.login)
        self.alyx.loggedIn.connect(self._onLoggedIn)
        self.alyx.loggedOut.connect(self._onLoggedOut)
        self.alyx.tokenMissing.connect(self._onTokenMissing)

    def login(self):
        """Attempt to log in to Alyx with the entered username.

        If the username field is empty, the login attempt is ignored.
        """
        if len(self.text()) == 0:
            return
        self.alyx.login(username=self.text())

    def _onLoggedIn(self, username: str):
        """Handle successful login by updating the UI.

        Parameters
        ----------
        username : str
            The username of the logged-in user.
        """
        self.addAction(self._checkIcon, self.ActionPosition.TrailingPosition)
        self.setText(username)
        self.setReadOnly(True)
        self.setStyleSheet('background-color: rgb(246, 245, 244);')

    def _onLoggedOut(self):
        """Handle logout by resetting the UI elements."""
        for action in self.actions():
            self.removeAction(action)
        self.setText('')
        self.setReadOnly(False)
        self.setStyleSheet('')

    def _onTokenMissing(self, username: str):
        """Prompt the user for password when the authentication token is missing.

        Parameters
        ----------
        username : str
            The username for which the token is missing.
        """
        AlyxLoginDialog(self.alyx, username, self, self._cache).exec_()


class AlyxLoginWidget(QWidget):
    """A widget used for managing the connection to Alyx.

    This widget contains an :class:`AlyxUserEdit` for entering a username and a
    :class:`StatefulButton` for logging in and out.
    """

    def __init__(
        self,
        alyx: QAlyx | str,
        parent: QWidget | None = None,
        cache: UseTokenCache = UseTokenCache.ASK,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        alyx : QAlyx | str
            The Alyx instance or the base URL for the Alyx API.
        parent : QWidget
            The parent widget.
        cache : UseTokenCache
            Strategy for handling the token cache. Defaults to UseTokenCache.ASK.
        """
        super().__init__(parent)

        if isinstance(alyx, QAlyx):
            self.alyx = alyx
        else:
            self.alyx = QAlyx(base_url=alyx, parent=self)

        # edit field for username
        self.userEdit = AlyxUserEdit(alyx=self.alyx, cache=cache, parent=self)
        self.userEdit.textChanged.connect(self._onTextChanged)

        # stateful button that is used for, both, logging in and logging out
        self.button = StatefulButton(
            textActive='Logout', textInactive='Login', parent=self
        )
        self.button.setEnabled(False)
        self.button.clickedWhileInactive.connect(self.userEdit.login)
        self.button.clickedWhileActive.connect(self.alyx.logout)
        self.alyx.statusChanged.connect(self.button.setActive)

        QHBoxLayout(self)
        self.layout().addWidget(self.userEdit)
        self.layout().addWidget(self.button)

    def _onTextChanged(self, username: str):
        """Only enable the login button when a username has been entered."""
        text_entered = len(username) > 0
        if self.button.isEnabled() ^ text_entered:
            self.button.setEnabled(text_entered)


class AlyxLoginDialog(QDialog):
    """A password dialog window used for authenticating with Alyx."""

    def __init__(
        self,
        alyx: QAlyx,
        username: str | None = None,
        parent: QWidget | None = None,
        cache: UseTokenCache = UseTokenCache.ASK,
    ) -> None:
        """Initialize the widget.

        Parameters
        ----------
        alyx : QAlyx
            The alyx instance.
        username : str
            The username.
        parent : QWidget
            The parent widget.
        cache : UseTokenCache
            Strategy for handling the token cache. Defaults to ASK.
        """
        super().__init__(parent)
        self.setWindowTitle('Login')

        self._alyx = alyx
        self._alyx.authenticationFailed.connect(self._onAuthentificationFailed)
        self._alyx.loggedIn.connect(self._onAuthentificationSucceeded)

        if cache == UseTokenCache.ALWAYS:
            self._cache = True
        else:
            self._cache = False

        form_widget = QWidget(self)
        self.userEdit = QLineEdit(username or '', form_widget)
        self.passEdit = QLineEdit(form_widget)
        self.userEdit.textChanged.connect(self._onTextChanged)
        self.passEdit.setFocus()
        self.passEdit.setEchoMode(QLineEdit.Password)
        self.passEdit.textChanged.connect(self._onTextChanged)
        form_layout = QFormLayout(form_widget)
        form_layout.addRow('Server', QLabel(self._alyx.client.base_url, form_widget))
        form_layout.addRow('Username', self.userEdit)
        form_layout.addRow('Password', self.passEdit)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_widget.setLayout(form_layout)

        control_widget = QWidget(self)
        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, control_widget
        )
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.accept)
        self.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        control_layout = QHBoxLayout(control_widget)
        if cache == UseTokenCache.ASK:
            check_cache = QCheckBox('Remember me', control_widget)
            check_cache.stateChanged.connect(self._setCache)
            control_layout.addWidget(check_cache)
        control_layout.addItem(
            QSpacerItem(20, 0, QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        )
        control_layout.addWidget(self.buttonBox)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_widget.setLayout(control_layout)

        QVBoxLayout(self)
        self.layout().addWidget(form_widget)
        self.layout().addItem(
            QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)
        )
        self.layout().addWidget(control_widget)
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

    def _setCache(self, do_cache: int) -> None:
        self._cache = do_cache == Qt.CheckState.Checked

    def accept(self):
        """Hide the dialog and set the result code to Accepted."""
        self._alyx.login(self.userEdit.text(), self.passEdit.text(), self._cache)

    def _onAuthentificationSucceeded(self, _: str) -> None:
        super().accept()

    def _onAuthentificationFailed(self, _: str):
        """Show :class:`QMessageBox` on authentication failure."""
        self.passEdit.setText('')
        QMessageBox.critical(
            self,
            'Login Failed',
            'Authentication has failed.\nPlease try again.',
        )

    def _onTextChanged(self):
        """Only enable the OK button when text has been entered in both fields."""
        ok_button = self.buttonBox.button(QDialogButtonBox.Ok)
        text_entered = all([len(x.text()) > 0 for x in (self.userEdit, self.passEdit)])
        if ok_button.isEnabled() ^ text_entered:
            ok_button.setEnabled(text_entered)


class ThresholdProgressBar(QProgressBar):
    """A progress bar that changes color based on a threshold value."""

    thresholdChanged = Signal(int)
    thresholdCrossed = Signal(bool)

    def __init__(
        self,
        threshold: int,
        color_critical: QColor | None = None,
        color_default: QColor | None = None,
        **kwargs,
    ):
        """
        Initialize ThresholdProgressBar.

        Parameters
        ----------
        threshold : int
            The threshold at which the progress bar changes color.
        color_critical : QColor, optional
            The color to use when the threshold is crossed. Defaults to red.
        color_default : QColor, optional
            The default color of the progress bar when below the threshold.
        **kwargs : dict
            Arbitrary keyword arguments (passed to QProgressBar).
        """
        super().__init__(**kwargs)
        self._color_critical = color_critical or QColor('red')
        self._color_default = color_default or self.palette().color(QPalette.Highlight)
        self._above_threshold = False
        self.valueChanged.connect(self._check_value)
        self.thresholdChanged.connect(self._check_threshold)
        self.thresholdCrossed.connect(self._set_color)
        self._threshold = threshold
        self._set_color(self.aboveThreshold())

    def aboveThreshold(self) -> bool:
        """
        Check if the current value is above the threshold.

        Returns
        -------
        bool
            True if the current value is above the threshold, False otherwise.
        """
        return self.value() > self._threshold

    def threshold(self) -> int:
        """Get the current threshold value."""
        return self._threshold

    def setThreshold(self, value: int) -> None:
        """Set a new threshold value."""
        self._threshold = value
        self.thresholdChanged.emit(value)

    @Slot(int)
    def _check_value(self, value: int) -> None:
        self._check(self._threshold, value)

    @Slot(int)
    def _check_threshold(self, threshold: int) -> None:
        self._check(threshold, self.value())

    def _check(self, threshold: int, value: int) -> None:
        above_threshold = value > threshold
        if self._above_threshold ^ above_threshold:
            self.thresholdCrossed.emit(above_threshold)
            self._above_threshold = above_threshold

    @Slot(bool)
    def _set_color(self, above_threshold: bool) -> None:
        palette = self.palette()
        palette.setColor(
            QPalette.Highlight,
            self._color_critical if above_threshold else self._color_default,
        )
        self.setPalette(palette)


class DiskSpaceIndicator(ThresholdProgressBar):
    """A progress bar widget that indicates the disk space usage of a directory."""

    _directory: Path

    def __init__(
        self, directory: Path | str | None = None, percent_threshold: int = 90, **kwargs
    ) -> None:
        """
        Initialize DiskSpaceIndicator.

        Parameters
        ----------
        directory : Path, str, optional
            The directory to monitor for disk space usage.
            Default is the root of the current working directory.
        percent_threshold : int, optional
            The threshold percentage at which the progress bar changes color to red.
            Default is 90.
        **kwargs : dict
            Arbitrary keyword arguments (passed to QProgressBar).
        """
        super().__init__(threshold=percent_threshold, **kwargs)
        self.setRange(0, 100)
        self.setDirectory(directory or Path.cwd().anchor)

    def directory(self) -> str:
        """
        Get the directory being monitored for disk space usage.

        Returns
        -------
        str
            The path of the directory being monitored.
        """
        return str(self._directory)

    def setDirectory(self, directory: Path | str) -> None:
        """Set the directory to monitor for disk space usage.

        Parameters
        ----------
        directory : Path, str
            The directory path to monitor.
        """
        self._directory = Path(directory).resolve()
        self.updateData()

    def updateData(self) -> None:
        """Update the disk space information."""
        worker = Worker(disk_usage, self._directory.anchor)
        worker.signals.result.connect(self._on_result)
        QThreadPool.globalInstance().start(worker)

    def _on_result(self, result: _ntuple_diskusage) -> None:
        percent = round(result.used / result.total * 100)
        self.setValue(percent)
        self.setToolTip(f'{format_bytes(result.free)} available')


class RestrictedWebView(QWidget):
    """A browser widget that restricts navigation to a trusted URL prefix."""

    def __init__(
        self,
        url: QUrl | str,
        trusted_url_prefix: str | None = None,
        use_tool_tips: bool = True,
        use_status_tips: bool = False,
        parent: QWidget | None = None,
    ):
        """
        Initialize the RestrictedWebView.

        Parameters
        ----------
        url : QUrl or str
            The initial URL to load in the browser.
        trusted_url_prefix : str, optional
            Prefix of trusted URLs. Clicking on links to matching URLs will open
            in the widget's web engine view, while other links will open in the default
            web browser. Defaults to the value of `url`.
        use_tool_tips : bool, optional
            Whether to set default tool tips for the buttons. Default is True.
        use_status_tips : bool, optional
            Whether to set default status tips for the buttons. Default is False.
        parent : QWidget or None, optional
            The parent widget.
        """
        super().__init__(parent)
        if isinstance(url, str):
            url = QUrl(url)
        if trusted_url_prefix is None:
            trusted_url_prefix = url.toString()

        # The import of QWebEngineView is handled locally as QtWebEngineWidgets is a
        # separate package in case of PyQt5 and may not be available in all environments
        from qtpy.QtWebEngineWidgets import QWebEngineView

        # Create a QWebEngineView to display the web content
        self.webEngineView = QWebEngineView(self)
        self.webEngineView.setContextMenuPolicy(Qt.NoContextMenu)
        self.webEngineView.setAcceptDrops(False)
        self.webEnginePage = RestrictedWebEnginePage(
            parent=self.webEngineView, trusted_url_prefix=trusted_url_prefix
        )
        self.webEngineView.setPage(self.webEnginePage)

        # Create a horizontal line to separate the web view from the navigation buttons
        line = QFrame(self)
        line.setFrameShadow(QFrame.Sunken)
        line.setFrameShape(QFrame.HLine)

        # Create a frame to hold the navigation buttons
        frame = QFrame(self)
        frame.setFrameShape(QFrame.NoFrame)

        # Create navigation buttons, set initial state
        self.uiPushBack = QPushButton(QIcon(':/icon/previous'), '', frame)
        self.uiPushForward = QPushButton(QIcon(':/icon/next'), '', frame)
        self.uiPushHome = QPushButton(QIcon(':/icon/home'), '', frame)
        self.uiPushBrowser = QPushButton(QIcon(':/icon/globe'), '', frame)
        self.uiPushBack.setEnabled(False)
        self.uiPushForward.setEnabled(False)

        # Assemble the layout
        horizontalLayout = QHBoxLayout(frame)
        horizontalLayout.addWidget(self.uiPushBack)
        horizontalLayout.addWidget(self.uiPushForward)
        horizontalLayout.addWidget(self.uiPushHome)
        horizontalLayout.addStretch(1)
        horizontalLayout.addWidget(self.uiPushBrowser)
        verticalLayout = QVBoxLayout(self)
        verticalLayout.setContentsMargins(0, 0, 0, 0)
        verticalLayout.addWidget(self.webEngineView)
        verticalLayout.addWidget(line)
        verticalLayout.addWidget(frame)
        verticalLayout.setStretch(0, 1)
        verticalLayout.setSpacing(0)

        # connect signals to slots
        self.uiPushHome.clicked.connect(lambda: self.webEngineView.load(QUrl(url)))
        self.uiPushBack.clicked.connect(lambda: self.webEngineView.back())
        self.uiPushForward.clicked.connect(lambda: self.webEngineView.forward())
        self.uiPushBrowser.clicked.connect(
            lambda: webbrowser.open(self.webEngineView.url().url())
        )
        self.webEngineView.urlChanged.connect(self._on_url_changed)

        # set tool tips / status tips
        if use_tool_tips:
            self.uiPushHome.setToolTip('Go to home page')
            self.uiPushBack.setToolTip('Go back one page')
            self.uiPushForward.setToolTip('Go forward one page')
            self.uiPushBrowser.setToolTip('Open page in default browser')
        if use_status_tips:
            self.uiPushHome.setStatusTip('Go to home page')
            self.uiPushBack.setStatusTip('Go back one page')
            self.uiPushForward.setStatusTip('Go forward one page')
            self.uiPushBrowser.setStatusTip('Open page in default browser')

        # set prefix and initial URL
        self.setUrl(url)

    def _on_url_changed(self, url: QUrl):
        self.uiPushBack.setEnabled(self.webEngineView.history().canGoBack())
        self.uiPushForward.setEnabled(self.webEngineView.history().canGoForward())

    def setUrl(self, url: QUrl | str) -> bool:
        """
        Set the URL to be displayed in the web engine view.

        Parameters
        ----------
        url : QUrl or str
            The URL to load in the web engine view.

        Returns
        -------
        bool
            True if the navigation request is accepted, False otherwise.
        """
        if isinstance(url, str):
            url = QUrl(url)
        if url.toString().startswith(self.trustedUrlPrefix()):
            self.webEngineView.setUrl(url)
            return True
        else:
            return False

    def url(self) -> QUrl:
        """
        Get the current URL displayed in the web engine view.

        Returns
        -------
        QUrl
            The current URL.
        """
        return self.webEngineView.url()

    def setTrustedUrlPrefix(self, trusted_url_prefix: str):
        """
        Set the trusted URL prefix.

        Parameters
        ----------
        trusted_url_prefix : str
            The prefix of trusted URLs. Clicking on links to matching URLs will open
            in the widget's web engine view, while other links will open in the default
            web browser.
        """
        self.webEnginePage.setTrustedUrlPrefix(trusted_url_prefix)

    def trustedUrlPrefix(self) -> str:
        """
        Get the trusted URL prefix.

        Returns
        -------
        str
            The prefix of trusted URLs. Clicking on links to matching URLs will open
            in the widget's web engine view, while other links will open in the default
            web browser.
        """
        return self.webEnginePage.trustedUrlPrefix()


class SlideToggle(QAbstractButton):
    """
    A sliding toggle switch.

    Adapted from :class:`Switch` class by Stefan Scherfke
    https://stackoverflow.com/a/51825815
    """

    def __init__(self, parent: QWidget | None = None, height: int = 20) -> None:
        """
        Initialize the toggle switch.

        Parameters
        ----------
        parent : QWidget or None, optional
            The parent widget of the switch. Defaults to None.

        height : int, optional
            The height of the switch in pixels. The overall size and
            internal geometry are derived from this value. Defaults to 20.
        """
        super().__init__(parent=parent)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._radius = height / 2.0
        thumb_radius = self._radius * 0.8
        self._margin = self._radius - thumb_radius
        self._thumb_size = QSizeF(2 * thumb_radius, 2 * thumb_radius)
        self._relative_position_value = 0.0
        font = self.font()
        font.setPixelSize(round(1.5 * thumb_radius))
        self.setFont(font)

        self._animation = QPropertyAnimation(self, b'_relative_position', self)
        self._animation.setDuration(120)
        self.toggled.connect(self._setup_animation)

    def _setup_animation(self, checked: bool):
        self._animation.setEndValue(float(checked))
        self._animation.start()

    @Property(float)
    def _relative_position(self) -> float:
        return self._relative_position_value

    @_relative_position.setter  # type: ignore
    def _relative_position(self, value: float):
        self._relative_position_value = value
        self.update()

    def sizeHint(self) -> QSize:
        """
        Return the recommended size for the widget.

        Returns
        -------
        QSize
            The recommended size of the widget as a QSize object
        """
        return QSize(round(4 * self._radius), round(2 * self._radius))

    def paintEvent(self, event: QPaintEvent):
        """
        Handle the painting of the custom toggle switch.

        The appearance adapts to the widget's enabled/disabled state and palette.

        Parameters
        ----------
        event : QPaintEvent
            The paint event that triggered this repaint.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Determine colors based on state
        palette = self.palette()
        if not self.isEnabled():
            color_background = palette.dark().color()
            color_background.setAlphaF(0.6)
            color_foreground = palette.window().color()
        elif self.isChecked():
            color_background = palette.highlight().color()
            color_foreground = palette.light().color()
        else:
            color_background = palette.dark().color()
            color_foreground = palette.light().color()

        # Draw track
        painter.setPen(Qt.NoPen)
        painter.setBrush(color_background)
        painter.drawRoundedRect(self.rect(), self._radius, self._radius)

        # Draw thumb
        x = (self.width() - 2 * self._radius) * self._relative_position + self._margin
        thumb_rect = QRectF(QPointF(x, self._margin), self._thumb_size)
        painter.setBrush(color_foreground)
        painter.drawEllipse(thumb_rect)

        # Draw symbol
        painter.setPen(color_background)
        painter.setFont(self.font())
        symbol = '✔' if self.isChecked() else '✘'
        painter.drawText(thumb_rect, Qt.AlignCenter, symbol)
