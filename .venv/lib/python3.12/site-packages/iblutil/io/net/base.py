import asyncio
import re
import json
import socket
import warnings
import logging
from asyncio import isfuture
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import urllib.request
import ipaddress
from operator import or_
from functools import reduce
from enum import IntFlag, IntEnum, auto  # py3.11 STRICT

LISTEN_PORT = 11001  # listen for commands on this port
PROTOCOL_VERSION = '1.0.0'  # Versioning for ExpMessage, ExpStatus enumerations, and Service base class


def is_success(future: asyncio.Future) -> bool:
    """Check if future successfully resolved."""
    return future.done() and not future.cancelled() and future.exception() is None


def external_ip():
    """
    Fetch WAN IP address.

    NB: Requires internet.

    Returns
    -------
    ipaddress.IPv4Address, ipaddress.IPv6Address
        The computer's default WAN IP address.
    """
    return ipaddress.ip_address(urllib.request.urlopen('https://ident.me').read().decode('utf8'))


def is_valid_ip(ip_address) -> bool:
    """
    Test whether IP address is valid.

    Parameters
    ----------
    ip_address : str
        An IP address to validate.

    Returns
    -------
    bool
        True is IP address is valid.
    """
    try:
        ipaddress.ip_address(ip_address)
        return True
    except ValueError:
        return False


def hostname2ip(hostname=None):
    """
    Resolve hostname to IP address.

    Parameters
    ----------
    hostname : str, optional
        The hostname to resolve.  If None, resolved this computer's hostname.

    Returns
    -------
    ipaddress.IPv4Address, ipaddress.IPv6Address
        The resolved IP address.

    Raises
    ------
    ValueError
        Failed to resolve IP for hostname.
    """
    hostname = hostname or socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
        return ipaddress.ip_address(ip_address)
    except (socket.error, socket.gaierror):
        raise ValueError(f'Failed to resolve IP for hostname "{hostname}"')


def validate_uri(uri, resolve_host=True, default_port=LISTEN_PORT, default_proc='udp'):
    """
    Ensure URI is complete and correct.

    Parameters
    ----------
    uri : str, ipaddress.IPv4Address, ipaddress.IPv6Address
        A full URI, hostname or hostname and port.
    resolve_host : bool
        If the URI is not an IP address, attempt to resolve hostname to IP.
    default_port : int, str
        If the port is absent from the URI, append this one.
    default_proc : str
        If the URI scheme is missing, prepend this one.

    Returns
    -------
    str
        The complete URI.

    Raises
    ------
    TypeError
        URI type not supported.
    ValueError
        Failed to resolve host name to IP address.
        URI host contains invalid characters (expects only alphanumeric + hyphen).
        Port number not within range (must be > 1, <= 65535).
    """
    # Validate URI scheme
    if not isinstance(uri, (str, ipaddress.IPv4Address, ipaddress.IPv6Address)):
        raise TypeError(f'Unsupported URI "{uri}" of type {type(uri)}')

    if isinstance(uri, str) and (proc := re.match(r'(?P<proc>^[a-zA-Z]+(?=://))', uri)):
        proc = proc.group()
        uri = uri[len(proc) + 3 :]
    else:
        proc = default_proc
    # Validate hostname
    if isinstance(uri, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        host = str(uri)
        port = default_port
    elif ':' in uri:
        host, port = uri.split(':', 1)
    else:
        host = uri
        port = None
    if isinstance(uri, str) and not is_valid_ip(host):
        if resolve_host:
            host = hostname2ip(host)
        elif not re.match(r'^[a-z0-9-]+$', host):
            raise ValueError(f'Invalid hostname "{host}"')
    # Validate port
    try:
        port = int(port or default_port)
        assert 1 <= port <= 65535
    except (AssertionError, ValueError):
        raise ValueError(f'Invalid port number: {port or default_port}')
    return f'{proc or default_proc}://{host}:{port}'


# class ExpMessage(IntFlag, boundary=STRICT):  # py3.11
class ExpMessage(IntFlag):
    """A set of standard experiment messages for communicating between rigs."""

    """Experiment is initializing."""
    EXPINIT = auto()
    """Experiment has begun."""
    EXPSTART = auto()
    """Experiment has stopped."""
    EXPEND = auto()
    """Experiment cleanup begun."""
    EXPCLEANUP = auto()
    """Experiment interrupted."""
    EXPINTERRUPT = auto()
    """Experiment status."""
    EXPSTATUS = auto()
    """Experiment info, including task protocol start and end."""
    EXPINFO = auto()
    """Alyx token."""
    ALYX = auto()
    __version__ = PROTOCOL_VERSION

    @classmethod
    def any(cls) -> 'ExpMessage':
        """Return enumeration comprising all possible messages.

        NB: This doesn't include the null ExpMessage (0), used to indicate API errors.
        """
        return reduce(or_, cls)

    @staticmethod
    def validate(event, allow_bitwise=True):
        """
        Validate an event message, returning a corresponding enumeration if valid and raising an
        exception if not.

        Parameters
        ----------
        event : str, int, ExpMessage
            An event message to validate.
        allow_bitwise : bool
            If false, raise if event is the result of a bitwise operation.

        Returns
        -------
        ExpMessage:
            The corresponding event enumeration.

        Raises
        ------
        TypeError
            event is neither a string, integer nor enumeration.
        ValueError
            event does not correspond to any ExpMessage enumeration, neither in its integer form
            nor its string name, or `allow_bitwise` is false and value is combination of events.

        Examples
        --------
        >>> ExpMessage.validate('expstart')
        ExpMessage.EXPSTART

        >>> ExpMessage.validate(10)
        ExpMessage.EXPINIT

        >>> ExpMessage.validate(ExpMessage.EXPEND)
        ExpMessage.EXPEND
        """
        if not isinstance(event, ExpMessage):
            try:
                if isinstance(event, str):
                    event = ExpMessage[event.strip().upper()]
                elif isinstance(event, int):
                    event = ExpMessage(event)
                else:
                    raise TypeError(f'Unknown event type {type(event)}')
            except KeyError:
                raise ValueError(f'Unrecognized event "{event}". Choices: {tuple(ExpMessage.__members__.keys())}')
        if not allow_bitwise and event not in list(ExpMessage):
            raise ValueError(f'Compound (bitwise) events not permitted. Choices: {tuple(ExpMessage)}')
        return event

    def __iter__(self):  # py3.11 remove this method
        """Iterate over the individual bits in the enumeration.

        NB: This method is copied from Python 3.11 which supports iteration of Enum objects.
        """
        num = self.value
        while num:
            b = num & (~num + 1)
            yield b
            num ^= b


class ExpStatus(IntEnum):
    """A set of standard statuses for communicating between rigs."""

    """Service is connected."""
    CONNECTED = 0
    """Service is initialized."""
    INITIALIZED = 10
    """Service is running."""
    RUNNING = 20
    """Experiment has stopped."""
    STOPPED = 30
    __version__ = PROTOCOL_VERSION


class Service(ABC):
    """An abstract base class for auxiliary experiment services."""

    __version__ = PROTOCOL_VERSION
    __slots__ = 'name'

    @abstractmethod
    def init(self, data=None):
        """
        Initialize an experiment.

        This is intended to specify the expected message signature. The subclassed method should
        serialize the returned values and pass them to the transport layer.

        Parameters
        ----------
        data : any
            Optional extra data to send to the remote server.

        Returns
        -------
        ExpMessage.EXPINIT
            The EXPINIT event.
        any, None
            Optional extra data.
        """
        return ExpMessage.EXPINIT, data

    @abstractmethod
    def start(self, exp_ref, data=None):
        """
        Start an experiment.

        This is intended to specify the expected message signature. The subclassed method should
        serialize the returned values and pass them to the transport layer.

        Parameters
        ----------
        exp_ref : str
            An experiment reference string in the form yyyy-mm-dd_n_subject.
        data : any
            Optional extra data to send to the remote server.

        Returns
        -------
        ExpMessage.EXPSTART
            The EXPSTART event.
        str
            The experiment reference string.
        any, None
            Optional extra data.
        """
        exp_ref = exp_ref or None
        if isinstance(exp_ref, dict):
            exp_ref = '_'.join(map(str, (exp_ref['date'], int(exp_ref['sequence']), exp_ref['subject'])))
        return ExpMessage.EXPSTART, exp_ref, data

    @abstractmethod
    def stop(self, data=None, immediately=False):
        """
        Stop an experiment.

        This is intended to specify the expected message signature. The subclassed method should
        serialize the returned values and pass them to the transport layer.

        Parameters
        ----------
        data : any
            Optional extra data to send to the remote server.
        immediately : bool
            If True, an EXPINTERRUPT message is returned.

        Returns
        -------
        ExpMessage.EXPINTERRUPT, ExpMessage.EXPEND
            The EXPEND event, or EXPINTERRUPT if immediately is True.
        any, None
            Optional extra data.
        """
        return ExpMessage.EXPINTERRUPT if immediately else ExpMessage.EXPEND, data

    @abstractmethod
    def status(self, status):
        """
        Report experiment status.

        NB: This is intended to be lightweight. For more detail and custom data use the info
        method.

        This is intended to specify the expected message signature. The subclassed method should
        serialize the returned values and pass them to the transport layer.

        Parameters
        ----------
        status : ExpStatus
            The experiment status enumeration.

        Returns
        -------
        ExpMessage.EXPSTATUS
            The EXPSTATUS event.
        ExpStatus
            The validated experiment status.
        """
        if not isinstance(status, ExpStatus):
            status = ExpStatus(status) if isinstance(status, int) else ExpStatus[status]
        return ExpMessage.EXPSTATUS, status

    @abstractmethod
    def info(self, status, data=None):
        """
        Report experiment information.

        This is intended to specify the expected message signature. The subclassed method should
        serialize the returned values and pass them to the transport layer.

        Parameters
        ----------
        status : ExpStatus
            The experiment status enumeration.
        data : any
            Optional extra data to send to the remote server.

        Returns
        -------
        ExpMessage.EXPINFO
            The EXPINFO event.
        ExpStatus
            The validated experiment status.
        any, None
            Optional extra data.
        """
        return ExpMessage.EXPINFO, Service.status(self, status)[1], data

    @abstractmethod
    def cleanup(self, data=None):
        """
        Clean up an experiment.

        This is intended to specify the expected message signature. The subclassed method should
        serialize the returned values and pass them to the transport layer.

        Parameters
        ----------
        data : any
            Optional extra data to send to the remote server.

        Returns
        -------
        ExpMessage.EXPCLEANUP
            The EXPCLEANUP event.
        any, None
            Optional extra data.
        """
        return ExpMessage.EXPCLEANUP, data

    @abstractmethod
    def alyx(self, alyx):
        """
        Request/send Alyx token.

        This is intended to specify the expected message signature. The subclassed method should
        serialize the returned values and pass them to the transport layer.

        Parameters
        ----------
        alyx : one.webclient.AlyxClient
            Optional instance of Alyx to send.

        Returns
        -------
        ExpMessage.ALYX
            The ALYX event.
        str
            The Alyx database URL.
        dict
            The Alyx token in the form {user: token}.
        """
        base_url = alyx.base_url if alyx else None
        token = {alyx.user: alyx._token} if alyx and alyx.is_logged_in else {}
        return ExpMessage.ALYX, base_url, token


class Communicator(Service):
    """A base class for communicating between experimental rigs.

    Attributes
    ----------
    name : str
        An arbitrary label for the remote host
    server_uri : str
        The full URI of the remote device, e.g. udp://192.168.0.1:1001
    """

    __slots__ = ('server_uri', '_callbacks', 'logger', 'name')

    def __init__(self, server_uri, name=None, logger=None):
        self.server_uri = validate_uri(server_uri)
        self.name = name or server_uri
        self.logger = logger or logging.getLogger(self.name)
        # Init callbacks map of ExpMessage -> list, including null ExpMessage for processing callback errors
        self._callbacks = dict(map(lambda item: (item, []), (ExpMessage(0), *ExpMessage.__members__.values())))

    def assign_callback(self, event, callback):
        """
        Assign a callback to be called when an event occurs.

        NB: Unlike with futures, an assigned callback may be triggered multiple times, whereas
        coroutines may only be set once after which they are cleared.

        Parameters
        ----------
        event : str, int, iblutil.io.net.base.ExpMessage
            The event for which the callback is registered.
        callback : function, asyncio.Future
            A function or Future to trigger when an event occurs.

        See Also
        --------
        EchoProtocol.receive
            The method that processes the callbacks upon receiving a message.
        """
        event = ExpMessage.validate(event)
        if not (callable(callback) or isfuture(callback)):
            raise TypeError('Callback must be callable or a Future')
        if event is ExpMessage(0):
            self._callbacks.setdefault(event, []).append((callback, False))
        else:
            return_event = event not in ExpMessage
            for e in event:  # iterate over enum as bitwise ops permitted
                self._callbacks.setdefault(e, []).append((callback, return_event))

    def clear_callbacks(self, event, callback=None, cancel_futures=True):
        """
        For a given event, remove the provided callback, or all callbacks if none were provided.

        Parameters
        ----------
        event : str, int, iblutil.io.net.base.ExpMessage
            The event for which the callback was registered.
        callback : function, asyncio.Future
            The callback or future to remove.
        cancel_futures : bool
            If True and callback is a Future, cancel before removing.

        Returns
        -------
        int
            The number of callbacks removed.
        """
        i = 0
        event = ExpMessage.validate(event)
        if callback:  # clear specific callback
            # Wrapped callables have an id attribute containing the hash of the inner function
            for evt in event:  # iterate as bitwise enums permitted, e.g. ~ExpMessage.ALYX
                ids = [getattr(cb, 'id', None) or hash(cb) for cb, _ in self._callbacks[evt]]
                cb_id = getattr(callback, 'id', None) or hash(callback)
                while True:
                    try:
                        idx = ids.index(cb_id)
                        if cancel_futures and isfuture(cb := self._callbacks[evt][idx][0]):
                            cb.cancel()
                        del self._callbacks[evt][idx]
                        del ids[idx]
                        i += 1
                    except (IndexError, ValueError):
                        break
        else:  # clear all callbacks for event
            for evt in event:
                if cancel_futures:
                    for cb in filter(isfuture, map(lambda x: x[0], self._callbacks[evt])):
                        cb.cancel()
                i += len(self._callbacks[evt])
                del self._callbacks[evt][:]
        self.logger.debug('[%s] %i callbacks cleared', self.name, i)
        return i

    async def on_event(self, event):
        """
        Await an event from the remote host.

        Parameters
        ----------
        event : str, int, iblutil.io.net.base.ExpMessage
            The event to wait on.

        Returns
        -------
        any
            The response data returned by the remote host.

        Examples
        --------
        >>> data, addr = await com.on_event('EXPSTART')

        >>> event = await asyncio.create_task(com.on_event('EXPSTART'))
        >>> ...
        >>> data = await event

        Await more than one event

        >>> data, addr, event = await com.on_event(ExpMessage.EXPEND | ExpMessage.EXPINTERRUPT)
        """
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.assign_callback(event, fut)
        return await fut

    @property
    def port(self) -> int:
        """int: the remote port"""
        return int(urlparse(self.server_uri).port)

    @property
    def hostname(self) -> str:
        """str: the remote hostname or IP address"""
        return urlparse(self.server_uri).hostname

    @property
    def protocol(self) -> str:
        """str: the protocol scheme, e.g. udp, ws"""
        return urlparse(self.server_uri).scheme

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """bool: True if the remote device is connected"""
        pass

    @abstractmethod
    def send(self, data, addr=None):
        """Serialize and pass data to the transport layer"""
        pass

    def _receive(self, data, addr):
        """
        Process data received from remote host and notify event listeners.

        This is called by the transport layer when a message is received and should not be called
        by the user.

        Parameters
        ----------
        data : bytes
            The serialized data received by the transport layer.
        addr : (str, int)
            The source address as (hostname, port)

        Warnings
        --------
        Warnings
            Expects the deserialized data to be a tuple where the first element is an ExpMessage.
            A warning is thrown if the data is not a tuple, or has fewer than two elements.

        TODO Perhaps for every event only the first future should be set.
        """
        data = self.decode(data)
        if isinstance(data, (list, tuple)) and len(data) > 1:
            event, *data = data
            event = ExpMessage.validate(event, allow_bitwise=False) if event else ExpMessage(0)
            if event is ExpMessage(0):
                # An error in the remote callback function occurred
                err, evt = data
                self.logger.error('Callback for %s on %s://%s:%i failed with %s', ExpMessage(evt).name, self.protocol, *addr, err)
            if event not in self._callbacks or not self._callbacks[event]:
                self.logger.debug('No callbacks to execute for event %s', event.name)
                return
            for f, return_event in self._callbacks[event].copy():
                if isfuture(f):
                    if f.done():
                        self.logger.warning('Future %s already resolved', f)
                    elif not f.cancelled():
                        f.set_result((data, addr, event) if return_event else (data, addr))
                    self.clear_callbacks(event, f)  # Remove future from list
                else:
                    try:
                        f(data, addr, event) if return_event else f(data, addr)
                    except Exception as ex:
                        self.logger.error('Callback "%s" failed with error "%s"', f, ex)
                        message = self.encode([0, f'{type(ex).__name__}: {ex}', event])
                        self.send(message, addr)
                        break
        else:
            warnings.warn(f'Expected list, got {data}', RuntimeWarning)

    @staticmethod
    def encode(data) -> bytes:
        """
        Serialize data for transmission.

        None-string or -bytes objects are encoded as JSON before converting to bytes.

        Parameters
        ----------
        data : any
            The data to serialize.

        Returns
        -------
        bytes
            The encoded data.
        """
        if isinstance(data, bytes):
            return data
        if not isinstance(data, str):
            data = json.dumps(data)
        return data.encode()

    @staticmethod
    def decode(data: bytes):
        """
        De-serialize and parse bytes data.

        This function attempts to decode the data as a JSON object.  On failing that, returns a
        string.

        Parameters
        ----------
        data : bytes
            The data to de-serialize.

        Returns
        -------
        any
            Deserialized data.
        """
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            warnings.warn('Failed to decode as JSON')
            data = data.decode()
        return data

    def close(self):
        """De-register all callbacks and cancel futures"""
        for event in self._callbacks:
            for fut in filter(isfuture, map(lambda x: x[0], self._callbacks[event])):
                fut.cancel('Close called on communicator')
            del self._callbacks[event][:]
