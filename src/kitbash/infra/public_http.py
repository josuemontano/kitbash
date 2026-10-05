"""Public-only HTTP transport: validated DNS answers are the addresses actually connected.

The OS resolver, socket implementation, and platform TLS trust store are trusted. No
proxy or second hostname lookup sits between address validation and socket.connect.
"""

import ipaddress
import socket
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit

import httpcore
import httpx


class UnsafeURL(ValueError):
    """A destination is outside the public HTTP image acquisition policy."""


def _public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if not address.is_global or any((address.is_reserved, address.is_multicast, address.is_unspecified,
                                    address.is_loopback, address.is_private, address.is_link_local)):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        # Transition addresses can tunnel to an otherwise forbidden IPv4 destination.
        if address.ipv4_mapped or address.sixtofour or address.teredo or "%" in value:
            return False
        if address in ipaddress.ip_network("64:ff9b::/96") or address in ipaddress.ip_network("64:ff9b:1::/48"):
            return False
    return True


def validate_public_url(value: str | httpx.URL) -> httpx.URL:
    """Check URL syntax and literal addresses; DNS is checked at connection time."""
    raw = str(value)
    if "\\" in raw or any(ord(char) <= 32 or ord(char) == 127 for char in raw):
        raise UnsafeURL("Invalid URL characters")
    try:
        parsed = urlsplit(raw)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise UnsafeURL("Only absolute HTTP(S) URLs are supported")
        if parsed.username is not None or parsed.password is not None:
            raise UnsafeURL("URL credentials are not allowed")
        expected_port = 443 if parsed.scheme == "https" else 80
        if parsed.port not in (None, expected_port):
            raise UnsafeURL("Only the scheme's standard port is allowed")
        host = parsed.hostname
        if "%" in host or host.rstrip(".").lower() == "localhost" or host.lower().endswith(".localhost"):
            raise UnsafeURL("Local or scoped hostnames are not allowed")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass  # The connection backend validates every DNS result, including odd IPv4 spellings.
        else:
            if not _public_address(host):
                raise UnsafeURL("Non-public destination")
        return httpx.URL(raw)
    except (ValueError, httpx.InvalidURL) as exc:
        raise UnsafeURL(str(exc)) from exc


class _SocketStream(httpcore.NetworkStream):
    def __init__(self, sock: socket.socket) -> None:
        self._socket = sock

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        try:
            self._socket.settimeout(timeout)
            return self._socket.recv(max_bytes)
        except TimeoutError as exc:
            raise httpcore.ReadTimeout(str(exc)) from exc
        except OSError as exc:
            raise httpcore.ReadError(str(exc)) from exc

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        try:
            self._socket.settimeout(timeout)
            self._socket.sendall(buffer)
        except TimeoutError as exc:
            raise httpcore.WriteTimeout(str(exc)) from exc
        except OSError as exc:
            raise httpcore.WriteError(str(exc)) from exc

    def close(self) -> None:
        self._socket.close()

    def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: str | None = None,
                  timeout: float | None = None) -> httpcore.NetworkStream:
        try:
            self._socket.settimeout(timeout)
            return _SocketStream(ssl_context.wrap_socket(self._socket, server_hostname=server_hostname))
        except TimeoutError as exc:
            self.close()
            raise httpcore.ConnectTimeout(str(exc)) from exc
        except OSError as exc:
            self.close()
            raise httpcore.ConnectError(str(exc)) from exc

    def get_extra_info(self, info: str):
        if info == "ssl_object" and isinstance(self._socket, ssl.SSLSocket):
            return self._socket
        if info == "client_addr":
            return self._socket.getsockname()
        if info == "server_addr":
            return self._socket.getpeername()
        if info == "socket":
            return self._socket
        return None


class _PublicNetworkBackend(httpcore.NetworkBackend):
    def connect_tcp(self, host: str, port: int, timeout: float | None = None,
                    local_address: str | None = None, socket_options=None) -> httpcore.NetworkStream:
        if port not in (80, 443) or local_address is not None:
            raise UnsafeURL("Unsupported connection destination")
        try:
            answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        except OSError as exc:
            raise httpcore.ConnectError(str(exc)) from exc
        if not answers or any(family not in (socket.AF_INET, socket.AF_INET6) or not _public_address(address[0])
                              or address[1] != port for family, _, _, _, address in answers):
            # Reject the entire set, not just private answers in a mixed set.
            raise UnsafeURL("DNS returned a non-public destination")
        error: OSError | None = None
        for family, socktype, proto, _, address in answers:
            sock = socket.socket(family, socktype, proto)
            try:
                sock.settimeout(timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                for option in socket_options or ():
                    sock.setsockopt(*option)
                # address is a numeric sockaddr from the validated DNS snapshot. This
                # does not invoke getaddrinfo again, even if DNS changes immediately.
                sock.connect(address)
                return _SocketStream(sock)
            except OSError as exc:
                sock.close()
                error = exc
        if isinstance(error, TimeoutError):
            raise httpcore.ConnectTimeout(str(error)) from error
        raise httpcore.ConnectError(str(error)) from error

    def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options=None) -> httpcore.NetworkStream:
        raise UnsafeURL("Unix sockets are not supported")


@contextmanager
def _http_errors() -> Iterator[None]:
    try:
        yield
    except httpcore.TimeoutException as exc:
        raise httpx.TimeoutException(str(exc)) from exc
    except httpcore.NetworkError as exc:
        raise httpx.NetworkError(str(exc)) from exc
    except httpcore.ProtocolError as exc:
        raise httpx.ProtocolError(str(exc)) from exc


class _ResponseStream(httpx.SyncByteStream):
    def __init__(self, response: httpcore.Response) -> None:
        self._response = response

    def __iter__(self) -> Iterator[bytes]:
        with _http_errors():
            yield from self._response.iter_stream()

    def close(self) -> None:
        with _http_errors():
            self._response.close()


class PublicHTTPTransport(httpx.BaseTransport):
    """HTTPX adapter with public-only pinned sockets and normal hostname-verified TLS."""

    def __init__(self) -> None:
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl.create_default_context(), network_backend=_PublicNetworkBackend(),
            # Every redirect/request gets a fresh, validated resolution, including
            # same-host redirects. TLS always receives the original hostname.
            max_keepalive_connections=0,
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = validate_public_url(request.url)
        # Do not accept a caller-supplied SNI override or alternate Host authority.
        extensions = {key: value for key, value in request.extensions.items() if key != "sni_hostname"}
        headers = [(key, value) for key, value in request.headers.raw if key.lower() != b"host"]
        headers.insert(0, (b"Host", url.netloc))
        with _http_errors():
            response = self._pool.handle_request(httpcore.Request(
                method=request.method, url=httpcore.URL(scheme=url.raw_scheme, host=url.raw_host,
                                                       port=url.port, target=url.raw_path),
                headers=headers, content=request.stream, extensions=extensions,
            ))
        return httpx.Response(response.status, headers=response.headers, stream=_ResponseStream(response),
                              extensions=response.extensions)

    def close(self) -> None:
        self._pool.close()
