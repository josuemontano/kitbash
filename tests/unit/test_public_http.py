import io
import socket
import ssl

import pytest
from PIL import Image

from kitbash.infra.image_search import Candidate, ImageDownloader, make_http_client
from kitbash.infra.public_http import UnsafeURL

PUBLIC_IP = "93.184.216.34"


def dns_answer(address, port=80):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
    return family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr


def image_response() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 80, 120)).save(buffer, "PNG")
    body = buffer.getvalue()
    return b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body


class SocketPeer:
    """A wire-level peer; HTTP parsing, transport, redirects, and decoding stay real."""

    def __init__(self, response):
        self.response = response
        self.connected = None
        self.written = b""
        self.closed = False

    def settimeout(self, timeout):
        pass

    def setsockopt(self, *args):
        pass

    def connect(self, address):
        self.connected = address

    def recv(self, maximum):
        result, self.response = self.response[:maximum], self.response[maximum:]
        return result

    def sendall(self, data):
        self.written += data

    def close(self):
        self.closed = True

    def getsockname(self):
        return "192.0.2.10", 50000

    def getpeername(self):
        return self.connected


def install_peers(monkeypatch, responses):
    peers = []
    pending = iter(responses)

    def create_socket(*args):
        peer = SocketPeer(next(pending))
        peers.append(peer)
        return peer

    monkeypatch.setattr(socket, "socket", create_socket)
    return peers


@pytest.mark.parametrize("addresses", [
    ["127.0.0.1"], ["10.0.0.1"], ["172.16.0.1"], ["192.168.0.1"], ["169.254.169.254"], ["100.64.0.1"],
    ["224.0.0.1"], ["240.0.0.1"], ["0.0.0.0"], ["192.0.2.1"], ["::1"], ["fe80::1"], ["fc00::1"], ["ff02::1"],
    [PUBLIC_IP, "127.0.0.1"], [PUBLIC_IP, "::1"], ["::ffff:127.0.0.1"], ["64:ff9b::7f00:1"],
])
def test_dns_nonpublic_or_mixed_answers_never_connect(monkeypatch, addresses):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kwargs: [dns_answer(address, port) for address in addresses])
    monkeypatch.setattr(socket, "socket", lambda *args: pytest.fail("Forbidden DNS answers reached socket creation"))
    with make_http_client(2) as client, pytest.raises(UnsafeURL, match="DNS"):
        client.get("http://images.example/image.png")


@pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f000001"])
def test_alternate_ipv4_spellings_are_checked_after_resolution(monkeypatch, host):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kwargs: [dns_answer("127.0.0.1", port)])
    monkeypatch.setattr(socket, "socket", lambda *args: pytest.fail("Noncanonical loopback reached a socket"))
    with make_http_client(2) as client, pytest.raises(UnsafeURL):
        client.get(f"http://{host}/image.png")


def test_rebinding_cannot_change_validated_connect_address(monkeypatch, tmp_path):
    resolutions = []

    def resolve(host, port, **kwargs):
        resolutions.append(host)
        # A second resolver call would rebind even the original hostname to loopback.
        return [dns_answer(PUBLIC_IP if len(resolutions) == 1 else "127.0.0.1", port)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    peers = install_peers(monkeypatch, [image_response()])
    with make_http_client(2) as client:
        result = ImageDownloader(client, max_bytes=10000, min_side=16).fetch(
            Candidate("web", "lamp", "http://images.example/image.png"), tmp_path / "lamp.png")
    assert result is not None and (result.width, result.height) == (64, 48)
    assert resolutions == ["images.example"]
    assert peers[0].connected == (PUBLIC_IP, 80)
    assert b"Host: images.example\r\n" in peers[0].written
    assert peers[0].closed


def test_same_host_redirect_revalidates_dns_and_blocks_rebinding(monkeypatch, tmp_path):
    resolutions = []

    def resolve(host, port, **kwargs):
        resolutions.append(host)
        return [dns_answer(PUBLIC_IP if len(resolutions) == 1 else "127.0.0.1", port)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    peers = install_peers(monkeypatch, [b"HTTP/1.1 302 Found\r\nLocation: /final\r\nContent-Length: 0\r\n\r\n"])
    with make_http_client(2) as client:
        result = ImageDownloader(client, max_bytes=10000, min_side=16).fetch(
            Candidate("web", "lamp", "http://images.example/start"), tmp_path / "lamp.png")
    assert result is None
    assert resolutions == ["images.example", "images.example"]
    assert len(peers) == 1 and peers[0].connected == (PUBLIC_IP, 80)
    assert not (tmp_path / "lamp.png").exists()


def test_network_client_does_not_follow_api_redirects_or_environment_proxies(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:9999")
    hosts = []

    def resolve(host, port, **kwargs):
        hosts.append(host)
        return [dns_answer(PUBLIC_IP, port)]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    peers = install_peers(monkeypatch, [b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/private\r\nContent-Length: 0\r\n\r\n"])
    with make_http_client(2) as client:
        response = client.get("http://images.example/api")
    assert response.status_code == 302
    assert hosts == ["images.example"]
    assert len(peers) == 1 and peers[0].connected == (PUBLIC_IP, 80)


def test_https_retains_hostname_verification_and_rejects_authority_overrides(monkeypatch):
    peers = install_peers(monkeypatch, [image_response()])
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kwargs: [dns_answer(PUBLIC_IP, port)])
    tls_names = []

    def verified_wrap(context, sock, *, server_hostname):
        assert context.check_hostname is True
        assert context.verify_mode == ssl.CERT_REQUIRED
        if server_hostname != "images.example":
            raise ssl.SSLCertVerificationError("certificate does not match requested hostname")
        tls_names.append(server_hostname)
        return sock

    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", verified_wrap)
    with make_http_client(2) as client:
        response = client.get("https://images.example/image.png", headers={"Host": "attacker.example"},
                              extensions={"sni_hostname": "attacker.example"})
    assert response.status_code == 200
    assert tls_names == ["images.example"]
    assert peers[0].connected == (PUBLIC_IP, 443)
    assert b"Host: images.example\r\n" in peers[0].written
    assert b"attacker.example" not in peers[0].written


def test_certificate_failure_rejects_download_and_closes_socket(monkeypatch, tmp_path):
    peers = install_peers(monkeypatch, [image_response()])
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kwargs: [dns_answer(PUBLIC_IP, port)])

    def invalid_certificate(context, sock, *, server_hostname):
        raise ssl.SSLCertVerificationError("untrusted certificate")

    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", invalid_certificate)
    with make_http_client(2) as client:
        result = ImageDownloader(client, max_bytes=10000, min_side=16).fetch(
            Candidate("web", "lamp", "https://images.example/image.png"), tmp_path / "lamp.png")
    assert result is None
    assert peers[0].closed
    assert not (tmp_path / "lamp.png").exists()
