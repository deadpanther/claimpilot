import httpx
import pytest
import respx

from claimpilot.clients.attachment_guard import fetch_attachment, validate_attachment_url
from claimpilot.config import settings


def test_validate_attachment_url_rejects_disallowed_host():
    with pytest.raises(ValueError, match="not allowlisted"):
        validate_attachment_url("http://169.254.169.254/secret.png")


def test_validate_attachment_url_accepts_allowlisted_host():
    validate_attachment_url(f"https://{settings.allowed_attachment_host}/shipbob-fde-mock/case-1001/x.png")


async def test_fetch_attachment_rejects_disallowed_host_without_network_call():
    with respx.mock(assert_all_called=False):
        with pytest.raises(ValueError, match="not allowlisted"):
            await fetch_attachment("https://evil.example.com/payload.png")


async def test_fetch_attachment_rejects_oversized_body():
    url = f"https://{settings.allowed_attachment_host}/shipbob-fde-mock/case-1001/huge.png"
    oversized_content = b"x" * (settings.max_attachment_bytes + 1024)
    with respx.mock:
        respx.get(url).mock(
            return_value=httpx.Response(
                200, headers={"content-type": "image/png"}, content=oversized_content
            )
        )
        with pytest.raises(ValueError, match="exceeds max size"):
            await fetch_attachment(url)


async def test_fetch_attachment_rejects_non_image_content_type():
    url = f"https://{settings.allowed_attachment_host}/shipbob-fde-mock/case-1001/not-an-image.txt"
    with respx.mock:
        respx.get(url).mock(
            return_value=httpx.Response(200, headers={"content-type": "text/plain"}, content=b"hello")
        )
        with pytest.raises(ValueError, match="content-type"):
            await fetch_attachment(url)


# --- path traversal / unsafe cache key guardrails ---------------------------
#
# attachment_id feeds directly into a filesystem path under
# fixtures/images/<attachment_id>. It's collection-sourced today, but
# fetch_attachment() is designed to be reused by a real httpx
# client, where attachment metadata comes from a live (untrusted) API
# response -- so a malicious attachment_id must never be able to escape the
# cache directory.


async def test_fetch_attachment_rejects_relative_path_traversal_attachment_id():
    url = f"https://{settings.allowed_attachment_host}/shipbob-fde-mock/case-1001/x.png"
    with respx.mock(assert_all_called=False):
        with pytest.raises(ValueError, match="unsafe attachment_id"):
            await fetch_attachment(url, attachment_id="../../etc/passwd")


async def test_fetch_attachment_rejects_absolute_path_attachment_id():
    url = f"https://{settings.allowed_attachment_host}/shipbob-fde-mock/case-1001/x.png"
    with respx.mock(assert_all_called=False):
        with pytest.raises(ValueError, match="unsafe attachment_id"):
            await fetch_attachment(url, attachment_id="/etc/passwd")


async def test_fetch_attachment_rejects_backslash_attachment_id():
    url = f"https://{settings.allowed_attachment_host}/shipbob-fde-mock/case-1001/x.png"
    with respx.mock(assert_all_called=False):
        with pytest.raises(ValueError, match="unsafe attachment_id"):
            await fetch_attachment(url, attachment_id="..\\..\\windows\\system32")


async def test_fetch_attachment_rejects_dot_dot_attachment_id():
    url = f"https://{settings.allowed_attachment_host}/shipbob-fde-mock/case-1001/x.png"
    with respx.mock(assert_all_called=False):
        with pytest.raises(ValueError, match="unsafe attachment_id"):
            await fetch_attachment(url, attachment_id="..")
