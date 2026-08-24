"""Resolution must fall through to the local build when the bucket is unusable.

A DNS failure on the (placeholder) distribution bucket used to escape
`_download_from_s3` as `httpx.ConnectError`, killing injection with
`SandboxInjectionError` before the local-build fallback could run.
"""

import httpx
import pytest

from inspect_ai.tool._sandbox_tools_utils import sandbox as sandbox_mod


async def test_default_url_attempts_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-tree default URL is a REAL distribution point and must be tried.

    (A previous revision short-circuited on the default when it was a
    placeholder; that guard must never come back now that the default works.)
    """
    requested: list[str] = []

    class _FakeResponse:
        status_code = 404
        content = b""

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("GET", "https://x"),
                response=httpx.Response(404),
            )

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None: ...
        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None: ...
        async def get(self, url: str) -> _FakeResponse:
            requested.append(url)
            return _FakeResponse()

    # NOT overriding _BUCKET_BASE_URL: the point is that the DEFAULT gets tried.
    assert sandbox_mod._BUCKET_BASE_URL == sandbox_mod._DEFAULT_BUCKET_BASE_URL
    monkeypatch.setattr("httpx.AsyncClient", _FakeClient)
    assert (
        await sandbox_mod._download_from_s3("inspect-sandbox-tools-amd64-v26-tl1")
        is False  # 404 degrades gracefully...
    )
    assert requested == [
        f"{sandbox_mod._DEFAULT_BUCKET_BASE_URL}/inspect-sandbox-tools-amd64-v26-tl1"
    ]  # ...but the network WAS attempted against the default URL.


async def test_unreachable_bucket_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport errors (DNS, refused, timeout) mean 'not available', not fatal."""
    monkeypatch.setattr(
        sandbox_mod,
        "_BUCKET_BASE_URL",
        "https://definitely-not-resolvable.invalid",
    )
    assert (
        await sandbox_mod._download_from_s3("inspect-sandbox-tools-amd64-v26-tl1")
        is False
    )


async def test_http_500_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-404/403 HTTP errors keep raising: the bucket exists but is broken."""

    class _FakeResponse:
        status_code = 500
        content = b""

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "boom",
                request=httpx.Request("GET", "https://x"),
                response=httpx.Response(500),
            )

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None: ...
        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None: ...
        async def get(self, url: str) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr(sandbox_mod, "_BUCKET_BASE_URL", "https://bucket.example")
    monkeypatch.setattr("httpx.AsyncClient", _FakeClient)
    with pytest.raises(httpx.HTTPStatusError):
        await sandbox_mod._download_from_s3("inspect-sandbox-tools-amd64-v26-tl1")
