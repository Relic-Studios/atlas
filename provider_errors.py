"""Safe, structured provider failures shared by setup and live inference."""
from __future__ import annotations

import requests


class ProviderError(requests.RequestException):
    def __init__(self, provider: str, code: str, message: str, *, reset_at=None, retry_after=None):
        super().__init__(message)
        self.provider, self.code = provider, code
        self.reset_at, self.retry_after = reset_at, retry_after

    def public(self):
        return {"provider": self.provider, "code": self.code, "message": str(self),
                "reset_at": self.reset_at, "retry_after": self.retry_after,
                "retryable": self.code in {"usage_limit", "offline", "timeout"}}


def failure(provider: str, detail, *, status=None, reset_at=None, retry_after=None):
    """Inspect untrusted diagnostics, but expose only fixed, credential-free messages."""
    text = str(detail).lower()
    if status == 429 or any(s in text for s in ("usage limit", "rate limit", "rate_limit", "usage_limit", "usagelimitexceeded", "ratelimitexceeded", "quota exceeded", "hit your limit")):
        return ProviderError(provider, "usage_limit", "This account has reached its usage limit. Wait for it to reset or choose another provider. Atlas has not changed billing or retried the request.", reset_at=reset_at, retry_after=retry_after)
    if status in (401, 403) or any(s in text for s in ("not logged in", "not authenticated", "authentication", "unauthorized", "login required", "sign in")):
        return ProviderError(provider, "sign_in_required", "Sign in with the provider's Connect button, then verify the account again.")
    if any(s in text for s in ("model_not_found", "model not found", "model is not supported", "model access", "modelnotfound")):
        return ProviderError(provider, "model_access", "This account cannot use the selected model. Refresh models or choose a different model.")
    if any(s in text for s in ("timeout", "timed out")):
        return ProviderError(provider, "timeout", "The provider did not finish in time. Retry explicitly when you are ready.")
    if any(s in text for s in ("connection", "network", "dns", "offline", "econnrefused")):
        return ProviderError(provider, "offline", "The provider is unreachable. Check the connection, then retry.")
    return ProviderError(provider, "provider_error", "The provider rejected this request. Check the account and selected model, then retry explicitly.")
