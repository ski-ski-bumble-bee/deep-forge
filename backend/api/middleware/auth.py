import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

PASSWORD = os.environ.get("FORGE_PASSWORD", "")   # empty = auth disabled

class PasswordMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not PASSWORD:                            # disabled if no password set
            return await call_next(request)
        # Allow preflight
        if request.method == "OPTIONS":
            return await call_next(request)
        token = (
            request.headers.get("X-Forge-Password") or
            request.query_params.get("password")
        )
        if token != PASSWORD:
            raise HTTPException(401, "Unauthorized")
        return await call_next(request)
