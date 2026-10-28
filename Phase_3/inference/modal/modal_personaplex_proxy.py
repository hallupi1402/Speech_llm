"""Local authenticated HTTP/WebSocket proxy for the deployed PersonaPlex UI.

Run with:
    uv run --with aiohttp python modal_personaplex_proxy.py

The Modal Proxy Token is requested interactively and kept in this process only.
The local server binds to loopback so the token is not exposed as a public service.
"""

from __future__ import annotations

import asyncio
from getpass import getpass

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web


REMOTE_BASE = (
    "https://hyennguyen140205--personaplex-serve-personaplex.us-east.modal.run"
)
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8765
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


async def proxy_request(request: web.Request) -> web.StreamResponse:
    if request.host not in {f"{LOCAL_HOST}:{LOCAL_PORT}", f"localhost:{LOCAL_PORT}"}:
        raise web.HTTPMisdirectedRequest()

    session: ClientSession = request.app["session"]
    auth_headers: dict[str, str] = request.app["auth_headers"]
    remote_url = f"{REMOTE_BASE}{request.rel_url}"

    if request.headers.get("Upgrade", "").lower() == "websocket":
        origin = request.headers.get("Origin", "")
        allowed_origins = {
            f"http://{LOCAL_HOST}:{LOCAL_PORT}",
            f"http://localhost:{LOCAL_PORT}",
        }
        if origin not in allowed_origins:
            raise web.HTTPForbidden(text="Unexpected WebSocket origin")

        local_ws = web.WebSocketResponse(heartbeat=30, max_msg_size=0)
        await local_ws.prepare(request)

        try:
            async with session.ws_connect(
                remote_url,
                headers={**auth_headers, "Origin": origin},
                heartbeat=30,
                max_msg_size=0,
            ) as remote_ws:

                async def relay(source: web.WebSocketResponse, destination) -> None:
                    async for message in source:
                        if message.type == WSMsgType.TEXT:
                            await destination.send_str(message.data)
                        elif message.type == WSMsgType.BINARY:
                            await destination.send_bytes(message.data)
                        elif message.type == WSMsgType.PING:
                            await destination.ping()
                        elif message.type == WSMsgType.PONG:
                            await destination.pong()
                        elif message.type in (
                            WSMsgType.CLOSE,
                            WSMsgType.CLOSED,
                            WSMsgType.ERROR,
                        ):
                            break

                tasks = [
                    asyncio.create_task(relay(local_ws, remote_ws)),
                    asyncio.create_task(relay(remote_ws, local_ws)),
                ]
                _, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:
            print(f"WebSocket proxy error: {type(exc).__name__}: {exc}")
            await local_ws.close(code=1011, message=b"Upstream WebSocket failed")
        finally:
            if not local_ws.closed:
                await local_ws.close()
        return local_ws

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
        and key.lower() not in {"host", "modal-key", "modal-secret", "authorization"}
    }
    headers.update(auth_headers)

    async with session.request(
        request.method,
        remote_url,
        headers=headers,
        data=await request.read(),
        allow_redirects=False,
    ) as upstream:
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        return web.Response(
            body=await upstream.read(),
            status=upstream.status,
            headers=response_headers,
        )


async def main() -> None:
    token_id = input("Modal Proxy Token ID (wk-...): ").strip()
    token_secret = getpass("Modal Proxy Token Secret (ws-...): ").strip()
    if not token_id or not token_secret:
        raise SystemExit("Both Proxy Token ID and Secret are required.")

    app = web.Application(client_max_size=0)
    app["auth_headers"] = {
        "Modal-Key": token_id,
        "Modal-Secret": token_secret,
    }
    app["session"] = ClientSession(
        timeout=ClientTimeout(total=None, sock_connect=30),
        auto_decompress=False,
    )
    app.router.add_route("*", "/{path:.*}", proxy_request)

    async def close_session(application: web.Application) -> None:
        await application["session"].close()

    app.on_cleanup.append(close_session)
    print(f"Open http://{LOCAL_HOST}:{LOCAL_PORT} in your browser. Press Ctrl+C to stop.")
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, LOCAL_HOST, LOCAL_PORT)
    await site.start()
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
