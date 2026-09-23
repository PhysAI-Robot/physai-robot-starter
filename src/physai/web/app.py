"""FastAPI application factory for the browser viewer."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .actions import action_from_payload
from .host import Host
from .telemetry import build_mesh_payload

_CAMERA_STREAM_PERIOD = 1.0 / 30  # matches Host._CAMERA_PERIOD's capture cadence


def create_app(*, host: Host):
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, Response, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError(
            "web dependencies are missing; install with `uv sync --extra web`"
        ) from exc

    static_dir = Path(__file__).with_name("static")
    assets_dir = Path(__file__).resolve().parents[3] / "assets"

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            # The process that created the host owns its lifecycle.
            pass

    app = FastAPI(title="PhysAI Web Viewer", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/api/robots")
    async def robots() -> list[dict[str, Any]]:
        return host.list_robots()

    @app.get("/api/scene")
    async def scene(robot: str | None = None) -> dict[str, Any]:
        del robot
        return host.scene()

    @app.get("/api/state")
    async def state(robot: str | None = None) -> dict[str, Any] | None:
        del robot
        return host.latest_state()

    @app.get("/api/mesh/{mesh_id}")
    async def mesh(mesh_id: int, robot: str | None = None) -> Response:
        del robot
        return Response(
            build_mesh_payload(host.model, mesh_id),
            media_type="application/octet-stream",
        )

    @app.get("/api/camera/{name}.jpg")
    async def camera(name: str, robot: str | None = None) -> Response:
        return Response(
            host.camera_jpeg(name, instance_id=robot), media_type="image/jpeg"
        )

    @app.get("/api/camera/{name}/stream")
    async def camera_stream(name: str, robot: str | None = None) -> StreamingResponse:
        try:
            host.camera_jpeg(name, instance_id=robot)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def frames():
            while True:
                try:
                    frame = host.camera_jpeg(name, instance_id=robot)
                except ValueError:
                    return
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
                await asyncio.sleep(_CAMERA_STREAM_PERIOD)

        return StreamingResponse(
            frames(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        errors: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        active_robot = host.default_instance_id
        client_id = uuid.uuid4().hex

        async def receive_commands() -> None:
            try:
                while True:
                    message = await websocket.receive_json()
                    if message.get("type") == "select_robot":
                        if message.get("robot") not in host.instances:
                            await errors.put(
                                {"type": "error", "message": "unknown robot"}
                            )
                        else:
                            nonlocal active_robot
                            active_robot = message["robot"]
                    elif message.get("type") == "command":
                        try:
                            target = message.get("robot", active_robot)
                            if target != active_robot:
                                raise ValueError("command target is not selected")
                            action = action_from_payload(message["action"])
                            host.submit(action, instance_id=target, source=client_id)
                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                            PermissionError,
                        ) as exc:
                            await errors.put({"type": "error", "message": str(exc)})
                    elif message.get("type") == "reset":
                        host.reset()
                    elif message.get("type") == "pause":
                        host.set_paused(bool(message.get("value", True)))
                    elif message.get("type") == "release_control":
                        host.release_control(client_id)
            except WebSocketDisconnect:
                return

        receiver = asyncio.create_task(receive_commands())
        try:
            while True:
                state = host.latest_state()
                if state is not None:
                    await websocket.send_json(state)
                while not errors.empty():
                    await websocket.send_json(errors.get_nowait())
                await asyncio.sleep(0.01)
        except WebSocketDisconnect:
            pass
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            host.release_control(client_id)

    return app
