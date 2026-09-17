"""FastAPI application factory for the browser viewer."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..robots.registry import create_robot
from .runtime import SimulationSession, action_from_payload
from .telemetry import build_mesh_payload


def create_app(
    *,
    robot_name: str = "so101",
    robot_names: tuple[str, ...] | None = None,
    robot: Any = None,
    demo: bool = False,
):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, Response
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError(
            "web dependencies are missing; install with `uv sync --extra web`"
        ) from exc

    names = tuple(dict.fromkeys(robot_names or (robot_name,)))
    if not names:
        raise ValueError("at least one robot must be configured")
    if robot is not None and len(names) != 1:
        raise ValueError("an injected robot can only be used with one robot name")

    sessions: dict[str, SimulationSession] = {}
    for name in names:
        configured_robot = robot if robot is not None else None
        if configured_robot is None:
            from ..robots.registry import create_env_config

            config = create_env_config(name, render=True)
            if hasattr(config, "camera_stride"):
                config = replace(config, camera_stride=5)
            configured_robot = create_robot(name, config=config)
        policy = None
        if demo:
            if name != "so101" or len(names) != 1:
                raise ValueError("--demo currently supports only one robot: so101")
            from ..robots.so101.expert import SO101PickPlaceExpert
            from ..tasks import TaskRuntime, create_task

            configured_robot = TaskRuntime(configured_robot, create_task("pick_place"))
            policy = SO101PickPlaceExpert(configured_robot.kin, configured_robot)
        sessions[name] = SimulationSession(
            configured_robot, robot_name=name, policy=policy
        )
    static_dir = Path(__file__).with_name("static")
    assets_dir = Path(__file__).resolve().parents[3] / "assets"
    app = FastAPI(title="PhysAI Web Viewer")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    def get_session(name: str | None) -> SimulationSession:
        selected = name or names[0]
        try:
            return sessions[selected]
        except KeyError as exc:
            raise ValueError(f"unknown robot {selected!r}") from exc

    @app.on_event("startup")
    async def start_session() -> None:
        for session in sessions.values():
            session.start()

    @app.on_event("shutdown")
    async def stop_session() -> None:
        for session in sessions.values():
            session.stop()

    @app.get("/api/robots")
    async def robots() -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "kind": session.robot.robot_spec.kind,
                "action_modes": list(session.robot.robot_spec.action_modes),
                "capabilities": list(session.robot.robot_spec.capabilities),
                "cameras": list(session.robot.robot_spec.camera_frames),
            }
            for name, session in sessions.items()
        ]

    @app.get("/api/scene")
    async def scene(robot: str | None = None) -> dict[str, Any]:
        return get_session(robot).scene()

    @app.get("/api/state")
    async def state(robot: str | None = None) -> dict[str, Any] | None:
        return get_session(robot).latest_state()

    @app.get("/api/mesh/{mesh_id}")
    async def mesh(mesh_id: int, robot: str | None = None) -> Response:
        selected = get_session(robot)
        return Response(
            build_mesh_payload(selected.model, mesh_id),
            media_type="application/octet-stream",
        )

    @app.get("/api/camera/{name}.jpg")
    async def camera(name: str, robot: str | None = None) -> Response:
        return Response(get_session(robot).camera_jpeg(name), media_type="image/jpeg")

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        errors: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        active_robot = names[0]

        async def receive_commands() -> None:
            try:
                while True:
                    message = await websocket.receive_json()
                    if message.get("type") == "select_robot":
                        if message.get("robot") not in sessions:
                            await errors.put(
                                {"type": "error", "message": "unknown robot"}
                            )
                        else:
                            nonlocal active_robot
                            active_robot = message["robot"]
                    elif message.get("type") == "command":
                        try:
                            selected = get_session(message.get("robot", active_robot))
                            selected.submit(action_from_payload(message["action"]))
                        except (KeyError, TypeError, ValueError) as exc:
                            await errors.put({"type": "error", "message": str(exc)})
                    elif message.get("type") == "reset":
                        get_session(message.get("robot", active_robot)).reset()
                    elif message.get("type") == "pause":
                        get_session(message.get("robot", active_robot)).set_paused(
                            bool(message.get("value", True))
                        )
            except WebSocketDisconnect:
                return

        receiver = asyncio.create_task(receive_commands())
        try:
            while True:
                state = sessions[active_robot].latest_state()
                if state is not None:
                    await websocket.send_json(state)
                while not errors.empty():
                    await websocket.send_json(errors.get_nowait())
                await asyncio.sleep(0.01)
        except WebSocketDisconnect:
            pass
        finally:
            receiver.cancel()

    return app
