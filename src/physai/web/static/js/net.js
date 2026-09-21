let socket = null;
const handlers = { open: null, state: null, error: null, close: null };

export function on(event, callback) {
  handlers[event] = callback;
}

export function isOpen() {
  return !!socket && socket.readyState === WebSocket.OPEN;
}

export function send(payload) {
  if (isOpen()) socket.send(JSON.stringify(payload));
}

export function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onopen = () => handlers.open?.();
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "error") {
      handlers.error?.(message.message);
      return;
    }
    handlers.state?.(message);
  };
  socket.onclose = () => {
    handlers.close?.();
    setTimeout(connect, 1000);
  };
}

window.addEventListener("beforeunload", () => {
  if (isOpen()) send({ type: "release_control" });
});
