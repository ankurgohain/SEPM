"""
websocket_handler.py
====================
WebSocket endpoint: ws://host/ws/stream

Enables the LMS instructor dashboard to receive real-time prediction
events without polling.  Each connected client can:

  1. Send a SESSION event with a learner's latest session data
     → receives back a PREDICTION event with LSTM output.

  2. Send a PING → receives a PONG (keep-alive).

  3. Subscribe to the broadcast channel (INTERVENTION events are
     automatically pushed to all connected clients whenever a
     high-risk learner is scored via any channel).

Message format (JSON, both directions)
──────────────────────────────────────
Incoming:
  { "event": "prediction",
    "learner_id": "...",
    "sessions": [ { ...LearnerSession... }, ... ] }

  { "event": "ping" }

Outgoing:
  { "event": "prediction",  "payload": { ...PredictResponse... } }
  { "event": "intervention","payload": { learner_id, intervention, ... } }
  { "event": "pong",        "payload": {} }
  { "event": "error",       "payload": { "detail": "..." } }
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from src.api.model_registry import ModelRegistry, get_registry
from src.api.routers.intervention import store_prediction
from src.api.schemas import (
    LearnerSession,
    ModuleID,
    WSEventType,
    WSIncomingMessage,
)

logger = logging.getLogger("learnflow.ws")

router = APIRouter(tags=["WebSocket"])

SEQ_LEN      = 10
NUM_FEATURES = 6


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Tracks all active WebSocket connections and provides
    broadcast / targeted send helpers.
    """

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}   # connection_id → ws

    async def connect(self, connection_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections[connection_id] = ws
        logger.info("WS connected: %s (total=%d)", connection_id, len(self._connections))

    def disconnect(self, connection_id: str) -> None:
        self._connections.pop(connection_id, None)
        logger.info("WS disconnected: %s (total=%d)", connection_id, len(self._connections))

    async def send(self, connection_id: str, message: dict) -> None:
        ws = self._connections.get(connection_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as exc:
                logger.warning("Send failed (%s): %s", connection_id, exc)

    async def broadcast(self, message: dict) -> None:
        """Push a message to every connected client."""
        dead = []
        for cid, ws in self._connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self.disconnect(cid)

    @property
    def count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sessions_to_arrays(
    sessions: list[LearnerSession],
) -> tuple[np.ndarray, np.ndarray]:
    """Same pre-padding logic as the REST route."""
    T = len(sessions)
    num_raw = np.zeros((SEQ_LEN, NUM_FEATURES), dtype=np.float32)
    cat_raw = np.zeros((SEQ_LEN,),              dtype=np.int32)
    offset  = SEQ_LEN - T

    for i, s in enumerate(sessions):
        idx = offset + i
        num_raw[idx] = [
            s.quiz_score,
            s.engagement_rate,
            s.hint_count,
            s.session_duration,
            s.correct_attempts,
            s.incorrect_attempts,
        ]
        cat_raw[idx] = int(s.module_id)

    return num_raw, cat_raw


async def _run_prediction(
    registry:   ModelRegistry,
    learner_id: str,
    sessions:   list[LearnerSession],
) -> dict:
    """Run inference and return a serialisable result dict."""
    num_seq, cat_seq = _sessions_to_arrays(sessions)
    preds = registry.predict(
        num_sequences=num_seq[np.newaxis],
        cat_sequences=cat_seq[np.newaxis],
    )
    result = preds[0]
    result["learner_id"] = learner_id
    return result


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/stream")
async def websocket_stream(
    ws:       WebSocket,
    registry: ModelRegistry = Depends(get_registry),
):
    """
    WebSocket endpoint for real-time prediction streaming.

    Connect with:  ws://localhost:8000/ws/stream
    Auth:          pass ?token=<api_key> as a query parameter
                   (header-based auth is not possible in the WS handshake)

    The connection is assigned a server-side UUID. The client does not
    need to manage connection IDs — they are echoed in every response
    payload for debugging purposes.
    """
    import uuid
    connection_id = str(uuid.uuid4())[:8]

    await manager.connect(connection_id, ws)

    # Welcome message
    await ws.send_json({
        "event": "connected",
        "payload": {
            "connection_id": connection_id,
            "message":       "LearnFlow LSTM stream ready.",
            "model_version": registry.version,
            "active_connections": manager.count,
        },
    })

    try:
        while True:
            raw = await ws.receive_text()

            # ── Parse incoming message ───────────────────────────────────
            try:
                data = json.loads(raw)
                msg  = WSIncomingMessage(**data)
            except (json.JSONDecodeError, ValidationError) as exc:
                await manager.send(connection_id, {
                    "event":   WSEventType.error,
                    "payload": {"detail": f"Invalid message: {exc}"},
                })
                continue

            # ── PING / PONG ──────────────────────────────────────────────
            if msg.event == WSEventType.ping:
                await manager.send(connection_id, {
                    "event":   WSEventType.pong,
                    "payload": {"connection_id": connection_id},
                })
                continue

            # ── PREDICTION ───────────────────────────────────────────────
            if msg.event == WSEventType.prediction:
                if not msg.learner_id or not msg.sessions:
                    await manager.send(connection_id, {
                        "event":   WSEventType.error,
                        "payload": {"detail": "prediction event requires learner_id and sessions"},
                    })
                    continue

                if not registry.is_loaded:
                    await manager.send(connection_id, {
                        "event":   WSEventType.error,
                        "payload": {"detail": "Model not ready"},
                    })
                    continue

                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: asyncio.run(_run_prediction(
                            registry, msg.learner_id, msg.sessions
                        ))
                    )
                except Exception as exc:
                    logger.exception("WS prediction failed for %s", msg.learner_id)
                    await manager.send(connection_id, {
                        "event":   WSEventType.error,
                        "payload": {"detail": str(exc)},
                    })
                    continue

                # Cache for the intervention route
                store_prediction(msg.learner_id, result)

                await manager.send(connection_id, {
                    "event":   WSEventType.prediction,
                    "payload": result,
                })

                # Broadcast high-risk interventions to all connected dashboards
                if result.get("dropout_tier") in ("high", "medium"):
                    await manager.broadcast({
                        "event": WSEventType.intervention,
                        "payload": {
                            "learner_id":    msg.learner_id,
                            "intervention":  result["intervention"],
                            "dropout_tier":  result["dropout_tier"],
                            "dropout_risk":  result["dropout_risk"],
                            "mastery_prob":  result["mastery_prob"],
                            "source":        "websocket",
                        },
                    })
                continue

            # ── Unknown event ────────────────────────────────────────────
            await manager.send(connection_id, {
                "event":   WSEventType.error,
                "payload": {"detail": f"Unknown event type: {msg.event}"},
            })

    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as exc:
        logger.exception("Unexpected WS error for connection %s", connection_id)
        manager.disconnect(connection_id)