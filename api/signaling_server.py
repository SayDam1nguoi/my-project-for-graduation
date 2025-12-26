"""
WebRTC Signaling Server for Video Call

Server để kết nối 2 người với nhau qua WebRTC.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Set
import json
import logging
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Call Signaling Server",
    description="WebRTC signaling server for emotion analysis video calls",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
# Format: {room_id: {user_id: websocket}}
rooms: Dict[str, Dict[str, WebSocket]] = {}

# Store room metadata
# Format: {room_id: {created_at, users: [user_ids]}}
room_metadata: Dict[str, dict] = {}


@app.get("/")
def root():
    """Health check."""
    return {
        "message": "Video Call Signaling Server",
        "status": "running",
        "active_rooms": len(rooms),
        "total_users": sum(len(users) for users in rooms.values())
    }


@app.get("/api/rooms")
def list_rooms():
    """List all active rooms."""
    return {
        "rooms": [
            {
                "room_id": room_id,
                "users": len(users),
                "created_at": room_metadata.get(room_id, {}).get("created_at")
            }
            for room_id, users in rooms.items()
        ]
    }


@app.websocket("/ws/{room_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user_id: str):
    """
    WebSocket endpoint for signaling.
    
    Messages format:
    - offer: WebRTC offer from caller
    - answer: WebRTC answer from callee
    - ice-candidate: ICE candidate for connection
    - emotion-data: Emotion analysis data
    """
    await websocket.accept()
    
    # Create room if not exists
    if room_id not in rooms:
        rooms[room_id] = {}
        room_metadata[room_id] = {
            "created_at": datetime.now().isoformat(),
            "users": []
        }
        logger.info(f"Created room: {room_id}")
    
    # Add user to room
    rooms[room_id][user_id] = websocket
    room_metadata[room_id]["users"].append(user_id)
    
    logger.info(f"User {user_id} joined room {room_id}")
    logger.info(f"Room {room_id} now has {len(rooms[room_id])} users")
    
    # Notify other users in room
    await broadcast_to_room(
        room_id,
        {
            "type": "user-joined",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        },
        exclude_user=user_id
    )
    
    # Send current room state to new user
    await websocket.send_json({
        "type": "room-state",
        "room_id": room_id,
        "users": [uid for uid in rooms[room_id].keys() if uid != user_id],
        "timestamp": datetime.now().isoformat()
    })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            target_user = message.get("target")
            
            logger.info(f"Received {message_type} from {user_id} to {target_user}")
            
            # Forward message to target user
            if target_user and target_user in rooms[room_id]:
                await rooms[room_id][target_user].send_json({
                    **message,
                    "from": user_id,
                    "timestamp": datetime.now().isoformat()
                })
            elif message_type == "emotion-data":
                # Broadcast emotion data to all users in room
                await broadcast_to_room(
                    room_id,
                    {
                        **message,
                        "from": user_id,
                        "timestamp": datetime.now().isoformat()
                    },
                    exclude_user=user_id
                )
            
    except WebSocketDisconnect:
        logger.info(f"User {user_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"Error in websocket: {e}")
    finally:
        # Remove user from room
        if room_id in rooms and user_id in rooms[room_id]:
            del rooms[room_id][user_id]
            room_metadata[room_id]["users"].remove(user_id)
            
            # Notify other users
            await broadcast_to_room(
                room_id,
                {
                    "type": "user-left",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Clean up empty room
            if len(rooms[room_id]) == 0:
                del rooms[room_id]
                del room_metadata[room_id]
                logger.info(f"Deleted empty room: {room_id}")


async def broadcast_to_room(room_id: str, message: dict, exclude_user: str = None):
    """Broadcast message to all users in room."""
    if room_id not in rooms:
        return
    
    for user_id, ws in rooms[room_id].items():
        if user_id != exclude_user:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {user_id}: {e}")


@app.post("/api/create-room")
def create_room():
    """Create a new room and return room ID."""
    room_id = str(uuid.uuid4())[:8]
    
    rooms[room_id] = {}
    room_metadata[room_id] = {
        "created_at": datetime.now().isoformat(),
        "users": []
    }
    
    logger.info(f"Created room: {room_id}")
    
    return {
        "room_id": room_id,
        "created_at": room_metadata[room_id]["created_at"]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting Video Call Signaling Server...")
    print("=" * 60)
    print()
    print("📍 Server URL: http://localhost:8001")
    print("🔌 WebSocket: ws://localhost:8001/ws/{room_id}/{user_id}")
    print()
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
