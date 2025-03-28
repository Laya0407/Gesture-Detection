import argparse
import asyncio
import json
import logging
import uuid
import time
import numpy as np
import cv2
import mediapipe as mp
import weakref
import base64
import os
from io import BytesIO

from aiohttp import web
import aiohttp

"""
WebSocket-based Touchless Interaction System:

1. Video Processing Service - Handles WebSocket connections and video frame processing
2. Hand Detection Service - Detects hand landmarks using MediaPipe
3. Gesture Recognition Service - Interprets hand landmarks into gestures
4. Cursor Control Service - Translates hand position to cursor coordinates
5. Interaction Feedback Service - Provides visual feedback to the client
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("touchless-interaction")

# Global state instead of database
client_dimensions = {}
active_connections = weakref.WeakSet()

# Global configuration loaded from secrets
config = {}

def load_secrets():
    """
    Load secrets from Docker secrets or environment variables
    Returns a configuration dictionary
    """
    logger.info("Loading configuration from secrets")
    secrets = {}
    
    # List of secrets to load
    secret_names = [
        "hand_detection_confidence",
        "hand_tracking_confidence", 
        "model_complexity",
        "max_num_hands",
        "feature_gesture_pinch_enabled",
        "feature_gesture_point_enabled",
        "experiment_name",
        "experiment_version"
    ]
    
    # Docker secrets are mounted at /run/secrets/SECRET_NAME
    for secret_name in secret_names:
        # Try to load from Docker secret
        secret_path = f"/run/secrets/{secret_name}"
        env_var_name = secret_name.upper()
        
        if os.path.exists(secret_path):
            try:
                with open(secret_path, 'r') as secret_file:
                    value = secret_file.read().strip()
                    logger.info(f"Loaded secret {secret_name} from Docker secrets")
                    secrets[secret_name] = value
            except Exception as e:
                logger.error(f"Error reading secret {secret_name}: {e}")
                # Fallback to environment variable
                secrets[secret_name] = os.environ.get(env_var_name)
        else:
            # If secret doesn't exist, try environment variable
            secrets[secret_name] = os.environ.get(env_var_name)
            
            # If still not found, use default values
            if secrets[secret_name] is None:
                logger.warning(f"Secret {secret_name} not found, using default value")
                # Default values
                if secret_name == "hand_detection_confidence":
                    secrets[secret_name] = "0.5"
                elif secret_name == "hand_tracking_confidence":
                    secrets[secret_name] = "0.5"
                elif secret_name == "model_complexity":
                    secrets[secret_name] = "0"
                elif secret_name == "max_num_hands":
                    secrets[secret_name] = "1"
                elif secret_name == "feature_gesture_pinch_enabled":
                    secrets[secret_name] = "true"
                elif secret_name == "feature_gesture_point_enabled":
                    secrets[secret_name] = "false"
                elif secret_name == "experiment_name":
                    secrets[secret_name] = "touchless_default"
                elif secret_name == "experiment_version":
                    secrets[secret_name] = "1.0.0"
    
    # Convert types as needed
    try:
        secrets["hand_detection_confidence"] = float(secrets["hand_detection_confidence"])
        secrets["hand_tracking_confidence"] = float(secrets["hand_tracking_confidence"])
        secrets["model_complexity"] = int(secrets["model_complexity"])
        secrets["max_num_hands"] = int(secrets["max_num_hands"])
        secrets["feature_gesture_pinch_enabled"] = secrets["feature_gesture_pinch_enabled"].lower() == "true"
        secrets["feature_gesture_point_enabled"] = secrets["feature_gesture_point_enabled"].lower() == "true"
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting secret values: {e}")
        # Provide fallback values if conversion fails
        if "hand_detection_confidence" not in secrets or not isinstance(secrets["hand_detection_confidence"], float):
            secrets["hand_detection_confidence"] = 0.5
        if "hand_tracking_confidence" not in secrets or not isinstance(secrets["hand_tracking_confidence"], float):
            secrets["hand_tracking_confidence"] = 0.5
        if "model_complexity" not in secrets or not isinstance(secrets["model_complexity"], int):
            secrets["model_complexity"] = 0
        if "max_num_hands" not in secrets or not isinstance(secrets["max_num_hands"], int):
            secrets["max_num_hands"] = 1
        if "feature_gesture_pinch_enabled" not in secrets or not isinstance(secrets["feature_gesture_pinch_enabled"], bool):
            secrets["feature_gesture_pinch_enabled"] = True
        if "feature_gesture_point_enabled" not in secrets or not isinstance(secrets["feature_gesture_point_enabled"], bool):
            secrets["feature_gesture_point_enabled"] = False
    
    logger.info(f"Configuration loaded successfully: experiment {secrets['experiment_name']} v{secrets['experiment_version']}")
    return secrets


class VideoProcessingService:
    """
    Video Processing Service

    - Processing WebSocket video streams
    - Processing frames from clients
    - Maintaining connection state
    """
    
    def __init__(self):
        """Initialize the Video Processing Service"""
        self.start_time = time.time()
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "video-processing",
            "status": "online",
            "activeConnections": len(active_connections),
            "uptime": time.time() - self.start_time
        }
        
    def process_frame(self, frame_data):
        """Process raw video frame from base64 data"""
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img_frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            # Flip horizontally for natural interaction
            rgb_frame = cv2.flip(img_frame, 1)
            
            return rgb_frame
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None


class HandDetectionService:
    """
    Hand Detection Service

    - Detecting and tracking hand presence in video frames
    - Identifying key hand landmarks
    - Filtering and smoothing detection results
    """
    
    def __init__(self, config):
        """Initialize the Hand Detection Service with MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Use configuration from secrets
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config["max_num_hands"],
            min_detection_confidence=config["hand_detection_confidence"],
            min_tracking_confidence=config["hand_tracking_confidence"],
            model_complexity=config["model_complexity"]
        )
        self.start_time = time.time()
        self.detection_count = 0
        self.frame_count = 0
        
    def get_status(self):
        """Get current service status"""
        detection_rate = 0
        if self.frame_count > 0:
            detection_rate = (self.detection_count / self.frame_count) * 100
            
        return {
            "service": "hand-detection",
            "status": "online",
            "detectionRate": f"{detection_rate:.2f}%",
            "framesProcessed": self.frame_count,
            "uptime": time.time() - self.start_time,
            "config": {
                "maxNumHands": config["max_num_hands"],
                "detectionConfidence": config["hand_detection_confidence"],
                "trackingConfidence": config["hand_tracking_confidence"],
                "modelComplexity": config["model_complexity"]
            }
        }
        
    def detect_hands(self, frame):
        """Detect hand landmarks in the given frame"""
        self.frame_count += 1
        
        # Process with MediaPipe
        results = self.hands.process(frame)
        
        if results.multi_hand_landmarks:
            self.detection_count += 1
            # Return the first detected hand's landmarks
            return results.multi_hand_landmarks[0].landmark
        
        return None


class GestureRecognitionService:
    """
    Gesture Recognition Service

    - Interpreting hand positions as specific interaction gestures (pinch)
    - Mapping recognized gestures to system commands
    """
    
    def __init__(self, config):
        """Initialize the Gesture Recognition Service"""
        self.start_time = time.time()
        self.gesture_counts = {
            "pinch": 0,
            "point": 0
        }
        # Feature flags from configuration
        self.pinch_enabled = config["feature_gesture_pinch_enabled"]
        self.point_enabled = config["feature_gesture_point_enabled"]
        
    def get_status(self):
        """Get current service status"""
        supported_gestures = []
        if self.pinch_enabled:
            supported_gestures.append("pinch")
        if self.point_enabled:
            supported_gestures.append("point")
            
        return {
            "service": "gesture-recognition",
            "status": "online",
            "supportedGestures": supported_gestures,
            "gesturesCounted": self.gesture_counts,
            "uptime": time.time() - self.start_time,
            "experimentInfo": {
                "name": config["experiment_name"],
                "version": config["experiment_version"]
            }
        }
    
    def detect_pinch(self, landmarks):
        """
        Detect pinch gesture between thumb and ring finger
        """
        if not landmarks or len(landmarks) < 21 or not self.pinch_enabled:
            return False
            
        thumb_tip = landmarks[4]
        ring_tip = landmarks[16]
        
        # Calculate distance between thumb and ring finger
        distance = np.sqrt(
            (thumb_tip.x - ring_tip.x)**2 +
            (thumb_tip.y - ring_tip.y)**2
        )
        
        is_pinching = distance < 0.08  # threshold
        
        if is_pinching:
            self.gesture_counts["pinch"] += 1
            
        return bool(is_pinching)  # Convert np.bool_ to Python bool
        
    def detect_point(self, landmarks):
        """
        Pending implementation
        """
        if not landmarks or len(landmarks) < 21 or not self.point_enabled:
            return False
            
        self.gesture_counts["point"] += 1
        return True


class CursorControlService:
    """
    Cursor Control Service

    - Translating hand position to cursor coordinates
    - Applying linear interpolation for natural movement
    - Handling virtual-trackpad boundary conditions
    """
    
    def __init__(self):
        """Initialize the Cursor Control Service"""
        self.start_time = time.time()
        self.cursor_positions = {}  # Last cursor position by client_id
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "cursor-control",
            "status": "online",
            "activeCursors": len(self.cursor_positions),
            "uptime": time.time() - self.start_time
        }
    
    def calculate_cursor_position(self, landmarks, client_id):
        """
        Calculate cursor position based on index finger tip position
        """
        if client_id not in client_dimensions:
            return None, None
        
        dims = client_dimensions[client_id]
        screen_width = dims.get('screenWidth')
        screen_height = dims.get('screenHeight')
        
        # Get virtual box dimensions
        cam_width = dims.get('videoWidth', 1280)
        cam_height = dims.get('videoHeight', 720)
        
        # Make the virtual box larger for easier use
        pad_size = min(cam_width, cam_height) * 0.7
        pad_x = (cam_width - pad_size) / 2
        pad_y = (cam_height - pad_size) / 2
        
        # Get index finger tip position (landmark 8)
        if len(landmarks) <= 8:
            return None, None
            
        index_tip = landmarks[8]
        
        # Transform normalized coordinates to pixel coordinates
        px = int((index_tip.x) * cam_width) 
        py = int(index_tip.y * cam_height)
        
        # Check if finger is in virtual box
        if (pad_x <= px <= pad_x + pad_size and pad_y <= py <= pad_y + pad_size):
            # Map position to screen coordinates
            norm_x = (px - pad_x) / pad_size
            norm_y = (py - pad_y) / pad_size
            
            cursor_x = int(norm_x * screen_width)
            cursor_y = int(norm_y * screen_height)
            
            return self.apply_smoothing(client_id, cursor_x, cursor_y)
        
        return None, None
    
    def apply_smoothing(self, client_id, cursor_x, cursor_y):
        """
        Apply lerp to cursor position
        """
        if cursor_x is None or cursor_y is None:
            return None, None
            
        if client_id in self.cursor_positions:
            last_x, last_y = self.cursor_positions[client_id]
            
            # smoothing with 0.5 factor
            smooth_x = last_x + (cursor_x - last_x) * 0.5
            smooth_y = last_y + (cursor_y - last_y) * 0.5
            
            cursor_x, cursor_y = int(smooth_x), int(smooth_y)
        
        # Store current position for next frame
        self.cursor_positions[client_id] = (cursor_x, cursor_y)
        
        return cursor_x, cursor_y


class InteractionFeedbackService:
    """
    Interaction Feedback Service

    - Providing visual confirmation of detected gestures
    - Guiding users with interface highlights
    - Creating confidence in the touchless interaction
    """
    
    def __init__(self):
        """Initialize the Interaction Feedback Service"""
        self.start_time = time.time()
        self.feedback_count = 0
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "interaction-feedback",
            "status": "online",
            "feedbackCount": self.feedback_count,
            "uptime": time.time() - self.start_time
        }
    
    def prepare_fingertip_data(self, landmarks, frame_shape):
        """
        Prepare fingertip information for client visualization
        Returns list of fingertip coordinates
        """
        if not landmarks:
            return []
            
        # Extract fingertip landmarks
        fingertips = [
            landmarks[4],   # thumb
            landmarks[8],   # index
            landmarks[12],  # middle
            landmarks[16],  # ring
            landmarks[20]   # pinky
        ]
        
        height, width = frame_shape[:2] if len(frame_shape) >= 2 else (720, 1280)
        
        # Convert to client-friendly format
        fingertip_data = []
        for i, tip in enumerate(fingertips):
            fingertip_data.append({
                'x': float(tip.x * width),
                'y': float(tip.y * height),
                'z': float(tip.z)
            })
            
        return fingertip_data
    
    def prepare_feedback_data(self, client_id, landmarks, frame_shape, cursor_x, cursor_y, pinch_status):
        """
        Prepare comprehensive feedback data for client
        Returns dict with cursor position, pinch status, and fingertip data
        """
        self.feedback_count += 1
        
        # Handle case where no hand is detected
        if not landmarks:
            return {
                'handVisible': False,
                'pinchStatus': False
            }
        
        # Prepare feedback with fingertip visualization data
        fingertips = self.prepare_fingertip_data(landmarks, frame_shape)
        
        return {
            'cursorX': cursor_x,
            'cursorY': cursor_y,
            'pinchStatus': bool(pinch_status),  # Convert np.bool_ to Python bool
            'handVisible': True,
            'fingertips': fingertips
        }


class SystemIntegrationService:
    """
    System Integration Service

    - Interfacing with existing software
    - Translating gestures into standard input events
    - Coordinating between services
    """
    
    def __init__(self, config):
        """Initialize system integration and all services"""
        # Initialize all services with configuration
        self.video_service = VideoProcessingService()
        self.hand_detection = HandDetectionService(config)
        self.gesture_recognition = GestureRecognitionService(config)
        self.cursor_control = CursorControlService()
        self.feedback_service = InteractionFeedbackService()
        
        # For serializing NumPy types
        self.numpy_encoder = NumpyEncoder()
        
        # WebSocket connections
        self.websockets = {}
        
    async def handle_websocket(self, request):
        """Handle WebSocket connection"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = f"client-{uuid.uuid4()}"
        self.websockets[client_id] = ws
        active_connections.add(ws)
        
        logger.info(f"New WebSocket connection established: {client_id}")
        
        try:
            # Send initial confirmation
            await ws.send_json({"type": "connected", "clientId": client_id})
            
            # Process incoming messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        message_type = data.get('type')
                        
                        if message_type == 'config':
                            # Handle client configuration
                            client_dimensions[client_id] = {
                                'screenWidth': data.get('screenWidth', 1920),
                                'screenHeight': data.get('screenHeight', 1080),
                                'videoWidth': data.get('videoWidth', 1280),
                                'videoHeight': data.get('videoHeight', 720)
                            }
                            logger.info(f"Client config updated for {client_id}")
                            
                        elif message_type == 'frame':
                            # Process video frame
                            frame_data = data.get('frameData')
                            if frame_data:
                                # Convert base64 frame to image
                                rgb_frame = self.video_service.process_frame(frame_data)
                                
                                if rgb_frame is not None:
                                    # Detect hands
                                    landmarks = self.hand_detection.detect_hands(rgb_frame)
                                    
                                    # Process if hand is detected
                                    if landmarks:
                                        # Detect gestures
                                        pinch_status = self.gesture_recognition.detect_pinch(landmarks)
                                        
                                        # Calculate cursor position
                                        cursor_x, cursor_y = self.cursor_control.calculate_cursor_position(
                                            landmarks, client_id
                                        )
                                        
                                        # Prepare feedback
                                        feedback_data = self.feedback_service.prepare_feedback_data(
                                            client_id,
                                            landmarks,
                                            rgb_frame.shape,
                                            cursor_x,
                                            cursor_y,
                                            pinch_status
                                        )
                                        
                                        # Send feedback data
                                        # Use dumps with the custom encoder instead of send_json
                                        await ws.send_str(json.dumps(feedback_data, cls=NumpyEncoder))
                                    else:
                                        # Send hand not visible status
                                        await ws.send_json({
                                            'handVisible': False,
                                            'pinchStatus': False
                                        })
                                    
                        else:
                            logger.warning(f"Unknown message type: {message_type}")
                    
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception: {ws.exception()}")
                    break
            
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
        finally:
            # Clean up when connection is closed
            if client_id in self.websockets:
                del self.websockets[client_id]
            if client_id in client_dimensions:
                del client_dimensions[client_id]
            if ws in active_connections:
                active_connections.remove(ws)
            logger.info(f"WebSocket connection closed: {client_id}")
        
        return ws
    
    def get_system_status(self):
        """Get comprehensive status of all services"""
        return {
            "system": "touchless-interaction",
            "services": [
                self.video_service.get_status(),
                self.hand_detection.get_status(),
                self.gesture_recognition.get_status(),
                self.cursor_control.get_status(),
                self.feedback_service.get_status()
            ],
            "activeConnections": len(active_connections),
            "registeredClients": len(client_dimensions),
            "experiment": {
                "name": config["experiment_name"],
                "version": config["experiment_version"]
            }
        }


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)  # Convert NumPy boolean to Python boolean
        return super(NumpyEncoder, self).default(obj)


# API Routes

async def index(request):
    """Serve the index page"""
    content = open("index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

async def status(request):
    """Return system status"""
    return web.Response(
        content_type="application/json",
        text=json.dumps(system_service.get_system_status(), cls=NumpyEncoder)
    )

async def get_config(request):
    """Return configuration information (excluding sensitive values)"""
    # Create a sanitized version of config for public viewing
    safe_config = {
        "experimentName": config["experiment_name"],
        "experimentVersion": config["experiment_version"],
        "features": {
            "pinchEnabled": config["feature_gesture_pinch_enabled"],
            "pointEnabled": config["feature_gesture_point_enabled"]
        },
        "modelInfo": {
            "complexity": config["model_complexity"],
            "maxHands": config["max_num_hands"]
        }
    }
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(safe_config)
    )

async def cleanup_inactive_connections():
    """Periodically check and clean up inactive connections"""
    while True:
        try:
            # Log current connection count every minute
            if len(active_connections) > 0:
                logger.info(f"Active connections: {len(active_connections)}")
            
            # Check each connection
            to_remove = []
            for ws in active_connections:
                if ws.closed:
                    to_remove.append(ws)
            
            # Remove dead connections
            for ws in to_remove:
                active_connections.remove(ws)
                
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(60)  # Continue even if there's an error

async def start_background_tasks(app):
    """Start background tasks when the app starts"""
    app['cleanup_task'] = asyncio.create_task(cleanup_inactive_connections())
    logger.info("Started background cleanup task")

async def cleanup_background_tasks(app):
    """Clean up background tasks when the app stops"""
    app['cleanup_task'].cancel()
    try:
        await app['cleanup_task']
    except asyncio.CancelledError:
        pass
    logger.info("Canceled background cleanup task")
    
    # Close all active connections
    for ws in list(active_connections):
        await ws.close()
    logger.info("Closed all active connections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Touchless Interaction System (WebSocket Version)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server")
    args = parser.parse_args()

    # Load configuration from secrets
    config = load_secrets()

    # Initialize system with loaded configuration
    system_service = SystemIntegrationService(config)

    app = web.Application()
    
    app.router.add_get("/", index)
    app.router.add_get("/ws", system_service.handle_websocket)
    app.router.add_get("/status", status)
    app.router.add_get("/config", get_config)

    # Background tasks to cleanup stale connections
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    # Initialize MediaPipe with a test frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        system_service.hand_detection.detect_hands(dummy_frame)
    logger.info("MediaPipe hands initialized")
    
    logger.info(f"Starting Touchless Interaction System - Experiment: {config['experiment_name']} v{config['experiment_version']}")
    
    web.run_app(
        app, 
        host=args.host, 
        port=args.port
    )
