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

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay


"""

1. Video Processing Service - Handles WebRTC connections and video frame processing
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

# Disable noisy loggers
logging.getLogger('aiortc.rtcrtpreceiver').setLevel(logging.CRITICAL)
logging.getLogger('aioice').setLevel(logging.CRITICAL)

# Global state (would be separate databases/caches in a true microservice architecture)
client_dimensions = {}
active_connections = weakref.WeakSet()


class VideoProcessingService:
    """
    Video Processing Service

    - Capturing and processing webcam feed in real-time
    - Handling WebRTC connections
    - Processing frames from clients
    - Maintaining connection state
    """
    
    def __init__(self):
        """Initialize the Video Processing Service"""
        self.media_relay = MediaRelay()
        self.start_time = time.time()
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "video-processing",
            "status": "online",
            "activeConnections": len(active_connections),
            "uptime": time.time() - self.start_time
        }
    
    def create_peer_connection(self, client_id):
        """Create a new WebRTC peer connection"""
        pc = RTCPeerConnection()
        active_connections.add(pc)
        
        return pc
        
    def subscribe_to_track(self, track):
        """Subscribe to a media track using MediaRelay"""
        return self.media_relay.subscribe(track)
        
    def process_frame(self, frame):
        """Process raw video frame"""
        # Convert YUV to RGB for MediaPipe processing
        if hasattr(frame, 'to_ndarray'):
            # aiortc frame
            img_frame = frame.to_ndarray(format="yuv420p")
            # Use BGR2RGB instead of YUV420P2RGB
            rgb_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
        else:
            # Already ndarray
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Flip horizontally for natural interaction
        rgb_frame = cv2.flip(rgb_frame, 1)
        
        return rgb_frame


class HandDetectionService:
    """
    Hand Detection Service

    - Detecting and tracking hand presence in video frames
    - Identifying key hand landmarks
    - Filtering and smoothing detection results
    """
    
    def __init__(self):
        """Initialize the Hand Detection Service with MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
            model_complexity=1
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
            "uptime": time.time() - self.start_time
        }
        
    def detect_hands(self, frame):
        """Detect hand landmarks in the given frame"""
        self.frame_count += 1
        
        # Add debug print
        if self.frame_count % 100 == 0:  # Print every 100 frames to avoid flooding
            print(f"Processing frame {self.frame_count}, shape: {frame.shape}")
        
        # Process with MediaPipe
        results = self.hands.process(frame)
        
        # Add more debugging
        if self.frame_count % 100 == 0:
            print(f"Hand detected: {results.multi_hand_landmarks is not None}")
        
        if results.multi_hand_landmarks:
            self.detection_count += 1
            # Return the first detected hand's landmarks
            return results.multi_hand_landmarks[0].landmark
        
        return None

class GestureRecognitionService:
    """
    Enhanced Gesture Recognition Service

    - Interpreting hand positions as specific interaction gestures
    - Supporting multiple gestures: pinch, swipe, hover hold, finger spread, fist
    - Mapping recognized gestures to system commands
    """
    
    def __init__(self):
        """Initialize the Gesture Recognition Service"""
        self.start_time = time.time()
        self.gesture_counts = {
            "pinch": 0,
            "swipe": 0,
            "hover_hold": 0,
            "finger_spread": 0,
            "fist": 0
        }
        
        # State tracking for gestures that need temporal information
        self.previous_landmarks = None
        self.hover_start_time = None
        self.hover_position = None
        self.swipe_start_position = None
        self.swipe_direction = None
        self.swipe_cooldown = 0
        
    def get_status(self):
        """Get current service status"""
        return {
            "service": "gesture-recognition",
            "status": "online",
            "supportedGestures": ["pinch", "swipe", "hover_hold", "finger_spread", "fist"],
            "gesturesCounted": self.gesture_counts,
            "uptime": time.time() - self.start_time
        }
    
    def detect_pinch(self, landmarks):
        """
        Detect pinch gesture between thumb and index finger
        """
        if not landmarks or len(landmarks) < 21:
            return False
            
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2
        )
        
        is_pinching = distance < 0.05  # adjusted threshold
        
        if is_pinching:
            self.gesture_counts["pinch"] += 1

         # Add print trigger functionality
            if cursor_x is not None and cursor_y is not None:
             element = document.elementFromPoint(cursor_x, cursor_y)
            if element and hasattr(element, 'id') and element.id == 'print-button':
            # Trigger print action
             feedback_data["action"] = {
                "type": "print",
                "documentId": "current-document"
             } 
        return is_pinching
    
    def detect_swipe(self, landmarks, frame_shape):
        """
        Detect swipe gesture based on quick lateral movement of index finger
        Returns: None, "left", "right", "up", or "down"
        """
        if not landmarks or len(landmarks) < 21:
            # Reset state if hand not detected
            self.swipe_start_position = None
            self.swipe_cooldown = max(0, self.swipe_cooldown - 1)
            return None
        
        # Don't detect swipes during cooldown
        if self.swipe_cooldown > 0:
            self.swipe_cooldown -= 1
            return None
            
        # Get current index finger position
        index_tip = landmarks[8]
        current_pos = (index_tip.x, index_tip.y)
        
        # Initialize start position if not set
        if self.swipe_start_position is None:
            self.swipe_start_position = current_pos
            return None
            
        # Calculate displacement
        dx = current_pos[0] - self.swipe_start_position[0]
        dy = current_pos[1] - self.swipe_start_position[1]
        
        # Convert to pixel space for frame_shape
        pixel_dx = dx * frame_shape[1]
        pixel_dy = dy * frame_shape[0]
        
        # Determine if swipe distance threshold is met
        distance = np.sqrt(pixel_dx**2 + pixel_dy**2)
        min_swipe_distance = min(frame_shape[0], frame_shape[1]) * 0.15  # 15% of frame size
        
        swipe_direction = None
        
        if distance > min_swipe_distance:
            # Determine swipe direction based on dominant axis
            if abs(dx) > abs(dy):
                swipe_direction = "left" if dx < 0 else "right"
            else:
                swipe_direction = "up" if dy < 0 else "down"
                
            # Reset start position and set cooldown
            self.swipe_start_position = None
            self.swipe_cooldown = 10  # Frames to wait before detecting another swipe
            
            # Count the gesture
            self.gesture_counts["swipe"] += 1
        else:
            # Update start position for small movements (allow for swipe continuation)
            self.swipe_start_position = current_pos
            
        return swipe_direction
    
    def detect_hover_hold(self, landmarks, frame_shape, hold_time_threshold=1.5):
        """
        Detect hover hold gesture (keeping hand still in one position)
        Returns: (is_holding, hold_duration)
        """
        if not landmarks or len(landmarks) < 21:
            # Reset hover state if hand not detected
            self.hover_start_time = None
            self.hover_position = None
            return False, 0
        
        # Get index finger position
        index_tip = landmarks[8]
        current_pos = (index_tip.x, index_tip.y)
        
        # Convert to pixel space
        pixel_x = current_pos[0] * frame_shape[1]
        pixel_y = current_pos[1] * frame_shape[0]
        
        # Initialize hover position if not set
        if self.hover_position is None:
            self.hover_position = (pixel_x, pixel_y)
            self.hover_start_time = time.time()
            return False, 0
        
        # Calculate movement from hover start position
        movement = np.sqrt(
            (pixel_x - self.hover_position[0])**2 + 
            (pixel_y - self.hover_position[1])**2
        )
        
        # Define hover tolerance (how still the hand must be)
        hover_tolerance = min(frame_shape[0], frame_shape[1]) * 0.03  # 3% of frame size
        
        # Check if movement is within tolerance
        if movement < hover_tolerance:
            # Hand is holding position
            hold_duration = time.time() - self.hover_start_time
            is_hovering = hold_duration > hold_time_threshold
            
            if is_hovering and hold_duration - hold_time_threshold < 0.1:  # Count only once when crossing threshold
                self.gesture_counts["hover_hold"] += 1
                
            return is_hovering, hold_duration
        else:
            # Reset hover because hand moved too much
            self.hover_position = (pixel_x, pixel_y)
            self.hover_start_time = time.time()
            return False, 0
    
    def detect_finger_spread(self, landmarks):
        """
        Detect finger spread gesture (all fingers extended and separated)
        """
        if not landmarks or len(landmarks) < 21:
            return False
        
        # Get fingertip positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Get finger mid positions for reference
        wrist = landmarks[0]
        
        # Check if all fingers are extended by comparing fingertip y-position with wrist
        # Remember: in image coordinates, lower y means higher position
        thumb_extended = thumb_tip.x > wrist.x + 0.1  # Thumb extends more horizontally
        index_extended = index_tip.y < wrist.y
        middle_extended = middle_tip.y < wrist.y
        ring_extended = ring_tip.y < wrist.y
        pinky_extended = pinky_tip.y < wrist.y
        
        all_fingers_extended = thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended
        
        if not all_fingers_extended:
            return False
        
        # Calculate average distances between adjacent fingertips
        distances = [
            np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2),
            np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2),
            np.sqrt((middle_tip.x - ring_tip.x)**2 + (middle_tip.y - ring_tip.y)**2),
            np.sqrt((ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2)
        ]
        
        avg_distance = sum(distances) / len(distances)
        
        # Check if fingers are spread enough
        is_spread = avg_distance > 0.08  # Threshold for spread
        
        if is_spread:
            self.gesture_counts["finger_spread"] += 1
            
        return is_spread
    
    def detect_fist(self, landmarks):
        """
        Detect fist gesture (all fingers folded into palm)
        """
        if not landmarks or len(landmarks) < 21:
            return False
        
        # Get fingertip positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Get corresponding middle finger joints
        thumb_ip = landmarks[2]  # Thumb IP joint (closest equivalent)
        index_pip = landmarks[6]  # Index PIP joint
        middle_pip = landmarks[10]  # Middle PIP joint
        ring_pip = landmarks[14]  # Ring PIP joint
        pinky_pip = landmarks[18]  # Pinky PIP joint
        
        # Get wrist position
        wrist = landmarks[0]
        
        # For a fist, fingertips should be closer to the wrist than their PIP joints
        # Calculate distances from wrist to each point
        wrist_to_thumb_tip = np.sqrt((wrist.x - thumb_tip.x)**2 + (wrist.y - thumb_tip.y)**2)
        wrist_to_index_tip = np.sqrt((wrist.x - index_tip.x)**2 + (wrist.y - index_tip.y)**2)
        wrist_to_middle_tip = np.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
        wrist_to_ring_tip = np.sqrt((wrist.x - ring_tip.x)**2 + (wrist.y - ring_tip.y)**2)
        wrist_to_pinky_tip = np.sqrt((wrist.x - pinky_tip.x)**2 + (wrist.y - pinky_tip.y)**2)
        
        wrist_to_thumb_ip = np.sqrt((wrist.x - thumb_ip.x)**2 + (wrist.y - thumb_ip.y)**2)
        wrist_to_index_pip = np.sqrt((wrist.x - index_pip.x)**2 + (wrist.y - index_pip.y)**2)
        wrist_to_middle_pip = np.sqrt((wrist.x - middle_pip.x)**2 + (wrist.y - middle_pip.y)**2)
        wrist_to_ring_pip = np.sqrt((wrist.x - ring_pip.x)**2 + (wrist.y - ring_pip.y)**2)
        wrist_to_pinky_pip = np.sqrt((wrist.x - pinky_pip.x)**2 + (wrist.y - pinky_pip.y)**2)
        
        # Check if all fingertips are closer to the wrist than their PIP joints
        # Thumb is special case - just check if it's close to the palm
        thumb_folded = wrist_to_thumb_tip < 0.15  # Threshold for thumb
        index_folded = wrist_to_index_tip < wrist_to_index_pip
        middle_folded = wrist_to_middle_tip < wrist_to_middle_pip
        ring_folded = wrist_to_ring_tip < wrist_to_ring_pip
        pinky_folded = wrist_to_pinky_tip < wrist_to_pinky_pip
        
        # Consider it a fist if at least 4 out of 5 fingers are folded
        folded_count = sum([thumb_folded, index_folded, middle_folded, ring_folded, pinky_folded])
        is_fist = folded_count >= 4
        
        if is_fist:
            self.gesture_counts["fist"] += 1
            
        return is_fist
    
    def process_all_gestures(self, landmarks, frame_shape):
        """
        Process all gestures at once and return a dictionary of results
        """
        if not landmarks:
            return {
                "pinch": False,
                "swipe": None,
                "hover_hold": (False, 0),
                "finger_spread": False,
                "fist": False
            }
        
        # Detect all gestures
        pinch_detected = self.detect_pinch(landmarks)
        swipe_direction = self.detect_swipe(landmarks, frame_shape)
        hover_status = self.detect_hover_hold(landmarks, frame_shape)
        spread_detected = self.detect_finger_spread(landmarks)
        fist_detected = self.detect_fist(landmarks)
        
        # Update previous landmarks for next frame
        self.previous_landmarks = landmarks
        
        return {
            "pinch": pinch_detected,
            "swipe": swipe_direction,
            "hover_hold": hover_status,
            "finger_spread": spread_detected,
            "fist": fist_detected
        }


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
    Enhanced Interaction Feedback Service

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
    
    def prepare_hand_skeleton_data(self, landmarks, frame_shape):
        """
        Prepare full hand skeleton for visualization
        Returns list of all landmark coordinates and connections
        """
        if not landmarks:
            return {"landmarks": [], "connections": []}
            
        height, width = frame_shape[:2] if len(frame_shape) >= 2 else (720, 1280)
        
        # Convert all landmarks to client-friendly format
        landmark_data = []
        for i, lm in enumerate(landmarks):
            landmark_data.append({
                'id': i,
                'x': float(lm.x * width),
                'y': float(lm.y * height),
                'z': float(lm.z)
            })
        
        # Define connections for hand skeleton visualization
        # These connections create the standard MediaPipe hand skeleton
        connections = [
            # Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            # Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            # Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            # Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            # Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
            # Palm
            [0, 5], [5, 9], [9, 13], [13, 17]
        ]
        
        return {
            "landmarks": landmark_data, 
            "connections": connections
        }
        
    def prepare_feedback_data(self, client_id, landmarks, frame_shape, cursor_x, cursor_y, gesture_results):
        """
        Prepare comprehensive feedback data for client
        Returns dict with cursor position, gesture statuses, and visualization data
        """
        self.feedback_count += 1
        
        # Handle case where no hand is detected
        if not landmarks:
            return {
                'handVisible': False,
                'gestureDetected': False
            }
        
        # Get hover hold details
        hover_active, hover_duration = gesture_results.get("hover_hold", (False, 0))
        
        # Extract swipe direction
        swipe_direction = gesture_results.get("swipe")
        
        # Prepare feedback with fingertip and skeleton visualization data
        fingertips = self.prepare_fingertip_data(landmarks, frame_shape)
        hand_skeleton = self.prepare_hand_skeleton_data(landmarks, frame_shape)
        
        # Check if any gesture is active
        any_gesture_active = (
            gesture_results.get("pinch", False) or 
            swipe_direction is not None or
            hover_active or
            gesture_results.get("finger_spread", False) or
            gesture_results.get("fist", False)
        )
        
        return {
            'cursorX': cursor_x,
            'cursorY': cursor_y,
            'handVisible': True,
            'gestureDetected': any_gesture_active,
            'gestures': {
                'pinch': gesture_results.get("pinch", False),
                'swipe': {
                    'active': swipe_direction is not None,
                    'direction': swipe_direction
                },
                'hover': {
                    'active': hover_active,
                    'duration': hover_duration
                },
                'fingerSpread': gesture_results.get("finger_spread", False),
                'fist': gesture_results.get("fist", False)
            },
            'fingertips': fingertips,
            'handSkeleton': hand_skeleton
        }


class SystemIntegrationService:
    """
    System Integration Service

    - Interfacing with existing software
    - Translating gestures into standard input events
    - Coordinating between services
    """
    
    def __init__(self):
        """Initialize system integration and all services"""
        # Initialize all services
        self.video_service = VideoProcessingService()
        self.hand_detection = HandDetectionService()
        self.gesture_recognition = GestureRecognitionService()
        self.cursor_control = CursorControlService()
        self.feedback_service = InteractionFeedbackService()
        
        # For serializing NumPy types
        self.NumpyEncoder = NumpyEncoder
        
    async def process_offer(self, offer_params):
        """Process WebRTC offer and setup connection"""
        offer = RTCSessionDescription(sdp=offer_params["sdp"], type=offer_params["type"])
        client_id = offer_params.get("clientId", f"client-{uuid.uuid4()}")
        
        # Store client dimensions if provided
        if "screenWidth" in offer_params and "screenHeight" in offer_params:
            client_dimensions[client_id] = {
                'screenWidth': offer_params.get('screenWidth', 1920),
                'screenHeight': offer_params.get('screenHeight', 1080),
                'videoWidth': offer_params.get('videoWidth', 1280),
                'videoHeight': offer_params.get('videoHeight', 720)
            }
        
        # Create peer connection using Video Processing Service
        pc = self.video_service.create_peer_connection(client_id)
        pc_id = f"PeerConnection({uuid.uuid4()})"
        
        # State for this connection
        data_channel = None
        frame_processing_active = True
        
        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)
        
        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info(f"Connection state changed to {pc.connectionState}")
            nonlocal frame_processing_active
            
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                # Stop processing if connection is closed or failed
                frame_processing_active = False
                
                # Clean up if this connection is in our active set
                if pc in active_connections:
                    active_connections.remove(pc)
                    log_info("Connection removed from active set")
        
        # Handle data channel creation
        @pc.on("datachannel")
        def on_datachannel(channel):
            nonlocal data_channel
            data_channel = channel
            log_info("Data channel opened")
            
            @channel.on("close")
            def on_close():
                nonlocal frame_processing_active
                frame_processing_active = False
                log_info("Data channel closed")
        
        # Handle video track
        @pc.on("track")
        def on_track(track):
            if track.kind != "video":
                return

            log_info(f"Video track added, waiting for media to start flowing")
            
            # Use Video Processing Service to subscribe to track
            receiver = self.video_service.subscribe_to_track(track)
            
            # Store last sent cursor position for smoothing
            last_cursor_x, last_cursor_y = None, None
            
            # Set up task for frame processing
            task = None
            
            async def process_frames():
                nonlocal frame_processing_active, last_cursor_x, last_cursor_y
                frame_count = 0
                
                try:
                    while frame_processing_active:
                        if pc.connectionState in ["failed", "closed"]:
                            log_info("Connection closed, stopping frame processing")
                            break
                            
                        frame_start_time = time.time()
                        
                        try:
                            # Consistent timeout for frame reception
                            frame = await asyncio.wait_for(receiver.recv(), timeout=1.0)
                            
                            # Log first frame received
                            if frame_count == 0:
                                log_info("First frame received - media flow established")
                            
                            # Process frame with Video Processing Service
                            rgb_frame = self.video_service.process_frame(frame)
                            height, width, _ = rgb_frame.shape
                            
                            # Detect hand landmarks with Hand Detection Service
                            landmarks = self.hand_detection.detect_hands(rgb_frame)
                            
                            # Process landmarks if hand is detected
                            if landmarks and data_channel and data_channel.readyState == "open":
                                # Process all gestures with enhanced Gesture Recognition Service
                                gesture_results = self.gesture_recognition.process_all_gestures(
                                    landmarks, rgb_frame.shape
                                )
                                
                                # Calculate cursor position with Cursor Control Service
                                cursor_x, cursor_y = self.cursor_control.calculate_cursor_position(
                                    landmarks, client_id
                                )
                                
                                # Special handling for different gestures
                                is_pinching = gesture_results.get("pinch", False)
                                is_fist = gesture_results.get("fist", False)
                                swipe_direction = gesture_results.get("swipe")
                                is_finger_spread = gesture_results.get("finger_spread", False)
                                hover_active, hover_duration = gesture_results.get("hover_hold", (False, 0))
                                
                                # Prepare feedback with enhanced Interaction Feedback Service
                                feedback_data = self.feedback_service.prepare_feedback_data(
                                    client_id,
                                    landmarks,
                                    rgb_frame.shape,
                                    cursor_x,
                                    cursor_y,
                                    gesture_results
                                )
                                
                                # Add additional UI interaction commands based on gestures
                                if swipe_direction:
                                    feedback_data["action"] = {
                                        "type": "scroll",
                                        "direction": swipe_direction,
                                        # Adjust scroll amount based on your needs
                                        "amount": 100 if swipe_direction in ["up", "down"] else 100
                                    }
                                
                                if is_finger_spread:
                                    feedback_data["action"] = {
                                        "type": "zoom",
                                        "factor": 1.1  # Zoom in slightly
                                    }
                                
                                if is_fist and cursor_x is not None and cursor_y is not None:
                                    feedback_data["action"] = {
                                        "type": "drag",
                                        "active": True,
                                        "x": cursor_x,
                                        "y": cursor_y
                                    }
                                
                                if hover_active:
                                    feedback_data["action"] = {
                                        "type": "hover",
                                        "duration": hover_duration,
                                        "x": cursor_x,
                                        "y": cursor_y
                                    }
                                
                                # Send enhanced cursor and gesture data
                                try:
                                    data_channel.send(json.dumps(feedback_data, cls=self.NumpyEncoder))
                                except Exception as e:
                                    log_info(f"Data send error: {e}")
                                    if "closed" in str(e).lower():
                                        # Channel is closed, stop processing
                                        frame_processing_active = False
                                        break
                            elif data_channel and data_channel.readyState == "open":
                                # Send hand not visible status via Feedback Service
                                try:
                                    data_channel.send(json.dumps({
                                        'handVisible': False,
                                        'gestureDetected': False
                                    }))
                                except Exception as e:
                                    log_info(f"Data send error: {e}")
                                    if "closed" in str(e).lower():
                                        # Channel is closed, stop processing
                                        frame_processing_active = False
                                        break
                            
                            # Log performance metrics occasionally
                            if frame_count % 300 == 0:  # Further reduce log frequency 
                                processing_time = time.time() - frame_start_time
                                log_info(f"Frame processing time - {processing_time:.4f}s")
                            
                            frame_count += 1
                            
                        except asyncio.TimeoutError:
                            # Simplified timeout handling
                            if frame_count == 0:
                                log_info("Waiting for initial video frames...")
                            
                            # Check connection state
                            if pc.connectionState in ["failed", "closed"] or (data_channel and data_channel.readyState != "open"):
                                frame_processing_active = False
                                break
                        except Exception as e:
                            log_info(f"Frame processing error: {str(e)}")
                            # Continue processing other frames instead of breaking
                
                except asyncio.CancelledError:
                    log_info("Frame processing task cancelled")
                finally:
                    log_info("Frame processing stopped")
                    frame_processing_active = False
                    
                    # Clean up if this connection is in our active set
                    if pc in active_connections:
                        active_connections.remove(pc)
            
            # Create and store the task
            task = asyncio.create_task(process_frames())
            
            # Clean up when track ends
            @track.on("ended")
            async def on_ended():
                nonlocal frame_processing_active
                frame_processing_active = False
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        # Handle WebRTC offer
        try:
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return {
                "sdp": pc.localDescription.sdp, 
                "type": pc.localDescription.type
            }
        except Exception as e:
            logger.exception(f"Error handling offer: {e}")
            raise e
    
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
            "registeredClients": len(client_dimensions)
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
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Initialize the main system integration service
system_service = SystemIntegrationService()

# API Routes

async def index(request):
    """Serve the index page"""
    content = open("indexpml.html", "r", encoding="utf-8").read()
    return web.Response(content_type="text/html", text=content)

async def client_config(request):
    """Handle client configuration updates"""
    params = await request.json()
    client_id = params.get("clientId")
    
    if client_id:
        client_dimensions[client_id] = {
            'screenWidth': params.get('screenWidth', 1920),
            'screenHeight': params.get('screenHeight', 1080),
            'videoWidth': params.get('videoWidth', 1280),
            'videoHeight': params.get('videoHeight', 720)
        }
        return web.Response(text=json.dumps({"status": "success"}))
    
    return web.Response(status=400, text=json.dumps({"status": "error", "message": "Client ID required"}))

async def offer(request):
    """Handle WebRTC offer"""
    params = await request.json()
    
    try:
        result = await system_service.process_offer(params)
        return web.Response(
            content_type="application/json",
            text=json.dumps(result)
        )
    except Exception as e:
        logger.exception(f"Error handling offer: {e}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": str(e)})
        )

async def status(request):
    """Return system status"""
    return web.Response(
        content_type="application/json",
        text=json.dumps(system_service.get_system_status(), cls=NumpyEncoder)
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
            for pc in active_connections:
                if pc.connectionState in ["failed", "closed"]:
                    to_remove.append(pc)
            
            # Remove dead connections
            for pc in to_remove:
                active_connections.remove(pc)
                
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
    for pc in list(active_connections):
        pc.close()
    logger.info("Closed all active connections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Touchless Interaction System")
    parser.add_argument("--port", type=int, default=8080, help="Port for the web server (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the web server (default: 0.0.0.0)")
    args = parser.parse_args()
    
    # Create web application
    app = web.Application()
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    
    # Setup routes
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/config", client_config)
    app.router.add_get("/status", status)
    
    # Start the server
    logger.info(f"Starting server on {args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port)

    # Add near the imports at the top
import threading
import cv2
import mediapipe as mp

# Add this function somewhere in the file
def debug_hand_detection():
    """Debug function to test hand detection directly"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        model_complexity=1
    )
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the image horizontally for a selfie-view
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(rgb_image)
        
        # Draw hand landmarks on the image
        if results.multi_hand_landmarks:
            print("Hand detected directly in OpenCV window")
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display the resulting frame
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start this in a thread at the end of the "__main__" block
threading.Thread(target=debug_hand_detection, daemon=True).start()