# Touchless Interaction System

A WebSocket-based, machine learning-powered system for touchless interaction with digital interfaces through hand gestures. This project uses MediaPipe for real-time hand landmark detection and enables natural interaction without physical contact.

## 🌟 Features

- **Real-time hand tracking** using MediaPipe
- **Gesture recognition** (pinch, point) for interaction
- **Virtual cursor control** via hand movements
- **Visual feedback** system with fingertip tracking
- **ML Ops pipelines** for data collection and model serving
- **Docker containerization** for easy deployment
- **Configurable via environment variables** or Docker secrets

## 🏗️ System Architecture

The system employs a microservice-inspired architecture with six core services:

1. **Video Processing Service**: Handles WebSocket video stream processing
2. **Hand Detection Service**: Detects hand landmarks using MediaPipe
3. **Gesture Recognition Service**: Interprets hand positions as gestures
4. **Cursor Control Service**: Translates hand position to cursor coordinates
5. **Interaction Feedback Service**: Provides visual feedback to users
6. **System Integration Service**: Orchestrates all services

Additionally, the system includes two ML Ops pipelines:
- **Data Collection Pipeline**: Gathers hand landmark data for training
- **Model Serving Pipeline**: Manages model deployment and configuration

![Architecture Diagram](docs/architecture.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenCV
- MediaPipe
- Docker (optional)

### Installation

#### Using Docker Compose (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/touchless-interaction.git
cd touchless-interaction
```

2. Create secrets directory and configuration files:
```bash
mkdir -p secrets
echo "0.5" > secrets/hand_detection_confidence.txt
echo "0.5" > secrets/hand_tracking_confidence.txt
echo "0" > secrets/model_complexity.txt
echo "1" > secrets/max_num_hands.txt
echo "true" > secrets/feature_gesture_pinch_enabled.txt
echo "false" > secrets/feature_gesture_point_enabled.txt
echo "touchless_default" > secrets/experiment_name.txt
echo "1.0.0" > secrets/experiment_version.txt
echo "true" > secrets/data_collection_enabled.txt
```

3. Launch with Docker Compose:
```bash
docker-compose up -d
```

4. Access the web interface at `http://localhost:8081`

#### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/touchless-interaction.git
cd touchless-interaction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the web interface at `http://localhost:8081`

## 🎮 Usage

1. Open the web interface in a browser that supports WebRTC (Chrome, Firefox, Edge)
2. Click "Start Camera" to begin the detection
3. Place your hand in the virtual box shown on the video feed
4. Move your hand to control the cursor
5. Use pinch gesture (thumb and ring finger) to click

### Supported Gestures

- **Pinch**: Join your thumb and ring finger - equivalent to a click
- **Point**: Extend your index finger while keeping other fingers closed

## ⚙️ Configuration

The system can be configured using Docker secrets or environment variables:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hand_detection_confidence` | MediaPipe hand detection confidence threshold | 0.5 |
| `hand_tracking_confidence` | MediaPipe hand tracking confidence threshold | 0.5 |
| `model_complexity` | MediaPipe model complexity (0, 1, or 2) | 0 |
| `max_num_hands` | Maximum number of hands to detect | 1 |
| `feature_gesture_pinch_enabled` | Enable pinch gesture detection | true |
| `feature_gesture_point_enabled` | Enable point gesture detection | false |
| `experiment_name` | Current experiment name | touchless_default |
| `experiment_version` | Current experiment version | 1.0.0 |
| `data_collection_enabled` | Enable data collection for ML training | true |

## 📊 API Endpoints

The system provides RESTful API endpoints for configuration and monitoring:

- `GET /status` - Get system status
- `GET /config` - Get current configuration
- `POST /api/data-collection/start` - Start data collection session
- `POST /api/data-collection/stop` - Stop data collection session
- `GET /api/data-collection/datasets` - List available datasets
- `DELETE /api/data-collection/datasets/{filename}` - Delete a dataset
- `GET /api/model-serving/config` - Get model configuration
- `POST /api/model-serving/config/{model_name}` - Update model configuration
- `POST /api/model-serving/active-model` - Set active model
- `POST /api/model-serving/export-config/{model_name}` - Export model configuration
- `POST /api/model-serving/import-config` - Import model configuration

## 🧪 Development

### Project Structure

```
touchless-interaction/
├── app.py              # Main application
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── requirements.txt    # Python dependencies
├── index.html          # Web client interface
├── secrets/            # Configuration secrets
├── collected_data/     # Collected hand landmark data
└── models/             # Stored model configurations
```

### Adding New Gestures

To add a new gesture:

1. Create a new detection method in the `GestureRecognitionService` class
2. Add a feature flag for the gesture in the configuration
3. Update the client-side feedback visualization
4. Add gesture to the data collection pipeline

## 🔍 ML Capabilities

### Hand Detection

The system uses MediaPipe Hands, a machine learning pipeline for high-fidelity hand tracking. It employs:
- Palm detection model
- Hand landmark model (21 3D landmarks)
- Real-time performance optimization

### Data Collection

The data collection pipeline enables:
- Collection of labeled hand landmark data
- Dataset versioning and management
- Export for model training

### Model Serving

The model serving pipeline provides:
- Model configuration management
- Model versioning
- A/B testing capability
- Performance monitoring
