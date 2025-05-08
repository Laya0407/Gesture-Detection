# Touchless Interaction System

A WebSocket-based, machine learning-powered system for touchless interaction with digital interfaces through hand gestures. This project uses MediaPipe for real-time hand landmark detection and enables natural interaction without physical contact.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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

#### Launch with Docker Compose:
docker-compose up -d
