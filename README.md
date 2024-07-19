# Road Irregularities Detection 

This project is a real-time road irregularities detection system designed to identify potholes, open manholes, and unmarked bumps using YOLOv10 and FastAPI. The system processes video streams from mobile cameras or webcams to enhance road safety monitoring.

## Features

- Real-time detection of road irregularities.
- Processes video streams from mobile cameras or webcams.
- Efficient object detection and annotation using YOLOv10.
- Provides a video streaming server using FastAPI.


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Krishnapopat/Road-Irregularities-Detection
    cd road-irregularities-detection
    ```

2. Clone the YOLOv10 repository:
    ```sh
    git clone https://github.com/ultralytics/yolov10.git
    cd yolov10
    pip install .
    cd ..
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Server

1. Start the FastAPI server with Uvicorn:
    ```sh
    uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-certfile="cert.pem" --ssl-keyfile="key.pem"
    ```

2. Access the server from your mobile browser using the IP address of your laptop. Replace `<ip-address>` with your laptop's IP address:
    ```sh
    https://<ip-address>:8000
    ```

## Project Structure

- `server.py`: Main server file that sets up the FastAPI server and handles video streams.
- `static/`: Directory containing static files like `index.html` and `favicon.ico`.
- `requirements.txt`: List of required Python packages.

## Acknowledgements

- [Ultralytics YOLOv10](https://github.com/ultralytics/yolov10)
- [FastAPI](https://fastapi.tiangolo.com/)
- [aiortc](https://aiortc.readthedocs.io/)

## License

This project is licensed under the MIT License.
