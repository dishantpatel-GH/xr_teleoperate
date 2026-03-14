#!/usr/bin/env python3
"""
Multi-Camera Image Server

Supports:
- RealSense RGB + Depth (head camera)
- Multiple OpenCV/Arducam wrist cameras

Usage:
    # Head only (RealSense RGB + Depth)
    python multi_camera_server.py --no-wrist
    
    # Head + 1 wrist camera
    python multi_camera_server.py --wrist 0
    
    # Head + 2 wrist cameras  
    python multi_camera_server.py --wrist 0 8
    
    # Custom resolution
    python multi_camera_server.py --wrist 0 8 --width 640 --height 480
"""

import cv2
import zmq
import time
import struct
import argparse
import numpy as np
import pyrealsense2 as rs
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class RealSenseCamera:
    """Intel RealSense camera with RGB and optional Depth"""
    
    def __init__(self, width=640, height=480, fps=30, serial=None, enable_depth=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.enable_depth = enable_depth
        self.pipeline = None
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        
    def start(self):
        """Initialize and start the RealSense pipeline"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        if self.serial:
            config.enable_device(self.serial)
            
        # Check USB type and adjust FPS
        actual_fps = self.fps
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if self.serial and dev.get_info(rs.camera_info.serial_number) == self.serial:
                    usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
                    if usb_type.startswith("2"):
                        actual_fps = 15
                        logger.warning(f"[RealSense] USB {usb_type} detected, using {actual_fps}fps")
                    break
        except Exception as e:
            logger.warning(f"[RealSense] Could not check USB type: {e}")
        
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, actual_fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, actual_fps)
        
        logger.info(f"[RealSense] Starting: {self.width}x{self.height} @ {actual_fps}fps, depth={self.enable_depth}")
        
        try:
            profile = self.pipeline.start(config)
            
            if self.enable_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                logger.info(f"[RealSense] Depth scale: {self.depth_scale}")
            
            # Warm-up frames
            logger.info("[RealSense] Warming up...")
            for i in range(15):
                try:
                    self.pipeline.wait_for_frames(timeout_ms=2000)
                except:
                    pass
            logger.info("[RealSense] Ready!")
            return True
            
        except Exception as e:
            logger.error(f"[RealSense] Failed to start: {e}")
            return False
    
    def get_frames(self):
        """Get RGB and depth frames"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            
            color_frame = aligned.get_color_frame()
            if not color_frame:
                return None, None
                
            rgb = np.asanyarray(color_frame.get_data())
            
            depth = None
            if self.enable_depth:
                depth_frame = aligned.get_depth_frame()
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data())
            
            return rgb, depth
            
        except Exception as e:
            logger.warning(f"[RealSense] Frame error: {e}")
            return None, None
    
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()


class OpenCVCamera:
    """OpenCV-based camera (Arducam, USB webcam, etc.)"""
    
    def __init__(self, device_id, width=640, height=480, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_open = False
        
    def start(self):
        """Initialize and open the camera"""
        logger.info(f"[Camera {self.device_id}] Opening...")
        
        # Try V4L2 backend first
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            logger.warning(f"[Camera {self.device_id}] V4L2 failed, trying default backend")
            self.cap = cv2.VideoCapture(self.device_id)
        
        if not self.cap.isOpened():
            logger.error(f"[Camera {self.device_id}] Failed to open!")
            return False
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify with test read
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.error(f"[Camera {self.device_id}] Cannot read frames!")
            self.cap.release()
            return False
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"[Camera {self.device_id}] Opened: {actual_w}x{actual_h}")
        
        self.is_open = True
        return True
    
    def get_frame(self):
        """Get a single frame"""
        if not self.is_open or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def stop(self):
        if self.cap:
            self.cap.release()
        self.is_open = False


class MultiCameraServer:
    """Image server supporting RealSense head + multiple wrist cameras"""
    
    def __init__(self, width=640, height=480, fps=30, port=5555, 
                 realsense_serial=None, wrist_ports=None, enable_depth=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.enable_depth = enable_depth
        
        # Initialize RealSense (head camera)
        self.realsense = RealSenseCamera(
            width=width, height=height, fps=fps,
            serial=realsense_serial, enable_depth=enable_depth
        )
        
        # Initialize wrist cameras
        self.wrist_cameras = []
        self.wrist_ports = wrist_ports or []
        
        # ZMQ
        self.context = None
        self.socket = None
        
    def start(self):
        """Start all cameras and ZMQ server"""
        logger.info("=" * 60)
        logger.info("Multi-Camera Image Server")
        logger.info("=" * 60)
        
        # Start RealSense
        if not self.realsense.start():
            logger.error("Failed to start RealSense!")
            return False
        
        # Start wrist cameras
        for port in self.wrist_ports:
            cam = OpenCVCamera(port, self.width, self.height, self.fps)
            if cam.start():
                self.wrist_cameras.append(cam)
                logger.info(f"[Wrist] Camera {port} ready")
            else:
                logger.warning(f"[Wrist] Camera {port} failed to open, skipping")
        
        # Start ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        
        # Summary
        logger.info("-" * 60)
        logger.info(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
        logger.info(f"RealSense: RGB + {'Depth' if self.enable_depth else 'No Depth'}")
        logger.info(f"Wrist cameras: {len(self.wrist_cameras)} of {len(self.wrist_ports)} requested")
        logger.info(f"ZMQ: tcp://*:{self.port}")
        
        # Calculate output dimensions
        num_head = 2 if self.enable_depth else 1  # RGB + Depth or just RGB
        num_wrist = len(self.wrist_cameras)
        total_width = (num_head + num_wrist) * self.width
        logger.info(f"Output: {total_width}x{self.height} ({num_head} head + {num_wrist} wrist)")
        logger.info("=" * 60)
        
        return True
    
    def run(self):
        """Main loop - capture and send frames"""
        logger.info("Starting stream... (Ctrl+C to stop)")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frames = []
                
                # Get RealSense frames
                rgb, depth = self.realsense.get_frames()
                if rgb is None:
                    logger.warning("[RealSense] No frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Add depth first (colorized), then RGB
                if self.enable_depth and depth is not None:
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    frames.append(depth_color)
                
                frames.append(rgb)
                
                # Get wrist camera frames
                for i, cam in enumerate(self.wrist_cameras):
                    wrist_frame = cam.get_frame()
                    if wrist_frame is not None:
                        # Resize if needed
                        if wrist_frame.shape[0] != self.height or wrist_frame.shape[1] != self.width:
                            wrist_frame = cv2.resize(wrist_frame, (self.width, self.height))
                        frames.append(wrist_frame)
                    else:
                        # Use black frame as placeholder
                        logger.warning(f"[Wrist {i}] No frame, using placeholder")
                        frames.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
                
                # Concatenate all frames horizontally
                if len(frames) > 1:
                    combined = cv2.hconcat(frames)
                else:
                    combined = frames[0]
                
                # Encode and send
                ret, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    self.socket.send(buffer.tobytes())
                    frame_count += 1
                    
                    # Log FPS every 60 frames
                    if frame_count % 60 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        logger.info(f"[Stream] Frame {frame_count}, FPS: {fps:.1f}, Size: {combined.shape[1]}x{combined.shape[0]}")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Clean up resources"""
        logger.info("Shutting down...")
        
        self.realsense.stop()
        
        for cam in self.wrist_cameras:
            cam.stop()
        
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        
        logger.info("Server stopped.")


def auto_detect_realsense():
    """Auto-detect RealSense serial number"""
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            serial = devices[0].get_info(rs.camera_info.serial_number)
            name = devices[0].get_info(rs.camera_info.name)
            logger.info(f"[Auto-detect] Found RealSense: {name} ({serial})")
            return serial
    except Exception as e:
        logger.error(f"[Auto-detect] RealSense error: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Multi-Camera Image Server')
    parser.add_argument('--wrist', type=int, nargs='*', default=[], 
                        help='Wrist camera port(s), e.g., --wrist 0 8')
    parser.add_argument('--no-wrist', action='store_true',
                        help='Disable wrist cameras')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth stream')
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('--port', type=int, default=5555,
                        help='ZMQ port (default: 5555)')
    parser.add_argument('--realsense', type=str, default=None,
                        help='RealSense serial number (auto-detect if not specified)')
    args = parser.parse_args()
    
    # Auto-detect RealSense
    realsense_serial = args.realsense
    if not realsense_serial:
        realsense_serial = auto_detect_realsense()
        if not realsense_serial:
            logger.error("No RealSense camera found!")
            return 1
    
    # Determine wrist ports
    wrist_ports = [] if args.no_wrist else args.wrist
    
    # Create and run server
    server = MultiCameraServer(
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port,
        realsense_serial=realsense_serial,
        wrist_ports=wrist_ports,
        enable_depth=not args.no_depth
    )
    
    if server.start():
        server.run()
    else:
        logger.error("Failed to start server!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
