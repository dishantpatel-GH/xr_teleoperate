import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import logging
import glob
import subprocess
import pickle

# Optional: logging_mp (fallback to standard logging if not available)
try:
    import logging_mp
    logger_mp = logging_mp.getLogger(__name__)
except (ImportError, AttributeError):
    # Fallback to standard logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    logger_mp = logging.getLogger(__name__)

# Optional: RealSense support (only needed if using RealSense cameras)
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logger_mp.warning("pyrealsense2 not available - RealSense cameras disabled")


def detect_realsense_v4l2_nodes():
    """
    Detect RealSense RGB and Depth video nodes via v4l2.
    
    Returns:
        dict: {'rgb': [video_ids], 'depth': [video_ids], 'ir': [video_ids]}
    """
    result = {'rgb': [], 'depth': [], 'ir': []}
    video_devices = glob.glob('/dev/video*')
    
    print("\n" + "=" * 70)
    print("[V4L2 SCAN] Scanning all /dev/video* devices...")
    print("=" * 70)
    
    for device_path in sorted(video_devices, key=lambda x: int(x.replace('/dev/video', '')) if x.replace('/dev/video', '').isdigit() else 999):
        try:
            device_id = int(device_path.replace('/dev/video', ''))
        except ValueError:
            continue
        
        try:
            # Get formats for this device
            v4l_result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{device_id}', '--list-formats'],
                capture_output=True, text=True, timeout=2
            )
            
            formats_found = []
            if v4l_result.returncode == 0:
                output = v4l_result.stdout
                if 'Z16' in output:
                    formats_found.append('Z16(Depth)')
                if 'YUYV' in output:
                    formats_found.append('YUYV(RGB)')
                if 'GREY' in output:
                    formats_found.append('GREY(IR)')
                if 'UYVY' in output:
                    formats_found.append('UYVY')
                if 'MJPG' in output or 'Motion-JPEG' in output:
                    formats_found.append('MJPG')
            
            # Check if it's RealSense
            udev_result = subprocess.run(
                ['udevadm', 'info', '--query=property', f'--name=/dev/video{device_id}'],
                capture_output=True, text=True, timeout=2
            )
            
            is_realsense = 'RealSense' in udev_result.stdout
            is_arducam = 'Arducam' in udev_result.stdout or 'arducam' in udev_result.stdout.lower()
            
            device_type = "RealSense" if is_realsense else ("Arducam" if is_arducam else "Unknown")
            formats_str = ', '.join(formats_found) if formats_found else 'No capture formats'
            
            print(f"  /dev/video{device_id:2d} | {device_type:10s} | {formats_str}")
            
            if not is_realsense:
                continue
            
            # Classify RealSense nodes
            if v4l_result.returncode == 0:
                output = v4l_result.stdout
                if 'Z16' in output or '16-bit Depth' in output:
                    result['depth'].append(device_id)
                elif 'YUYV' in output and 'GREY' not in output:
                    result['rgb'].append(device_id)
                elif 'GREY' in output or 'UYVY' in output:
                    result['ir'].append(device_id)
        except Exception as e:
            print(f"  /dev/video{device_id:2d} | ERROR: {e}")
            continue
    
    print("-" * 70)
    print(f"[V4L2 RESULT] RealSense nodes: RGB={result['rgb']}, Depth={result['depth']}, IR={result['ir']}")
    print("=" * 70 + "\n")
    return result


def detect_realsense_cameras():
    """
    Auto-detect all connected RealSense cameras using pyrealsense2 SDK.
    
    Returns:
        list: List of RealSense camera serial numbers
    """
    print("\n" + "=" * 70)
    print("[REALSENSE SDK] Detecting RealSense cameras via pyrealsense2...")
    print("=" * 70)
    
    if not REALSENSE_AVAILABLE:
        print("  ❌ pyrealsense2 not installed - skipping RealSense detection")
        print("=" * 70 + "\n")
        return []
    
    realsense_serials = []
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("  No RealSense devices found via SDK")
        
        for i, dev in enumerate(devices):
            serial = str(dev.get_info(rs.camera_info.serial_number))
            name = str(dev.get_info(rs.camera_info.name))
            fw = str(dev.get_info(rs.camera_info.firmware_version))
            usb = str(dev.get_info(rs.camera_info.usb_type_descriptor)) if dev.supports(rs.camera_info.usb_type_descriptor) else "N/A"
            realsense_serials.append(serial)
            print(f"  [{i}] {name}")
            print(f"      Serial: {serial}")
            print(f"      Firmware: {fw}")
            print(f"      USB Type: {usb}")
        
        print("-" * 70)
        print(f"[REALSENSE SDK RESULT] {len(realsense_serials)} cameras found: {realsense_serials}")
    except Exception as e:
        print(f"  ❌ RealSense SDK detection failed: {e}")
    
    print("=" * 70 + "\n")
    return realsense_serials


def detect_arduino_cameras():
    """
    Auto-detect all working Arduino/Arducam cameras.
    
    Returns:
        list: List of working Arduino camera device IDs
    """
    print("\n" + "=" * 70)
    print("[ARDUCAM] Detecting Arducam/Arduino cameras...")
    print("=" * 70)
    
    arduino_cameras = []
    video_devices = glob.glob('/dev/video*')
    
    for device_path in sorted(video_devices, key=lambda x: int(x.replace('/dev/video', '')) if x.replace('/dev/video', '').isdigit() else 999):
        try:
            device_id = int(device_path.replace('/dev/video', ''))
        except ValueError:
            continue
        
        # Check if it's Arduino
        is_arduino = False
        try:
            result = subprocess.run(
                ['udevadm', 'info', '--query=property', f'--name=/dev/video{device_id}'],
                capture_output=True, text=True, timeout=2
            )
            
            if result.returncode == 0:
                output = result.stdout
                is_arducam = ('Arducam' in output or 'arducam' in output.lower() or 'Arduino' in output)
                has_capture = 'ID_V4L_CAPABILITIES=:capture:' in output
                is_arduino = is_arducam and has_capture
        except:
            pass
        
        if not is_arduino:
            continue
        
        print(f"  Testing /dev/video{device_id} (Arducam detected)...")
        
        # Test if camera can reliably read frames (test 5 times with setup)
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"    ❌ Failed to open device")
            continue
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm up - discard first frame
        cap.read()
        time.sleep(0.1)
        
        # Test 5 times
        success_count = 0
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                success_count += 1
            time.sleep(0.05)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        time.sleep(0.1)  # Let camera close properly
        
        # Only accept if all 5 tests passed
        if success_count == 5:
            arduino_cameras.append(device_id)
            print(f"    ✓ PASS ({success_count}/5 frames) Resolution: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
        else:
            print(f"    ❌ FAIL ({success_count}/5 frames) - unstable")
    
    print("-" * 70)
    print(f"[ARDUCAM RESULT] {len(arduino_cameras)} working cameras: {arduino_cameras}")

    # Fallback: if no Arducam found, try all V4L2 capture devices with OpenCV
    if not arduino_cameras:
        print("\n[FALLBACK] No Arducam found. Testing all /dev/video* with OpenCV...")
        for device_path in sorted(video_devices, key=lambda x: int(x.replace('/dev/video', '')) if x.replace('/dev/video', '').isdigit() else 999):
            try:
                device_id = int(device_path.replace('/dev/video', ''))
            except ValueError:
                continue
            cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.read()
            time.sleep(0.1)
            ok = 0
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok += 1
                time.sleep(0.05)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            time.sleep(0.1)
            if ok >= 2:
                arduino_cameras.append(device_id)
                print(f"  ✓ /dev/video{device_id} works ({ok}/3 frames) {actual_w}x{actual_h}")

    print("=" * 70 + "\n")
    return arduino_cameras


class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        img_shape: [height, width]
        serial_number: serial number
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.init_realsense()

    def init_realsense(self):
        print("\n" + "-" * 70)
        print(f"[REALSENSE INIT] Initializing camera: {self.serial_number}")
        print("-" * 70)
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)
            print(f"  Device: {self.serial_number}")

        print(f"  Enabling RGB stream: {self.img_shape[1]}x{self.img_shape[0]} @ {self.fps}fps (BGR8)")
        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            print(f"  Enabling DEPTH stream: {self.img_shape[1]}x{self.img_shape[0]} @ {self.fps}fps (Z16)")
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        print(f"  Starting pipeline...", flush=True)
        try:
            profile = self.pipeline.start(config)
            print(f"  ✓ Pipeline started successfully", flush=True)
        except Exception as e:
            print(f"  ❌ Pipeline start FAILED: {e}", flush=True)
            raise
        
        self._device = profile.get_device()
        if self._device is None:
            print(f"  ❌ ERROR: get_device() returned None", flush=True)
        else:
            dev_name = self._device.get_info(rs.camera_info.name)
            usb_type = "N/A"
            try:
                usb_type = self._device.get_info(rs.camera_info.usb_type_descriptor)
            except:
                pass
            print(f"  Device Name: {dev_name}", flush=True)
            print(f"  USB Type: {usb_type}", flush=True)
        
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()
            print(f"  Depth Scale: {self.g_depth_scale} (1 unit = {self.g_depth_scale * 1000:.2f}mm)", flush=True)

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        print(f"  Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}, "
              f"ppx={self.intrinsics.ppx:.1f}, ppy={self.intrinsics.ppy:.1f}", flush=True)
        
        print(f"  ✓ RealSense {self.serial_number} ready: RGB={self.img_shape}, depth={'ENABLED' if self.enable_depth else 'DISABLED'}", flush=True)
        print("-" * 70 + "\n", flush=True)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None
        return color_image, depth_image

    def release(self):
        self.pipeline.stop()


class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        decive_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.working = False
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10
        
        print("\n" + "-" * 70)
        print(f"[OPENCV INIT] Initializing camera: /dev/video{self.id}")
        print("-" * 70)
        print(f"  Requested: {self.img_shape[1]}x{self.img_shape[0]} @ {self.fps}fps (MJPG)")
        
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"  ❌ ERROR: Cannot open /dev/video{self.id}")
            return
        print(f"  ✓ Device opened")
            
        # Configure video format
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Reduce USB bandwidth by tuning buffer size (smaller = less bandwidth spikes)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        
        # Optimize USB settings (platform-specific)
        try:
            # Disable auto-exposure to reduce bandwidth variance
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 150)     # Fixed exposure
        except:
            pass

        # Get actual settings
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"  Actual: {actual_w}x{actual_h} @ {actual_fps:.0f}fps ({fourcc_str})")

        # Test if the camera can read frames
        if self._can_read_frame():
            self.working = True
            print(f"  ✓ Frame test PASSED - camera ready")
        else:
            print(f"  ❌ Frame test FAILED - cannot read frames")
            self.release()
        print("-" * 70 + "\n")

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        if self.cap:
            self.cap.release()

    def get_frame(self):
        if not self.working:
            return None
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image


class ImageServer:
    def __init__(self, config, port = 5555, Unit_Test = False):
        """
        config example1:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            'wrist_camera_type': 'realsense', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # realsense camera's serial number
        }

        config example2:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'realsense',                                  # opencv or realsense
            'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
            'head_camera_id_numbers': ["218622271739"],                       # realsense camera's serial number
            'wrist_camera_type': 'opencv', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': [0,1],                                 # '/dev/video0' and '/dev/video1' (opencv)
        }

        If you are not using the wrist camera, you can comment out its configuration, like this below:
        config:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            #'wrist_camera_type': 'realsense', 
            #'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            #'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # serial number (realsense)
        }
        """
        print("\n" + "=" * 70)
        print("[IMAGE SERVER] Starting ImageServer initialization...")
        print("=" * 70)
        print(f"\n[CONFIG] Received configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
        
        self.config = config
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])      # (height, width)
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])

        self.wrist_camera_type = config.get('wrist_camera_type', None)
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])    # (height, width)
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', None)

        self.port = port
        self.Unit_Test = Unit_Test


        # Initialize head cameras
        self.head_cameras = []
        if self.head_camera_type == 'opencv':
            for device_id in self.head_camera_id_numbers:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                if camera.working:
                    self.head_cameras.append(camera)
                else:
                    logger_mp.warning(f"[Image Server] Skipping failed head camera {device_id}")
        elif self.head_camera_type == 'realsense':
            if not REALSENSE_AVAILABLE:
                raise ImportError("RealSense camera requested but pyrealsense2 not installed. Install: pip install pyrealsense2")
            for serial_number in self.head_camera_id_numbers:
                enable_depth = self.config.get('enable_depth', False)
                camera = RealSenseCamera(img_shape=self.head_image_shape, fps=self.fps, serial_number=serial_number, enable_depth=enable_depth)
                self.head_cameras.append(camera)
        else:
            logger_mp.warning(f"[Image Server] Unsupported head_camera_type: {self.head_camera_type}")

        # Initialize wrist cameras if provided
        self.wrist_cameras = []
        if self.wrist_camera_type and self.wrist_camera_id_numbers:
            if self.wrist_camera_type == 'opencv':
                for device_id in self.wrist_camera_id_numbers:
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.wrist_image_shape, fps=self.fps)
                    if camera.working:
                        self.wrist_cameras.append(camera)
                    else:
                        logger_mp.warning(f"[Image Server] Skipping failed wrist camera {device_id}")
            elif self.wrist_camera_type == 'realsense':
                if not REALSENSE_AVAILABLE:
                    raise ImportError("RealSense camera requested but pyrealsense2 not installed. Install: pip install pyrealsense2")
                for serial_number in self.wrist_camera_id_numbers:
                    enable_depth = self.config.get('enable_depth', False)
                    camera = RealSenseCamera(img_shape=self.wrist_image_shape, fps=self.fps, serial_number=serial_number, enable_depth=enable_depth)
                    self.wrist_cameras.append(camera)
            else:
                logger_mp.warning(f"[Image Server] Unsupported wrist_camera_type: {self.wrist_camera_type}")

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        # Print final camera summary
        print("\n" + "=" * 70)
        print("[CAMERA SUMMARY] Active cameras:")
        print("=" * 70)
        
        print(f"\n  HEAD CAMERAS ({len(self.head_cameras)} active):")
        for i, cam in enumerate(self.head_cameras):
            if isinstance(cam, OpenCVCamera):
                actual_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_fps = cam.cap.get(cv2.CAP_PROP_FPS)
                print(f"    [{i}] OpenCV /dev/video{cam.id}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
            elif isinstance(cam, RealSenseCamera):
                depth_str = "RGB+DEPTH" if cam.enable_depth else "RGB only"
                print(f"    [{i}] RealSense {cam.serial_number}: {cam.img_shape[1]}x{cam.img_shape[0]} @ {cam.fps}fps ({depth_str})")

        print(f"\n  WRIST CAMERAS ({len(self.wrist_cameras)} active):")
        if not self.wrist_cameras:
            print("    (none)")
        for i, cam in enumerate(self.wrist_cameras):
            if isinstance(cam, OpenCVCamera):
                actual_h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_fps = cam.cap.get(cv2.CAP_PROP_FPS)
                print(f"    [{i}] OpenCV /dev/video{cam.id}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
            elif isinstance(cam, RealSenseCamera):
                depth_str = "RGB+DEPTH" if cam.enable_depth else "RGB only"
                print(f"    [{i}] RealSense {cam.serial_number}: {cam.img_shape[1]}x{cam.img_shape[0]} @ {cam.fps}fps ({depth_str})")

        print(f"\n  ZMQ Server: tcp://*:{self.port}")
        print("=" * 70)
        print("\n[STREAMING] Starting frame capture... (Ctrl+C to stop)\n")



    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            logger_mp.info(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        for cam in self.head_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        logger_mp.info("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            # Check if we have any cameras at all
            if not self.head_cameras and not self.wrist_cameras:
                logger_mp.error("[Image Server] No cameras available! Exiting...")
                return
            
            consecutive_failures = 0
            max_failures = 30  # Exit after 30 consecutive failures (1 second at 30fps)
            self.frame_count = 0
            start_time = time.time()
            last_stats_time = start_time
            frames_since_stats = 0
            
            while True:
                self.frame_count += 1
                frames_since_stats += 1
                
                # Print stats every 5 seconds
                current_time = time.time()
                if current_time - last_stats_time >= 5.0:
                    elapsed = current_time - start_time
                    fps = frames_since_stats / (current_time - last_stats_time)
                    print(f"[STREAMING] Frame {self.frame_count} | FPS: {fps:.1f} | Elapsed: {elapsed:.1f}s")
                    last_stats_time = current_time
                    frames_since_stats = 0
                # Capture head camera frames (RGB + optional Depth)
                head_frames = []
                head_depth_frames = []
                if self.head_cameras:
                    for cam in self.head_cameras:
                        if self.head_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.warning(f"[Image Server] Head camera {cam.id} frame read failed, skipping frame")
                                continue  # Skip this camera, try others
                        elif self.head_camera_type == 'realsense':
                            color_image, depth_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.warning(f"[Image Server] Head camera {cam.serial_number} frame read failed, skipping frame")
                                continue  # Skip this camera, try others
                            if self.config.get('enable_depth', False) and depth_image is not None:
                                head_depth_frames.append(depth_image)
                        head_frames.append(color_image)
                
                # Capture wrist camera frames (RGB + optional Depth)
                wrist_frames = []
                wrist_depth_frames = []
                if self.wrist_cameras:
                    for cam in self.wrist_cameras:
                        if self.wrist_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.warning(f"[Image Server] Wrist camera {cam.id} frame read failed, skipping frame")
                                continue  # Skip this camera, try others
                        elif self.wrist_camera_type == 'realsense':
                            color_image, depth_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.warning(f"[Image Server] Wrist camera {cam.serial_number} frame read failed, skipping frame")
                                continue  # Skip this camera, try others
                            if self.config.get('enable_depth', False) and depth_image is not None:
                                wrist_depth_frames.append(depth_image)
                        wrist_frames.append(color_image)
                
                # Convert depth frames to colorized images for display
                head_depth_colored = []
                if head_depth_frames and self.config.get('enable_depth', False):
                    for depth_img in head_depth_frames:
                        # Normalize depth to 0-255 range for visualization
                        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                        depth_uint8 = depth_normalized.astype(np.uint8)
                        # Apply colormap for better visualization
                        depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                        head_depth_colored.append(depth_colormap)
                
                # Combine frames based on what's available
                # Order: [Head RGB] [Head Depth] [Wrist1] [Wrist2]
                all_frames = []
                
                if head_frames:
                    all_frames.extend(head_frames)
                
                if head_depth_colored:
                    all_frames.extend(head_depth_colored)
                
                if wrist_frames:
                    all_frames.extend(wrist_frames)
                
                if all_frames:
                    full_color = cv2.hconcat(all_frames)
                    consecutive_failures = 0  # Reset failure counter
                else:
                    # No frames captured
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger_mp.error(f"[Image Server] {consecutive_failures} consecutive frame failures. Cameras may have disconnected. Exiting...")
                        break
                    continue

                ret, buffer = cv2.imencode('.jpg', full_color)
                if not ret:
                    logger_mp.error("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                # Check if we have depth data to send
                enable_depth = self.config.get('enable_depth', False)
                has_depth = enable_depth and (head_depth_frames or wrist_depth_frames)
                
                if has_depth:
                    # Combine depth frames if multiple cameras
                    head_depth = None
                    wrist_depth = None
                    
                    if head_depth_frames:
                        if len(head_depth_frames) > 1:
                            head_depth = np.hstack(head_depth_frames)
                        else:
                            head_depth = head_depth_frames[0]
                    
                    if wrist_depth_frames:
                        if len(wrist_depth_frames) > 1:
                            wrist_depth = np.hstack(wrist_depth_frames)
                        else:
                            wrist_depth = wrist_depth_frames[0]
                    
                    # Build structured message with RGB + Depth using pickle
                    message_data = {
                        'rgb': jpg_bytes,
                        'head_depth': head_depth,
                        'wrist_depth': wrist_depth,
                        'frame_id': self.frame_count,
                        'timestamp': time.time()
                    }
                    message = pickle.dumps(message_data)
                    
                    # Log depth info periodically
                    if self.frame_count % 150 == 1:  # Every 5 seconds at 30fps
                        depth_info = []
                        if head_depth is not None:
                            depth_info.append(f"head_depth={head_depth.shape}")
                        if wrist_depth is not None:
                            depth_info.append(f"wrist_depth={wrist_depth.shape}")
                        print(f"[DEPTH] Transmitting: {', '.join(depth_info)}")
                else:
                    # RGB only - send as before for backward compatibility
                    if self.Unit_Test:
                        timestamp = time.time()
                        frame_id = self.frame_count
                        header = struct.pack('dI', timestamp, frame_id)  # 8-byte double, 4-byte unsigned int
                        message = header + jpg_bytes
                    else:
                        message = jpg_bytes

                self.socket.send(message)

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            logger_mp.warning("[Image Server] Interrupted by user.")
        finally:
            self._close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Server with Flexible Camera Configuration')
    parser.add_argument('--auto', action='store_true', help='Auto-detect Arduino cameras')
    parser.add_argument('--realsense', type=int, nargs='+', metavar='N', help='RealSense camera indices (e.g., --realsense 2 4)')
    parser.add_argument('--arduino', type=int, nargs='+', metavar='N', help='Arduino/OpenCV camera indices (e.g., --arduino 0 1)')
    parser.add_argument('--depth', action='store_true', help='Enable depth stream for RealSense cameras')
    parser.add_argument('--port', type=int, default=5555, help='Server port (default: 5555)')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate (default: 30)')
    args = parser.parse_args()
    
    # Priority: explicit camera args > auto-detection > manual config
    if args.realsense or args.arduino:
        # User specified cameras explicitly
        print("=" * 70)
        print("Using user-specified camera configuration")
        print("=" * 70)
        
        realsense_ids = args.realsense if args.realsense else []
        arduino_ids = args.arduino if args.arduino else []
        
        print(f"RealSense cameras: {realsense_ids if realsense_ids else 'None'}")
        print(f"Arduino cameras: {arduino_ids if arduino_ids else 'None'}")
        
        # Determine head and wrist assignment
        head_cameras = []
        head_type = None
        wrist_cameras = []
        wrist_type = None
        
        # If 2+ RealSense: use them as head (binocular)
        if len(realsense_ids) >= 2:
            head_cameras = realsense_ids[:2]  # First 2 for head
            head_type = 'realsense'
            # Remaining RealSense + Arduino go to wrist
            if len(realsense_ids) > 2:
                wrist_cameras.extend(realsense_ids[2:])
                wrist_type = 'realsense'
            if arduino_ids:
                if wrist_type is None:
                    wrist_type = 'opencv'
                    wrist_cameras = arduino_ids
                else:
                    # Mixed: can't mix types in same category, so add Arduino to wrist as opencv
                    logger_mp.warning("Mixed camera types detected. Using RealSense for head, Arduino for wrist.")
                    wrist_type = 'opencv'
                    wrist_cameras = arduino_ids
        # If 1 RealSense: use it as head, Arduino as wrist
        elif len(realsense_ids) == 1:
            head_cameras = realsense_ids
            head_type = 'realsense'
            if arduino_ids:
                wrist_cameras = arduino_ids
                wrist_type = 'opencv'
        # Only Arduino cameras
        elif arduino_ids:
            if len(arduino_ids) >= 3:
                # First for head, rest for wrist
                head_cameras = [arduino_ids[0]]
                wrist_cameras = arduino_ids[1:]
            elif len(arduino_ids) == 2:
                head_cameras = [arduino_ids[0]]
                wrist_cameras = [arduino_ids[1]]
            else:
                head_cameras = arduino_ids
                wrist_cameras = []
            head_type = 'opencv'
            wrist_type = 'opencv' if wrist_cameras else None
        
        print(f"\nAssignment:")
        print(f"  Head: {head_type} {head_cameras}")
        print(f"  Wrist: {wrist_type} {wrist_cameras if wrist_cameras else 'None'}")
        print("=" * 70)
        print()
        
        # Build config
        config = {'fps': args.fps, 'enable_depth': args.depth}
        
        if head_type == 'realsense':
            if not REALSENSE_AVAILABLE:
                print("❌ Error: pyrealsense2 not installed. Run: pip install pyrealsense2")
                exit(1)
            # Get serial numbers for RealSense
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) < len(head_cameras):
                print(f"❌ Error: Requested {len(head_cameras)} RealSense cameras but only {len(devices)} connected")
                exit(1)
            # Map video indices to serial numbers (approximate - uses device order)
            serials = [str(devices[i].get_info(rs.camera_info.serial_number)) for i in range(len(head_cameras))]
            config['head_camera_type'] = 'realsense'
            config['head_camera_image_shape'] = [480, 640]
            config['head_camera_id_numbers'] = serials
            if args.depth:
                print("  Depth: ENABLED for RealSense cameras")
        else:
            config['head_camera_type'] = 'opencv'
            config['head_camera_image_shape'] = [480, 640]
            config['head_camera_id_numbers'] = head_cameras
        
        if wrist_cameras:
            if wrist_type == 'realsense':
                ctx = rs.context()
                devices = ctx.query_devices()
                # Use remaining devices for wrist
                offset = len(head_cameras) if head_type == 'realsense' else 0
                serials = [str(devices[offset + i].get_info(rs.camera_info.serial_number)) for i in range(len(wrist_cameras))]
                config['wrist_camera_type'] = 'realsense'
                config['wrist_camera_image_shape'] = [480, 640]
                config['wrist_camera_id_numbers'] = serials
            else:
                config['wrist_camera_type'] = 'opencv'
                config['wrist_camera_image_shape'] = [480, 640]
                config['wrist_camera_id_numbers'] = wrist_cameras
    
    elif args.auto:
        # Auto-detect all cameras
        print("=" * 70)
        print("Auto-detecting cameras...")
        print("=" * 70)
        
        # Detect RealSense via v4l2 to get RGB and depth nodes
        realsense_nodes = detect_realsense_v4l2_nodes()
        realsense_serials = detect_realsense_cameras()
        arduino_ids = detect_arduino_cameras()
        
        print(f"\nDetection summary:")
        print(f"  RealSense RGB nodes: {realsense_nodes['rgb']}")
        print(f"  RealSense depth nodes: {realsense_nodes['depth']}")
        print(f"  RealSense devices (SDK): {len(realsense_serials)}")
        print(f"  Arduino cameras: {arduino_ids}")
        
        if not realsense_serials and not arduino_ids:
            print("\n❌ No cameras found!")
            print("   Check connections and permissions")
            exit(1)
        
        print(f"\n✓ Detected {len(realsense_serials)} RealSense + {len(arduino_ids)} Arduino cameras")
        
        # Auto-assign cameras based on what was found
        head_cameras = []
        head_type = None
        wrist_cameras = []
        wrist_type = None
        
        # Priority: Use RealSense for head if available (better quality)
        if len(realsense_serials) >= 2:
            # 2+ RealSense: use first 2 as binocular head
            head_cameras = realsense_serials[:2]
            head_type = 'realsense'
            # Remaining RealSense go to wrist
            if len(realsense_serials) > 2:
                wrist_cameras = realsense_serials[2:]
                wrist_type = 'realsense'
            # Arduino cameras go to wrist
            if arduino_ids:
                if wrist_type is None:
                    wrist_cameras = arduino_ids
                    wrist_type = 'opencv'
                else:
                    # Already have RealSense wrist, add Arduino too (but can't mix types)
                    wrist_cameras.extend(arduino_ids)
        elif len(realsense_serials) == 1:
            # 1 RealSense: use as head
            head_cameras = realsense_serials
            head_type = 'realsense'
            # Arduino cameras go to wrist
            if arduino_ids:
                wrist_cameras = arduino_ids
                wrist_type = 'opencv'
        else:
            # No RealSense, use Arduino
            if len(arduino_ids) >= 3:
                head_cameras = [arduino_ids[0]]
                wrist_cameras = arduino_ids[1:]
            elif len(arduino_ids) == 2:
                head_cameras = [arduino_ids[0]]
                wrist_cameras = [arduino_ids[1]]
            else:
                head_cameras = arduino_ids
                wrist_cameras = []
            head_type = 'opencv'
            wrist_type = 'opencv' if wrist_cameras else None
        
        print("\nAuto-configuration:")
        print(f"  Head: {head_type} {head_cameras}")
        print(f"  Wrist: {wrist_type} {wrist_cameras if wrist_cameras else 'None'}")
        if args.depth:
            print("  Depth: ENABLED for RealSense cameras")
        print("=" * 70)
        print()
        
        # Build config
        config = {'fps': args.fps, 'enable_depth': args.depth}
        
        if head_type == 'realsense':
            config['head_camera_type'] = 'realsense'
            config['head_camera_image_shape'] = [480, 640]
            config['head_camera_id_numbers'] = head_cameras
        else:
            config['head_camera_type'] = 'opencv'
            config['head_camera_image_shape'] = [480, 640]
            config['head_camera_id_numbers'] = head_cameras
        
        if wrist_cameras:
            if wrist_type == 'realsense':
                config['wrist_camera_type'] = 'realsense'
                config['wrist_camera_image_shape'] = [480, 640]
                config['wrist_camera_id_numbers'] = wrist_cameras
            else:
                config['wrist_camera_type'] = 'opencv'
                config['wrist_camera_image_shape'] = [480, 640]
                config['wrist_camera_id_numbers'] = wrist_cameras
    else:
        # Manual configuration
        config = {
            'fps': args.fps,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 1280],  # Head camera resolution
            'head_camera_id_numbers': [2],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            'wrist_camera_id_numbers': [1,4],  # Only use working cameras
        }
    
    server = ImageServer(config, port=args.port, Unit_Test=False)
    server.send_process()