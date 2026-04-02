from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
import threading
import time
from multiprocessing import Array, Lock

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

inspire_tip_indices = [4, 9, 14, 19, 24]
Inspire_Num_Motors = 6

kTopicInspireCtrlLeft    = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight   = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft   = "rt/inspire_hand/state/l"
kTopicInspireStateRight  = "rt/inspire_hand/state/r"
kTopicInspireTouchLeft   = "rt/inspire_hand/touch/l"
kTopicInspireTouchRight  = "rt/inspire_hand/touch/r"

TOUCH_FIELDS = [
    ("fingerone_tip_touch", 9),
    ("fingerone_top_touch", 96),
    ("fingerone_palm_touch", 80),
    ("fingertwo_tip_touch", 9),
    ("fingertwo_top_touch", 96),
    ("fingertwo_palm_touch", 80),
    ("fingerthree_tip_touch", 9),
    ("fingerthree_top_touch", 96),
    ("fingerthree_palm_touch", 80),
    ("fingerfour_tip_touch", 9),
    ("fingerfour_top_touch", 96),
    ("fingerfour_palm_touch", 80),
    ("fingerfive_tip_touch", 9),
    ("fingerfive_top_touch", 96),
    ("fingerfive_middle_touch", 9),
    ("fingerfive_palm_touch", 96),
    ("palm_touch", 112),
]
TOUCH_FIELD_NAMES = [name for name, _ in TOUCH_FIELDS]

class Inspire_Controller_FTP:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False):
        from inspire_sdkpy import inspire_dds, inspire_hand_defaut
        self.inspire_dds = inspire_dds
        self.inspire_hand_defaut = inspire_hand_defaut

        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)

        # Re-initialize DDS — required for inspire hand IDL types to register.
        # The arm controller already called this, but a second call is needed
        # to register the inspire_hand DDS types in the same participant.
        try:
            ChannelFactoryInitialize(0, "")
        except Exception as e:
            logger_mp.warning(f"[Inspire_Controller] ChannelFactoryInitialize note: {e}")

        # Publishers and subscribers — all in main process (using Thread, not Process)
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        self.LeftHandTouch_subscriber = ChannelSubscriber(kTopicInspireTouchLeft, inspire_dds.inspire_hand_touch)
        self.LeftHandTouch_subscriber.Init()
        self.RightHandTouch_subscriber = ChannelSubscriber(kTopicInspireTouchRight, inspire_dds.inspire_hand_touch)
        self.RightHandTouch_subscriber.Init()

        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        self._touch_lock = threading.Lock()
        self._left_touch_data = None
        self._right_touch_data = None

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state, daemon=True)
        self.subscribe_state_thread.start()

        # Wait for DDS state confirmation
        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0:
                logger_mp.info("[Inspire_Controller] Waiting for hand state DDS...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:
                break
        if any(self.left_hand_state_array) or any(self.right_hand_state_array):
            logger_mp.info(f"[Inspire_Controller] DDS state CONFIRMED. L={[f'{x:.3f}' for x in self.left_hand_state_array]}, R={[f'{x:.3f}' for x in self.right_hand_state_array]}")
        else:
            logger_mp.warning("[Inspire_Controller] DDS state TIMEOUT — is Headless_driver running on the robot?")

        # Thread (not Process) keeps DDS publishers in the same process
        hand_control_thread = threading.Thread(
            target=self._control_loop,
            args=(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array),
            daemon=True
        )
        hand_control_thread.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller] Subscribe thread started.")
        while True:
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0

            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0

            left_touch_msg = self.LeftHandTouch_subscriber.Read()
            if left_touch_msg is not None:
                td = {}
                for field_name, _ in TOUCH_FIELDS:
                    td[field_name] = list(getattr(left_touch_msg, field_name, []))
                with self._touch_lock:
                    self._left_touch_data = td

            right_touch_msg = self.RightHandTouch_subscriber.Read()
            if right_touch_msg is not None:
                td = {}
                for field_name, _ in TOUCH_FIELDS:
                    td[field_name] = list(getattr(right_touch_msg, field_name, []))
                with self._touch_lock:
                    self._right_touch_data = td

            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        left_cmd = self.inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd.angle_set = left_angle_cmd_scaled
        left_cmd.mode = 0b0001
        self.LeftHandCmd_publisher.Write(left_cmd)

        right_cmd = self.inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd.angle_set = right_angle_cmd_scaled
        right_cmd.mode = 0b0001
        self.RightHandCmd_publisher.Write(right_cmd)

    def get_tactile_data(self):
        """Return the latest tactile readings as a dict with left_ee / right_ee keys,
        or None if no touch data has been received yet."""
        with self._touch_lock:
            left = self._left_touch_data
            right = self._right_touch_data
        if left is None and right is None:
            return None
        return {
            "left_ee": left if left is not None else {name: [0] * size for name, size in TOUCH_FIELDS},
            "right_ee": right if right is not None else {name: [0] * size for name, size in TOUCH_FIELDS},
        }

    def _control_loop(self, left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array):
        logger_mp.info("[Inspire_Controller] Control thread started.")
        self.running = True

        left_q_target  = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)
        log_counter = 0

        try:
            while self.running:
                start_time = time.time()
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                state_data = np.concatenate((np.array(self.left_hand_state_array[:]), np.array(self.right_hand_state_array[:])))

                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])):
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx]  = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # Normalized 1.0=open, 0.0=closed → FTP scaled: 1000=open, 0=closed
                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target]

                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)

                log_counter += 1
                if log_counter % 500 == 1:
                    logger_mp.info(f"[Inspire_Controller] Cmd L={scaled_left_cmd} R={scaled_right_cmd}")

                time_elapsed = time.time() - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")
