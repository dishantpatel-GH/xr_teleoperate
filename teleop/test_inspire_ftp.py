"""
Simple test script for Inspire FTP hands via DDS.
Run the Headless_driver_double.py on the robot first, then run this.

Usage:
    python test_inspire_ftp.py

Commands:
    o  = open both hands  (angle_set = [0,0,0,0,0,0])
    c  = close both hands (angle_set = [1000,1000,1000,1000,1000,1000])
    h  = half open        (angle_set = [500,500,500,500,500,500])
    q  = quit
    Or type 6 comma-separated values, e.g.: 0,0,0,0,1000,1000
"""

import sys
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

kTopicCtrlLeft  = "rt/inspire_hand/ctrl/l"
kTopicCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicStateLeft  = "rt/inspire_hand/state/l"
kTopicStateRight = "rt/inspire_hand/state/r"

def main():
    print("=== Inspire FTP Hand Test ===")
    print("Initializing DDS...")
    ChannelFactoryInitialize(0)

    pub_l = ChannelPublisher(kTopicCtrlLeft, inspire_dds.inspire_hand_ctrl)
    pub_l.Init()
    pub_r = ChannelPublisher(kTopicCtrlRight, inspire_dds.inspire_hand_ctrl)
    pub_r.Init()

    sub_l = ChannelSubscriber(kTopicStateLeft, inspire_dds.inspire_hand_state)
    sub_l.Init()
    sub_r = ChannelSubscriber(kTopicStateRight, inspire_dds.inspire_hand_state)
    sub_r.Init()

    print("Waiting 2s for DDS discovery...")
    time.sleep(2.0)

    # Check if we can read state
    msg_l = sub_l.Read()
    msg_r = sub_r.Read()
    if msg_l is not None:
        print(f"  Left hand state:  angle_act={list(msg_l.angle_act)}")
    else:
        print("  Left hand state:  NOT received (Headless_driver running?)")
    if msg_r is not None:
        print(f"  Right hand state: angle_act={list(msg_r.angle_act)}")
    else:
        print("  Right hand state: NOT received (Headless_driver running?)")

    print("\nCommands: o=open, c=close, h=half, q=quit")
    print("Or type 6 values like: 0,0,0,0,1000,1000\n")

    while True:
        try:
            user_input = input("cmd> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input == 'q':
            break
        elif user_input == 'o':
            values = [0, 0, 0, 0, 0, 0]
        elif user_input == 'c':
            values = [1000, 1000, 1000, 1000, 1000, 1000]
        elif user_input == 'h':
            values = [500, 500, 500, 500, 500, 500]
        else:
            try:
                values = [int(x.strip()) for x in user_input.split(',')]
                if len(values) != 6:
                    print("  Need exactly 6 values")
                    continue
            except ValueError:
                print("  Invalid input")
                continue

        cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        cmd.angle_set = values
        cmd.mode = 0b0001

        # Send multiple times to ensure delivery
        for _ in range(10):
            ret_l = pub_l.Write(cmd)
            ret_r = pub_r.Write(cmd)
            time.sleep(0.02)

        print(f"  Sent angle_set={values} to both hands (Write returned L={ret_l}, R={ret_r})")

        # Read current state
        time.sleep(0.3)
        msg_l = sub_l.Read()
        msg_r = sub_r.Read()
        if msg_l:
            print(f"  Left  state now: angle_act={list(msg_l.angle_act)}")
        if msg_r:
            print(f"  Right state now: angle_act={list(msg_r.angle_act)}")

    print("Done.")

if __name__ == "__main__":
    main()
