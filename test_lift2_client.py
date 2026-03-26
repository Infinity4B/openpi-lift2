"""
LIFT2 client script for testing policy inference.

This script demonstrates how to query the policy server from a test machine.
"""

import argparse
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


def main():
    parser = argparse.ArgumentParser(description='Test LIFT2 policy client')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Policy server host')
    parser.add_argument('--port', type=int, default=8000,
                        help='Policy server port')
    parser.add_argument('--prompt', type=str, default='perform task',
                        help='Task prompt/instruction')
    args = parser.parse_args()

    # Initialize the policy client
    print(f"Connecting to policy server at {args.host}:{args.port}")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    # Example: Create dummy observations for testing
    # In real usage, replace these with actual camera images and robot state
    head_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    left_wrist_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right_wrist_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    state = np.random.rand(14).astype(np.float32)  # 14-dim EEF state

    # Construct observation dictionary
    # Resize images to 224x224 as expected by the model
    observation = {
        "observation.images.head": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(head_img, 224, 224)
        ),
        "observation.images.left_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(left_wrist_img, 224, 224)
        ),
        "observation.images.right_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(right_wrist_img, 224, 224)
        ),
        "observation.state": state,
        "prompt": args.prompt,
    }

    print(f"Sending observation to policy server with prompt: '{args.prompt}'")

    # Query the policy server
    result = client.infer(observation)
    action_chunk = result["actions"]

    print(f"Received action chunk with shape: {action_chunk.shape}")
    print(f"First action: {action_chunk[0]}")

    # In real usage, you would execute these actions on the robot
    # For example:
    # for action in action_chunk:
    #     robot.execute_action(action)

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
