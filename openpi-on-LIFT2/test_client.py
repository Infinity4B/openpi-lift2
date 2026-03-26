#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for OpenPI LIFT2 client
Tests policy server connection without requiring a real robot
"""

import numpy as np
import argparse
import time

from openpi_client import image_tools
from openpi_client import websocket_client_policy


def test_policy_server(host, port):
    """
    Test connection to policy server and perform a sample inference

    Args:
        host: Policy server host
        port: Policy server port
    """
    print("="*60)
    print("OpenPI LIFT2 Client Test")
    print("="*60)
    print(f"Policy Server: {host}:{port}")
    print()

    # Initialize client
    print("1. Connecting to policy server...")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port
        )
        print("   ✓ Connected successfully")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False

    # Create dummy observation
    print("\n2. Creating test observation...")
    head_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    left_wrist_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right_wrist_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Use training mean for left arm
    left_qpos = np.array([-0.0008, 0.0032, 0.0055, -0.0037, -0.0025, 0.0005, 4.8946], dtype=np.float32)
    right_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.9], dtype=np.float32)
    state = np.concatenate([left_qpos, right_qpos], axis=0)

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
        "prompt": "put items from plate to left",
    }
    print("   ✓ Test observation created")
    print(f"     - Image shapes: {head_img.shape}")
    print(f"     - State shape: {state.shape}")

    # Perform inference
    print("\n3. Performing test inference...")
    try:
        t0 = time.perf_counter()
        result = client.infer(observation)
        latency_ms = (time.perf_counter() - t0) * 1000

        action_chunk = result["actions"]
        print(f"   ✓ Inference successful")
        print(f"     - Latency: {latency_ms:.1f} ms")
        print(f"     - Action chunk shape: {action_chunk.shape}")
        print(f"     - First action: {action_chunk[0]}")
        print(f"     - Left gripper: {action_chunk[0][6]:.3f}")
        print(f"     - Right gripper: {action_chunk[0][13]:.3f}")

    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        return False

    # Multiple inference test
    print("\n4. Testing multiple inferences...")
    latencies = []
    for i in range(5):
        t0 = time.perf_counter()
        result = client.infer(observation)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        print(f"   Inference {i+1}: {latency_ms:.1f} ms")

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    print(f"\n   Average latency: {avg_latency:.1f} ± {std_latency:.1f} ms")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✓ Policy server connection: OK")
    print("✓ Inference functionality: OK")
    print(f"✓ Average latency: {avg_latency:.1f} ms")

    if avg_latency < 100:
        print("\n✓ Latency is excellent (< 100ms)")
    elif avg_latency < 200:
        print("\n✓ Latency is good (< 200ms)")
    else:
        print("\n⚠ Latency is high (> 200ms)")
        print("  Consider using wired connection or reducing image resolution")

    print("\n✓ All tests passed! Ready to run on robot.")
    print("="*60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Test OpenPI LIFT2 client setup')
    parser.add_argument('--host', type=str, default='192.168.101.101',
                        help='Policy server host')
    parser.add_argument('--port', type=int, default=7777,
                        help='Policy server port')
    args = parser.parse_args()

    success = test_policy_server(args.host, args.port)

    if not success:
        print("\n✗ Tests failed. Please check:")
        print("  1. Policy server is running")
        print("  2. Network connectivity")
        print("  3. Firewall settings")
        exit(1)


if __name__ == '__main__':
    main()
