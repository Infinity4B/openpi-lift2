import dataclasses
import enum
import logging
import pathlib
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Enable debug printing of model inputs and outputs.
    debug_print: bool = False
    # Print debug info every N steps (only used when debug_print=True).
    debug_interval: int = 1

    # Fix left arm state to training data mean (for right-arm-only tasks).
    fix_left_arm: bool = False
    # Fix right arm state to training data mean (for left-arm-only tasks).
    fix_right_arm: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Get local IP for display purposes
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        # Fallback: try to get IP by connecting to external address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"

    logging.info("Creating server (host: 0.0.0.0, local_ip: %s, port: %d)", local_ip, args.port)
    logging.info("LAN clients can connect to: %s:%d", local_ip, args.port)

    if args.debug_print:
        logging.info("Debug printing enabled (interval: %d)", args.debug_interval)

    # Compute fixed arm state values from norm_stats if needed
    fix_state_mask = None
    fix_state_values = None
    if args.fix_left_arm or args.fix_right_arm:
        if not isinstance(args.policy, Checkpoint):
            raise ValueError("--fix-left-arm / --fix-right-arm requires policy:checkpoint mode")
        checkpoint_dir = pathlib.Path(args.policy.dir)
        # Find norm_stats.json in assets directory
        assets_dir = checkpoint_dir / "assets"
        norm_stats_files = list(assets_dir.glob("*/norm_stats.json"))
        if not norm_stats_files:
            raise ValueError(f"Cannot find norm_stats.json in {assets_dir}")
        norm_stats_path = norm_stats_files[0].parent
        norm_stats = _normalize.load(norm_stats_path)
        if norm_stats is None or "state" not in norm_stats:
            raise ValueError("Cannot load state norm_stats for fixing arm state")
        state_mean = np.array(norm_stats["state"].mean, dtype=np.float32)
        state_dim = state_mean.shape[-1]
        fix_state_mask = np.zeros(state_dim, dtype=bool)
        fix_state_values = np.zeros(state_dim, dtype=np.float32)
        if args.fix_left_arm and state_dim >= 7:
            fix_state_mask[:7] = True
            fix_state_values[:7] = state_mean[:7]
            logging.info("Fixing left arm state to training mean: %s", state_mean[:7])
        if args.fix_right_arm and state_dim >= 14:
            fix_state_mask[7:14] = True
            fix_state_values[7:14] = state_mean[7:14]
            logging.info("Fixing right arm state to training mean: %s", state_mean[7:14])

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        debug_print=args.debug_print,
        debug_interval=args.debug_interval,
        fix_state_mask=fix_state_mask,
        fix_state_values=fix_state_values,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
