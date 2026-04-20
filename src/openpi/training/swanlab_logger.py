"""Optional SwanLab experiment logger, activated by environment variables.

wandb is always used as the primary logger (controlled by `wandb_enabled` config).
SwanLab is an **additional** logger that is enabled when any SWANLAB_* env var is set.

Supported environment variables:
    SWANLAB_API_KEY    - API key for cloud authentication (setting this enables cloud mode)
    SWANLAB_MODE       - Running mode: "cloud", "local", or "disabled" (default: "cloud")
    SWANLAB_LOG_DIR    - Directory for parsed log files (default: ./swanlog)
    SWANLAB_SAVE_DIR   - Directory for SwanLab global folder (default: ~/.swanlab)
    SWANLAB_LOG_LEVEL  - Logging output level (e.g. "DEBUG", "INFO", "WARNING")
    SWANLAB_WEB_HOST   - Self-hosted server address (e.g. "http://your-server:8000")
    SWANLAB_WORKSPACE  - Workspace name (personal or organization)

Usage:
    # Cloud mode (requires API key)
    export SWANLAB_API_KEY=your_key
    uv run scripts/train.py ...

    # Local mode (no API key needed, view via `swanlab watch`)
    export SWANLAB_MODE=local
    uv run scripts/train.py ...

    # Self-hosted mode
    export SWANLAB_API_KEY=your_key
    export SWANLAB_WEB_HOST=http://your-server:8000
    uv run scripts/train.py ...
"""

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# All recognized SWANLAB_* environment variables.
_SWANLAB_ENV_VARS = (
    "SWANLAB_API_KEY",
    "SWANLAB_MODE",
    "SWANLAB_LOG_DIR",
    "SWANLAB_SAVE_DIR",
    "SWANLAB_LOG_LEVEL",
    "SWANLAB_WEB_HOST",
    "SWANLAB_WORKSPACE",
)


def _swanlab_available() -> bool:
    """Check if SwanLab should be activated based on environment variables."""
    return any(os.environ.get(var) for var in _SWANLAB_ENV_VARS)


class SwanLabLogger:
    """Optional SwanLab logger that mirrors wandb logging."""

    def __init__(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def init(
        self,
        *,
        name: str | None = None,
        project: str | None = None,
        config: dict[str, Any] | None = None,
        logdir: str | None = None,
        ckpt_dir=None,
        resuming: bool = False,
    ):
        """Initialize SwanLab if environment variables are set. No-op otherwise."""
        if not _swanlab_available():
            return

        try:
            import swanlab
        except ImportError:
            logger.warning("SWANLAB env vars detected but swanlab package not installed. Skipping SwanLab logging.")
            return

        mode = os.environ.get("SWANLAB_MODE", "cloud")
        # SWANLAB_LOG_DIR → logdir (env var takes precedence over argument)
        resolved_logdir = os.environ.get("SWANLAB_LOG_DIR") or logdir
        kwargs: dict[str, Any] = {
            "project": project,
            "experiment_name": name,
            "config": config,
            "mode": mode,
            "logdir": resolved_logdir,
        }

        if resuming and ckpt_dir is not None:
            run_id_file = ckpt_dir / "swanlab_run_id.txt"
            if run_id_file.exists():
                run_id = run_id_file.read_text().strip()
                kwargs["resume"] = run_id
                logger.info(f"SwanLab resuming run: {run_id}")

        # SWANLAB_WORKSPACE → workspace
        workspace = os.environ.get("SWANLAB_WORKSPACE")
        if workspace:
            kwargs["workspace"] = workspace

        # SWANLAB_WEB_HOST → login to self-hosted server before init
        web_host = os.environ.get("SWANLAB_WEB_HOST")
        if web_host:
            api_key = os.environ.get("SWANLAB_API_KEY")
            try:
                swanlab.login(api_key=api_key, host=web_host)
                logger.info(f"SwanLab logged in to self-hosted server: {web_host}")
            except Exception:
                logger.warning(f"Failed to login to SwanLab self-hosted server: {web_host}", exc_info=True)

        try:
            swanlab.init(**kwargs)
            self._enabled = True
            detected = {var: os.environ[var] for var in _SWANLAB_ENV_VARS if var in os.environ}
            logger.info(
                f"SwanLab initialized (mode={mode}, project={project}, name={name}, env={detected})"
            )

            if ckpt_dir is not None:
                try:
                    run_id = swanlab.run.id
                    if run_id:
                        (ckpt_dir / "swanlab_run_id.txt").write_text(run_id)
                        logger.info(f"SwanLab run ID saved: {run_id}")
                except Exception:
                    logger.warning("Failed to save SwanLab run ID.", exc_info=True)
        except Exception:
            logger.warning("Failed to initialize SwanLab.", exc_info=True)

    def log(self, data: dict[str, Any], step: int | None = None):
        if not self._enabled:
            return
        import swanlab
        swanlab.log(data, step=step)

    def log_images(self, key: str, images: list[np.ndarray], step: int | None = None):
        if not self._enabled:
            return
        import swanlab
        self.log({key: [swanlab.Image(img) for img in images]}, step=step)

    def finish(self):
        if not self._enabled:
            return
        import swanlab
        swanlab.finish()
        self._enabled = False


# Global instance
_swanlab_logger: SwanLabLogger | None = None


def get_swanlab_logger() -> SwanLabLogger:
    global _swanlab_logger
    if _swanlab_logger is None:
        _swanlab_logger = SwanLabLogger()
    return _swanlab_logger
