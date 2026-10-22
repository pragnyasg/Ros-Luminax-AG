"""LUMI Voice Assistant — Application entry point."""

from __future__ import annotations

import sys
import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LUMI Voice Assistant — Speech-to-Text, Conversational AI, and Text-to-Speech",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to user config YAML file (default: config/user.yaml)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI) — for testing or ROS 2 bridge",
    )
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="Launch ROS 2 bridge node",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for LUMI Voice Assistant."""
    args = parse_args()

    from lumi_voice.config import load_config
    from lumi_voice.logging_config import setup_logging

    # Load config
    overrides = {}
    if args.log_level:
        overrides["logging"] = {"level": args.log_level}

    config = load_config(user_path=args.config, overrides=overrides if overrides else None)

    # Setup logging
    setup_logging(level=config.logging.level, log_file=config.logging.file)

    logger.info("LUMI Voice Assistant v0.1.0 starting")
    logger.info("Python %s on %s", sys.version, sys.platform)

    if args.ros2:
        # Launch ROS 2 bridge
        logger.info("Starting ROS 2 bridge mode")
        from lumi_voice.pipeline import VoicePipeline
        from lumi_voice.ros2_bridge.node import launch_ros2_bridge, is_ros2_available

        if not is_ros2_available():
            logger.error(
                "ROS 2 (rclpy) is not available. "
                "Install ROS 2 Jazzy and source its setup script."
            )
            return 1

        pipeline = VoicePipeline(config=config)
        pipeline.initialize()
        launch_ros2_bridge(pipeline, namespace=config.ros2.namespace)
        return 0

    if args.headless:
        # Headless mode — initialize and run pipeline interactively via stdin
        logger.info("Running in headless mode")
        from lumi_voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(
            config=config,
            on_transcript=lambda t: print(f"\n[You]: {t}"),
            on_response=lambda r: print(f"[LUMI]: {r}"),
            on_error=lambda e: print(f"[ERROR]: {e}"),
        )
        results = pipeline.initialize()
        for comp, (ok, msg) in results.items():
            print(f"  {'✅' if ok else '⚠️ '} {comp}: {msg}")

        print("\nType messages to LUMI (or 'quit' to exit):\n")
        try:
            while True:
                text = input("> ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if text:
                    pipeline.process_text_input(text)
                    # Wait briefly for async processing
                    import time
                    time.sleep(0.5)
                    # Wait for pipeline to finish
                    while pipeline.state.value not in ("idle",):
                        time.sleep(0.1)
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            pipeline.shutdown()
        return 0

    # GUI mode
    from lumi_voice.ui.main_window import launch_ui
    return launch_ui(config)


if __name__ == "__main__":
    sys.exit(main())
