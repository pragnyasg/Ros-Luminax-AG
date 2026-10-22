"""ROS 2 node for LUMI Voice Assistant.

Publishes recognized speech and LUMI responses as ROS 2 topics.
Subscribes to voice commands for triggering speech synthesis.

Topics:
    Published:
        /<namespace>/recognized_speech  (std_msgs/String): Transcribed user speech
        /<namespace>/response_text      (std_msgs/String): LUMI's text response
        /<namespace>/voice_status       (std_msgs/String): Pipeline status JSON

    Subscribed:
        /<namespace>/voice_command      (std_msgs/String): Commands to the voice system
            - JSON format: {"action": "speak", "text": "Hello world"}
            - JSON format: {"action": "listen"}
            - JSON format: {"action": "stop"}
            - JSON format: {"action": "status"}

Services:
    /<namespace>/speak               (std_srvs/Trigger-like): Speak text via TTS

Requires: ROS 2 Jazzy (or compatible), rclpy, std_msgs
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS2_AVAILABLE = True
except ImportError:
    logger.info("ROS 2 (rclpy) not available. ROS 2 bridge disabled.")

if TYPE_CHECKING:
    from lumi_voice.pipeline import VoicePipeline


def is_ros2_available() -> bool:
    """Check if ROS 2 runtime is available."""
    return ROS2_AVAILABLE


if ROS2_AVAILABLE:

    class LumiVoiceNode(Node):  # type: ignore[misc]
        """ROS 2 node that bridges the voice pipeline to the ROS 2 ecosystem."""

        def __init__(
            self,
            pipeline: "VoicePipeline",
            namespace: str = "/lumi",
            node_name: str = "lumi_voice",
        ) -> None:
            super().__init__(node_name, namespace=namespace)
            self.pipeline = pipeline

            # Publishers
            self.speech_pub = self.create_publisher(String, "recognized_speech", 10)
            self.response_pub = self.create_publisher(String, "response_text", 10)
            self.status_pub = self.create_publisher(String, "voice_status", 10)

            # Subscribers
            self.command_sub = self.create_subscription(
                String, "voice_command", self._on_command, 10
            )

            # Status timer (publish status every 5 seconds)
            self.status_timer = self.create_timer(5.0, self._publish_status)

            # Wire pipeline callbacks
            self.pipeline._on_transcript = self._on_transcript
            self.pipeline._on_response = self._on_response

            self.get_logger().info("LUMI Voice ROS 2 node initialized")

        def _on_transcript(self, text: str) -> None:
            """Called when speech is transcribed."""
            msg = String()
            msg.data = text
            self.speech_pub.publish(msg)
            self.get_logger().info(f"Published transcript: {text}")

        def _on_response(self, text: str) -> None:
            """Called when LUMI generates a response."""
            msg = String()
            msg.data = text
            self.response_pub.publish(msg)
            self.get_logger().info(f"Published response: {text[:100]}")

        def _on_command(self, msg: String) -> None:
            """Handle incoming voice commands."""
            try:
                cmd = json.loads(msg.data)
                action = cmd.get("action", "")

                if action == "speak":
                    text = cmd.get("text", "")
                    if text:
                        self.pipeline.process_text_input(text)
                elif action == "listen":
                    self.pipeline.start_continuous_listening()
                elif action == "stop":
                    self.pipeline.stop_all()
                elif action == "status":
                    self._publish_status()
                else:
                    self.get_logger().warning(f"Unknown command action: {action}")

            except json.JSONDecodeError:
                # Treat as plain text to speak
                self.pipeline.process_text_input(msg.data)

        def _publish_status(self) -> None:
            """Publish current pipeline status."""
            msg = String()
            status = {
                "state": self.pipeline.state.value,
                "components": self.pipeline.status.to_dict(),
            }
            msg.data = json.dumps(status)
            self.status_pub.publish(msg)

        def shutdown(self) -> None:
            """Clean shutdown."""
            self.pipeline.shutdown()
            self.get_logger().info("LUMI Voice ROS 2 node shutting down")


    def launch_ros2_bridge(
        pipeline: "VoicePipeline",
        namespace: str = "/lumi",
    ) -> None:
        """Launch the ROS 2 bridge node."""
        rclpy.init()
        node = LumiVoiceNode(pipeline, namespace=namespace)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.shutdown()
            node.destroy_node()
            rclpy.shutdown()

else:

    def launch_ros2_bridge(*args, **kwargs) -> None:  # type: ignore[misc]
        """Stub when ROS 2 is not available."""
        logger.error(
            "ROS 2 (rclpy) is not installed. "
            "Install ROS 2 Jazzy and source its setup script to use the bridge."
        )
