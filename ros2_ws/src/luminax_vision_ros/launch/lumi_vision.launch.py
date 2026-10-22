import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """Launch file for LUMI Vision ROS 2 Node."""

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Incoming camera video stream topic'
    )

    monitored_target_arg = DeclareLaunchArgument(
        'monitored_target',
        default_value='',
        description='Name of specific enrolled person to monitor (empty for general monitoring)'
    )

    publish_annotated_arg = DeclareLaunchArgument(
        'publish_annotated',
        default_value='true',
        description='Whether to publish annotated debug frames on /vision/image_annotated'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock'
    )

    vision_node = Node(
        package='luminax_vision_ros',
        executable='lumi_vision_node',
        name='lumi_vision_node',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'monitored_target': LaunchConfiguration('monitored_target'),
            'publish_annotated': LaunchConfiguration('publish_annotated'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    return LaunchDescription([
        camera_topic_arg,
        monitored_target_arg,
        publish_annotated_arg,
        use_sim_time_arg,
        vision_node
    ])
