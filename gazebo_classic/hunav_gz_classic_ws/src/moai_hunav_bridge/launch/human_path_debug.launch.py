from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="moai_hunav_bridge",
                executable="human_path_debug_node",
                name="human_path_debug_node",
                output="screen",
                parameters=[
                    {
                        "human_states_topic": "/human_states",
                        "marker_topic": "/moai/debug_human_paths",
                        "obs_len": 8,
                        "pred_len": 12,
                        "prediction_dt": 0.4,
                        "history_timeout": 3.0,
                        "publish_rate": 5.0,
                    }
                ],
            )
        ]
    )
