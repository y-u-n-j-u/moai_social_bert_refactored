from glob import glob
from setuptools import find_packages, setup

package_name = "moai_hunav_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="junwoo",
    maintainer_email="junwoo@example.com",
    description="Bridge utilities for connecting HuNavSim human states to MOAI trajectory prediction.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cmd_vel_passthrough_node = moai_hunav_bridge.cmd_vel_passthrough_node:main",
            "ground_truth_localization_node = moai_hunav_bridge.ground_truth_localization_node:main",
            "human_path_debug_node = moai_hunav_bridge.human_path_debug_node:main",
            "human_obstacle_cloud_node = moai_hunav_bridge.human_obstacle_cloud_node:main",
            "social_bert_compute_agents_node = moai_hunav_bridge.social_bert_compute_agents_node:main",
            "spubert_compute_agents_node = moai_hunav_bridge.social_bert_compute_agents_node:main",
            "spubert_jackal_controller_node = moai_hunav_bridge.spubert_jackal_controller_node:main",
            "spubert_nav2_bridge_node = moai_hunav_bridge.spubert_nav2_bridge_node:main",
            "jackal_teleop_dataset_logger_node = moai_hunav_bridge.jackal_teleop_dataset_logger_node:main",
            "pedestrian_dataset_logger_node = moai_hunav_bridge.pedestrian_dataset_logger_node:main",
            "random_goal_publisher_node = moai_hunav_bridge.random_goal_publisher_node:main",
        ],
    },
)
