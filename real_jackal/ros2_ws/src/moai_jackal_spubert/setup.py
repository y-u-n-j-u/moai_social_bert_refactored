from glob import glob

from setuptools import find_packages, setup


package_name = "moai_jackal_spubert"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="junwoo",
    maintainer_email="junwoo@example.com",
    description="Fail-closed real Jackal adapter for guided SPU-BERT navigation.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "real_jackal_spubert_bridge = "
            "moai_jackal_spubert.real_jackal_spubert_bridge_node:main",
            "safe_path_tracker = moai_jackal_spubert.safe_path_tracker_node:main",
        ],
    },
)
