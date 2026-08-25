ARG BASE_IMAGE=spubert_deploy:latest
FROM ${BASE_IMAGE}

ENV ROS_DOMAIN_ID=1 \
    MOAI_MODEL_ROOT=/root/moai_social_bert_refactored

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-slam-toolbox \
    && rm -rf /var/lib/apt/lists/*

RUN rm -rf /root/moai_social_bert_refactored
COPY model_repo /root/moai_social_bert_refactored
COPY moai_jackal_spubert /root/robot_ws/src/moai_jackal_spubert

WORKDIR /root/robot_ws
RUN source /opt/ros/humble/setup.bash \
    && source /root/robot_ws/install/setup.bash \
    && colcon build --symlink-install --packages-select moai_jackal_spubert

COPY scripts /root/real_jackal_scripts
RUN chmod +x /root/real_jackal_scripts/*.bash \
    && printf '%s\n' \
        'source /opt/ros/humble/setup.bash' \
        'source /root/robot_ws/install/setup.bash' \
        >> /root/.bashrc

# A successful colcon build does not prove that the checkpoint can be loaded
# with the Python packages in the image.  Run one complete CPU inference while
# building so an incompatible torch/transformers combination fails early.
RUN source /opt/ros/humble/setup.bash \
    && source /root/robot_ws/install/setup.bash \
    && /usr/bin/python3 /root/real_jackal_scripts/model_runtime_smoke.py

COPY docker_entrypoint.sh /real_jackal_entrypoint.sh
RUN chmod +x /real_jackal_entrypoint.sh

WORKDIR /root
ENTRYPOINT ["/real_jackal_entrypoint.sh"]
CMD ["bash"]
