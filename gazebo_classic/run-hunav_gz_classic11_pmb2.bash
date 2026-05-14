
xhost +local:docker

#Capture the current working directory
cwd=$(pwd)

docker run -it \
    --name hunavsim_pmb2 \
    --gpus all \
    --env="RMW_IMPLEMENTATION=rmw_fastrtps_cpp" \
    --env="HUNAV_ROBOT_TYPE=${HUNAV_ROBOT_TYPE:-pmb2}" \
    --env="HUNAV_ROBOT_NAME=${HUNAV_ROBOT_NAME:-${HUNAV_ROBOT_TYPE:-pmb2}}" \
    --env="HUNAV_AGENT_MOTION_MODEL=${HUNAV_AGENT_MOTION_MODEL:-hunav}" \
    --env="HUNAV_NAVIGATION=${HUNAV_NAVIGATION:-False}" \
    --env="HUNAV_UPDATE_RATE=${HUNAV_UPDATE_RATE:-50.0}" \
    --env="HUNAV_JACKAL_REALSENSE=${HUNAV_JACKAL_REALSENSE:-0}" \
    --env="HUNAV_JACKAL_SPUBERT_CONTROLLER=${HUNAV_JACKAL_SPUBERT_CONTROLLER:-False}" \
    --env="HUNAV_SOCIAL_BERT_PREDICTOR=${HUNAV_SOCIAL_BERT_PREDICTOR:-spubert}" \
    --env="HUNAV_SPUBERT_MODEL_PATH=${HUNAV_SPUBERT_MODEL_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/ethucy/univ/spubert.pth}" \
    --env="HUNAV_SPUBERT_REPO_PATH=${HUNAV_SPUBERT_REPO_PATH:-/home/hunav_gz_classic_ws/src/SPUBERT}" \
    --env="HUNAV_MOAI_SPUBERT_REPO_PATH=${HUNAV_MOAI_SPUBERT_REPO_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored}" \
    --env="HUNAV_SPUBERT_CUDA=${HUNAV_SPUBERT_CUDA:-false}" \
    --env="HUNAV_SPUBERT_K_SAMPLE=${HUNAV_SPUBERT_K_SAMPLE:-20}" \
    --env="HUNAV_SPUBERT_D_SAMPLE=${HUNAV_SPUBERT_D_SAMPLE:-200}" \
    --env="HUNAV_SPUBERT_CACHE_TTL=${HUNAV_SPUBERT_CACHE_TTL:-1.0}" \
    --env="HUNAV_SOCIAL_BERT_LOOKAHEAD_STEP=${HUNAV_SOCIAL_BERT_LOOKAHEAD_STEP:-6}" \
    --env="HUNAV_SOCIAL_BERT_MIN_GOAL_SPEED=${HUNAV_SOCIAL_BERT_MIN_GOAL_SPEED:-0.9}" \
    --env="HUNAV_SOCIAL_BERT_MAX_SPEED=${HUNAV_SOCIAL_BERT_MAX_SPEED:-1.8}" \
    --env="HUNAV_SOCIAL_BERT_MAX_ACCEL=${HUNAV_SOCIAL_BERT_MAX_ACCEL:-1.8}" \
    --env="HUNAV_SOCIAL_BERT_GOAL_VELOCITY_BLEND=${HUNAV_SOCIAL_BERT_GOAL_VELOCITY_BLEND:-0.7}" \
    --env="HUNAV_SOCIAL_BERT_AGENT_PERSONAL_SPACE=${HUNAV_SOCIAL_BERT_AGENT_PERSONAL_SPACE:-1.25}" \
    --env="HUNAV_SOCIAL_BERT_AGENT_AVOIDANCE_GAIN=${HUNAV_SOCIAL_BERT_AGENT_AVOIDANCE_GAIN:-0.9}" \
    --env="HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_DISTANCE=${HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_DISTANCE:-1.35}" \
    --env="HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_GAIN=${HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_GAIN:-0.45}" \
    --env="HUNAV_SOCIAL_BERT_OBSTACLE_COLLISION_BUFFER=${HUNAV_SOCIAL_BERT_OBSTACLE_COLLISION_BUFFER:-0.42}" \
    --env="HUNAV_SOCIAL_BERT_VELOCITY_SMOOTHING_ALPHA=${HUNAV_SOCIAL_BERT_VELOCITY_SMOOTHING_ALPHA:-0.35}" \
    --env="HUNAV_SOCIAL_BERT_COLLISION_BUFFER=${HUNAV_SOCIAL_BERT_COLLISION_BUFFER:-0.16}" \
    --env="HUNAV_SOCIAL_BERT_MAX_LATERAL_SPEED_RATIO=${HUNAV_SOCIAL_BERT_MAX_LATERAL_SPEED_RATIO:-0.25}" \
    --env="HUNAV_SOCIAL_BERT_MAX_YAW_RATE=${HUNAV_SOCIAL_BERT_MAX_YAW_RATE:-1.0}" \
    --env="HUNAV_GZPOSE_X=${HUNAV_GZPOSE_X:--22.0}" \
    --env="HUNAV_GZPOSE_Y=${HUNAV_GZPOSE_Y:--5.0}" \
    --env="HUNAV_GZPOSE_Z=${HUNAV_GZPOSE_Z:-0.25}" \
    --env="HUNAV_GZPOSE_YAW=${HUNAV_GZPOSE_YAW:-0.0}" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="__GLX_VENDOR_LIBRARY_NAME=nvidia" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --net=host \
    --privileged \
    --mount type=bind,source=$cwd/entrypoint.bash,target=/entrypoint.bash,readonly \
    --mount type=bind,source=$cwd/hunav_gz_classic_ws,target=/home/hunav_gz_classic_ws \
    -v ~/jackal_ws:/root/jackal_ws \
    pmb2_hunavsim \
    bash
    
docker rm hunavsim_pmb2
