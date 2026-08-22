ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ROS_DISTRO=humble

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    ca-certificates \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-rviz2 \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-dev-tools \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    libegl1 \
    libopengl0 \
    libosmesa6-dev \
    libglfw3 \
    libglew2.2 \
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt pyproject.toml ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python3 -m pip install --no-cache-dir -e ".[dev]"

ENV MUJOCO_GL=osmesa
ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
