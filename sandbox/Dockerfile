FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:1 \
    RESOLUTION=1280x800x24

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    fluxbox \
    x11vnc \
    xdotool \
    imagemagick \
    mousepad \
    xterm \
    drawing \
    python3 \
    python3-pip \
    novnc \
    websockify \
    net-tools \
    dbus-x11 \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir requests pillow opencv-python-headless numpy

WORKDIR /agent
COPY agent_sandbox.py entrypoint.sh memory.py ocr.py vision_zoom.py /agent/
COPY apps/form_app.py /usr/bin/form_app
COPY fluxbox_menu /root/.fluxbox/menu
RUN chmod +x /agent/entrypoint.sh /usr/bin/form_app
RUN mkdir -p /agent/memory

EXPOSE 6080

ENTRYPOINT ["/agent/entrypoint.sh"]
