#!/bin/bash
# ============================================================
#  FRC Pit Safety Monitor - Easy Pi Setup
# ============================================================
#  Usage: chmod +x install.sh && ./install.sh
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

INSTALL_DIR="$HOME/pit_safety"

clear
echo -e "${BLUE}"
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║     FRC Pit Safety Monitor - Pi Setup             ║"
echo "  ╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# Check Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

echo -e "${GREEN}Step 1/5:${NC} Installing system packages..."
sudo apt-get update -qq
# Bookworm no longer ships libatlas-base-dev; use OpenBLAS instead.
BLAS_PKG="libopenblas-dev"
if apt-cache policy libatlas-base-dev 2>/dev/null | grep -q "Candidate: [^n]"; then
    # Only use ATLAS if apt has an installable candidate.
    if ! apt-cache policy libatlas-base-dev 2>/dev/null | grep -q "Candidate: (none)"; then
        BLAS_PKG="libatlas-base-dev"
    fi
fi

sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-tk \
    python3-picamera2 \
    libcamera-dev \
    "$BLAS_PKG" > /dev/null

echo -e "${GREEN}Step 2/5:${NC} Creating installation directory..."
mkdir -p "$INSTALL_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}Step 3/5:${NC} Copying files..."
if [[ "$SCRIPT_DIR" == "$INSTALL_DIR" ]]; then
    echo -e "         ${YELLOW}!${NC} Already running from $INSTALL_DIR (skipping copy)"
else
    cp -f "$SCRIPT_DIR/pi_deploy.py" "$INSTALL_DIR/"
    cp -f "$SCRIPT_DIR/launcher.py" "$INSTALL_DIR/"
    cp -f "$SCRIPT_DIR/config.env" "$INSTALL_DIR/"
fi

if ls "$SCRIPT_DIR"/*.tflite 1>/dev/null 2>&1; then
    if [[ "$SCRIPT_DIR" == "$INSTALL_DIR" ]]; then
        echo -e "         ${GREEN}✓${NC} Model file already in place"
    else
        cp -f "$SCRIPT_DIR"/*.tflite "$INSTALL_DIR/"
        echo -e "         ${GREEN}✓${NC} Model file copied"
    fi
else
    echo -e "         ${YELLOW}!${NC} No .tflite model found - copy manually later"
fi

echo -e "${GREEN}Step 4/5:${NC} Setting up Python environment..."
cd "$INSTALL_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet opencv-python numpy ai-edge-litert
deactivate

echo -e "${GREEN}Step 5/5:${NC} Setting up auto-start..."

# Create desktop autostart entry
mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/pit_safety.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Pit Safety Monitor
Exec=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/launcher.py
Path=$INSTALL_DIR
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=2
EOF

# Create launcher script
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export DISPLAY="${DISPLAY:-:0}"
python3 launcher.py
EOF
chmod +x "$INSTALL_DIR/start.sh"

# Create desktop shortcut
mkdir -p "$HOME/Desktop"
cat > "$HOME/Desktop/PitSafety.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Pit Safety Monitor
Icon=camera
Exec=$INSTALL_DIR/start.sh
Path=$INSTALL_DIR
Terminal=false
Categories=Utility;
EOF
chmod +x "$HOME/Desktop/PitSafety.desktop" 2>/dev/null || true

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Installation Complete!                  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Install location: ${BLUE}$INSTALL_DIR${NC}"
echo ""
echo -e "  ${GREEN}What happens on boot:${NC}"
echo "  1. Fullscreen launcher (DEMO_MODE waits for camera, then auto-starts)"
echo "  2. Monitor runs fullscreen; plug/unplug camera without keyboard"
echo "  3. Set DEMO_MODE=0 or FULLSCREEN=0 in config.env for bench use"
echo ""
echo -e "  ${GREEN}Manual start:${NC} $INSTALL_DIR/start.sh"
echo ""

if ! ls "$INSTALL_DIR"/*.tflite 1>/dev/null 2>&1; then
    echo -e "  ${YELLOW}⚠ IMPORTANT: Copy your .tflite model to $INSTALL_DIR${NC}"
    echo ""
fi

read -p "Start the monitor now? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    cd "$INSTALL_DIR"
    source venv/bin/activate
    export DISPLAY="${DISPLAY:-:0}"
    python3 launcher.py
fi
