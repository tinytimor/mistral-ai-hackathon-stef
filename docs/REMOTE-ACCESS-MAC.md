# 🖥️ Remote Access: Mac → Orin Nano (VNC + SSH)

> Connect to your Orin Nano from your MacBook - view the GUI, run commands,
> and monitor experiments from the train to NYC.

---

## Quick Reference

| Method | Use Case | Latency | Bandwidth |
|--------|----------|---------|-----------|
| **SSH** | Terminal commands, port forwarding | Lowest | Minimal |
| **VNC** | Full desktop GUI | Medium | Moderate |
| **Tailscale** | Secure remote access over cellular | Low | Minimal |
| **SSH + Port Forward** | Access Orin web UIs from Mac | Low | Minimal |

---

## 1. SSH Setup (Do This First)

### On the Orin Nano

```bash
# SSH server is installed by default on JetPack 6.2
# Verify it's running:
sudo systemctl status ssh

# If not running:
sudo systemctl enable ssh
sudo systemctl start ssh

# Get the Orin's IP address:
hostname -I
# Example: 192.168.1.100
```

### On Your Mac

```bash
# Test SSH connection (replace with your Orin's IP)
ssh orin@192.168.1.100

# For convenience, add to ~/.ssh/config:
cat >> ~/.ssh/config << 'EOF'

Host orin
    HostName 192.168.1.100
    User orin
    ForwardAgent yes
    # Port forwarding for Ollama, Bridge, Memory, and Jupyter
    LocalForward 11434 localhost:11434
    LocalForward 8000 localhost:8000
    LocalForward 8100 localhost:8100
EOF

# Now you can just do:
ssh orin
```

### SSH Key Setup (passwordless)

```bash
# On Mac - generate key if you don't have one:
ssh-keygen -t ed25519 -C "mac-to-orin"

# Copy key to Orin:
ssh-copy-id orin@192.168.1.100

# Test passwordless login:
ssh orin
```

---

## 2. VNC Server Setup (Orin Nano GUI)

### Option A: TigerVNC (Recommended for JetPack)

```bash
# SSH into the Orin
ssh orin

# Install TigerVNC server
sudo apt update
sudo apt install -y tigervnc-standalone-server tigervnc-common

# Set VNC password
vncpasswd
# Enter a password (you'll use this from your Mac)
# Say "no" to view-only password

# Create VNC config
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XDG_SESSION_TYPE=x11
exec startxfce4 &
EOF
chmod +x ~/.vnc/xstartup

# Start VNC server on display :1 (port 5901)
vncserver :1 -geometry 1920x1080 -depth 24

# Verify it's running:
vncserver -list
# Should show:  :1   (port 5901)

# To stop:
# vncserver -kill :1
```

### Option B: x11vnc (Share Existing Desktop)

If you want to see the **same desktop** that's on the Orin's HDMI output:

```bash
ssh orin

# Install x11vnc
sudo apt install -y x11vnc

# Set password
x11vnc -storepasswd

# Start (shares the current HDMI display)
x11vnc -display :0 -auth /var/run/lightdm/root/:0 \
       -usepw -forever -repeat -shared \
       -rfbport 5900 &

# Or run as a service:
sudo tee /etc/systemd/system/x11vnc.service << 'EOF'
[Unit]
Description=x11vnc VNC Server
After=display-manager.service

[Service]
ExecStart=/usr/bin/x11vnc -display :0 -auth /var/run/lightdm/root/:0 -usepw -forever -repeat -shared -rfbport 5900
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable x11vnc
sudo systemctl start x11vnc
```

### On Your Mac - Connect with VNC Viewer

**Built-in macOS Screen Sharing:**
```bash
# Open Screen Sharing directly:
open vnc://192.168.1.100:5901

# Or from Finder:
# Go → Connect to Server → vnc://192.168.1.100:5901
```

**Or use a dedicated VNC client:**
- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/) (free)
- [TigerVNC Viewer](https://tigervnc.org/) (open source)

```bash
# Install RealVNC via brew:
brew install --cask vnc-viewer

# Connect to: 192.168.1.100:5901
```

### Auto-Start VNC on Boot

```bash
ssh orin

# Create systemd service for TigerVNC
sudo tee /etc/systemd/system/vncserver@.service << 'EOF'
[Unit]
Description=TigerVNC Server for display %i
After=syslog.target network.target

[Service]
Type=forking
User=orin
Group=orin
WorkingDirectory=/home/orin
ExecStartPre=-/usr/bin/vncserver -kill :%i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver :%i -geometry 1920x1080 -depth 24
ExecStop=/usr/bin/vncserver -kill :%i
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable for display :1
sudo systemctl daemon-reload
sudo systemctl enable vncserver@1
sudo systemctl start vncserver@1

# Verify:
sudo systemctl status vncserver@1
```

---

## 3. Tailscale (Remote Access from Train / NYC)

Tailscale creates a secure WireGuard VPN mesh so you can access the Orin
from anywhere - even on cellular from the train.

### On the Orin Nano

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Authenticate (opens a browser URL to log in)
sudo tailscale up

# Get the Tailscale IP:
tailscale ip -4
# Example: 100.100.100.50

# Enable SSH via Tailscale:
sudo tailscale up --ssh
```

### On Your Mac

```bash
# Install Tailscale
brew install --cask tailscale

# Or download from: https://tailscale.com/download/mac

# Open Tailscale, sign in with the same account
# Your Orin will appear in the device list

# Now you can SSH via Tailscale IP from ANYWHERE:
ssh orin@100.100.100.50

# VNC via Tailscale (from the train!):
open vnc://100.100.100.50:5901
```

### On the 5090 Desktop (so you can monitor training remotely too)

```bash
# Install Tailscale on the 5090 machine
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Now from your Mac on the train, you can:
- SSH into the Orin: `ssh orin@100.x.x.x`
- SSH into the 5090: `ssh user@100.y.y.y`
- View Orin GUI: `open vnc://100.x.x.x:5901`
- Check W&B dashboard: browser → wandb.ai
- Access Ollama API: `curl http://100.x.x.x:11434/api/...`

---

## 4. Port Forwarding Cheat Sheet

Forward Orin services to your Mac:

```bash
# All-in-one SSH tunnel (run on Mac):
ssh -N -L 11434:localhost:11434 \
       -L 8000:localhost:8000 \
       -L 8100:localhost:8100 \
       -L 5901:localhost:5901 \
       orin@192.168.1.100

# Now on your Mac:
# Ollama API:    http://localhost:11434
# Bridge API:    http://localhost:8000
# Memory API:    http://localhost:8100
# VNC:           vnc://localhost:5901

# With Tailscale (from anywhere):
ssh -N -L 11434:localhost:11434 \
       -L 8000:localhost:8000 \
       -L 8100:localhost:8100 \
       orin@100.x.x.x
```

---

## 5. Monitoring from Mac

### Watch GPU usage on Orin (via SSH)

```bash
ssh orin "watch -n 1 tegrastats"
```

### Watch Ollama inference

```bash
ssh orin "watch -n 2 'ollama ps'"
```

### Check bridge server logs

```bash
ssh orin "tail -f ~/reachy-bridge/server.log"
```

### Test the full pipeline from Mac

```bash
# Health check
curl http://localhost:8000/health

# Chat with Reachy (via tunnel)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Reachy!"}'

# Check memory
curl http://localhost:8100/memory/stats
```

---

## 6. Quick Setup Script

Save this on your Mac and run it to establish all connections:

```bash
#!/bin/bash
# connect-to-orin.sh - Run on Mac to connect to Orin Nano

ORIN_IP="${ORIN_IP:-192.168.1.100}"  # or Tailscale IP

echo "🔌 Connecting to Orin Nano at $ORIN_IP..."

# Start SSH tunnel in background
ssh -N -f \
    -L 11434:localhost:11434 \
    -L 8000:localhost:8000 \
    -L 8100:localhost:8100 \
    orin@$ORIN_IP

echo "✅ SSH tunnels established:"
echo "   Ollama:  http://localhost:11434"
echo "   Bridge:  http://localhost:8000"
echo "   Memory:  http://localhost:8100"

# Open VNC
echo "🖥️  Opening VNC..."
open vnc://$ORIN_IP:5901

echo "Done! You can now work with Orin from your Mac."
```

---

## Troubleshooting

### VNC: "Connection refused"
```bash
# Check if VNC server is running on Orin:
ssh orin "vncserver -list"
# If empty, start it:
ssh orin "vncserver :1 -geometry 1920x1080 -depth 24"
```

### VNC: Black screen
```bash
# Make sure a desktop environment is installed:
ssh orin "sudo apt install -y xfce4 xfce4-goodies"
# Restart VNC:
ssh orin "vncserver -kill :1 && vncserver :1"
```

### SSH: Connection timeout (from train)
```bash
# Use Tailscale instead of local IP
ssh orin@100.x.x.x

# Or if SSH keeps disconnecting, add to ~/.ssh/config:
Host orin
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

### Tailscale: Not connecting
```bash
# On Orin, check status:
tailscale status
# If offline, restart:
sudo systemctl restart tailscaled
sudo tailscale up
```
