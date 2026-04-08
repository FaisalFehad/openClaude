#!/bin/bash
set -e

INSTALL_DIR="${1:-/usr/local/bin}"
CONFIG_DIR="$HOME/.config/openClaude"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find Go
GO=""
for candidate in go /usr/local/go/bin/go /opt/homebrew/bin/go; do
  if command -v "$candidate" &>/dev/null; then
    GO="$candidate"
    break
  fi
done
if [ -z "$GO" ]; then
  # Search Homebrew cellar
  GO=$(find /opt/homebrew/Cellar/go -name "go" -type f 2>/dev/null | head -1)
fi
if [ -z "$GO" ]; then
  echo "Error: Go not found. Install Go first: https://go.dev/dl/"
  exit 1
fi
echo "Using Go: $GO"

# Build
echo "Building openClaude..."
cd "$SCRIPT_DIR"
"$GO" build -o openClaude .
echo "Built successfully."

# Install binary
echo "Installing to $INSTALL_DIR/openClaude..."
if [ -w "$INSTALL_DIR" ]; then
  cp openClaude "$INSTALL_DIR/openClaude"
else
  sudo cp openClaude "$INSTALL_DIR/openClaude"
fi
chmod +x "$INSTALL_DIR/openClaude"

# Set up config directory
mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/config.json" ]; then
  cp config.example.json "$CONFIG_DIR/config.json"
  echo ""
  echo "Config created at $CONFIG_DIR/config.json"
  echo "Edit it to add your providers and API keys."
else
  echo "Config already exists at $CONFIG_DIR/config.json (not overwritten)."
fi

echo ""
echo "Done! Set your API key(s) and run:"
echo "  openClaude --model <model>"
echo ""
echo "List available models:"
echo "  openClaude list"
