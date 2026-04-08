#!/bin/bash
set -e

INSTALL_DIR="${1:-/usr/local/bin}"
CONFIG_DIR="$HOME/.config/openClaude"

# Remove binary
if [ -f "$INSTALL_DIR/openClaude" ]; then
  echo "Removing $INSTALL_DIR/openClaude..."
  if [ -w "$INSTALL_DIR" ]; then
    rm "$INSTALL_DIR/openClaude"
  else
    sudo rm "$INSTALL_DIR/openClaude"
  fi
  echo "Binary removed."
else
  echo "Binary not found at $INSTALL_DIR/openClaude"
fi

# Ask about config
if [ -d "$CONFIG_DIR" ]; then
  read -p "Remove config at $CONFIG_DIR? [y/N] " answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    rm -r "$CONFIG_DIR"
    echo "Config removed."
  else
    echo "Config kept at $CONFIG_DIR"
  fi
fi

echo "Done."
