#!/bin/bash
# setup_launchd.sh
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"

mkdir -p "$BASE_DIR/logs"

echo "Copying plist files to $PLIST_DIR..."
cp "$BASE_DIR"/*.plist "$PLIST_DIR/"

echo "Cleaning up legacy agents..."
LEGACY_PLISTS=("com.telegramdaily.pharma30d.plist" "com.telegramdaily.pipeline.plist" "com.telegramdaily.energyweekly.plist")
for plist in "${LEGACY_PLISTS[@]}"; do
    if [ -f "$PLIST_DIR/$plist" ]; then
        echo "  Unloading and removing $plist..."
        launchctl unload "$PLIST_DIR/$plist" 2>/dev/null || true
        rm "$PLIST_DIR/$plist"
    fi
done

echo "Loading agents..."
launchctl unload "$PLIST_DIR/com.telegramdaily.macro.plist" 2>/dev/null || true
launchctl unload "$PLIST_DIR/com.telegramdaily.energy.plist" 2>/dev/null || true
launchctl unload "$PLIST_DIR/com.telegramdaily.medical.plist" 2>/dev/null || true

launchctl load "$PLIST_DIR/com.telegramdaily.macro.plist"
launchctl load "$PLIST_DIR/com.telegramdaily.energy.plist"
launchctl load "$PLIST_DIR/com.telegramdaily.medical.plist"

echo "✅ Launchd setup complete."
echo "Daily Macro: 08:00 every day"
echo "Weekly Energy: 09:00 every Monday"
echo "Weekly Medical: 10:00 every Monday"
echo "Check status with: launchctl list | grep telegramdaily"
