#!/bin/bash

echo "Searching for NetLogo installation..."

# Check common locations
LOCATIONS=(
    "$HOME/NetLogo-6.3.0"
    "$HOME/NetLogo"
    "/usr/local/NetLogo-6.3.0"
    "/usr/local/NetLogo"
    "/opt/NetLogo-6.3.0"
    "/opt/NetLogo"
)

for loc in "${LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "✅ Found NetLogo at: $loc"
        echo "   Use this path with: python dqn.py --netlogo-path \"$loc\""
        exit 0
    fi
done

echo "❌ NetLogo not found in common locations"
echo "Install with:"
echo "  cd ~"
echo "  wget https://ccl.northwestern.edu/netlogo/6.3.0/NetLogo-6.3.0-64.tgz"
echo "  tar -xzf NetLogo-6.3.0-64.tgz"