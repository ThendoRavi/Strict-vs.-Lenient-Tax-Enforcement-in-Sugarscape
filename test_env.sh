#!/bin/bash
# Simple test script to verify environment setup

echo "=== Environment Test ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

echo -e "\n=== Python Detection ==="
for cmd in python3 python python3.9 python3.8; do
    if command -v $cmd &> /dev/null; then
        echo "✓ Found: $cmd ($(which $cmd))"
        $cmd --version
    else
        echo "✗ Not found: $cmd"
    fi
done

echo -e "\n=== Available Modules ==="
if command -v module &> /dev/null; then
    echo "Module system available"
    echo "Python modules:"
    module avail python 2>&1 | head -10
    echo "Java modules:"
    module avail java 2>&1 | head -5
else
    echo "No module system found"
fi

echo -e "\n=== Files Check ==="
for file in dqn.py requirements.txt "Sugarscape 2 Constant Growback.nlogo"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
    fi
done

echo -e "\n=== Test Complete ==="