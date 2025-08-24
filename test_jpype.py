#!/usr/bin/env python3
"""
Test script to verify JPype and pynetlogo installation
"""

print("Testing JPype and pynetlogo installation...")

# Test 1: JPype basic functionality
try:
    import jpype
    print("✓ JPype imported successfully")
    
    # Try to start JVM
    if not jpype.isJVMStarted():
        jpype.startJVM()
        print("✓ JVM started successfully")
    else:
        print("✓ JVM was already running")
        
    # Test basic Java functionality
    java_string = jpype.JClass("java.lang.String")
    test_str = java_string("Hello from Java!")
    print(f"✓ Java String test: {test_str}")
    
    jpype.shutdownJVM()
    print("✓ JVM shutdown successfully")
    
except Exception as e:
    print(f"✗ JPype error: {e}")

print()

# Test 2: pynetlogo import
try:
    import pynetlogo
    print("✓ pynetlogo imported successfully")
    print(f"✓ pynetlogo version: {pynetlogo.__version__}")
except Exception as e:
    print(f"✗ pynetlogo error: {e}")

print()
print("Installation test completed!")
