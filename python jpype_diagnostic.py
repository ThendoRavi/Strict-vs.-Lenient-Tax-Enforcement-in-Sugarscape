import os
import sys
import subprocess

def diagnose_and_fix():
    print("🔍 DIAGNOSING JPYPE ISSUES...")
    print("=" * 50)
    
    # Step 1: Check Python version
    print(f"🐍 Python version: {sys.version}")
    
    # Step 2: Check Java
    print("\n☕ Checking Java installation...")
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        print("✅ Java is installed:")
        print(result.stderr.split('\n')[0])  # Java version is in stderr
    except FileNotFoundError:
        print("❌ Java is NOT installed!")
        print("   Please install Java from: https://www.java.com/download/")
        return False
    
    # Step 3: Check current installations
    print("\n📦 Checking current packages...")
    try:
        import jpype
        print(f"✅ JPype1 version: {jpype.__version__}")
        print(f"   JPype location: {jpype.__file__}")
    except ImportError:
        print("❌ JPype1 is NOT installed")
    
    try:
        import pynetlogo
        print(f"✅ pynetlogo version: {pynetlogo.__version__}")
    except ImportError:
        print("❌ pynetlogo is NOT installed")
    except AttributeError:
        print("✅ pynetlogo is installed (no version info)")
    
    # Step 4: Check JAVA_HOME
    print("\n🏠 Checking JAVA_HOME...")
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        print(f"✅ JAVA_HOME is set: {java_home}")
    else:
        print("⚠️  JAVA_HOME is not set")
        # Try to find Java automatically
        common_java_paths = [
            r"C:\Program Files\Java",
            r"C:\Program Files (x86)\Java",
            r"C:\Program Files\Eclipse Adoptium",
            r"C:\Program Files\Microsoft\jdk"
        ]
        
        java_found = False
        for base_path in common_java_paths:
            if os.path.exists(base_path):
                for item in os.listdir(base_path):
                    if item.startswith(('jdk', 'jre')):
                        java_path = os.path.join(base_path, item)
                        print(f"   Found Java at: {java_path}")
                        java_found = True
                        break
                if java_found:
                    break
    
    print("\n" + "=" * 50)
    print("🔧 RECOMMENDED FIXES:")
    
    print("\n1️⃣ COMPLETE REINSTALL (recommended):")
    print("   pip uninstall pynetlogo JPype1 -y")
    print("   pip install --upgrade pip")
    print("   pip install JPype1==1.4.1")
    print("   pip install pynetlogo")
    
    print("\n2️⃣ ALTERNATIVE: Use conda instead of pip:")
    print("   conda install -c conda-forge jpype1")
    print("   conda install -c conda-forge pynetlogo")
    
    print("\n3️⃣ IF STILL FAILING: Set JAVA_HOME manually")
    print("   1. Find your Java folder (probably in C:\\Program Files\\Java\\)")
    print("   2. Add environment variable: JAVA_HOME = [path to Java]")
    print("   3. Restart your command prompt")
    
    print("\n4️⃣ LAST RESORT: Use different approach")
    print("   Instead of pynetlogo, we can use NetLogo's built-in Python extension")
    
    return True

def test_jpype_directly():
    """Test JPype without pynetlogo"""
    print("\n🧪 TESTING JPYPE DIRECTLY...")
    try:
        import jpype
        print("✅ JPype imported successfully")
        
        # Try to start JVM
        if not jpype.isJVMStarted():
            jpype.startJVM()
            print("✅ JVM started successfully")
        else:
            print("✅ JVM already running")
            
        print("✅ JPype is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ JPype test failed: {e}")
        return False

if __name__ == "__main__":
    diagnose_and_fix()
    test_jpype_directly()