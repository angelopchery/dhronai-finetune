import os
import sys
import subprocess
import platform
import shutil


def get_system_python():
    """
    Get the system Python executable, NOT the PyInstaller exe.
    """
    # Try python3 first (Unix), then python (Windows)
    for name in ["python3", "python"]:
        if shutil.which(name):
            return name
    print("❌ No system Python found")
    sys.exit(1)


def run_command(cmd_list):
    """
    Run command safely without shell=True
    """
    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd_list)}")
        sys.exit(1)


def check_python():
    print("🔍 Checking Python...")

    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)

    print(f"✅ Python {sys.version.split()[0]} detected")


def create_venv():
    print("📦 Creating virtual environment...")

    if not os.path.exists("dhron_env"):
        run_command([get_system_python(), "-m", "venv", "dhron_env"])
    else:
        print("⚠️ venv already exists, skipping")


def get_base_path():
    """
    Handles PyInstaller temp path vs normal execution
    """
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.abspath(".")


def get_venv_python():
    """
    Get correct Python executable inside venv
    """
    if platform.system() == "Windows":
        return os.path.join("dhron_env", "Scripts", "python.exe")
    else:
        return os.path.join("dhron_env", "bin", "python")


def install_package():
    print("⬇️ Installing DhronAI...")

    base_path = get_base_path()
    dist_path = os.path.join(base_path, "dist")

    print(f"📁 Looking for dist at: {dist_path}")

    if not os.path.exists(dist_path):
        print("❌ dist folder not found inside installer")
        sys.exit(1)

    wheels = [f for f in os.listdir(dist_path) if f.endswith(".whl")]

    if not wheels:
        print("❌ No wheel found in dist/")
        sys.exit(1)

    wheel_path = os.path.join(dist_path, wheels[0])

    print(f"📦 Found wheel: {wheel_path}")

    venv_python = get_venv_python()

    run_command([venv_python, "-m", "pip", "install", wheel_path])


def install_dependencies():
    print("📚 Installing optional training dependencies...")

    venv_python = get_venv_python()

    run_command([venv_python, "-m", "pip", "install", "dhronai[train]"])


def main():
    print("\n🚀 DhronAI Installer\n")

    check_python()
    create_venv()
    install_package()
    install_dependencies()

    print("\n✅ Installation Complete!\n")

    if platform.system() == "Windows":
        print("👉 Activate with: dhron_env\\Scripts\\activate")
    else:
        print("👉 Activate with: source dhron_env/bin/activate")

    print("👉 Then run: dhronai --help\n")


if __name__ == "__main__":
    main()