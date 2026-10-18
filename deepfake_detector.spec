# PyInstaller build: pyinstaller deepfake_detector.spec --noconfirm
# Output: dist/DeepfakeDetector/DeepfakeDetector.exe (run it from that folder, which holds its _internal libraries)
from PyInstaller.utils.hooks import collect_all

# MediaPipe loads its native library and graph files at runtime, which import analysis cannot see.
mp_datas, mp_binaries, mp_hiddenimports = collect_all("mediapipe")

a = Analysis(
    ["main.py"],
    datas=mp_datas + [
        # Bundled so detection needs no download on first run.
        ("runtime/detection/blaze_face_short_range.tflite", "runtime/detection"),
        # The keys ship inside the build so it runs on double-click; anyone given the folder can read
        # them in _internal/.env, which is also where they can be replaced without rebuilding.
        (".env", "."),
    ],
    binaries=mp_binaries,
    # plyer picks its Windows notification backend by name at runtime.
    hiddenimports=mp_hiddenimports + ["plyer.platforms.win.notification"],
    excludes=["pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DeepfakeDetector", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="DeepfakeDetector")
