import os
import platform

if platform.system() == 'Darwin':  # macOS
    os.system("""
    osascript -e '
    tell application "System Events"
        tell application process "QuickTime Player"
            set frontmost to true
            tell window 1
                set position to {0, 0}
                set size to {1300, 1300}
            end tell
        end tell
    end tell'
    """)
elif platform.system() == 'Windows':
    try:
        import pygetwindow as gw

        windows = gw.getWindowsWithTitle('TradingApp')
        if not windows:
            windows = gw.getWindowsWithTitle('Media Player')
        

        if windows:
            win = windows[0]
            win.moveTo(0, 0)
            win.resizeTo(1600, 1000)
            print("Trading App window repositioned.")
        else:
            print("Trading App window not found.")
    except ImportError:
        print("pygetwindow not installed. Run 'pip install pygetwindow'")
    except Exception as e:
        print(f"Error manipulating window: {e}")
else:
    print("Unsupported OS.")
