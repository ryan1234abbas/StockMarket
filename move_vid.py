import os

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
