import os

os.system("""
osascript -e '
tell application "System Events"
    tell application process "QuickTime Player"
        set frontmost to true
        tell window 1
            set position to {100, 100}
            set size to {800, 600}
        end tell
    end tell
end tell'
""")
