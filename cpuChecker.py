import psutil, time

while True:
    usage = psutil.cpu_percent(interval=1)
    freq = psutil.cpu_freq()
    print(f"CPU: {usage}% | {freq.current:.0f} MHz / {freq.max:.0f} MHz")
