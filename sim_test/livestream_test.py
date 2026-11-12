"""Isaac Sim WebRTC livestream 启动测试：服务器 headless 跑，笔记本浏览器看。"""
import time

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True, livestream=1)  # 1 = WebRTC public
app = launcher.app
print("APP_STARTED", flush=True)

for i in range(1200):
    app.update()
    if i % 20 == 0:
        print("tick", i, flush=True)
    time.sleep(0.5)

app.close()
