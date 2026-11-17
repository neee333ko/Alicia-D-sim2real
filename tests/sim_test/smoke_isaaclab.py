"""Isaac Lab 冒烟测试：起 headless 仿真，走 10 步物理，打印结果。"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
sim.reset()
for _ in range(10):
    sim.step()

print("SMOKE_OK", "dt=", sim.get_physics_dt(), "device=", sim.device)
simulation_app.close()
