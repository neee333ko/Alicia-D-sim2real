# 仿真侧物体真值：expert 用它算抓取/放置位姿，detector 用它做 ground_truth 对拍。
# 不给质量：脚本专家是纯运动学的，质量不参与任何控制计算；打滑/掉落由物理和 detector 负责。

std_msgs/Header header

# 杯子。pose 取杯口中心：z 直接就是杯口高度，release 从它往上加。
geometry_msgs/Point cup_pose
float32 cup_height      # 杯高，detector/可视化用

# 方块。pose 取几何中心。
geometry_msgs/Point cube_pose
float32 cube_edge_m     # 边长，用来算抓取高度和夹爪开口余量
