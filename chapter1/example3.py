import sys
sys.path.append("..")
from grid import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    
    # 动作空间：[down(0,1), right(1,0), up(0,-1), left(-1,0), stay(0,0)]
    # 动作索引：0: down, 1: right, 2: up, 3: left, 4: stay
    
    # 根据图片中的箭头方向构建策略矩阵
    # 图片坐标从1开始，代码坐标从0开始
    # 图片中的(row, col)对应代码中的(col-1, row-1)
    # 状态索引计算：state_index = y * env_size[0] + x (x是列，y是行)
    
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    
    # 根据图片描述构建策略（图片坐标转代码坐标）
    # Row 1 (图片) -> y=0 (代码)
    policy_matrix[0 * 5 + 0, 1] = 1.0  # (1,1) -> right -> (0,0)
    policy_matrix[0 * 5 + 1, 1] = 1.0  # (1,2) -> right -> (1,0)
    policy_matrix[0 * 5 + 2, 1] = 1.0  # (1,3) -> right -> (2,0)
    policy_matrix[0 * 5 + 3, 0] = 1.0  # (1,4) -> down -> (3,0)
    policy_matrix[0 * 5 + 4, 0] = 1.0  # (1,5) -> down -> (4,0)
    
    # Row 2 (图片) -> y=1 (代码)
    policy_matrix[1 * 5 + 0, 2] = 1.0  # (2,1) -> up -> (0,1)
    policy_matrix[1 * 5 + 1, 2] = 1.0  # (2,2) -> up -> (1,1)
    policy_matrix[1 * 5 + 2, 1] = 1.0  # (2,3) -> right -> (2,1)
    policy_matrix[1 * 5 + 3, 0] = 1.0  # (2,4) -> down -> (3,1)
    policy_matrix[1 * 5 + 4, 0] = 1.0  # (2,5) -> down -> (4,1)
    
    # Row 3 (图片) -> y=2 (代码)
    policy_matrix[2 * 5 + 0, 2] = 1.0  # (3,1) -> up -> (0,2)
    policy_matrix[2 * 5 + 1, 3] = 1.0  # (3,2) -> left -> (1,2)
    policy_matrix[2 * 5 + 2, 0] = 1.0  # (3,3) -> down -> (2,2)
    policy_matrix[2 * 5 + 3, 1] = 1.0  # (3,4) -> right -> (3,2)
    policy_matrix[2 * 5 + 4, 0] = 1.0  # (3,5) -> down -> (4,2)
    
    # Row 4 (图片) -> y=3 (代码)
    policy_matrix[3 * 5 + 0, 2] = 1.0  # (4,1) -> up -> (0,3)
    policy_matrix[3 * 5 + 1, 1] = 1.0  # (4,2) -> right -> (2,3)
    policy_matrix[3 * 5 + 2, 4] = 1.0  # (4,3) -> stay -> (2,3)
    policy_matrix[3 * 5 + 3, 3] = 1.0  # (4,4) -> left -> (2,3)
    policy_matrix[3 * 5 + 4, 0] = 1.0  # (4,5) -> down -> (4,3)
    
    # Row 5 (图片) -> y=4 (代码)
    policy_matrix[4 * 5 + 0, 2] = 1.0  # (5,1) -> up -> (0,4)
    policy_matrix[4 * 5 + 1, 1] = 1.0  # (5,2) -> right -> (1,4)
    policy_matrix[4 * 5 + 2, 2] = 1.0  # (5,3) -> up -> (2,4)
    policy_matrix[4 * 5 + 3, 3] = 1.0  # (5,4) -> left -> (3,4)
    policy_matrix[4 * 5 + 4, 3] = 1.0  # (5,5) -> left -> (4,4)
    
    # 值迭代算法参数
    gamma = 0.9  # 折扣因子
    theta = 1e-4  # 收敛阈值
    max_iterations = 1000
    
    # 初始化状态价值函数
    state_values = np.zeros(env.num_states)
    
    # 值迭代主循环
    for iteration in range(max_iterations):
        delta = 0
        new_state_values = np.zeros(env.num_states)
        
        # 遍历所有状态
        for state_idx in range(env.num_states):
            # 跳过目标状态和禁止状态
            x = state_idx % env.env_size[0]
            y = state_idx // env.env_size[0]
            state = (x, y)
            
            # 根据策略选择动作
            action_probs = policy_matrix[state_idx]
            action_idx = np.argmax(action_probs)
            action = env.action_space[action_idx]
            
            # 计算下一个状态和奖励
            next_state, reward = env._get_next_state_and_reward(state, action)
            next_x, next_y = next_state
            next_state_idx = next_y * env.env_size[0] + next_x
            
            # 贝尔曼方程更新
            new_state_values[state_idx] = reward + gamma * state_values[next_state_idx]
            
            # 更新最大变化量
            delta = max(delta, abs(new_state_values[state_idx] - state_values[state_idx]))
        
        # 更新状态价值函数
        state_values = new_state_values.copy()
        
        # 检查收敛
        if delta < theta:
            print(f"值迭代收敛于第 {iteration + 1} 次迭代")
            break
    
    # 输出保留2位小数的状态价值函数
    state_values_rounded = np.round(state_values, 1)
    print(f"最终状态价值函数（经过 {iteration + 1} 次迭代）:")
    print(state_values_rounded.reshape(env.env_size[1], env.env_size[0]))
    
    # 初始化环境用于可视化（设置traj为只包含起始点的列表，避免render报错）
    env.reset()
    env.traj = [env.start_state]  # 只包含起始点，避免空列表导致zip报错
    
    # 可视化
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(state_values_rounded, precision=2)
    
    # 隐藏agent和轨迹（因为我们只显示价值函数和策略）
    env.agent_star.set_data([], [])
    env.traj_obj.set_data([], [])
    
    env.render(animation_interval=5)
    