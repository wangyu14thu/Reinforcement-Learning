import sys
sys.path.append("..")
from grid import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    
    # 值迭代算法参数
    gamma = 0.9  # 折扣因子 γ
    theta = 1e-4  # 收敛阈值
    max_iterations = 1000
    
    # 初始化：v0 - 初始状态价值函数
    v_k = np.zeros(env.num_states)  # vk: 当前迭代的状态价值函数
    v_k_prev = np.zeros(env.num_states)  # vk-1: 上一次迭代的状态价值函数
    
    # 初始化策略矩阵
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    
    # 值迭代主循环：while ||vk - vk-1|| > theta
    for k in range(max_iterations):
        v_k_prev = v_k.copy()  # 保存上一次的值
        
        # 对每个状态 s ∈ S
        for state_idx in range(env.num_states):
            x = state_idx % env.env_size[0]
            y = state_idx // env.env_size[0]
            state = (x, y)
            
            # 对每个动作 a ∈ A(s)，计算 qk(s, a)
            q_values = []  # 存储所有动作的Q值
            
            for action_idx, action in enumerate(env.action_space):
                # 计算 qk(s, a) = Σr p(r|s,a)r + γ Σs' p(s'|s,a)vk(s')
                # 在GridWorld中，转移是确定性的，所以简化为：
                # qk(s, a) = r + γ * vk(s')
                next_state, reward = env._get_next_state_and_reward(state, action)
                next_x, next_y = next_state
                next_state_idx = next_y * env.env_size[0] + next_x
                
                # Q值计算：即时奖励 + 折扣因子 * 下一状态价值
                q_k_s_a = reward + gamma * v_k_prev[next_state_idx]
                q_values.append(q_k_s_a)
            
            # a*(s) = arg max_a qk(s, a) - 找到最优动作
            a_star_idx = np.argmax(q_values)
            
            # 更新策略：πk+1(a|s) = 1 if a = a*, else 0
            policy_matrix[state_idx, :] = 0  # 先清零
            policy_matrix[state_idx, a_star_idx] = 1.0
            
            # 更新价值：vk+1(s) = max_a qk(s, a)
            v_k[state_idx] = max(q_values)
        
        # 检查收敛：||vk - vk-1|| < theta
        delta = np.max(np.abs(v_k - v_k_prev))
        
        if delta < theta:
            print(f"值迭代收敛于第 {k + 1} 次迭代，||vk - vk-1|| = {delta:.6f}")
            break
    
    # 输出保留1位小数的状态价值函数
    state_values_rounded = np.round(v_k, 1)
    print(f"\n最终状态价值函数（经过 {k + 1} 次迭代）:")
    print(state_values_rounded.reshape(env.env_size[1], env.env_size[0]))
    print("\n最优策略已通过值迭代算法得到")
    
    # 初始化环境用于可视化（设置traj为只包含起始点的列表，避免render报错）
    env.reset()
    env.traj = [env.start_state]  # 只包含起始点，避免空列表导致zip报错
    
    # 可视化
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(state_values_rounded, precision=1)
    
    # 隐藏agent和轨迹（因为我们只显示价值函数和策略）
    env.agent_star.set_data([], [])
    env.traj_obj.set_data([], [])
    
    env.render(animation_interval=5)
    