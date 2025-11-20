#include "solver.h"

#include <Eigen/Core>

using Eigen::Vector3f;

KineticState operator*(const KineticState& state, float scalar)
{
    return KineticState(
        state.position * scalar, state.velocity * scalar, state.acceleration * scalar
    );
}

KineticState operator+(const KineticState& a, const KineticState& b)
{
    return KineticState(
        a.position + b.position, a.velocity + b.velocity, a.acceleration + b.acceleration
    );
}

// External Force does not changed.

// Function to calculate the derivative of KineticState
KineticState derivative(const KineticState& state)
{
    return KineticState(state.velocity, state.acceleration, Eigen::Vector3f(0, 0, 0));
}

// Function to perform a single Forward Euler step
// 前向欧拉法：显式积分，不稳定但简单
KineticState
forward_euler_step([[maybe_unused]] const KineticState& previous, const KineticState& current)
{
    // x(t+1) = x(t) + v(t) * dt
    // v(t+1) = v(t) + a(t) * dt
    Vector3f next_pos = current.position + current.velocity * time_step;
    Vector3f next_vel = current.velocity + current.acceleration * time_step;

    return KineticState(next_pos, next_vel, current.acceleration);
}

// Function to perform a single Runge-Kutta step
// 四阶龙格-库塔法：通过四次采样计算加权平均斜率，精度极高
KineticState
runge_kutta_step([[maybe_unused]] const KineticState& previous, const KineticState& current)
{
    // 定义状态步进函数：y' = f(y) -> [v, a]
    // k1 = f(y)
    KineticState k1 = derivative(current);

    // k2 = f(y + 0.5 * dt * k1)
    KineticState s2 = current + k1 * (0.5f * time_step);
    KineticState k2 = derivative(s2);

    // k3 = f(y + 0.5 * dt * k2)
    KineticState s3 = current + k2 * (0.5f * time_step);
    KineticState k3 = derivative(s3);

    // k4 = f(y + dt * k3)
    KineticState s4 = current + k3 * time_step;
    KineticState k4 = derivative(s4);

    // y(t+1) = y(t) + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    // KineticState 重载了运算符，可以直接相加
    KineticState next_state = current + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (time_step / 6.0f);

    // 位置更新需要特别注意，标准RK4是对状态向量(x, v)做积分
    // derivative返回的是 (v, a)，积分后得到 (x, v)
    return next_state;
}

// Function to perform a single Symplectic Euler step
// 半隐式欧拉法：在计算位置时使用“新速度”，对保持轨道能量守恒很有帮助
KineticState symplectic_euler_step(const KineticState& previous, const KineticState& current)
{
    (void)previous;
    // 1. 先更新速度：v(t+1) = v(t) + a(t) * dt
    Vector3f next_vel = current.velocity + current.acceleration * time_step;

    // 2. 再用新速度更新位置：x(t+1) = x(t) + v(t+1) * dt
    Vector3f next_pos = current.position + next_vel * time_step;

    return KineticState(next_pos, next_vel, current.acceleration);
}

// // Function to perform a single Runge-Kutta step
// KineticState
// runge_kutta_step([[maybe_unused]] const KineticState& previous, const KineticState& current)
// {
//     return current;
// }

// Function to perform a single Backward Euler step
KineticState
backward_euler_step([[maybe_unused]] const KineticState& previous, const KineticState& current)
{
    return current;
}

// // Function to perform a single Symplectic Euler step
// KineticState symplectic_euler_step(const KineticState& previous, const KineticState& current)
// {
//     (void)previous;
//     return current;
// }
