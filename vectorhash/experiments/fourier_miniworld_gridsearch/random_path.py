import math
import numpy as np
import matplotlib.pyplot as plt

# Functions for trajectory generation

mu = 0
sigma = 330
b_ = 0.13
dt = 0.1
min_distance = 0.03
rho_rh = 0.25
limits = np.array([1, 1])
t = 100
num_samples = int(t / dt)

# Inputs:
# scene - contains description of the environment
# num_samples - number of samples to take

# Outputs:
# Position: Array of position vectors
# Velocity: Array of velocities


def distance_from_wall(position, direction_vector, limits):
    x, y = position[0], position[1]
    lim_x, lim_y = limits[0], limits[1]
    # 0=left, 1=right, 2=top, 3=down
    dists = np.array([x, lim_x - x, y, lim_y - y])
    distance_wall = np.min(dists)

    closest_wall = np.argmin(dists)
    if closest_wall == 1:
        normal = np.array([1, 0])
    elif closest_wall == 0:
        normal = np.array([-1, 0])
    elif closest_wall == 3:
        normal = np.array([0, 1])
    elif closest_wall == 2:
        normal = np.array([0, -1])

    angle_wall = math.acos(np.dot(direction_vector, normal))
    return distance_wall, angle_wall


def turn(direction_vector, angle, dt):
    new_angle = np.arctan2(direction_vector[1], direction_vector[0]) + dt * np.deg2rad(
        angle
    )
    return np.array([np.cos(new_angle), np.sin(new_angle)])


def generate_rat_trajectory(limits, num_samples, start_location):
    random_turn = np.random.normal(mu, sigma, num_samples)
    random_velocity = np.random.rayleigh(b_, num_samples)

    v = random_velocity[0]
    direction_vector = np.array(
        [np.cos(np.deg2rad(random_turn[0])), np.sin(np.deg2rad(random_turn[0]))]
    )

    position = np.zeros((num_samples, 2))
    position[0] = start_location
    velocity = np.zeros((num_samples, 2))
    velocity[0] = v * direction_vector

    for i in range(1, num_samples):
        distance_wall, angle_wall = distance_from_wall(
            position[i - 1, :], direction_vector, limits
        )

        if distance_wall < min_distance and np.abs(angle_wall) < np.pi / 2:
            angle = (
                np.sign(angle_wall) * (np.pi / 2 - np.abs(angle_wall)) + random_turn[i]
            )
            v = (1 - rho_rh) * v
        else:
            angle = random_turn[i]
            v = random_velocity[i]

        position[i, :] = position[i - 1, :] + direction_vector * v * dt
        velocity[i, :] = direction_vector * v * dt

        if position[i, 0] < 0:
            diff = np.abs(position[i, 0])
            velocity[i, 0] -= diff
            position[i, 0] = 0
        if position[i, 0] > limits[0]:
            diff = np.abs(position[i, 0] - limits[0])
            velocity[i, 0] -= diff
            position[i, 0] = limits[0]
        if position[i, 1] < 0:
            diff = np.abs(position[i, 1])
            velocity[i, 1] -= diff
            position[i, 1] = 0
        if position[i, 1] > limits[1]:
            diff = np.abs(position[i, 1] - limits[1])
            velocity[i, 1] -= diff
            position[i, 1] = limits[1]

        if (
            position[i, 0] < 0
            or position[i, 0] > limits[0]
            or position[i, 1] < 0
            or position[i, 1] > limits[1]
        ):
            print("OUTSIDE BOUNDARIES: t=" + str(i * dt))

        direction_vector = turn(direction_vector, angle, dt)

    return [position, velocity]


if __name__ == "__main__":
    pos, _ = generate_rat_trajectory(limits, num_samples, start_location=limits / 2)

    vel = pos[1:, :] - pos[:-1, :]
    noisy_vel = vel + np.random.normal(0, 0.01, vel.shape)

    fig1, ax1 = plt.subplots()

    noisy_pos = np.cumsum(np.concatenate([np.zeros((1, 2)), noisy_vel]), axis=0) + pos[0, :]

    fig, ax = plt.subplots()
    ax.plot(noisy_pos[:, 0], noisy_pos[:, 1], label="Noisy Trajectory")
    ax.plot(pos[:, 0], pos[:, 1], label="True Trajectory")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_xlim(0, limits[0])
    ax.set_ylim(0, limits[1])
    ax.legend()
    fig.savefig("trajectory.png")

    err_vs_t = np.linalg.norm(noisy_pos - pos, axis=1)

    plt.figure()
    plt.plot(err_vs_t)
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.title("Error vs Time")
    plt.savefig("error_vs_time.png")
    plt.close()
