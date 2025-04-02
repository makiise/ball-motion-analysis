import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

good_lighting_video = "good_lighting.mp4"
bad_lighting_video = "bad_lighting.mp4"

edge_params = {
    "good": {"low_threshold": 50, "high_threshold": 150},
    "bad": {"low_threshold": 30, "high_threshold": 100},
}

def detect_ball_diameter(frame, edge_param):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, edge_param['low_threshold'], edge_param['high_threshold'])

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea, default=None)
        if max_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            diameter = 2 * radius
            return diameter, (x, y)
    return None, None

def estimate_scaling_factor(reference_object_real_size, reference_object_pixel_size):
    scaling_factor = reference_object_real_size / reference_object_pixel_size
    return scaling_factor


def compute_ball_mass(diameter_in_pixels, scaling_factor, density):
    radius_real_world = (diameter_in_pixels / 2) * scaling_factor
    volume = (4 / 3) * np.pi * (radius_real_world ** 3)
    mass = density * volume
    return mass


def compute_velocity(positions, dt, scaling_factor):
    velocities = []
    for i in range(1, len(positions)):
        delta_x = positions[i][0] - positions[i - 1][0]
        delta_y = positions[i][1] - positions[i - 1][1]

        displacement_meters = np.sqrt(delta_x ** 2 + delta_y ** 2) * scaling_factor

        velocity = displacement_meters / dt
        velocities.append(velocity)

    return np.mean(velocities) if velocities else 0


def estimate_drag_coefficient(velocity, ball_diameter, air_density=1.225):
    radius = ball_diameter / 2
    area = np.pi * radius ** 2

    mass = compute_ball_mass(ball_diameter, 1, 1200)
    drag_force = mass * 9.81

    if velocity > 0:
        drag_coefficient = (2 * drag_force) / (air_density * area * velocity ** 2)
    else:
        drag_coefficient = 0
    return drag_coefficient


def process_video_and_compute_properties(video_path, reference_object_real_size, reference_object_pixel_size,
                                         ball_pixel_diameter, edge_param):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / frame_rate
    positions = []
    time_stamps = []
    frame_index = 0
    ball_diameter_pixels_list = []

    scaling_factor = estimate_scaling_factor(reference_object_real_size, reference_object_pixel_size)
    print(f"Scaling Factor: {scaling_factor:.4f}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        diameter_in_pixels, position = detect_ball_diameter(frame, edge_param)
        if diameter_in_pixels is not None:
            ball_diameter_pixels_list.append(diameter_in_pixels)
            positions.append(position)
        frame_index += 1

    cap.release()

    if not positions:
        print("No ball detected in the video.")
        return

    average_ball_diameter = np.mean(ball_diameter_pixels_list)
    ball_mass = compute_ball_mass(ball_pixel_diameter, scaling_factor, 1200)
    print(f"Estimated Ball Mass: {ball_mass:.5f} kg")

    velocity = compute_velocity(positions, dt, scaling_factor)
    print(f"Average Ball Velocity: {velocity:.2f} m/s")

    drag_coefficient = estimate_drag_coefficient(velocity, average_ball_diameter * scaling_factor)
    print(f"Estimated Drag Coefficient: {drag_coefficient:.4f}")

reference_object_real_size = 0.15
reference_object_pixel_size = 45

ball_pixel_diameter = 33

print("Processing Good Lighting Video:")
process_video_and_compute_properties(good_lighting_video, reference_object_real_size, reference_object_pixel_size,
                                     ball_pixel_diameter, edge_params['good'])

print("\nProcessing Bad Lighting Video:")
process_video_and_compute_properties(bad_lighting_video, reference_object_real_size, reference_object_pixel_size,
                                     ball_pixel_diameter, edge_params['bad'])

g = 9.81
air_density = 1.225
radius = 0.12
mass = 0.83629
drag_coefficient = 64.7187

A = np.pi * radius ** 2


def motion_equations(t, state, drag_coefficient, air_density, radius, mass, g=9.81):
    x, y, vx, vy = state

    speed = np.sqrt(vx ** 2 + vy ** 2)

    drag_force_x = -0.5 * air_density * A * drag_coefficient * speed * vx
    drag_force_y = -0.5 * air_density * A * drag_coefficient * speed * vy

    ax = drag_force_x / mass
    ay = (-g + drag_force_y / mass)

    return [vx, vy, ax, ay]


def runge_kutta_integration(t0, t_final, dt, initial_state, drag_coefficient, air_density, radius, mass):
    times = np.arange(t0, t_final, dt)
    state = initial_state
    states = []

    for t in times:
        states.append(state)

        k1 = np.array(motion_equations(t, state, drag_coefficient, air_density, radius, mass))
        k2 = np.array(
            motion_equations(t + 0.5 * dt, state + 0.5 * dt * k1, drag_coefficient, air_density, radius, mass))
        k3 = np.array(
            motion_equations(t + 0.5 * dt, state + 0.5 * dt * k2, drag_coefficient, air_density, radius, mass))
        k4 = np.array(motion_equations(t + dt, state + dt * k3, drag_coefficient, air_density, radius, mass))

        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        if state[1] < 0:
            state[1] = 0
            state[3] = -state[3] * 0.8

    return np.array(states)

def compute_initial_velocity(positions, dt, scaling_factor):
    if len(positions) >= 2:
        delta_x = positions[1][0] - positions[0][0]
        delta_y = positions[1][1] - positions[0][1]

        displacement_meters = np.sqrt(delta_x ** 2 + delta_y ** 2) * scaling_factor

        initial_velocity = displacement_meters / dt
        return initial_velocity
    else:
        print("Not enough positions to calculate initial velocity.")
        return 0

video_path = good_lighting_video
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
dt = 1 / frame_rate
positions = []
scaling_factor = estimate_scaling_factor(reference_object_real_size, reference_object_pixel_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    diameter_in_pixels, position = detect_ball_diameter(frame, edge_params['good'])
    if diameter_in_pixels is not None:
        positions.append(position)

cap.release()

initial_velocity = compute_initial_velocity(positions, dt, scaling_factor)
print(f"Initial Velocity for the good lighting video: {initial_velocity:.2f} m/s")

initial_position = [0, 1.5]
initial_velocity = [0.25, 0]
initial_state = initial_position + initial_velocity

t0 = 0
t_final = 8
dt = 1/120

states = runge_kutta_integration(t0, t_final, dt, initial_state, drag_coefficient, air_density, radius, mass)

plt.plot(states[:, 0], states[:, 1])
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Ball Trajectory with Drag and Gravity')
plt.grid(True)
plt.show()
