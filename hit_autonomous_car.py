import random

import carla
import time
import numpy as np
import cv2
import pygame
from simple_pid import PID

# Initialize pygame
pygame.init()

# Pygame display settings
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
IM_WIDTH = 1500
IM_HEIGHT = 1500
KP = 3
KI = 0.00003
KD = 5
DEFAULT_SPEED = 4
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
# SPAWN_POINT_NUM = random.randint(0,301)
SPAWN_POINT_NUM = 152
print(f'spawn point number {SPAWN_POINT_NUM}')

# PID Controller
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.p_error = self.d_error = self.i_error = 0.0
        self.prev_error = 0.0
        self.counter = 0
        self.errorSum = 0.0
        self.minError = 999999999
        self.maxError = -99999999

    def update(self, error):
        self.p_error = error

        self.i_error += error

        self.d_error = error - self.prev_error
        self.prev_error = error

        self.errorSum += error
        self.counter += 1

        if error > self.maxError:
            self.maxError = error
        if error < self.minError:
            self.minError = error
        return (self.p_error * self.kp) + (self.i_error * self.ki) + (self.d_error * self.kd)


# Autonomous Car Class
class AutonomousCar:
    def __init__(self, vehicle, front_sensor_range):
        self.vehicle = vehicle
        self.front_sensor_range = front_sensor_range
        self.front_distance = float('inf')
        self.speed = DEFAULT_SPEED  # default speed
        self.lane_detector_pid = PIDController(KP, KI, KD)  # PID for lane keeping
        self.speed_pid = PID(1.0, 0.1, 0.05, setpoint=self.speed)
        self.speed_pid.output_limits = (0, 30)  # Speed range
        self.collided = False

    # Front distance sensor callback
    def on_distance_measurement(self, event):
        if hasattr(event, 'distance'):
            self.front_distance = event.distance
            print(f"Front Distance: {self.front_distance}")

    # Collision event callback
    def on_collision(self, event):
        if not self.collided:
            self.collided = True
            collision_actor = event.other_actor
            impulse = event.normal_impulse
            print(f"Collision detected! - Collided with: {collision_actor.type_id} ; Collision impulse: {impulse}")

    # Lane detection method
    def detect_lanes(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = edges.shape
        region_of_interest_vertices = [
            (0, height),
            (width * 0.25, height * 0.44),
            (width * 0.75, height * 0.44),
            (width, height)
        ]
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=1000)

        # Crosswalk detection (assuming white color)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Find contours for crosswalks
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)  # Draw crosswalks
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
            error = self.calculate_lane_error(lines, width)
            return error, image
        return 0, image

    # Calculate the lane error
    def calculate_lane_error(self, lines, width):
        # Average x-coordinates of the detected lane lines
        left_lane_x = []
        right_lane_x = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    if slope < 0:
                        left_lane_x.append((x1 + x2) / 2)
                    else:
                        right_lane_x.append((x1 + x2) / 2)

        if left_lane_x and right_lane_x:
            lane_center = (np.mean(left_lane_x) + np.mean(right_lane_x)) / 2
            image_center = width / 2
            error = lane_center - image_center
            return error
        return 0

    # Update control logic
    def control(self, image):
        lane_error, lane_image = self.detect_lanes(image)
        steering_adjustment = self.lane_detector_pid.update(lane_error)

        # Speed control based on front sensor
        if self.front_distance < 5:  # Minimum safe distance
            self.speed = 0
            print("Too close to an object, stopping the car.")
        elif self.front_distance < 20:  # Slow down when something is near
            self.speed = max(self.front_distance, 4)
            print(f"Slowing down, speed set to {self.speed}")
        else:
            self.speed = 4  # Normal speed
            # print(f"Normal speed set to {self.speed}")

        # Set the PID target to the current speed and calculate throttle
        self.speed_pid.setpoint = self.speed
        throttle = self.speed_pid(self.vehicle.get_velocity().length())

        # Ensure throttle is not too low
        if throttle < 0.2 and self.speed > 0:  # Set a minimum throttle to avoid zero
            throttle = 0.2

        # print(f"Throttle: {throttle}, Speed: {self.speed}")

        # Apply controls
        control = carla.VehicleControl()
        control.steer = steering_adjustment
        control.throttle = throttle
        control.brake = 0 if self.speed > 0 else 1
        self.vehicle.apply_control(control)

        return lane_image


# Set up CARLA environment
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    print(len(world.get_map().get_spawn_points()))
    spawn_point = world.get_map().get_spawn_points()[SPAWN_POINT_NUM]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is None:
        print("Vehicle not spawned!")
        return

    front_sensor_bp = blueprint_library.find('sensor.other.collision')
    front_sensor = world.spawn_actor(front_sensor_bp, carla.Transform(), attach_to=vehicle)

    # Set up collision sensor
    collision_sensor_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=vehicle)
    car = AutonomousCar(vehicle, front_sensor_range=20)
    front_sensor.listen(car.on_distance_measurement)
    collision_sensor.listen(car.on_collision)

    # Camera to capture lane images
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera_transform = carla.Transform(carla.Location(x=4, y=0, z=1.5), carla.Rotation(pitch=-30))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    camera.listen(lambda image: process_camera_image(car, image))

    try:
        while True:
            world.tick()
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            time.sleep(0.05)  # control loop delay
    except KeyboardInterrupt:
        pass
    finally:
        front_sensor.destroy()
        collision_sensor.destroy()
        camera.destroy()
        vehicle.destroy()
        pygame.quit()


# Function to process the camera image and display lane detection
def process_camera_image(car, image):
    # Convert raw data to a numpy array (RGB format)
    image_np = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_np = image_np.reshape((image.height, image.width, 4))[:, :, :3]
    # Make the image writable (deep copy of the image)
    writable_image = image_np.copy()

    # Apply control logic and get lane-detected image
    lane_image = car.control(writable_image)

    # Display the image using pygame
    lane_image_resized = cv2.resize(lane_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    lane_image_rgb = cv2.cvtColor(lane_image_resized, cv2.COLOR_BGR2RGB)
    pygame_image = pygame.surfarray.make_surface(lane_image_rgb.swapaxes(0, 1))

    # Show the image in the Pygame window
    display.blit(pygame_image, (0, 0))
    pygame.display.flip()


if __name__ == "__main__":
    main()
