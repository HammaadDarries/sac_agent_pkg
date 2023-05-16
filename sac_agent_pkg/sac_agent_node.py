#! /usr/bin/env python 

import rclpy
import csv
import numpy as np
import time
import os

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist

import torch

MAX_SPEED = 2
MAX_STEER = 0.4
NOISE_FACTOR = 0.002

class SACAgent:
    def __init__(self, filepath):
        self.actor = torch.load(filepath)

    def act(self, state):
        state = torch.FloatTensor(state)
        action, log_prob = self.actor(state)
        
        return action.detach().numpy()

class SACAgentNode (Node):
    def __init__(self):
        super().__init__('sac_agent_node')

        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = None
        nn_path = '/home/hammaad/sim_ws/src/sac_agent_pkg/Data/Main_Agent/gamma_0.99.pth'
        self.scan_buffer = np.zeros((2, 20))
        self.lidar_state = np.zeros(40)
        self.lap_complete = False
        self.start_position = None

        self.agent = SACAgent(nn_path)
        self.position_history = []
        self.speed_history = []
        self.steering_angle_history = []        
        self.speed_output_history = []

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = self.add_noise(ranges, NOISE_FACTOR)
        
        # Normalize ranges
        scaled_ranges = ranges/30
        # get 20 beams out of the 1080 by averaging every 54 beams
        ranges = np.clip(scaled_ranges, 0, 1)
        mean_ranges = np.mean(ranges.reshape(-1, 54), axis=1)
        
        # FOR REAL CAR #
        # mean_ranges = np.mean(ranges[:-1].reshape(-1, 54), axis=1)
        scan = mean_ranges

        if self.scan_buffer.all() ==0: # first reading
            for i in range(2):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        lidar_state = np.reshape(self.scan_buffer, (20 * 2))

        self.lidar_state = lidar_state

    def timer_callback(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'Attempt_1_Lesgooo'

        nn_state = self.lidar_state
        nn_action = self.agent.act(nn_state)

        steering_angle = nn_action[0] * MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED)

        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed

        self.steering_angle_history.append(steering_angle)
        self.speed_output_history.append(speed)

        self.drive_pub.publish(msg)


    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        self.twist = msg.twist.twist

        self.position_history.append((self.pose.position.x, self.pose.position.y))
        self.speed_history.append(self.twist.linear.x)

        current_position = (self.pose.position.x, self.pose.position.y)
        if self.start_time is None:
            self.start_time = self.get_clock().now()

        if self.check_lap_complete(current_position):
            time_since_start = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            self.get_logger().info(f'Lap completed in {time_since_start} seconds.')
            self.start_time = self.get_clock().now()

        self.position_history.append(current_position)
        self.speed_history.append(self.twist.linear.x)


    # NEW #
    def add_noise(self, ranges, noise_factor):
        noise = np.random.normal(0, noise_factor, ranges.shape)
        noisy_ranges = ranges + noise
        return noisy_ranges

    # def save_data_to_file(self):
    #     with open('position_history.csv', 'w', newline='') as position_file, open('speed_history.csv', 'w', newline='') as speed_file:
    #         position_writer = csv.writer(position_file)
    #         speed_writer = csv.writer(speed_file)
        
    #         position_writer.writerow(['x', 'y'])
    #         speed_writer.writerow(['speed'])

    #         for pos, speed in zip(self.position_history, self.speed_history):
    #             position_writer.writerow(pos)
    #             speed_writer.writerow([speed])

    # def save_data_to_file(self):
    #     folder_name = f"PLOTS/speed_{MAX_SPEED}_{NOISE_FACTOR}_data"
    #     os.makedirs(folder_name, exist_ok=True)

    #     position_file_path = os.path.join(folder_name, 'position_history.csv')
    #     speed_file_path = os.path.join(folder_name, 'speed_history.csv')

    #     with open(position_file_path, 'w', newline='') as position_file, open(speed_file_path, 'w', newline='') as speed_file:
    #         position_writer = csv.writer(position_file)
    #         speed_writer = csv.writer(speed_file)

    #         position_writer.writerow(['x', 'y'])
    #         speed_writer.writerow(['speed'])

    #         for pos, speed in zip(self.position_history, self.speed_history):
    #             position_writer.writerow(pos)
    #             speed_writer.writerow([speed])

    def save_data_to_file(self):
        folder_name = f"PLOTS/speed_{MAX_SPEED}_{NOISE_FACTOR}_data"
        os.makedirs(folder_name, exist_ok=True)

        position_file_path = os.path.join(folder_name, 'position_history.csv')
        speed_file_path = os.path.join(folder_name, 'speed_history.csv')
        steering_angle_file_path = os.path.join(folder_name, 'steering_angle_history.csv')
        speed_output_file_path = os.path.join(folder_name, 'speed_output_history.csv')

        with open(position_file_path, 'w', newline='') as position_file, open(speed_file_path, 'w', newline='') as speed_file, open(steering_angle_file_path, 'w', newline='') as steering_angle_file, open(speed_output_file_path, 'w', newline='') as speed_output_file:
            position_writer = csv.writer(position_file)
            speed_writer = csv.writer(speed_file)
            steering_angle_writer = csv.writer(steering_angle_file)
            speed_output_writer = csv.writer(speed_output_file)

            position_writer.writerow(['x', 'y'])
            speed_writer.writerow(['speed'])
            steering_angle_writer.writerow(['steering_angle'])
            speed_output_writer.writerow(['speed_output'])

            for pos, speed, angle, speed_output in zip(self.position_history, self.speed_history, self.steering_angle_history, self.speed_output_history):
                position_writer.writerow(pos)
                speed_writer.writerow([speed])
                steering_angle_writer.writerow([angle])
                speed_output_writer.writerow([speed_output])


    def check_lap_complete(self, current_position):
        if self.start_position is None:
            self.start_position = current_position
            return False

        distance_to_start = np.linalg.norm(np.array(self.start_position) - np.array(current_position))
        if not self.lap_complete and distance_to_start < 1.0:
            self.lap_complete = True
        elif self.lap_complete and distance_to_start > 2.0:
            self.lap_complete = False
            return True

        return False

def main(args = None):
    rclpy.init(args = args)
    sac_agent_node = SACAgentNode()
    
    try:
        rclpy.spin(sac_agent_node)
    except KeyboardInterrupt:
        pass
    finally:
        # sac_agent_node.save_data_to_file()
        sac_agent_node.destroy_node()
        rclpy.shutdown()



