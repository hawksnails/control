# PID Control Algorithm for AI Challenge
#######################################################
## Read Me
## This code is for the control of the vehicle using PID control.
## The vehicle is controlled to follow the path.
## The state variables are the position, velocity, and acceleration of the vehicle.
## The path is given by the waypoints.
## First edited by Michitoshi TSUBAKI on 2024/08/07
## Code to generate Animation is written by ChatGPT
## Last edited by Michitoshi TSUBAKI on 2024/08/08
#######################################################

#モジュールのインポート
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#車両の仕様を設定
WB = 1.087 #[m] ホイールベース
MAX_STEER = math.radians(80) #最大ステアリング転舵角は80度
MAX_ACCEL = 3.2 #[m/ss] 最大加速度
MAX_SPEED = 8.3333 #[m/s] 最大速度

#シミュレータのパラメタを設定
dt = 0.1 #[s] シミュレーションの時間間隔
nearest_index = 0

#様々なパラメタの設定
Kp1 = 1.0
Ki1 = 0.01
Kd1 = 0.01
Kp2 = 0.9
Ki2 = 0.1
Kd2 = 0.01
K = 0.2

#誤差の初期化
lateral_error = 0.0
steer_error = 0.0
velocity_error = 0.0
velocity_error_integral = 0.0
velocity_error_differential = 0.0
velocity_error_was = 0.0
steer_error_integral = 0.0
steer_error_differential = 0.0
steer_error_was = 0.0

#車両の状態を表すクラス
class Car:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.yaw = 0.0
        self.delta = 0.0
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw -= self.v / WB * np.tan(delta) * dt
        if self.yaw > math.pi:
            self.yaw -= 2.0 * math.pi
        elif self.yaw < -math.pi:
            self.yaw += 2.0 * math.pi
        if self.v > MAX_SPEED:
            self.v = MAX_SPEED
        elif self.v < 0:
            self.v = 0
        else:
            self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

#マップのクラス
class Map:
    def __init__(self):
        self.X = np.arange(0, 100, 0.1)
        self.Y = np.sin(self.X / 5.0) * self.X / 2.0

    def make_map(self):
        return self.X, self.Y

    def calc_target_yaw(self):
        target_yaw = np.zeros(len(self.X))
        for i in range(len(self.X)-1):
            target_yaw[i] = math.atan2(self.Y[i+1] - self.Y[i], self.X[i+1] - self.X[i])
        target_yaw[-1] = target_yaw[-2]
        return target_yaw

    def calc_velocity(self):
        velocity = np.zeros(len(self.X))
        for i in range(len(self.X)-1):
            velocity[i] = 8.0
        velocity[-1] = velocity[-2]
        return velocity

#PID制御を行う関数
def PID_control_vel(car, map, Kp1, Ki1, Kd1):
    global lateral_error
    global steer_error
    global velocity_error
    global velocity_error_integral
    global velocity_error_differential
    global velocity_error_was
    global nearest_index
    global MAX_ACCEL
    # 車両の位置に最も近いマップ上の点を見つける
    nearest_index_kouho = np.argmin(np.sqrt((map.X - car.rear_x)**2 + (map.Y - car.rear_y)**2))
    if nearest_index_kouho > nearest_index:
        nearest_index = nearest_index_kouho
    else:
        nearest_index = nearest_index
    nearest_x = map.X[nearest_index]
    nearest_y = map.Y[nearest_index]
    #目標速度はmapの速度を参照
    target_v = map.calc_velocity()[nearest_index]
    #目標速度と現在速度の差を計算
    velocity_error = target_v - car.v
    #目標速度と現在速度の差を積分
    velocity_error_integral += velocity_error * dt
    #目標速度と現在速度の差を微分
    velocity_error_differential = (velocity_error - velocity_error_was) / dt
    #目標速度と現在速度の差をPID制御
    a = Kp1 * velocity_error + Ki1 * velocity_error_integral + Kd1 * velocity_error_differential
    #1コマ前の目標速度と現在速度の差を保存
    velocity_error_was = velocity_error
    if a > MAX_ACCEL:
        a = MAX_ACCEL
    elif a < -MAX_ACCEL:
        a = -MAX_ACCEL
    else:
        a = a
    return a

def PID_control_steer(car, map, Kp2, Ki2, Kd2,K):
    global lateral_error
    global steer_error
    global velocity_error
    global velocity_error_integral
    global velocity_error_differential
    global velocity_error_was
    global steer_error_integral
    global steer_error_differential
    global steer_error_was
    global nearest_index
    global MAX_STEER
    # 車両の位置に最も近いマップ上の点を見つける
    nearest_x = map.X[nearest_index]
    nearest_y = map.Y[nearest_index]
    # 目標ステア角はmapの接線を参照
    target_steer = map.calc_target_yaw()[nearest_index]
    # 目標ステア角と現在ステア角の差を計算
    steer_error = (car.yaw - target_steer )#* lateral_error/2
    # 車両の位置と最も近いマップ上の点との距離を計算
    lateral_error = car.calc_distance(nearest_x, nearest_y) 
    #車両の位置と最も近いマップ上の点との角度を計算
    lateral_angle_error = math.atan2(nearest_y - car.rear_y, nearest_x - car.rear_x) - car.yaw
    # 目標ステア角と現在ステア角の差を積分
    steer_error_integral += steer_error * dt
    # 目標ステア角と現在ステア角の差を微分
    steer_error_differential = (steer_error - steer_error_was) / dt
    # 目標ステア角と現在ステア角の差をPID制御
    delta = Kp2 * steer_error + Ki2 * steer_error_integral + Kd2 * steer_error_differential
    # 1コマ前の目標ステア角と現在ステア角の差を保存
    steer_error_was = steer_error
    out = delta - lateral_error * lateral_angle_error * K
        
    if out > MAX_STEER:
        out = MAX_STEER
    elif out < -MAX_STEER:
        out = -MAX_STEER
    else:
        out = out
    #print("lateral_error: ", lateral_error)
    #print("car.yaw: ", car.yaw)
    #print("target_steer: ", target_steer)
    #print("steer_error: ", steer_error)
    #print("out: ", out)
    return out


#車両のシミュレーションを行う関数を作成
def simulate(map, num_steps):
    global car
    global steer_error
    global nearest_index
    x_positions = []
    y_positions = []
    velocity_of_car = []
    steer_errors = []
    for i in range(num_steps):
        #現在の車両の状態を保存
        x_positions.append(car.x)
        y_positions.append(car.y)
        velocity_of_car.append(car.v)
        steer_errors.append(steer_error)
        #PID制御を使って加速度とステアリング角を計算
        accel = PID_control_vel(car, map, Kp1, Ki1, Kd1)
        steer = PID_control_steer(car, map, Kp2, Ki2, Kd2,K)
        
        #車両を更新
        car.update(accel, steer)
        if nearest_index == len(map.X) - 1:
            break
    return x_positions, y_positions, velocity_of_car, steer_errors

#mapのインスタンスを作成
map = Map()
X, Y = map.make_map()
tangent = map.calc_target_yaw()

#車両のインスタンスを作成
car = Car()

#シミュレーションのステップ数
num_steps = 500

#シミュレーションを実行
x_positions, y_positions, velocity_of_car, steer_errors = simulate(map, num_steps)

#結果をプロット
plt.plot(X, Y, label="map")
plt.plot(x_positions, y_positions, label="car")
plt.legend()
plt.show()
