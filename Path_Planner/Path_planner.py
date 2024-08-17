import numpy as np
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as ani

#CSVから地図データを読み込む
df = pd.read_csv('osm_nodes_coordinates_with_wall_info.csv')

#中央線、左壁、右壁の情報を抽出
centerline_coords = df[df['Wall Info'] == 'Centerline'][['X (meters)', 'Y (meters)']].values
left_wall_coords = df[df['Wall Info'] == 'Left'][['X (meters)', 'Y (meters)']].values
right_wall_coords = df[df['Wall Info'] == 'Right'][['X (meters)', 'Y (meters)']].values

#デバッグ用マップ描画
#plt.figure(figsize=(10, 10))
#plt.scatter(centerline_coords[:, 0], centerline_coords[:, 1], c='blue', #marker='o', edgecolor='k')
#plt.scatter(left_wall_coords[:, 0], left_wall_coords[:, 1], c='red', marker='o', edgecolor='k')
#plt.scatter(right_wall_coords[:, 0], right_wall_coords[:, 1], c='green', marker='o', edgecolor='k')
#plt.xlabel('X (meters)')
#plt.ylabel('Y (meters)')
#plt.title('OSM Nodes for Specific Relation in Meter Coordinate System')
#plt.gca().set_aspect('equal')
#plt.grid(True)
#plt.show()

#定数設定
MAX_ACCEL = 3.2  # 最大加速度 [m/s^2]
MAX_SPEED = 8.3333  # 最大速度 [m/s]
MIN_DISTANCE_TO_WALL = 0.8  # 壁からの最小距離 [m]
N = 0  # 経由点の数（後で設定）
dt = 0.1  # タイムステップ
INIT_DIFF = 2  # 中央線と解の距離の最大値
min_distance = 1.5  # 経由点間の最小距離


def downsample_coordinates(coords, min_distance):
    """指定された最小距離以上に間引いた座標リストを返す"""
    if len(coords) < 2:
        return coords
    
    downsampled = [coords[0]]  # 最初の点を追加
    last_point = coords[0]
    
    for point in coords[1:]:
        distance = np.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
        if distance >= min_distance:
            downsampled.append(point)
            last_point = point
    
    return np.array(downsampled)


#centerlineを間引く
centerline_coords = downsample_coordinates(centerline_coords, min_distance)
centerline_coords = np.array_split(centerline_coords, 4)[0]


#Optiオブジェクトの作成
opti = ca.Opti()

#経由点の数を中央線のノード数に設定
N = len(centerline_coords)-1
#print("N:", N) #デバッグ用

#状態変数と制御入力
X = opti.variable(6, N+1)  # X = [[x, y, th, vx, vy, vth], ...]
U = opti.variable(3, N)  # U = [[ux, uy, uth], ...]
T = opti.variable(N+1)  # T は各経由点での時刻

T_total = ca.sum1(T)

#logの準備
log = []
opti.callback(lambda i: log.append((i, opti.debug)))  # logの追加

#制約条件############################################
#運動方程式
def subject_to_dynamics():
    dt = T_total / N
    for i in range(N):
        x_next = ca.vertcat(
            X[0,i] + dt * X[3,i] + 0.5 * dt**2 * U[0,i],  # x
            X[1,i] + dt * X[4,i] + 0.5 * dt**2 * U[1,i],  # y
            X[2,i] + dt * X[5,i] + 0.5 * dt**2 * U[2,i],  # th
            X[3,i] + dt * U[0,i],  # vx
            X[4,i] + dt * U[1,i],  # vy
            X[5,i] + dt * U[2,i]   # vth
        )
        opti.subject_to(X[:,i+1] == x_next)

#初期解(centerline)との距離制約
def subject_to_distance():
    for i in range(N):
        opti.subject_to((X[0,i] - centerline_coords[i, 0])**2 + (X[1,i] - centerline_coords[i, 1])**2 <= INIT_DIFF)

#壁制約
def subject_to_wall():
    for i in range(N):
        # 車両の位置を3次元ベクトルに拡張
        #car_pos = ca.vertcat(X[0,i], X[1,i], 0)
        
        # 左壁の制約
        #for j in range(len(left_wall_coords)-1):
        #    p1 = ca.vertcat(left_wall_coords[j][0], left_wall_coords[j][1], 0)
        #    p2 = ca.vertcat(left_wall_coords[j+1][0], left_wall_coords[j+1][1], 0)
        #    wall_vec = p2 - p1
        #    car_vec = car_pos - p1
        #    cross_product = ca.cross(wall_vec, car_vec)
        #    opti.subject_to(cross_product[2] >= 0)  # Z成分で左側の制約を表現
        
        # 右壁の制約
        #for j in range(len(right_wall_coords)-1):
        #    p1 = ca.vertcat(right_wall_coords[j][0], right_wall_coords[j][1], 0)
        #    p2 = ca.vertcat(right_wall_coords[j+1][0], right_wall_coords[j+1][1], 0)
        #    wall_vec = p2 - p1
        #    car_vec = car_pos - p1
        #    cross_product = ca.cross(wall_vec, car_vec)
        #    opti.subject_to(cross_product[2] <= 0)  # Z成分で右側の制約を表現
        
        for p in left_wall_coords:
            opti.subject_to((X[0,i] - p[0])**2 + (X[1,i] - p[1])**2 >= MIN_DISTANCE_TO_WALL**2)
        for p in right_wall_coords:
            opti.subject_to((X[0,i] - p[0])**2 + (X[1,i] - p[1])**2 >= MIN_DISTANCE_TO_WALL**2)

#障害物制約
# 障害物の中心と半径がリストで[x, y, r]と与えられたとき、半径+0.5m以上離れる
def subject_to_obstacles(L):
    for i in range(N):
        for p in L:
            opti.subject_to((X[0,i] - p[0])**2 + (X[1,i] - p[1])**2 >= (p[2]+0.5)**2)

#加速度制約
def subject_to_acc_limit():
    for i in range(N):
        opti.subject_to(ca.fabs(U[0,i])**2 + ca.fabs(U[1,i])**2 <= MAX_ACCEL**2)


#障害物の中心と半径
obstacle = [[20.6, 23.2, 0.8]]

#subject_to_distance()
subject_to_dynamics()
subject_to_wall()
subject_to_obstacles(obstacle)
subject_to_acc_limit()

# 終端拘束
# 始点は中央線の最初の点で停止している
opti.subject_to(X[0,0] == centerline_coords[0, 0]) 
opti.subject_to(X[1,0] == centerline_coords[0, 1])
opti.subject_to(X[2,0] == 0)  # th
opti.subject_to(X[3,0] == 0)  # vx
opti.subject_to(X[4,0] == 0)  # vy
opti.subject_to(X[5,0] == 0)  # vth
opti.subject_to(U[0,0] == 0)  # acc_x
opti.subject_to(U[1,0] == 0)  # acc_y
opti.subject_to(U[2,0] == 0)  # acc_th

# 終端拘束
# 終点は中央線の最後の点で停止している
opti.subject_to(X[0,N] == centerline_coords[-1, 0])
opti.subject_to(X[1,N] == centerline_coords[-1, 1])
opti.subject_to(X[2,N] == 0)  # th
opti.subject_to(X[3,N] == 0)  # vx
opti.subject_to(X[4,N] == 0)  # vy
opti.subject_to(X[5,N] == 0)  # vth
opti.subject_to(U[0,N-1] == 0)  # acc_x
opti.subject_to(U[1,N-1] == 0)  # acc_y
opti.subject_to(U[2,N-1] == 0)  # acc_th

# 目的関数
# 時間の最小化
opti.minimize(T_total)

# 初期解の設定
# 状態変数Xの初期解を中央線の座標で設定
init_x = np.zeros((6, N+1))
init_x[0, :] = centerline_coords[:, 0]
init_x[1, :] = centerline_coords[:, 1]
init_x[2, :] = 0  # th
init_x[3, :] = 0  # vx
init_x[4, :] = 0  # vy
init_x[5, :] = 0  # vth

#初期解の描画(デバッグ)
plt.plot(init_x[0,:], init_x[1,:])
plt.gca().set_aspect('equal')
plt.gca().add_patch(
    patch.Circle( (obstacle[0][0], obstacle[0][1]), obstacle[0][2], edgecolor="black", facecolor="r", alpha=0.8)
)
plt.scatter(centerline_coords[0, 0], centerline_coords[0, 1], c='blue', marker='o', edgecolor='k')
plt.scatter(centerline_coords[-1, 0], centerline_coords[-1, 1], c='blue', marker='o', edgecolor='k')
plt.legend(['Optimized Path', 'Obstacle'])
plt.show()
plt.show()


# 制御入力Uの初期解を0で設定
init_u = np.zeros((3, N))

# 各経由点での時刻の初期解
init_T = np.linspace(0, 100, N+1)

# 初期解をOptiに設定
opti.set_initial(X, init_x)
opti.set_initial(U, init_u)
opti.set_initial(T, init_T)



# 最適化問題を解く
opti.solver("ipopt")
try:
    opti.solve()
except:
    print("error")

#解の描画
plt.plot(opti.value(X[0,:]), opti.value(X[1,:]))
plt.gca().set_aspect('equal')
plt.gca().add_patch(
    #obstacleの描画
    patch.Circle( (obstacle[0][0], obstacle[0][1]), obstacle[0][2], edgecolor="black", facecolor="r", alpha=0.8)
)
#初期解(centerline)の描画
plt.plot(init_x[0,:], init_x[1,:])
#始点と終点の描画
plt.scatter(centerline_coords[0, 0], centerline_coords[0, 1], c='blue', marker='o', edgecolor='k')
plt.scatter(centerline_coords[-1, 0], centerline_coords[-1, 1], c='blue', marker='o', edgecolor='k')
#左と右の壁の描画
plt.scatter(left_wall_coords[:, 0], left_wall_coords[:, 1], c='red', marker='o', edgecolor='k')
plt.scatter(right_wall_coords[:, 0], right_wall_coords[:, 1], c='green', marker='o', edgecolor='k')
plt.legend(['Optimized Path', 'Obstacle', 'Initial Path', 'Start/Goal', 'Left Wall', 'Right Wall'])
plt.show()

print("log:",log)
#最適化のlogの描画
def draw_frame(i):
    j, debug = log[i]
    plt.cla()
    plt.gca().set_aspect('equal')
    plt.gca().add_patch(
        patch.Circle( (obstacle[0][0], obstacle[0][1]), obstacle[0][2], edgecolor="black", facecolor="r", alpha=0.8)
    )
    plt.scatter(debug.value(X[0,:]), debug.value(X[1,:]))
    plt.plot(debug.value(X[0,:]), debug.value(X[1,:]))

fig = plt.figure()
_ = ani.FuncAnimation(fig, draw_frame, frames=len(log), interval=10)
plt.show()