import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# XML ファイルのパスを指定します。
xml_file = 'map.osm'

# XML ファイルを解析します。
tree = ET.parse(xml_file)
root = tree.getroot()

# ノードの座標を格納するリストを初期化します。
latitudes = []
longitudes = []

# ノード要素をすべて取得し、座標をリストに追加します。
for node in root.findall('node'):
    lat = float(node.get('lat'))
    lon = float(node.get('lon'))
    latitudes.append(lat)
    longitudes.append(lon)

# ノードの位置をプロットします。
plt.figure(figsize=(10, 6))
plt.scatter(longitudes, latitudes, c='blue', marker='o', edgecolor='k')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('OSM Nodes')
plt.grid(True)
plt.show()