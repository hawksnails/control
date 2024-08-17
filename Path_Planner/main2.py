import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point

# Constants
EARTH_RADIUS = 6378137  # Earth's radius in meters

# File paths
xml_file = 'map.osm'

# Parse the XML file
tree = ET.parse(xml_file)
root = tree.getroot()

# Initialize containers for nodes
nodes_dict = {}
node_wall_info = {}

# Extract nodes
for node in root.findall('node'):
    node_id = node.get('id')
    lat = float(node.get('lat'))
    lon = float(node.get('lon'))
    nodes_dict[node_id] = (lat, lon)
    node_wall_info[node_id] = 'Unknown'  # Initial placeholder

# Identify ways based on relation roles
centerline_ways = set()
left_ways = set()
right_ways = set()

for relation in root.findall('relation'):
    for member in relation.findall('member'):
        role = member.get('role')
        way_id = member.get('ref')
        if role == 'centerline':
            centerline_ways.add(way_id)
        elif role == 'left':
            left_ways.add(way_id)
        elif role == 'right':
            right_ways.add(way_id)

# Collect nodes associated with each way
centerline_nodes = set()
left_nodes = set()
right_nodes = set()

for way in root.findall('way'):
    way_id = way.get('id')
    node_ids = [nd.get('ref') for nd in way.findall('nd')]
    if way_id in centerline_ways:
        centerline_nodes.update(node_ids)
    elif way_id in left_ways:
        left_nodes.update(node_ids)
    elif way_id in right_ways:
        right_nodes.update(node_ids)

# Convert lists of nodes to LineStrings
def nodes_to_linestring(node_ids):
    coords = [nodes_dict[node_id] for node_id in node_ids]
    return LineString(coords)

# Get the centerline as a LineString
centerline_node_ids = [node_id for node_id in centerline_nodes]
centerline = nodes_to_linestring(centerline_node_ids)

# Helper function to determine side relative to line
def classify_side(centerline, point):
    point = Point(point)
    if centerline.project(point, normalized=False) > 0:
        return 'Right'
    return 'Left'

# Update node_wall_info based on position relative to centerline
for node_id, (lat, lon) in nodes_dict.items():
    if node_id in left_nodes:
        node_wall_info[node_id] = 'Left'
    elif node_id in right_nodes:
        node_wall_info[node_id] = 'Right'
    elif node_id in centerline_nodes:
        node_wall_info[node_id] = 'Centerline'
    else:
        # Compute which side of the centerline the node is on
        side = classify_side(centerline, (lat, lon))
        node_wall_info[node_id] = side

# Exclude centerline nodes for plotting
filtered_latitudes = []
filtered_longitudes = []
filtered_wall_info = []

for node_id, (lat, lon) in nodes_dict.items():
    #if node_id not in centerline_nodes:
    filtered_latitudes.append(lat)
    filtered_longitudes.append(lon)
    filtered_wall_info.append(node_wall_info[node_id])

# Convert latitude and longitude to meters
def lat_lon_to_meters(lat, lon):
    x = lon * (np.pi / 180) * EARTH_RADIUS
    y = lat * (np.pi / 180) * EARTH_RADIUS
    return x, y

x_coords = []
y_coords = []

for lat, lon in zip(filtered_latitudes, filtered_longitudes):
    x, y = lat_lon_to_meters(lat, lon)
    x_coords.append(x)
    y_coords.append(y)

# Translate coordinates to make the minimum values zero
min_x = min(x_coords)
min_y = min(y_coords)
translated_x_coords = [x - min_x for x in x_coords]
translated_y_coords = [y - min_y for y in y_coords]

# Plot results
plt.figure(figsize=(10, 10))
plt.scatter(translated_x_coords, translated_y_coords, c='blue', marker='o', edgecolor='k')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('OSM Nodes (Excluding Centerline) in Meter Coordinate System')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.savefig('osm_nodes_map.png')
plt.show()

# Save results to CSV
df = pd.DataFrame({
    'X (meters)': translated_x_coords,
    'Y (meters)': translated_y_coords,
    'Wall Info': filtered_wall_info
})
df.to_csv('osm_nodes_coordinates_with_wall_info.csv', index=False)
