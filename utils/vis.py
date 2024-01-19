import pandas as pd
import numpy as np
import open3d as o3d

def load_filtered_labels(filepath):
    """
    Load filtered labels from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The filtered labels data.
    """
    return pd.read_csv(filepath)

def filter_point_cloud(data, trainid_to_keep):
    """
    Filter point cloud data based on train IDs to keep.

    Args:
        data (pandas.DataFrame): The point cloud data.
        trainid_to_keep (list): The train IDs to keep.

    Returns:
        pandas.DataFrame: The filtered point cloud data.
    """
    mask = np.isin(data['semantic'], trainid_to_keep)
    return data[mask]

def convert_color_string_to_tuple(color_string):
    """
    Convert a color string to a tuple.

    Args:
        color_string (str): The color string.

    Returns:
        tuple: The color tuple.
    """
    try:
        if pd.isnull(color_string):
            return None
        stripped_string = color_string.strip('()')
        tuple_values = tuple(map(int, stripped_string.split(',')))
        return tuple_values
    except Exception as e:
        print(f"Error converting color: {e}")
        return None

def normalize_color(color_tuple):
    """
    Normalize a color tuple.

    Args:
        color_tuple (tuple): The color tuple.

    Returns:
        tuple: The normalized color tuple.
    """
    if color_tuple is None:
        return None
    return tuple(c / 255.0 for c in color_tuple)

def create_point_cloud_dataframe(filtered_point_cloud, filtered_labels):
    """
    Create a DataFrame from filtered point cloud data and labels.

    Args:
        filtered_point_cloud (pandas.DataFrame): The filtered point cloud data.
        filtered_labels (pandas.DataFrame): The filtered labels data.

    Returns:
        pandas.DataFrame: The combined point cloud DataFrame.
    """
    data_dataframe = pd.DataFrame({
        'x': filtered_point_cloud['x'],
        'y': filtered_point_cloud['y'],
        'z': filtered_point_cloud['z'],
        'semantic': filtered_point_cloud['semantic']
    })
    id_to_color = filtered_labels.set_index('id')['color']
    data_dataframe['color'] = data_dataframe['semantic'].map(id_to_color)
    data_dataframe['color'] = data_dataframe['color'].apply(convert_color_string_to_tuple)
    return data_dataframe

def prepare_point_cloud_data(data_dataframe):
    """
    Prepare point cloud data for visualization.

    Args:
        data_dataframe (pandas.DataFrame): The combined point cloud DataFrame.

    Returns:
        numpy.ndarray: The points array.
        numpy.ndarray: The colors array.
    """
    points = np.vstack((data_dataframe['x'], data_dataframe['y'], data_dataframe['z'])).T 
    colors = np.array(data_dataframe['color'].apply(normalize_color).tolist())
    return points, colors

def visualize_point_cloud(points, colors):
    """
    Visualize a point cloud using Open3D.

    Args:
        points (numpy.ndarray): The points array.
        colors (numpy.ndarray): The colors array.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])