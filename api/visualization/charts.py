import plotly.graph_objects as go
import numpy as np
import cv2
from typing import List, Tuple


def brightness(color):
    """Compute the brightness of a color."""
    return 0.299*color[0] + 0.587*color[1] + 0.114*color[2]

def is_bright(color, threshold=30):
    """Check if a color is bright enough to be considered."""
    return np.mean(color) > threshold

def interpolate_palette(palette, num_colors):
    interpolated_palette = []
    for i in np.linspace(0, len(palette)-1, num_colors):
        base_color = palette[int(i)]
        if i.is_integer():
            interpolated_palette.append(base_color)
        else:
            next_color = palette[int(i)+1]
            alpha = i % 1  # Fractional part of i
            new_color = (1-alpha) * base_color + alpha * next_color
            interpolated_palette.append(new_color.astype(int))
    return interpolated_palette

def initialize_kmeans_plusplus(data, k, brightness_weight=1):
    """Initialize centroids using KMeans++ with a brightness bias."""
    centroids = [data[np.random.choice(len(data))]]
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(d-c) for c in centroids]) for d in data])
        brightness_weights = np.array([brightness(d) for d in data])
        
        # Combine the distance and brightness to compute the probabilities
        probabilities = ((1 - brightness_weight) * distances + brightness_weight * brightness_weights)
        probabilities /= probabilities.sum()
        
        next_centroid = data[np.random.choice(len(data), p=probabilities)]
        centroids.append(next_centroid)
    return np.array(centroids)


def kmeans_clustering(data, k, max_iters=100):
    # Initialize centroids using KMeans++
    centroids = initialize_kmeans_plusplus(data, k)
    for _ in range(max_iters):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels


def get_dominant_colors(image_path: str, k: int = 5, brightness_threshold: int = 75) -> List[Tuple[int, int, int]]:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    # Filter out dark colors
    bright_colors = np.array([color for color in image if is_bright(color, brightness_threshold)])
    
    # If there are not enough bright colors, return colors from the predefined palette
    if len(bright_colors) < k:
        PREDEFINED_PALETTE = np.array([
            [102, 189, 99],  # dark green
            [166, 217, 106], # green
            [217, 239, 139], # light green
            [255, 255, 191], # yellow
            [254, 224, 139], # light orange
            [253, 174, 97],  # orange
            [244, 109, 67],  # orange-red
            [215, 48, 39],   # red
            [165, 0, 38],    # dark red
        ])

        print(f"Warning: Not enough bright colors found ({len(bright_colors)}). Using predefined palette.")
        return [tuple(color) for color in interpolate_palette(PREDEFINED_PALETTE, k)]
    
    centroids, labels = kmeans_clustering(bright_colors, k)
    
    # Sort centroids by their frequency in the image
    sorted_labels = sorted([(np.sum(labels == i), i) for i in range(k)], reverse=True)
    dominant_colors = [tuple(map(int, centroids[label[1]])) for label in sorted_labels]
    
    return dominant_colors


def create_probability_chart(probs: List[float], labels: List[str], filename: str, img_path: str) -> None:
    """
    Create a bar chart visualizing prediction probabilities.

    Parameters:
    - probs (list): List of prediction probabilities.
    - labels (list): List of labels corresponding to each prediction.
    - filename (str): Path to save the resulting chart.
    - img_path (str): Path to the image used for prediction.

    Returns:
    - None: The function writes the chart to an HTML file.
    """
    probs = np.array(probs)
    labels = np.array(labels)
    sorted_indices = probs.argsort()[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    dominant_colors = get_dominant_colors(img_path, len(sorted_labels))
    colors = [f'rgb({color[0]}, {color[1]}, {color[2]})' for color in dominant_colors]


    fig = go.Figure(data=[
        go.Bar(
            y=sorted_labels,
            x=sorted_probs,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color=colors[0], width=1),
            ),
            hovertemplate=(
                "<b>Confidence Score</b>: %{x:.5f}<br>"
                "<b>Classification</b>: %{y}<extra></extra>"
            ),
        )
    ])

    fig.update_layout(
        title='Classification Confidence',
        xaxis=dict(
            title='Confidence Score',
            showgrid=True,
            zeroline=True,
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            zeroline=False
        ),
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="#f0f0f0"),
        showlegend=False
    )
    
    fig.write_html(filename)