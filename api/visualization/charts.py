import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2
from typing import List, Tuple


def get_dominant_colors(image_path: str, k: int = 5) -> List[Tuple[int, int, int]]:
    """
    Get the dominant colors of an image using KMeans clustering.

    Parameters:
    - image_path (str): Path to the image file.
    - k (int): Number of clusters (colors). Default is 5.

    Returns:
    - list: Dominant colors in the image.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    kmeans = MiniBatchKMeans(n_clusters=k, n_init='auto')
    kmeans.fit(image)

    sorted_labels = sorted([(sum(kmeans.cluster_centers_[label]), label) for label in set(kmeans.labels_)], reverse=True)
    dominant_colors = [kmeans.cluster_centers_[label[1]].astype(int) for label in sorted_labels]

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
    
    dominant_colors = get_dominant_colors(img_path, len(sorted_probs))
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

    # Layout aesthetics
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