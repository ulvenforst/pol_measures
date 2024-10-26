import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def plot_correlation_matrix(
    matrix: pd.DataFrame,
    title: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot correlation matrix as a heatmap.
    
    Parameters:
        matrix (pd.DataFrame): Correlation matrix
        title (str, optional): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Crear heatmap
    sns.heatmap(
        matrix,
        annot=True,
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        center=0,
        fmt='.3f'
    )
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()
