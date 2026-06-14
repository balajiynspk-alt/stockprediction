import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.logger import setup_logger

logger = setup_logger("visualizer")

class StockVisualizer:
    """
    Class responsible for generating publication-quality visualisations
    of the stock prediction model performance and data features.
    """
    def __init__(self, output_dir: str = "images"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory for images: {output_dir}")
            
        # Set a premium seaborn style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['font.family'] = 'sans-serif'

    def plot_actual_vs_predicted(self, y_true: pd.Series, y_pred: np.ndarray, ticker: str, save: bool = True) -> plt.Figure:
        """
        Generates an Actual vs Predicted Price line chart.
        """
        logger.info(f"Generating Actual vs Predicted plot for {ticker}...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Align index
        dates = y_true.index
        
        ax.plot(dates, y_true.values, label='Actual Next-Day Close', color='#1f77b4', linewidth=1.5)
        ax.plot(dates, y_pred, label='Predicted Next-Day Close', color='#ff7f0e', linewidth=1.5, linestyle='--')
        
        ax.set_title(f"{ticker} Stock Price: Actual vs Predicted (Next-Day Close)", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (USD / Local Currency)", fontsize=12)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f"{ticker}_actual_vs_predicted.png")
            fig.savefig(filepath, dpi=300)
            logger.info(f"Saved plot to {filepath}")
            
        plt.close(fig)
        return fig

    def plot_feature_importance(self, df_importance: pd.DataFrame, ticker: str, save: bool = True) -> plt.Figure:
        """
        Generates a horizontal bar chart of feature importances.
        """
        logger.info(f"Generating Feature Importance plot for {ticker}...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Take top 15 features for readability
        top_n = min(15, len(df_importance))
        df_top = df_importance.head(top_n)
        
        # Create a beautiful gradient palette
        colors = sns.color_palette("viridis_r", n_colors=top_n)
        
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=df_top, 
            ax=ax, 
            palette=colors,
            hue='Feature',
            legend=False
        )
        
        ax.set_title(f"Top {top_n} Feature Importances - {ticker}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Technical Indicators / Features", fontsize=12)
        
        fig.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f"{ticker}_feature_importance.png")
            fig.savefig(filepath, dpi=300)
            logger.info(f"Saved plot to {filepath}")
            
        plt.close(fig)
        return fig

    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray, ticker: str, save: bool = True) -> plt.Figure:
        """
        Generates a histogram with KDE showing the distribution of prediction residuals.
        """
        logger.info(f"Generating Residuals Distribution plot for {ticker}...")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        residuals = y_true.values - y_pred
        
        sns.histplot(residuals, kde=True, color='#2ca02c', ax=ax, bins=50, stat='density', alpha=0.6)
        
        # Add normal curve fit for comparison
        mean_res, std_res = np.mean(residuals), np.std(residuals)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = (1 / (np.sqrt(2 * np.pi) * std_res)) * np.exp(-((x - mean_res) ** 2) / (2 * std_res ** 2))
        ax.plot(x, p, 'r--', linewidth=1.5, label=f'Normal Fit (μ={mean_res:.2f}, σ={std_res:.2f})')
        
        ax.set_title(f"Residuals (Error) Distribution - {ticker}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Prediction Error (Actual - Predicted)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()
        
        fig.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f"{ticker}_residuals_distribution.png")
            fig.savefig(filepath, dpi=300)
            logger.info(f"Saved plot to {filepath}")
            
        plt.close(fig)
        return fig

    def plot_correlation_heatmap(self, df_features: pd.DataFrame, ticker: str, save: bool = True) -> plt.Figure:
        """
        Generates a correlation heatmap of the feature space.
        """
        logger.info(f"Generating Correlation Heatmap plot for {ticker}...")
        fig, ax = plt.subplots(figsize=(14, 12))
        
        corr = df_features.corr()
        
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Draw the heatmap with a clean blue-to-red palette
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=1.0, 
            vmin=-1.0, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .75},
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            ax=ax
        )
        
        ax.set_title(f"Feature Correlation Heatmap - {ticker}", fontsize=16, fontweight='bold', pad=20)
        fig.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, f"{ticker}_correlation_heatmap.png")
            fig.savefig(filepath, dpi=300)
            logger.info(f"Saved plot to {filepath}")
            
        plt.close(fig)
        return fig
