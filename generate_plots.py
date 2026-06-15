import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Set paths
workspace_dir = r"d:\stockprediction"
data_dir = os.path.join(workspace_dir, "data")
output_dir_workspace = os.path.join(workspace_dir, "plots")
output_dir_artifact = r"C:\Users\balaj\.gemini\antigravity\brain\5cbcf4ed-26b4-4635-86f6-65489f96a0f7"

# Create directories if they do not exist
os.makedirs(output_dir_workspace, exist_ok=True)
os.makedirs(output_dir_artifact, exist_ok=True)

# ----------------------------------------------------
# Global Styling
# ----------------------------------------------------
dark_theme = {
    "axes.facecolor": "#13151a",
    "figure.facecolor": "#0d0e12",
    "grid.color": "#21252d",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "text.color": "#e3e6eb",
    "axes.labelcolor": "#a9b2c3",
    "xtick.color": "#8f9aa9",
    "ytick.color": "#8f9aa9",
    "axes.edgecolor": "#21252d",
    "axes.titlecolor": "#ffffff",
    "font.sans-serif": ["Inter", "Roboto", "DejaVu Sans", "Arial"]
}

plt.rcParams.update(dark_theme)
sns.set_context("talk", font_scale=0.8)

# ----------------------------------------------------
# Data Loading & Preparation
# ----------------------------------------------------
# Find some CSV files in the data directory
all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
if not all_files:
    print("Error: No CSV files found in data folder.")
    sys.exit(1)

# Pick a primary stock (prefer 360ONE.NS if present, else first available)
primary_ticker = "360ONE.NS"
primary_file = f"{primary_ticker}_historical.csv"
if primary_file not in all_files:
    primary_file = all_files[0]
    primary_ticker = primary_file.replace("_historical.csv", "")

print(f"Primary ticker selected: {primary_ticker}")
df = pd.read_csv(os.path.join(data_dir, primary_file))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Calculate indicators
df["20_SMA"] = df["Close"].rolling(window=20).mean()
df["50_SMA"] = df["Close"].rolling(window=50).mean()
df["Daily_Return"] = df["Close"].pct_change() * 100

# Select last 2 years for clearer visualization of price history
df_recent = df.tail(500).copy()

def save_plot(filename):
    # Save to workspace
    p1 = os.path.join(output_dir_workspace, filename)
    plt.savefig(p1, dpi=300, bbox_inches="tight", facecolor="#0d0e12")
    # Save to brain artifact folder
    p2 = os.path.join(output_dir_artifact, filename)
    plt.savefig(p2, dpi=300, bbox_inches="tight", facecolor="#0d0e12")
    print(f"Saved plot: {filename}")
    plt.close()

# ----------------------------------------------------
# Plot 1: Closing Price & SMAs
# ----------------------------------------------------
plt.figure(figsize=(12, 6.5))
sns.lineplot(data=df_recent, x="Date", y="Close", color="#00f0ff", linewidth=2.0, label="Close Price")
sns.lineplot(data=df_recent, x="Date", y="20_SMA", color="#ffb703", linewidth=1.5, linestyle="--", label="20-day SMA")
sns.lineplot(data=df_recent, x="Date", y="50_SMA", color="#ff2a7f", linewidth=1.5, linestyle=":", label="50-day SMA")

plt.title(f"{primary_ticker} - Historical Close Price & Moving Averages", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Date", labelpad=10)
plt.ylabel("Price (INR / USD)", labelpad=10)

# Format X axis dates beautifully
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

# Customize Legend
plt.legend(frameon=True, facecolor="#1a1c23", edgecolor="#2d313f", loc="upper left")
plt.tight_layout()
save_plot("close_price_sma.png")

# ----------------------------------------------------
# Plot 2: Daily Returns Distribution
# ----------------------------------------------------
plt.figure(figsize=(10, 6))
# Calculate standard deviation and stats
mean_ret = df["Daily_Return"].mean()
std_ret = df["Daily_Return"].std()

sns.histplot(data=df.dropna(), x="Daily_Return", kde=True, color="#39ff14", bins=80, alpha=0.3, edgecolor="#39ff14")
plt.axvline(mean_ret, color="#ff2a7f", linestyle="--", linewidth=1.5, label=f"Mean Return: {mean_ret:.2f}%")
plt.axvline(mean_ret - std_ret, color="#00f0ff", linestyle=":", linewidth=1.2, label=f"-1 Std Dev: {mean_ret-std_ret:.2f}%")
plt.axvline(mean_ret + std_ret, color="#00f0ff", linestyle=":", linewidth=1.2, label=f"+1 Std Dev: {mean_ret+std_ret:.2f}%")

plt.title(f"{primary_ticker} - Daily Return Distribution & Volatility", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Daily Return (%)", labelpad=10)
plt.ylabel("Frequency", labelpad=10)
plt.xlim(-8, 8)  # Limit range to focus on main distribution

plt.legend(frameon=True, facecolor="#1a1c23", edgecolor="#2d313f")
plt.tight_layout()
save_plot("daily_returns_distribution.png")

# ----------------------------------------------------
# Plot 3: Correlation Heatmap between Multiple Stocks
# ----------------------------------------------------
# Select up to 6 stocks to correlate
tickers_to_correlate = [
    "360ONE.NS", "20MICRONS.NS", "3IINFOLTD.NS", "21STCENMGM.NS", "3BBLACKBIO.NS"
]
# Filter to ones actually present in dataset
available_tickers = [t for t in tickers_to_correlate if f"{t}_historical.csv" in all_files]

if len(available_tickers) >= 2:
    returns_df = pd.DataFrame()
    for ticker in available_tickers:
        temp_df = pd.read_csv(os.path.join(data_dir, f"{ticker}_historical.csv"))
        temp_df["Date"] = pd.to_datetime(temp_df["Date"])
        temp_df = temp_df.sort_values("Date").reset_index(drop=True)
        temp_df["Daily_Return"] = temp_df["Close"].pct_change() * 100
        returns_df[ticker] = temp_df.set_index("Date")["Daily_Return"]
    
    corr_matrix = returns_df.corr()
    
    plt.figure(figsize=(9, 7.5))
    # Diverging color palette with cyan-purple shades
    cmap = sns.diverging_palette(220, 320, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        square=True, 
        linewidths=1.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    plt.title("Stock Returns Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    save_plot("correlation_heatmap.png")
else:
    print("Not enough files for a correlation matrix. Skipping Plot 3.")

# ----------------------------------------------------
# Plot 4: Volatility vs Volume Scatter Plot
# ----------------------------------------------------
plt.figure(figsize=(10, 6.5))

# Filter extreme volume values for better plot scaling
volume_cap = df["Volume"].quantile(0.99)
df_filtered = df[(df["Volume"] < volume_cap) & (df["Volume"] > 0)].dropna()

# Convert volume to millions
df_filtered["Volume_M"] = df_filtered["Volume"] / 1_000_000

sns.scatterplot(
    data=df_filtered, 
    x="Volume_M", 
    y="Daily_Return", 
    hue="Daily_Return", 
    palette="viridis", 
    size="Volume_M",
    sizes=(20, 200),
    alpha=0.6,
    edgecolor="none"
)

plt.title(f"{primary_ticker} - Volatility vs. Trading Volume", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Volume (in Millions)", labelpad=10)
plt.ylabel("Daily Return (%)", labelpad=10)
plt.axhline(0, color="#ffffff", linestyle="-", linewidth=0.5, alpha=0.5)

# Place legend outside
plt.legend([], [], frameon=False) # Clear legend to avoid cluttering
plt.tight_layout()
save_plot("volume_vs_returns.png")

print("All plots generated successfully!")
