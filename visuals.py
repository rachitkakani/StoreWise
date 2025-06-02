import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

COLORS = {
    'primary': '#4facfe',
    'secondary': '#00f2fe',
    'accent': '#ff6b6b',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'error': '#ff6b6b',
    'info': '#339af0'
}

PLOTLY_COLORS = ['#4facfe', '#00f2fe', '#ff6b6b', '#51cf66', '#ffd43b', '#339af0', '#845ec2', '#f39c12']

# ===== MATPLOTLIB FUNCTIONS =====

def plot_monthly_trends(df):
    """Create monthly trends plot using matplotlib"""
    plt.figure(figsize=(12, 6))
    
    df_copy = df.copy()
    df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date'])
    df_copy['YearMonth'] = df_copy['Order Date'].dt.to_period('M')
    
    monthly_data = df_copy.groupby('YearMonth').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(monthly_data['YearMonth'], monthly_data['Sales'], 
             color=COLORS['primary'], marker='o', linewidth=3, markersize=6, label='Sales')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Sales ($)', color=COLORS['primary'], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(monthly_data['YearMonth'], monthly_data['Profit'], 
             color=COLORS['accent'], marker='s', linewidth=3, markersize=6, label='Profit')
    ax2.set_ylabel('Profit ($)', color=COLORS['accent'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    
    plt.title('Monthly Sales & Profit Trends', fontsize=16, fontweight='bold', pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_category_bar(df):
    """Create category performance bar chart"""
    plt.figure(figsize=(10, 6))
    
    df_sorted = df.sort_values('Sales', ascending=True)
    
    bars = plt.barh(df_sorted.index, df_sorted['Sales'], 
                    color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=2)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'${width:,.0f}', ha='left', va='center', fontweight='bold')
    
    plt.title('Sales by Category', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sales ($)', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return plt

def plot_region_bar(df):
    """Create region performance bar chart"""
    plt.figure(figsize=(10, 6))
    
    df_sorted = df.sort_values('Profit', ascending=False)
    
    colors = [COLORS['success'] if x >= 0 else COLORS['error'] for x in df_sorted['Profit']]
    
    bars = plt.bar(df_sorted.index, df_sorted['Profit'], 
                   color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + height*0.01 if height >= 0 else height - abs(height)*0.05,
                f'${height:,.0f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.title('Profit by Region', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return plt

def plot_segment_pie(df):
    """Create segment distribution pie chart"""
    seg_data = df.groupby('Segment')['Sales'].sum()
    
    plt.figure(figsize=(8, 8))
    
    wedges, texts, autotexts = plt.pie(seg_data.values, labels=seg_data.index, 
                                      autopct='%1.1f%%', startangle=90,
                                      colors=PLOTLY_COLORS[:len(seg_data)],
                                      explode=[0.05] * len(seg_data),
                                      shadow=True, textprops={'fontsize': 12})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Sales Distribution by Segment', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    return plt

def plot_discount_profit(df):
    """Create discount vs profit scatter plot"""
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(df['Discount'], df['Profit'], 
                         c=df['Sales'], cmap='viridis', 
                         alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sales ($)', fontsize=12)
    
    z = np.polyfit(df['Discount'], df['Profit'], 1)
    p = np.poly1d(z)
    plt.plot(df['Discount'], p(df['Discount']), "r--", alpha=0.8, linewidth=2)
    
    plt.title('Discount vs Profit Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Discount Rate', fontsize=12)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_ship_mode(df):
    """Create shipping mode performance chart"""
    plt.figure(figsize=(10, 6))
    
    df_sorted = df.sort_values('Sales', ascending=True)
    
    bars = plt.barh(df_sorted.index, df_sorted['Sales'], 
                    color=COLORS['info'], alpha=0.8, edgecolor='white', linewidth=2)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'${width:,.0f}', ha='left', va='center', fontweight='bold')
    
    plt.title('Sales by Shipping Mode', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sales ($)', fontsize=12)
    plt.ylabel('Shipping Mode', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return plt

# ===== PLOTLY FUNCTIONS =====

def plot_monthly_trends_plotly(df):
    """Create interactive monthly trends plot using Plotly"""
    df_copy = df.copy()
    df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date'])
    df_copy['YearMonth'] = df_copy['Order Date'].dt.to_period('M')
    
    monthly_data = df_copy.groupby('YearMonth').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['Sales'],
                  name='Sales', line=dict(color=COLORS['primary'], width=3),
                  marker=dict(size=8)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['Profit'],
                  name='Profit', line=dict(color=COLORS['accent'], width=3),
                  marker=dict(size=8)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
    
    fig.update_layout(
        title='Monthly Sales & Profit Trends',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def plot_category_bar_plotly(df):
    """Create interactive category bar chart"""
    df_sorted = df.sort_values('Sales', ascending=True)
    
    fig = px.bar(
        df_sorted.reset_index(), 
        x='Sales', 
        y='Category',
        orientation='h',
        title='Sales by Category',
        color='Sales',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_region_bar_plotly(df):
    """Create interactive region bar chart"""
    df_sorted = df.sort_values('Profit', ascending=False)
    
    colors = [COLORS['success'] if x >= 0 else COLORS['error'] for x in df_sorted['Profit']]
    
    fig = go.Figure(data=[
        go.Bar(x=df_sorted.index, y=df_sorted['Profit'], 
               marker_color=colors, name='Profit')
    ])
    
    fig.update_layout(
        title='Profit by Region',
        xaxis_title='Region',
        yaxis_title='Profit ($)',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_segment_pie_plotly(df):
    """Create interactive segment pie chart"""
    seg_data = df.groupby('Segment')['Sales'].sum().reset_index()
    
    fig = px.pie(
        seg_data, 
        values='Sales', 
        names='Segment',
        title='Sales Distribution by Segment',
        color_discrete_sequence=PLOTLY_COLORS
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    return fig

def plot_discount_profit_plotly(df):
    """Create interactive discount vs profit scatter plot"""
    fig = px.scatter(
        df, 
        x='Discount', 
        y='Profit',
        color='Sales',
        size='Sales',
        hover_data=['Category', 'Region'],
        title='Discount vs Profit Analysis',
        color_continuous_scale='Viridis'
    )
    
    z = np.polyfit(df['Discount'], df['Profit'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['Discount'].min(), df['Discount'].max(), 100)
    
    fig.add_trace(
        go.Scatter(x=x_trend, y=p(x_trend), 
                  mode='lines', name='Trend Line',
                  line=dict(color='red', dash='dash'))
    )
    
    fig.update_layout(template='plotly_white', height=500)
    
    return fig

def plot_ship_mode_plotly(df):
    """Create interactive shipping mode chart"""
    df_sorted = df.sort_values('Sales', ascending=True)
    
    fig = px.bar(
        df_sorted.reset_index(), 
        x='Sales', 
        y='Ship Mode',
        orientation='h',
        title='Sales by Shipping Mode',
        color='Sales',
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

# ===== ADVANCED VISUALIZATIONS =====

def plot_heatmap_correlation(df):
    """Create correlation heatmap for numerical columns"""
    plt.figure(figsize=(10, 8))
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Heatmap of Numerical Variables', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return plt

def plot_profit_margin_distribution(df):
    """Create profit margin distribution plot"""
    df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
    
    plt.figure(figsize=(12, 6))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(df['Profit_Margin'], bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax1.set_title('Profit Margin Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Profit Margin (%)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    if 'Category' in df.columns:
        sns.boxplot(data=df, x='Category', y='Profit_Margin', ax=ax2, palette='Set2')
        ax2.set_title('Profit Margin by Category', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Profit Margin (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_sales_forecast(df, periods=6):
    """Create a simple sales forecast visualization"""
    from datetime import datetime, timedelta
    
    plt.figure(figsize=(12, 6))
    
    df_copy = df.copy()
    df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date'])
    monthly_sales = df_copy.groupby(df_copy['Order Date'].dt.to_period('M'))['Sales'].sum()
    
    window = 3
    forecast_values = []
    for i in range(periods):
        if len(monthly_sales) >= window:
            forecast = monthly_sales.tail(window).mean()
            forecast_values.append(forecast)
            
            next_period = monthly_sales.index[-1] + 1
            monthly_sales[next_period] = forecast
    
    historical_dates = [pd.Timestamp(str(p)) for p in monthly_sales.index[:-periods]]
    forecast_dates = [pd.Timestamp(str(p)) for p in monthly_sales.index[-periods:]]
    
    plt.plot(historical_dates, monthly_sales.values[:-periods], 
             color=COLORS['primary'], linewidth=3, marker='o', label='Historical Sales')
    plt.plot(forecast_dates, forecast_values, 
             color=COLORS['accent'], linewidth=3, marker='s', linestyle='--', label='Forecast')
    
    plt.title('Sales Forecast (Simple Moving Average)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Month')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# ===== UTILITY FUNCTIONS =====

def save_all_plots(df, output_dir='plots'):
    """Save all plots to specified directory"""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plots = [
        (plot_monthly_trends, 'monthly_trends.png'),
        (plot_category_bar, 'category_performance.png'),
        (plot_region_bar, 'region_performance.png'),
        (plot_segment_pie, 'segment_distribution.png'),
        (plot_discount_profit, 'discount_analysis.png'),
        (plot_ship_mode, 'shipping_performance.png'),
        (plot_heatmap_correlation, 'correlation_heatmap.png'),
        (plot_profit_margin_distribution, 'profit_margin_dist.png')
    ]
    
    for plot_func, filename in plots:
        try:
            plot_func(df)
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {str(e)}")