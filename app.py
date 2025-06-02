import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="StoreWise Analytics", 
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data():
    """Generate sample retail data for demonstration"""
    np.random.seed(42)
    
    categories = ['Technology', 'Furniture', 'Office Supplies']
    subcategories = {
        'Technology': ['Phones', 'Computers', 'Accessories'],
        'Furniture': ['Chairs', 'Tables', 'Storage'],
        'Office Supplies': ['Paper', 'Binders', 'Art']
    }
    segments = ['Consumer', 'Corporate', 'Home Office']
    regions = ['West', 'East', 'Central', 'South']
    ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
    
    n_records = 1000
    data = []
    
    for i in range(n_records):
        category = np.random.choice(categories)
        subcategory = np.random.choice(subcategories[category])
        segment = np.random.choice(segments)
        region = np.random.choice(regions)
        ship_mode = np.random.choice(ship_modes)
        
        order_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
        sales = np.random.uniform(10, 1000)
        discount = np.random.uniform(0, 0.5)
        profit = sales * np.random.uniform(-0.2, 0.4)
        
        data.append({
            'Order Date': order_date,
            'Category': category,
            'Sub-Category': subcategory,
            'Segment': segment,
            'Region': region,
            'Ship Mode': ship_mode,
            'Sales': sales,
            'Profit': profit,
            'Discount': discount,
            'Product Name': f"{subcategory} Product {i+1}",
            'Customer Name': f"Customer {np.random.randint(1, 200)}"
        })
    
    return pd.DataFrame(data)

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return generate_sample_data()

def get_kpis(df):
    """Calculate key performance indicators"""
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = len(df)
    avg_order_value = total_sales / total_orders
    profit_margin = (total_profit / total_sales) * 100
    
    return {
        'Total Sales': f"{total_sales:,.0f}",
        'Total Profit': f"{total_profit:,.0f}",
        'Total Orders': f"{total_orders:,}",
        'Avg Order Value': f"{avg_order_value:.2f}",
        'Profit Margin': f"{profit_margin:.1f}"
    }

def create_monthly_trends_chart(df):
    """Create monthly trends visualization"""
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    monthly_data = df.groupby(df['Order Date'].dt.to_period('M')).agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    monthly_data['Order Date'] = monthly_data['Order Date'].astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_data['Order Date'],
        y=monthly_data['Sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#4facfe', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=monthly_data['Order Date'],
        y=monthly_data['Profit'],
        mode='lines+markers',
        name='Profit',
        line=dict(color='#00f2fe', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Monthly Sales and Profit Trends',
        xaxis_title='Month',
        yaxis_title='Sales ($)',
        yaxis2=dict(title='Profit ($)', overlaying='y', side='right'),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig

def create_category_chart(df):
    """Create category performance chart"""
    category_data = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = px.bar(
        category_data, 
        x='Category', 
        y='Sales',
        color='Profit',
        color_continuous_scale='Viridis',
        title='Category Performance'
    )
    fig.update_layout(template='plotly_white', height=400)
    return fig

def create_region_chart(df):
    """Create region performance chart"""
    region_data = df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = px.pie(
        region_data, 
        values='Sales', 
        names='Region',
        title='Sales Distribution by Region',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=400)
    return fig

def create_segment_chart(df):
    """Create a donut chart for customer segments"""
    segment_counts = df['Segment'].value_counts()
    
    fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segments Distribution",
        hole=0.4,  # This creates the donut effect
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_discount_impact_chart(df):
    """Create discount impact visualization"""
    df['Discount_Range'] = pd.cut(df['Discount'], bins=5, labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%'])
    discount_data = df.groupby('Discount_Range')['Profit'].mean().reset_index()
    
    fig = px.bar(
        discount_data,
        x='Discount_Range',
        y='Profit',
        title='Average Profit by Discount Range',
        color='Profit',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_layout(template='plotly_white', height=400)
    return fig

def main():
    st.markdown("""
    <div class="section-header">
        <h1>🛒 StoreWise – Advanced Retail Analytics Dashboard</h1>
        <p>Comprehensive insights into your retail performance</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📁 Data Source")
        uploaded_file = st.file_uploader(
            "Upload your CSV file", 
            type=["csv"],
            help="Upload a CSV file with retail data (Superstore format)"
        )
        
        if uploaded_file is None:
            st.info("📊 Using sample data for demonstration")
        
        st.markdown("### 🎛️ Filters")
        
    df = load_data(uploaded_file)
    
    with st.sidebar:
        if 'Order Date' in df.columns:
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            date_range = st.date_input(
                "Select Date Range",
                value=(df['Order Date'].min(), df['Order Date'].max()),
                min_value=df['Order Date'].min(),
                max_value=df['Order Date'].max()
            )
            
            if len(date_range) == 2:
                df = df[(df['Order Date'] >= pd.Timestamp(date_range[0])) & 
                       (df['Order Date'] <= pd.Timestamp(date_range[1]))]
        
        if 'Category' in df.columns:
            categories = st.multiselect(
                "Select Categories",
                options=df['Category'].unique(),
                default=df['Category'].unique()
            )
            df = df[df['Category'].isin(categories)]
        
        if 'Region' in df.columns:
            regions = st.multiselect(
                "Select Regions",
                options=df['Region'].unique(),
                default=df['Region'].unique()
            )
            df = df[df['Region'].isin(regions)]

    kpis = get_kpis(df)
    
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${kpis['Total Sales']}</p>
            <p class="metric-label">Total Sales</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${kpis['Total Profit']}</p>
            <p class="metric-label">Total Profit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{kpis['Total Orders']}</p>
            <p class="metric-label">Total Orders</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${kpis['Avg Order Value']}</p>
            <p class="metric-label">Avg Order Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{kpis['Profit Margin']}%</p>
            <p class="metric-label">Profit Margin</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Performance Analytics")
    
    st.plotly_chart(create_monthly_trends_chart(df), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_category_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(create_region_chart(df), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_segment_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(create_discount_impact_chart(df), use_container_width=True)
    
    st.markdown("### 📋 Detailed Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Products", "👥 Top Customers", "📉 Loss Makers", "📊 Raw Data"])
    
    with tab1:
        if 'Product Name' in df.columns:
            top_products = df.groupby('Product Name').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Order Date': 'count'
            }).rename(columns={'Order Date': 'Orders'}).sort_values('Sales', ascending=False).head(10)
            st.dataframe(top_products, use_container_width=True)
        else:
            st.info("Product data not available")
    
    with tab2:
        if 'Customer Name' in df.columns:
            top_customers = df.groupby('Customer Name').agg({
                'Sales': 'sum',
                'Profit': 'sum',
                'Order Date': 'count'
            }).rename(columns={'Order Date': 'Orders'}).sort_values('Sales', ascending=False).head(10)
            st.dataframe(top_customers, use_container_width=True)
        else:
            st.info("Customer data not available")
    
    with tab3:
        loss_products = df[df['Profit'] < 0].groupby('Product Name' if 'Product Name' in df.columns else 'Category').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order Date': 'count'
        }).rename(columns={'Order Date': 'Orders'}).sort_values('Profit').head(10)
        if not loss_products.empty:
            st.dataframe(loss_products, use_container_width=True)
        else:
            st.success("🎉 No loss-making products found!")
    
    with tab4:
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="storewise_filtered_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()