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

# ---------------------
# Utility helpers
# ---------------------

def safe_to_datetime(series):
    try:
        return pd.to_datetime(series, errors='coerce')
    except Exception:
        return pd.Series(pd.to_datetime(series.astype(str), errors='coerce'))


def ensure_numeric(df, col, fill=0.0):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill)
    else:
        df[col] = fill
    return df


@st.cache_data
def generate_sample_data(n_records: int = 1000):
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

    data = []
    for i in range(n_records):
        category = np.random.choice(categories)
        subcategory = np.random.choice(subcategories[category])
        segment = np.random.choice(segments)
        region = np.random.choice(regions)
        ship_mode = np.random.choice(ship_modes)

        order_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
        sales = round(np.random.uniform(10, 1000), 2)
        discount = round(np.random.uniform(0, 0.5), 2)
        profit = round(sales * np.random.uniform(-0.2, 0.4), 2)

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
    """Load CSV or Excel; robust encoding fallback and minimal cleaning"""
    if uploaded_file is None:
        df = generate_sample_data()
    else:
        # try common encodings and file types
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            try:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            except Exception:
                try:
                    df = pd.read_excel(uploaded_file)
                except Exception:
                    # last resort: read as latin1 with engine python
                    df = pd.read_csv(uploaded_file, encoding='latin1', engine='python')

    # normalize column names (strip + consistent casing)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # common columns fixups
    if 'Order Date' in df.columns:
        df['Order Date'] = safe_to_datetime(df['Order Date'])
    if 'Ship Date' in df.columns:
        df['Ship Date'] = safe_to_datetime(df['Ship Date'])

    # ensure numeric columns exist
    df = ensure_numeric(df, 'Sales', fill=0.0)
    df = ensure_numeric(df, 'Profit', fill=0.0)
    df = ensure_numeric(df, 'Discount', fill=0.0)

    # strip whitespace for text cols
    text_cols = ['Category', 'Sub-Category', 'Product Name', 'Customer Name', 'Region', 'State', 'City', 'Segment']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # If Segment missing, try to infer from available columns or create synthetic
    if 'Segment' not in df.columns:
        # simple heuristic: if Customer Name exists, create Regular/Premium split
        if 'Customer Name' in df.columns:
            df['Segment'] = np.random.choice(['Regular', 'Premium', 'Wholesale'], size=len(df), p=[0.65, 0.25, 0.10])
        else:
            df['Segment'] = 'Unknown'

    return df


# ---------------------
# KPI helpers
# ---------------------

def get_kpis(df):
    # safe aggregations
    total_sales = float(df['Sales'].sum()) if 'Sales' in df.columns else 0.0
    total_profit = float(df['Profit'].sum()) if 'Profit' in df.columns else 0.0
    total_orders = int(df.shape[0])
    avg_order_value = (total_sales / total_orders) if total_orders > 0 else 0.0
    profit_margin = ((total_profit / total_sales) * 100) if total_sales != 0 else 0.0

    return {
        'Total Sales': f"{total_sales:,.0f}",
        'Total Profit': f"{total_profit:,.0f}",
        'Total Orders': f"{total_orders:,}",
        'Avg Order Value': f"{avg_order_value:.2f}",
        'Profit Margin': f"{profit_margin:.1f}"
    }


# ---------------------
# Chart factories (robust to missing cols)
# ---------------------

def create_monthly_trends_chart(df):
    if 'Order Date' not in df.columns or df['Order Date'].dropna().empty:
        return None

    df_local = df.copy()
    df_local['Order Date'] = pd.to_datetime(df_local['Order Date'], errors='coerce')
    monthly = df_local.groupby(df_local['Order Date'].dt.to_period('M')).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    monthly['Order Date'] = monthly['Order Date'].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Sales'], mode='lines+markers', name='Sales'))
    if 'Profit' in monthly.columns:
        fig.add_trace(go.Scatter(x=monthly['Order Date'], y=monthly['Profit'], mode='lines+markers', name='Profit', yaxis='y2'))

    fig.update_layout(title='Monthly Sales and Profit Trends', xaxis_title='Month', yaxis_title='Sales', template='plotly_white', height=400)
    if 'Profit' in monthly.columns:
        fig.update_layout(yaxis2=dict(title='Profit', overlaying='y', side='right'))

    return fig


def create_category_chart(df):
    if 'Category' not in df.columns:
        return None
    ag = df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    fig = px.bar(ag, x='Category', y='Sales', color='Profit', color_continuous_scale='Viridis', title='Category Performance')
    fig.update_layout(template='plotly_white', height=400)
    return fig


def create_region_chart(df):
    if 'Region' not in df.columns:
        return None
    ag = df.groupby('Region').agg({'Sales': 'sum'}).reset_index()
    fig = px.pie(ag, values='Sales', names='Region', title='Sales Distribution by Region')
    fig.update_layout(height=400)
    return fig


def create_segment_chart(df):
    if 'Segment' not in df.columns:
        return None
    counts = df['Segment'].value_counts()
    if counts.empty:
        return None
    fig = px.pie(values=counts.values, names=counts.index, title='Customer Segments Distribution', hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig


def create_discount_impact_chart(df):
    if 'Discount' not in df.columns or 'Profit' not in df.columns:
        return None
    df_local = df.copy()
    try:
        df_local['Discount_Range'] = pd.cut(df_local['Discount'], bins=5, labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%'])
        ag = df_local.groupby('Discount_Range')['Profit'].mean().reset_index()
    except Exception:
        return None

    fig = px.bar(ag, x='Discount_Range', y='Profit', title='Average Profit by Discount Range')
    fig.update_layout(template='plotly_white', height=400)
    return fig


# ---------------------
# Main app
# ---------------------

def main():
    st.markdown("""
    <div style='background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); padding:12px; border-radius:8px;'>
        <h1 style='color:white; margin:0;'>🛒 StoreWise – Advanced Retail Analytics Dashboard</h1>
        <p style='color:white; margin:0.2rem 0 0;'>Handles multiple dataset formats and missing columns gracefully</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📁 Data Source")
        uploaded_file = st.file_uploader("Upload CSV / XLSX file", type=["csv", "xlsx"], help="Upload retail data (Superstore or similar)")

        st.markdown("### 🎛️ Filters (applied if available)")

    df = load_data(uploaded_file)

    # date filter
    with st.sidebar:
        if 'Order Date' in df.columns and not df['Order Date'].isna().all():
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            min_date = df['Order Date'].min()
            max_date = df['Order Date'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    df = df[(df['Order Date'] >= pd.Timestamp(date_range[0])) & (df['Order Date'] <= pd.Timestamp(date_range[1]))]

        if 'Category' in df.columns:
            categories = st.multiselect("Select Categories", options=sorted(df['Category'].dropna().unique()), default=sorted(df['Category'].dropna().unique()))
            if categories:
                df = df[df['Category'].isin(categories)]

        if 'Region' in df.columns:
            regions = st.multiselect("Select Regions", options=sorted(df['Region'].dropna().unique()), default=sorted(df['Region'].dropna().unique()))
            if regions:
                df = df[df['Region'].isin(regions)]

    kpis = get_kpis(df)

    st.markdown("### 📊 Key Performance Indicators")
    cols = st.columns(5)
    labels = ['Total Sales', 'Total Profit', 'Total Orders', 'Avg Order Value', 'Profit Margin']
    for c, lbl in zip(cols, labels):
        with c:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:12px; border-radius:8px; color:white; text-align:center;'>
                <div style='font-size:20px; font-weight:700;'>{kpis.get(lbl, '0')}</div>
                <div style='opacity:0.85;'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📈 Performance Analytics")

    # Monthly trends
    fig_monthly = create_monthly_trends_chart(df)
    if fig_monthly is not None:
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.info("Monthly trends not available (missing or invalid Order Date / Sales columns)")

    # Category and Region
    c1, c2 = st.columns(2)
    with c1:
        fig_cat = create_category_chart(df)
        if fig_cat is not None:
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Category chart not available (missing Category / Sales)")
    with c2:
        fig_reg = create_region_chart(df)
        if fig_reg is not None:
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("Region chart not available (missing Region / Sales)")

    # Segment and Discount impact
    c3, c4 = st.columns(2)
    with c3:
        fig_seg = create_segment_chart(df)
        if fig_seg is not None:
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.info("Segment chart not available (missing Segment column)")

    with c4:
        fig_disc = create_discount_impact_chart(df)
        if fig_disc is not None:
            st.plotly_chart(fig_disc, use_container_width=True)
        else:
            st.info("Discount impact chart not available (missing Discount / Profit)")

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
        if 'Profit' in df.columns:
            loss_products = df[df['Profit'] < 0]
            if not loss_products.empty:
                key = 'Product Name' if 'Product Name' in df.columns else 'Category' if 'Category' in df.columns else None
                if key:
                    loss_agg = loss_products.groupby(key).agg({'Sales': 'sum', 'Profit': 'sum', 'Order Date': 'count'}).rename(columns={'Order Date': 'Orders'}).sort_values('Profit').head(10)
                    st.dataframe(loss_agg, use_container_width=True)
                else:
                    st.dataframe(loss_products.head(50), use_container_width=True)
            else:
                st.success("🎉 No loss-making products found!")
        else:
            st.info("Profit column missing — cannot compute loss makers")

    with tab4:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button(label="📥 Download Filtered Data", data=csv, file_name="storewise_filtered_data.csv", mime="text/csv")


if __name__ == '__main__':
    main()
