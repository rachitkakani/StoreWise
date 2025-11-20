import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===== DATA LOADING AND PREPROCESSING =====

def load_data(file):
    """
    Load and preprocess retail data with enhanced date handling and validation
    """
    try:
        df = pd.read_csv(file)
        
        required_columns = ['Order Date', 'Sales', 'Profit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['Ship Date'] = pd.to_datetime(df.get('Ship Date', pd.NaT), errors='coerce')
        
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Quarter'] = df['Order Date'].dt.quarter
        df['Day_of_Week'] = df['Order Date'].dt.day_name()
        df['Week_of_Year'] = df['Order Date'].dt.isocalendar().week
        df['Is_Weekend'] = df['Order Date'].dt.dayofweek >= 5
        
        df['Profit_Margin'] = np.where(df['Sales'] != 0, (df['Profit'] / df['Sales']) * 100, 0)
        df['Discount_Amount'] = df['Sales'] * df.get('Discount', 0)
        
        if 'Ship Date' in df.columns and not df['Ship Date'].isna().all():
            df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days
        
        text_columns = ['Category', 'Sub-Category', 'Product Name', 'Customer Name', 'Region', 'State', 'City']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        print(f"✅ Data loaded successfully: {len(df)} records, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

# ===== KEY PERFORMANCE INDICATORS =====

def get_kpis(df):
    """
    Calculate comprehensive KPIs with growth metrics and trends
    """
    try:
        total_sales = df['Sales'].sum()
        total_profit = df['Profit'].sum()
        total_orders = df['Order ID'].nunique() if 'Order ID' in df.columns else len(df)
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
        
        total_customers = df['Customer Name'].nunique() if 'Customer Name' in df.columns else 0
        avg_profit_per_order = total_profit / total_orders if total_orders > 0 else 0
        total_products = df['Product Name'].nunique() if 'Product Name' in df.columns else 0
        total_quantity = df['Quantity'].sum() if 'Quantity' in df.columns else 0
        avg_discount = df['Discount'].mean() * 100 if 'Discount' in df.columns else 0
        
        growth_metrics = calculate_growth_metrics(df)
        
        kpis = {
            'Total Sales': round(total_sales, 2),
            'Total Profit': round(total_profit, 2),
            'Total Orders': total_orders,
            'Avg Order Value': round(avg_order_value, 2),
            'Profit Margin': round(profit_margin, 2),
            
            'Total Customers': total_customers,
            'Avg Profit per Order': round(avg_profit_per_order, 2),
            'Total Products': total_products,
            'Total Quantity': total_quantity,
            'Avg Discount': round(avg_discount, 2),
            
            'Revenue per Customer': round(total_sales / total_customers, 2) if total_customers > 0 else 0,
            'Orders per Customer': round(total_orders / total_customers, 2) if total_customers > 0 else 0,
        }
        
        kpis.update(growth_metrics)
        
        return kpis
        
    except Exception as e:
        print(f"❌ Error calculating KPIs: {str(e)}")
        return {}

def calculate_growth_metrics(df):
    """Calculate year-over-year and month-over-month growth"""
    growth_metrics = {}
    
    try:
        if 'Year' in df.columns and df['Year'].nunique() > 1:
            
            yearly_sales = df.groupby('Year')['Sales'].sum()
            if len(yearly_sales) >= 2:
                current_year_sales = yearly_sales.iloc[-1]
                previous_year_sales = yearly_sales.iloc[-2]
                yoy_growth = ((current_year_sales - previous_year_sales) / previous_year_sales) * 100
                growth_metrics['YoY Sales Growth'] = round(yoy_growth, 2)
        
        if 'Month' in df.columns and 'Year' in df.columns:
            df['YearMonth'] = df['Order Date'].dt.to_period('M')
            monthly_sales = df.groupby('YearMonth')['Sales'].sum()
            if len(monthly_sales) >= 2:
                current_month_sales = monthly_sales.iloc[-1]
                previous_month_sales = monthly_sales.iloc[-2]
                mom_growth = ((current_month_sales - previous_month_sales) / previous_month_sales) * 100
                growth_metrics['MoM Sales Growth'] = round(mom_growth, 2)
                
    except Exception as e:
        print(f"⚠️ Warning: Could not calculate growth metrics: {str(e)}")
    
    return growth_metrics

# ===== PERFORMANCE ANALYSIS =====

def category_performance(df):
    """Enhanced category analysis with multiple metrics"""
    try:
        category_stats = df.groupby('Category').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Profit': ['sum', 'mean'],
            'Quantity': 'sum' if 'Quantity' in df.columns else 'count',
            'Discount': 'mean' if 'Discount' in df.columns else lambda x: 0
        }).round(2)
        
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.reset_index()
        
        category_stats['Profit_Margin'] = (category_stats['Profit_sum'] / category_stats['Sales_sum'] * 100).round(2)
        
        return category_stats.sort_values('Sales_sum', ascending=False)
        
    except Exception as e:
        print(f"❌ Error in category analysis: {str(e)}")
        return pd.DataFrame()

def subcategory_performance(df, top_n=20):
    """Enhanced subcategory analysis"""
    try:
        if 'Sub-Category' not in df.columns:
            return pd.DataFrame()
            
        subcategory_stats = df.groupby(['Category', 'Sub-Category']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count'
        }).round(2)
        
        subcategory_stats.columns = ['Sales', 'Profit', 'Orders']
        subcategory_stats['Profit_Margin'] = (subcategory_stats['Profit'] / subcategory_stats['Sales'] * 100).round(2)
        
        return subcategory_stats.reset_index().sort_values('Sales', ascending=False).head(top_n)
        
    except Exception as e:
        print(f"❌ Error in subcategory analysis: {str(e)}")
        return pd.DataFrame()

def region_summary(df):
    """Enhanced regional analysis"""
    try:
        region_stats = df.groupby('Region').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count',
            'Customer Name': 'nunique' if 'Customer Name' in df.columns else lambda x: 0
        }).round(2)
        
        region_stats.columns = ['_'.join(col).strip() for col in region_stats.columns]
        region_stats = region_stats.reset_index()
        
        region_stats['Profit_Margin'] = (region_stats['Profit_sum'] / region_stats['Sales_sum'] * 100).round(2)
        region_stats['Revenue_per_Customer'] = (region_stats['Sales_sum'] / region_stats['Customer Name_nunique']).round(2)
        
        return region_stats.sort_values('Sales_sum', ascending=False)
        
    except Exception as e:
        print(f"❌ Error in region analysis: {str(e)}")
        return pd.DataFrame()

def state_summary(df, top_n=15):
    """Enhanced state-level analysis"""
    try:
        if 'State' not in df.columns:
            return pd.DataFrame()
            
        state_stats = df.groupby(['Region', 'State']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count'
        }).round(2)
        
        state_stats.columns = ['Sales', 'Profit', 'Orders']
        state_stats['Profit_Margin'] = (state_stats['Profit'] / state_stats['Sales'] * 100).round(2)
        
        return state_stats.reset_index().sort_values('Sales', ascending=False).head(top_n)
        
    except Exception as e:
        print(f"❌ Error in state analysis: {str(e)}")
        return pd.DataFrame()

def segment_summary(df):
    """Enhanced customer segment analysis"""
    try:
        segment_stats = df.groupby('Segment').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count',
            'Customer Name': 'nunique' if 'Customer Name' in df.columns else lambda x: 0
        }).round(2)
        
        segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns]
        segment_stats = segment_stats.reset_index()
        
        segment_stats['Profit_Margin'] = (segment_stats['Profit_sum'] / segment_stats['Sales_sum'] * 100).round(2)
        segment_stats['Avg_Order_Value'] = (segment_stats['Sales_sum'] / segment_stats['Order ID_nunique']).round(2)
        
        return segment_stats.sort_values('Sales_sum', ascending=False)
        
    except Exception as e:
        print(f"❌ Error in segment analysis: {str(e)}")
        return pd.DataFrame()

# ===== CUSTOMER AND PRODUCT ANALYSIS =====

def top_products(df, n=10, metric='Sales'):
    """Enhanced top products analysis with multiple metrics"""
    try:
        if 'Product Name' not in df.columns:
            return pd.DataFrame()
            
        product_stats = df.groupby('Product Name').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count',
            'Quantity': 'sum' if 'Quantity' in df.columns else 'count'
        }).round(2)
        
        product_stats.columns = ['Sales', 'Profit', 'Orders', 'Quantity']
        product_stats['Profit_Margin'] = (product_stats['Profit'] / product_stats['Sales'] * 100).round(2)
        product_stats['Avg_Sale_Value'] = (product_stats['Sales'] / product_stats['Orders']).round(2)
        
        return product_stats.reset_index().sort_values(metric, ascending=False).head(n)
        
    except Exception as e:
        print(f"❌ Error in top products analysis: {str(e)}")
        return pd.DataFrame()

def top_customers(df, n=10, metric='Sales'):
    """Enhanced top customers analysis"""
    try:
        if 'Customer Name' not in df.columns:
            return pd.DataFrame()
            
        customer_stats = df.groupby('Customer Name').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count',
            'Order Date': ['min', 'max']
        }).round(2)
        
        customer_stats.columns = ['Sales', 'Profit', 'Orders', 'First_Order', 'Last_Order']
        
        customer_stats['Customer_Lifetime_Days'] = (
            pd.to_datetime(customer_stats['Last_Order']) - 
            pd.to_datetime(customer_stats['First_Order'])
        ).dt.days
        
        customer_stats['Avg_Order_Value'] = (customer_stats['Sales'] / customer_stats['Orders']).round(2)
        customer_stats['Profit_per_Order'] = (customer_stats['Profit'] / customer_stats['Orders']).round(2)
        
        return customer_stats.reset_index().sort_values(metric, ascending=False).head(n)
        
    except Exception as e:
        print(f"❌ Error in top customers analysis: {str(e)}")
        return pd.DataFrame()

def loss_making_products(df, n=10):
    """Enhanced loss-making products analysis"""
    try:
        if 'Product Name' not in df.columns:
            return pd.DataFrame()
            
        product_stats = df.groupby('Product Name').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count'
        }).round(2)
        
        loss_products = product_stats[product_stats['Profit'] < 0]
        
        if loss_products.empty:
            return pd.DataFrame()
            
        loss_products.columns = ['Sales', 'Profit', 'Orders']
        loss_products['Profit_Margin'] = (loss_products['Profit'] / loss_products['Sales'] * 100).round(2)
        loss_products['Loss_per_Order'] = (abs(loss_products['Profit']) / loss_products['Orders']).round(2)
        
        return loss_products.reset_index().sort_values('Profit').head(n)
        
    except Exception as e:
        print(f"❌ Error in loss products analysis: {str(e)}")
        return pd.DataFrame()

# ===== TIME SERIES ANALYSIS =====

def monthly_trends(df):
    """Enhanced monthly trends with seasonal analysis"""
    try:
        monthly_stats = df.groupby(['Year', 'Month']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count'
        }).round(2)
        
        monthly_stats.columns = ['Sales', 'Profit', 'Orders']
        monthly_stats['Profit_Margin'] = (monthly_stats['Profit'] / monthly_stats['Sales'] * 100).round(2)
        monthly_stats['Avg_Order_Value'] = (monthly_stats['Sales'] / monthly_stats['Orders']).round(2)
        
        monthly_stats = monthly_stats.reset_index()
        monthly_stats['Month_Name'] = pd.to_datetime(monthly_stats[['Year', 'Month']].assign(day=1)).dt.strftime('%B')
        
        return monthly_stats
        
    except Exception as e:
        print(f"❌ Error in monthly trends analysis: {str(e)}")
        return pd.DataFrame()

def quarterly_trends(df):
    """Quarterly performance analysis"""
    try:
        quarterly_stats = df.groupby(['Year', 'Quarter']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count'
        }).round(2)
        
        quarterly_stats.columns = ['Sales', 'Profit', 'Orders']
        quarterly_stats['Profit_Margin'] = (quarterly_stats['Profit'] / quarterly_stats['Sales'] * 100).round(2)
        
        return quarterly_stats.reset_index()
        
    except Exception as e:
        print(f"❌ Error in quarterly trends analysis: {str(e)}")
        return pd.DataFrame()

def seasonal_analysis(df):
    """Analyze seasonal patterns in sales"""
    try:
        seasonal_stats = df.groupby('Month').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count'
        }).round(2)
        
        seasonal_stats.columns = ['_'.join(col).strip() for col in seasonal_stats.columns]
        seasonal_stats = seasonal_stats.reset_index()
        
        seasonal_stats['Month_Name'] = pd.to_datetime(seasonal_stats['Month'], format='%m').dt.strftime('%B')
        
        return seasonal_stats.sort_values('Sales_sum', ascending=False)
        
    except Exception as e:
        print(f"❌ Error in seasonal analysis: {str(e)}")
        return pd.DataFrame()

# ===== ADVANCED ANALYSIS =====

def discount_impact(df):
    """Enhanced discount impact analysis"""
    try:
        if 'Discount' not in df.columns:
            return pd.DataFrame()
            
        df['Discount_Range'] = pd.cut(df['Discount'], 
                                    bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0], 
                                    labels=['0-10%', '10-20%', '20-30%', '30-40%', '40%+'])
        
        discount_analysis = df.groupby('Discount_Range').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count'
        }).round(2)
        
        discount_analysis.columns = ['_'.join(col).strip() for col in discount_analysis.columns]
        discount_analysis = discount_analysis.reset_index()
        
        discount_analysis['Profit_Margin'] = (discount_analysis['Profit_sum'] / discount_analysis['Sales_sum'] * 100).round(2)
        
        return discount_analysis
        
    except Exception as e:
        print(f"❌ Error in discount analysis: {str(e)}")
        return pd.DataFrame()

def shipping_analysis(df):
    """Enhanced shipping mode analysis"""
    try:
        if 'Ship Mode' not in df.columns:
            return pd.DataFrame()
            
        shipping_stats = df.groupby('Ship Mode').agg({
            'Sales': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count',
            'Shipping_Days': 'mean' if 'Shipping_Days' in df.columns else lambda x: 0
        }).round(2)
        
        shipping_stats.columns = ['_'.join(col).strip() for col in shipping_stats.columns]
        shipping_stats = shipping_stats.reset_index()
        
        shipping_stats['Profit_Margin'] = (shipping_stats['Profit_sum'] / shipping_stats['Sales_sum'] * 100).round(2)
        
        return shipping_stats.sort_values('Sales_sum', ascending=False)
        
    except Exception as e:
        print(f"❌ Error in shipping analysis: {str(e)}")
        return pd.DataFrame()

def customer_segmentation_rfm(df):
    """RFM (Recency, Frequency, Monetary) customer segmentation"""
    try:
        if not all(col in df.columns for col in ['Customer Name', 'Order Date', 'Sales']):
            return pd.DataFrame()
            
        current_date = df['Order Date'].max()
        
        rfm = df.groupby('Customer Name').agg({
            'Order Date': lambda x: (current_date - x.max()).days, 
            'Order ID': 'nunique' if 'Order ID' in df.columns else 'count',  
            'Sales': 'sum' 
        }).round(2)
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['RFM_Score'] in ['111', '112', '121', '131', '141', '151']:
                return 'Lost Customers'
            else:
                return 'Others'
        
        rfm['Segment'] = rfm.apply(segment_customers, axis=1)
        
        return rfm.reset_index()
        
    except Exception as e:
        print(f"❌ Error in RFM analysis: {str(e)}")
        return pd.DataFrame()

# ===== PREDICTIVE ANALYSIS =====

def sales_forecast_simple(df, periods=6):
    """Simple sales forecasting using moving average"""
    try:
        monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
        
        window = min(3, len(monthly_sales))
        forecasts = []
        
        for i in range(periods):
            if len(monthly_sales) >= window:
                forecast = monthly_sales.tail(window).mean()
                forecasts.append(forecast)
                
                next_period = monthly_sales.index[-1] + 1
                monthly_sales[next_period] = forecast
        
        last_date = df['Order Date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Sales': forecasts
        })
        
        return forecast_df
        
    except Exception as e:
        print(f"❌ Error in sales forecast: {str(e)}")
        return pd.DataFrame()

# ===== UTILITY FUNCTIONS =====

def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    try:
        report = {
            'Data Overview': {
                'Total Records': len(df),
                'Date Range': f"{df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}",
                'Unique Products': df['Product Name'].nunique() if 'Product Name' in df.columns else 'N/A',
                'Unique Customers': df['Customer Name'].nunique() if 'Customer Name' in df.columns else 'N/A',
                'Categories': df['Category'].nunique() if 'Category' in df.columns else 'N/A'
            },
            'Financial Summary': get_kpis(df),
            'Top Performers': {
                'Best Category': category_performance(df).iloc[0]['Category'] if not category_performance(df).empty else 'N/A',
                'Best Region': region_summary(df).iloc[0]['Region'] if not region_summary(df).empty else 'N/A',
                'Best Month': monthly_trends(df).sort_values('Sales', ascending=False).iloc[0]['Month_Name'] if not monthly_trends(df).empty else 'N/A'
            }
        }
        
        return report
        
    except Exception as e:
        print(f"❌ Error generating summary report: {str(e)}")
        return {}

def data_quality_check(df):
    """Perform data quality checks"""
    try:
        quality_report = {
            'Missing Values': df.isnull().sum().to_dict(),
            'Duplicate Rows': df.duplicated().sum(),
            'Negative Values': {
                'Sales': (df['Sales'] < 0).sum(),
                'Profit': (df['Profit'] < 0).sum() if 'Profit' in df.columns else 0
            },
            'Data Types': df.dtypes.to_dict()
        }
        
        return quality_report
        
    except Exception as e:
        print(f"❌ Error in data quality check: {str(e)}")
        return {}