import pandas as pd
import streamlit as st
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import warnings
import numpy as np
from datetime import timedelta
import io
import scipy.stats as stats

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CashWise - Enhanced Forecasting App", layout="wide")
st.title("📊 CashWise - Enhanced Forecasting App")
st.markdown("### Smarter Forecasting for Everyone")

st.sidebar.header("📁 Upload Your Cashflow Data")

uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Demo data generation section
st.sidebar.markdown("---")
st.sidebar.header("🎲 Generate Demo Data")
st.sidebar.markdown("*Don't have data? Generate realistic sample data to test the app*")

demo_config = st.sidebar.expander("📝 Demo Data Configuration")
with demo_config:
    demo_countries = st.multiselect(
        "Countries", 
        ["Netherlands", "Germany", "France", "Belgium", "UK"],
        default=["Netherlands", "Germany"]
    )
    
    demo_components = st.multiselect(
        "Cost Components",
        ["Salaries", "Rent", "Marketing", "IT Services", "Office Supplies", "Travel", "Utilities"],
        default=["Salaries", "Rent", "Marketing"]
    )
    
    demo_currencies = st.multiselect(
        "Currencies",
        ["EUR", "USD", "GBP"],
        default=["EUR"]
    )
    
    demo_years = st.slider("Years of historical data", 1, 5, 3)
    demo_seasonality = st.selectbox(
        "Business type", 
        ["Stable", "Seasonal", "High Growth", "Weather Dependent"],
        help="Affects the data patterns generated"
    )

def generate_demo_data(countries, components, currencies, years, seasonality_type):
    """Generate realistic demo cashflow data"""
    np.random.seed(42)  # For reproducible demo data
    
    # Date range
    end_date = pd.Timestamp.now().replace(day=1) - pd.DateOffset(months=1)  # Last complete month
    start_date = end_date - pd.DateOffset(years=years) + pd.DateOffset(months=1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    data = []
    
    # Base amounts for different components (in thousands)
    component_bases = {
        "Salaries": 150,
        "Rent": 25,
        "Marketing": 40,
        "IT Services": 15,
        "Office Supplies": 5,
        "Travel": 12,
        "Utilities": 8
    }
    
    # Country multipliers (cost of living factors)
    country_multipliers = {
        "Netherlands": 1.0,
        "Germany": 0.9,
        "France": 0.95,
        "Belgium": 0.85,
        "UK": 1.1
    }
    
    # Currency conversion rates (relative to EUR)
    currency_rates = {
        "EUR": 1.0,
        "USD": 1.1,
        "GBP": 0.85
    }
    
    for country in countries:
        for component in components:
            for currency in currencies:
                base_amount = component_bases[component] * country_multipliers[country] / currency_rates[currency]
                
                # Employee count (affects salary scaling)
                if component == "Salaries":
                    employee_count = np.random.randint(20, 100)
                else:
                    employee_count = None
                
                for i, date in enumerate(date_range):
                    month = date.month
                    year_progress = i / len(date_range)
                    
                    # Base seasonal patterns
                    seasonal_factor = 1.0
                    if seasonality_type == "Seasonal":
                        # Higher costs in Q4, lower in summer
                        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (month - 3) / 12)
                        if component == "Marketing":
                            seasonal_factor += 0.3 if month in [11, 12] else 0
                    elif seasonality_type == "Weather Dependent":
                        # Strong quarterly patterns
                        seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * (month - 1) / 3)
                        if component in ["Utilities", "Travel"]:
                            seasonal_factor += 0.5 if month in [12, 1, 2, 6, 7, 8] else 0
                    
                    # Growth trends
                    growth_factor = 1.0
                    if seasonality_type == "High Growth":
                        growth_factor = 1 + 0.15 * year_progress  # 15% annual growth
                    elif seasonality_type == "Stable":
                        growth_factor = 1 + 0.03 * year_progress  # 3% annual growth
                    else:
                        growth_factor = 1 + 0.05 * year_progress  # 5% annual growth
                    
                    # Component-specific adjustments
                    component_factor = 1.0
                    if component == "Salaries":
                        # Annual salary increases
                        if month == 1:  # January increases
                            component_factor += 0.05
                    elif component == "Marketing":
                        # Higher marketing in certain months
                        if month in [3, 9, 11]:  # Campaign months
                            component_factor += 0.3
                    
                    # Calculate final amount
                    amount = (base_amount * seasonal_factor * growth_factor * component_factor * 
                             (1 + np.random.normal(0, 0.1)))  # 10% random variation
                    
                    # Ensure positive values
                    amount = max(amount, base_amount * 0.1)
                    
                    data.append({
                        'Date': date,
                        'Country': country,
                        'Component': component,
                        'Amount': round(amount * 1000, 2),  # Convert to actual amounts
                        'Currency': currency,
                        'EmployeeCount': employee_count if component == "Salaries" else None
                    })
    
    return pd.DataFrame(data)

# Generate demo data button
if st.sidebar.button("🎯 Generate Demo Data", type="primary"):
    if demo_countries and demo_components and demo_currencies:
        demo_df = generate_demo_data(demo_countries, demo_components, demo_currencies, demo_years, demo_seasonality)
        
        # Store demo data in session state
        st.session_state['demo_data'] = demo_df
        st.sidebar.success(f"✅ Generated {len(demo_df)} records!")
        
        # Show preview
        st.sidebar.markdown("**Preview:**")
        st.sidebar.dataframe(demo_df.head(3), use_container_width=True)
        
    else:
        st.sidebar.error("Please select at least one option for each category")

# Download demo data
if 'demo_data' in st.session_state:
    demo_buffer = io.BytesIO()
    st.session_state['demo_data'].to_csv(demo_buffer, index=False)
    
    st.sidebar.download_button(
        "📥 Download Demo Data (CSV)",
        data=demo_buffer.getvalue(),
        file_name=f"cashwise_demo_data_{demo_seasonality.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
    
    # Option to use demo data directly
    if st.sidebar.button("🚀 Use Demo Data Directly"):
        st.session_state['uploaded_data'] = st.session_state['demo_data']
        st.sidebar.success("Demo data loaded for analysis!")

st.sidebar.markdown("---")

# Enhanced configuration options
st.sidebar.header("⚙️ Configuration Options")

# Weather dependency option
weather_dependent = st.sidebar.checkbox(
    "🌤️ Weather-Dependent Business", 
    help="Check this if your business is significantly affected by seasonal weather patterns (e.g., retail, tourism, agriculture)"
)

# Data simulation option
enable_simulation = st.sidebar.checkbox(
    "🎲 Enable Data Simulation", 
    help="Generate additional synthetic data points based on historical patterns (useful for businesses with <5 years of data)"
)

if enable_simulation:
    simulation_years = st.sidebar.slider(
        "Years of synthetic data to generate", 
        min_value=1, max_value=3, value=2,
        help="Additional years of data to simulate based on historical patterns"
    )

# Scenario planning options
st.sidebar.header("📈 Scenario Planning")
scenario_planning = st.sidebar.checkbox(
    "🎯 Enable Scenario Planning",
    help="Generate optimistic, pessimistic, and realistic forecasts based on confidence intervals"
)

required_cols = ['Date', 'Country', 'Component', 'Amount', 'Currency']
model_logs = []
forecast_outputs = []
model_performance = {}

def generate_synthetic_data(ts_df, years_to_generate=2):
    """Generate synthetic data based on historical patterns"""
    if len(ts_df) < 12:  # Need at least 1 year of data
        return ts_df
    
    # Calculate trend and seasonality
    ts_df_indexed = ts_df.set_index('ds')
    monthly_avg = ts_df_indexed.groupby(ts_df_indexed.index.month)['y'].mean()
    
    # Calculate year-over-year growth
    if len(ts_df) >= 24:  # If we have at least 2 years
        yearly_growth = ts_df_indexed['y'].pct_change(12).median()
    else:
        yearly_growth = 0.05  # Default 5% growth
    
    # Generate synthetic data
    last_date = ts_df['ds'].max()
    synthetic_data = []
    
    for year in range(1, years_to_generate + 1):
        for month in range(1, 13):
            new_date = last_date + pd.DateOffset(months=(year-1)*12 + month)
            
            # Base value from historical monthly average
            base_value = monthly_avg[month]
            
            # Apply growth trend
            growth_factor = (1 + yearly_growth) ** year
            
            # Add some noise
            noise = np.random.normal(0, base_value * 0.1)
            
            synthetic_value = base_value * growth_factor + noise
            synthetic_data.append({'ds': new_date, 'y': max(0, synthetic_value)})
    
    synthetic_df = pd.DataFrame(synthetic_data)
    return pd.concat([ts_df, synthetic_df], ignore_index=True)

def calculate_model_metrics(actual, predicted):
    """Calculate comprehensive model performance metrics"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

def generate_confidence_intervals(forecast_values, model_name, historical_residuals=None):
    """Generate confidence intervals for scenario planning"""
    if historical_residuals is not None and len(historical_residuals) > 0:
        # Use historical prediction errors to estimate uncertainty
        std_error = np.std(historical_residuals)
    else:
        # Use 15% of mean forecast as default uncertainty
        std_error = np.mean(forecast_values) * 0.15
    
    # Generate scenarios
    confidence_level = 0.8  # 80% confidence interval
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    optimistic = forecast_values + z_score * std_error
    pessimistic = forecast_values - z_score * std_error
    realistic = forecast_values
    
    return {
        'Optimistic': optimistic,
        'Realistic': realistic,
        'Pessimistic': pessimistic
    }

# Check for data source (uploaded file or demo data)
data_source = None
df = None

if uploaded_file:
    data_source = "uploaded"
elif 'uploaded_data' in st.session_state:
    data_source = "demo"
    df = st.session_state['uploaded_data'].copy()

if data_source:
    try:
        if data_source == "uploaded":
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, parse_dates=["Date"])
            else:
                df = pd.read_excel(uploaded_file, parse_dates=["Date"])
        # For demo data, df is already loaded

        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Found columns: {df.columns.tolist()}")
        else:
            if 'EmployeeCount' not in df.columns:
                st.warning("Optional column 'EmployeeCount' not found. Proceeding without it.")

            latest_months = df.groupby(['Country', 'Component'])['Date'].max().dt.to_period("M")
            if not all(latest_months == latest_months.iloc[0]):
                st.error("All (Country, Component) series must end in the same month for consistent forecasting. Please align your data.")
            else:
                st.subheader("📊 Data Overview")
                
                # Show data source
                if data_source == "demo":
                    st.info(f"📊 Using demo data with {demo_seasonality.lower()} business patterns")
                else:
                    st.info("📁 Using uploaded data")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(df.head(10))
                
                with col2:
                    st.metric("Total Records", len(df))
                    st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
                    st.metric("Currencies", df['Currency'].nunique())
                
                st.subheader("🎯 Enhanced Forecasted Monthly Cashflow")

                forecast_horizon = 3
                
                for currency, currency_df in df.groupby("Currency"):
                    st.markdown(f"### 💰 Currency: {currency}")
                    forecasts = []
                    all_histories = []
                    scenario_data = []

                    for (country, component), group in currency_df.groupby(["Country", "Component"]):
                        group = group.sort_values("Date")
                        ts_df = group[["Date", "Amount"]].rename(columns={"Date": "ds", "Amount": "y"})
                        ts_df = ts_df.set_index("ds").resample("MS").sum().reset_index()

                        # Data simulation if enabled and insufficient data
                        original_length = len(ts_df)
                        if enable_simulation and len(ts_df) < 60:  # Less than 5 years
                            ts_df = generate_synthetic_data(ts_df, simulation_years)
                            st.info(f"Generated {len(ts_df) - original_length} synthetic data points for {country}-{component}")

                        if len(ts_df) < 12:  # Need at least 1 year
                            st.warning(f"Insufficient data for {country}-{component}. Need at least 12 months.")
                            continue

                        split_index = int(len(ts_df) * 0.8)
                        train_df = ts_df.iloc[:split_index]
                        test_df = ts_df.iloc[split_index:]

                        errors = {}
                        predictions = {}
                        residuals = {}

                        # Prophet with weather dependency adjustment
                        try:
                            prophet_model = Prophet(
                                changepoint_prior_scale=0.05,
                                seasonality_mode='multiplicative' if weather_dependent else 'additive',
                                yearly_seasonality=True,
                                weekly_seasonality=False,
                                daily_seasonality=False
                            )
                            
                            if weather_dependent:
                                prophet_model.add_seasonality(
                                    name='quarterly',
                                    period=91.25,
                                    fourier_order=8,
                                    mode='multiplicative'
                                )
                            
                            prophet_model.fit(train_df.rename(columns={"ds": "ds", "y": "y"}))
                            future = prophet_model.make_future_dataframe(periods=len(test_df), freq="MS")
                            forecast = prophet_model.predict(future)
                            pred = forecast.set_index("ds").loc[test_df["ds"]]["yhat"]
                            
                            metrics = calculate_model_metrics(test_df.set_index("ds")["y"], pred)
                            errors['Prophet'] = (metrics['MAE'], prophet_model, metrics)
                            predictions['Prophet'] = pred
                            residuals['Prophet'] = test_df.set_index("ds")["y"] - pred
                        except Exception as e:
                            model_logs.append(f"Prophet failed for {country}-{component}: {e}")

                        # SARIMA with weather adjustment
                        try:
                            seasonal_period = 4 if weather_dependent else 12  # Quarterly vs yearly
                            sarima_model = SARIMAX(
                                train_df.set_index("ds")["y"], 
                                order=(1, 1, 1), 
                                seasonal_order=(1, 1, 1, seasonal_period)
                            )
                            sarima_fit = sarima_model.fit(disp=False)
                            forecast = sarima_fit.forecast(steps=len(test_df))
                            
                            metrics = calculate_model_metrics(test_df.set_index("ds")["y"], forecast)
                            errors['SARIMA'] = (metrics['MAE'], sarima_model, metrics)
                            predictions['SARIMA'] = forecast
                            residuals['SARIMA'] = test_df.set_index("ds")["y"] - forecast
                        except Exception as e:
                            model_logs.append(f"SARIMA failed for {country}-{component}: {e}")

                        # Holt-Winters with weather adjustment
                        try:
                            seasonal_periods = 4 if weather_dependent else 12
                            hw_model = ExponentialSmoothing(
                                train_df["y"], 
                                seasonal="multiplicative" if weather_dependent else "add", 
                                seasonal_periods=min(seasonal_periods, len(train_df)//2)
                            )
                            hw_fit = hw_model.fit()
                            forecast = hw_fit.forecast(len(test_df))
                            
                            metrics = calculate_model_metrics(test_df["y"], forecast)
                            errors['Holt-Winters'] = (metrics['MAE'], hw_model, metrics)
                            predictions['Holt-Winters'] = forecast
                            residuals['Holt-Winters'] = test_df["y"] - forecast
                        except Exception as e:
                            model_logs.append(f"Holt-Winters failed for {country}-{component}: {e}")

                        # Random Forest
                        try:
                            rf_train = train_df.copy()
                            rf_train['month'] = rf_train['ds'].dt.month
                            rf_train['year'] = rf_train['ds'].dt.year
                            rf_train['quarter'] = rf_train['ds'].dt.quarter
                            
                            features = ['month', 'year']
                            if weather_dependent:
                                features.append('quarter')
                            
                            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf_model.fit(rf_train[features], rf_train['y'])

                            rf_test = test_df.copy()
                            rf_test['month'] = rf_test['ds'].dt.month
                            rf_test['year'] = rf_test['ds'].dt.year
                            rf_test['quarter'] = rf_test['ds'].dt.quarter
                            pred = rf_model.predict(rf_test[features])
                            
                            metrics = calculate_model_metrics(test_df['y'], pred)
                            errors['RandomForest'] = (metrics['MAE'], rf_model, metrics)
                            predictions['RandomForest'] = pred
                            residuals['RandomForest'] = test_df['y'] - pred
                        except Exception as e:
                            model_logs.append(f"RandomForest failed for {country}-{component}: {e}")

                        if not errors:
                            continue

                        # Select best model and display performance
                        best_model_name = min(errors, key=lambda k: errors[k][0])
                        best_model = errors[best_model_name][1]
                        best_metrics = errors[best_model_name][2]
                        
                        # Store model performance for display
                        model_performance[f"{country}-{component}"] = {
                            'best_model': best_model_name,
                            'metrics': {name: errors[name][2] for name in errors.keys()}
                        }
                        
                        model_logs.append(f"Best model for {country}-{component}: {best_model_name} (MAE: {best_metrics['MAE']:.2f}, R²: {best_metrics['R²']:.3f})")

                        try:
                            final_month = ts_df["ds"].max()
                            future_dates = pd.date_range(final_month + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq="MS")

                            if best_model_name == "Prophet":
                                final_model = Prophet(
                                    changepoint_prior_scale=0.05,
                                    seasonality_mode='multiplicative' if weather_dependent else 'additive'
                                )
                                if weather_dependent:
                                    final_model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
                                
                                final_model.fit(ts_df.rename(columns={"ds": "ds", "y": "y"}))
                                future_final = final_model.make_future_dataframe(periods=forecast_horizon, freq="MS")
                                final_forecast = final_model.predict(future_final)
                                forecast_result = final_forecast[["ds", "yhat"]].tail(forecast_horizon)
                                forecast_result.columns = ["Month", "Estimated Cashflow"]
                                forecast_values = forecast_result["Estimated Cashflow"].values

                            elif best_model_name == "SARIMA":
                                seasonal_period = 4 if weather_dependent else 12
                                final_model = SARIMAX(ts_df.set_index("ds")["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
                                final_fit = final_model.fit(disp=False)
                                forecast_values = final_fit.forecast(steps=forecast_horizon)
                                forecast_result = pd.DataFrame({"Month": future_dates, "Estimated Cashflow": forecast_values})

                            elif best_model_name == "Holt-Winters":
                                seasonal_periods = 4 if weather_dependent else 12
                                final_model = ExponentialSmoothing(ts_df["y"], seasonal="multiplicative" if weather_dependent else "add", seasonal_periods=min(seasonal_periods, len(ts_df)//2))
                                final_fit = final_model.fit()
                                forecast_values = final_fit.forecast(forecast_horizon)
                                forecast_result = pd.DataFrame({"Month": future_dates, "Estimated Cashflow": forecast_values})

                            else:  # RandomForest or Ridge
                                ml_all = ts_df.copy()
                                ml_all['month'] = ml_all['ds'].dt.month
                                ml_all['year'] = ml_all['ds'].dt.year
                                ml_all['quarter'] = ml_all['ds'].dt.quarter
                                
                                features = ['month', 'year']
                                if weather_dependent:
                                    features.append('quarter')
                                
                                final_model = best_model
                                final_model.fit(ml_all[features], ml_all['y'])

                                future_ml = pd.DataFrame({"Month": future_dates})
                                future_ml['month'] = future_ml['Month'].dt.month
                                future_ml['year'] = future_ml['Month'].dt.year
                                future_ml['quarter'] = future_ml['Month'].dt.quarter
                                forecast_values = final_model.predict(future_ml[features])
                                forecast_result = pd.DataFrame({"Month": future_dates, "Estimated Cashflow": forecast_values})

                            # Generate scenarios if enabled
                            if scenario_planning:
                                best_residuals = residuals.get(best_model_name, None)
                                scenarios = generate_confidence_intervals(forecast_values, best_model_name, best_residuals)
                                
                                scenario_result = pd.DataFrame({
                                    "Month": future_dates,
                                    "Optimistic": scenarios['Optimistic'],
                                    "Realistic": scenarios['Realistic'],
                                    "Pessimistic": scenarios['Pessimistic'],
                                    "Country": country,
                                    "Component": component
                                })
                                scenario_data.append(scenario_result)

                            history = ts_df[['ds', 'y']].rename(columns={'ds': 'Month', 'y': 'Estimated Cashflow'})
                            forecast_result['Country'] = country
                            forecast_result['Component'] = component
                            forecast_outputs.append(forecast_result.copy())
                            forecasts.append(forecast_result)
                            all_histories.append(history)
                            
                        except Exception as e:
                            model_logs.append(f"Final forecast failed for {country}-{component}: {e}")

                    if forecasts:
                        # Display forecasts
                        combined = pd.concat(forecasts)
                        currency_summary = combined.groupby("Month")["Estimated Cashflow"].sum().reset_index()
                        full_history = pd.concat(all_histories).groupby("Month")["Estimated Cashflow"].sum().reset_index()

                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.dataframe(currency_summary)
                        
                        with col2:
                            if scenario_planning and scenario_data:
                                combined_scenarios = pd.concat(scenario_data)
                                scenario_summary = combined_scenarios.groupby("Month")[["Optimistic", "Realistic", "Pessimistic"]].sum()
                                st.markdown("**📊 Scenario Summary**")
                                st.dataframe(scenario_summary)

                        # Enhanced visualization
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(full_history["Month"], full_history["Estimated Cashflow"], 
                               label="Historical", color='gray', linewidth=2)
                        ax.plot(currency_summary["Month"], currency_summary["Estimated Cashflow"], 
                               label="Forecast", color='blue', linewidth=2, marker='o')
                        
                        if scenario_planning and scenario_data:
                            combined_scenarios = pd.concat(scenario_data)
                            scenario_summary = combined_scenarios.groupby("Month")[["Optimistic", "Realistic", "Pessimistic"]].sum()
                            ax.fill_between(scenario_summary.index, 
                                          scenario_summary["Pessimistic"], 
                                          scenario_summary["Optimistic"], 
                                          alpha=0.3, color='lightblue', label='Confidence Band')
                            ax.plot(scenario_summary.index, scenario_summary["Optimistic"], 
                                   '--', color='green', label='Optimistic')
                            ax.plot(scenario_summary.index, scenario_summary["Pessimistic"], 
                                   '--', color='red', label='Pessimistic')
                        
                        ax.set_title(f"Enhanced 3-Month Cashflow Forecast for {currency}", fontsize=14, fontweight='bold')
                        ax.set_xlabel("Month")
                        ax.set_ylabel("Estimated Cashflow")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

                # Model Performance Dashboard
                if model_performance:
                    st.subheader("🎯 Model Performance Dashboard")
                    
                    performance_data = []
                    for series, data in model_performance.items():
                        for model_name, metrics in data['metrics'].items():
                            performance_data.append({
                                'Series': series,
                                'Model': model_name,
                                'MAE': metrics['MAE'],
                                'RMSE': metrics['RMSE'],
                                'MAPE (%)': metrics['MAPE'],
                                'R²': metrics['R²'],
                                'Best': '✅' if model_name == data['best_model'] else ''
                            })
                    
                    performance_df = pd.DataFrame(performance_data)
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Best models summary
                    best_models = [data['best_model'] for data in model_performance.values()]
                    best_model_counts = pd.Series(best_models).value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🏆 Best Model Distribution**")
                        for model, count in best_model_counts.items():
                            st.metric(model, f"{count} series")
                    
                    with col2:
                        avg_metrics = performance_df.groupby('Model')[['MAE', 'RMSE', 'MAPE (%)', 'R²']].mean()
                        st.markdown("**📈 Average Model Performance**")
                        st.dataframe(avg_metrics)

                # Model logs and download
                with st.sidebar.expander("📝 Model Logs"):
                    for log in model_logs:
                        st.write(log)

                if forecast_outputs:
                    full_result = pd.concat(forecast_outputs)
                    output_buffer = io.BytesIO()
                    
                    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                        full_result.to_excel(writer, sheet_name='Forecasts', index=False)
                        
                        if model_performance:
                            pd.DataFrame(performance_data).to_excel(writer, sheet_name='Model_Performance', index=False)
                        
                        if scenario_planning and scenario_data:
                            pd.concat(scenario_data).to_excel(writer, sheet_name='Scenarios', index=False)
                    
                    st.download_button(
                        "📥 Download Enhanced Forecasts as Excel", 
                        data=output_buffer.getvalue(), 
                        file_name="enhanced_forecast_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    # Aggregate View by Country
                    st.subheader("📋 Aggregate Forecast per Country (All Components)")
                    agg_country = full_result.groupby(["Month", "Country"])["Estimated Cashflow"].sum().reset_index()
                    st.dataframe(agg_country, use_container_width=True)
                   

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}. Common issues include: wrong date format, missing required columns, or non-numeric values in 'Amount'.")
else:
    st.info("👆 Please upload a CSV/Excel file or generate demo data to begin forecasting.")
    
    # Demo data showcase
    st.markdown("## 🎯 Try the Demo!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎲 Generate Sample Data
        Use the sidebar to generate realistic demo data:
        1. Select countries, components, and currencies
        2. Choose years of historical data (1-5 years)
        3. Pick business type for realistic patterns:
           - **Stable**: Steady 3% growth
           - **Seasonal**: Holiday/summer variations
           - **High Growth**: 15% annual growth
           - **Weather Dependent**: Strong quarterly patterns
        4. Click "Generate Demo Data"
        5. Download or use directly in the app
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Required Data Format
        Your file must contain these columns:
        - **Date**: Date column (YYYY-MM-DD format)
        - **Country**: Country identifier  
        - **Component**: Cost component (e.g., Salaries, Rent)
        - **Amount**: Monetary amount
        - **Currency**: Currency code (e.g., EUR, USD)
        
        Optional: **EmployeeCount** for enhanced modeling
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🚀 **New Enhanced Features**
    
    ### 🎲 **Smart Data Simulation**
    - Automatically detects businesses with limited historical data (<5 years)
    - Generates realistic synthetic data based on your patterns
    - Preserves seasonal trends and growth patterns
    - Configurable simulation period (1-3 additional years)
    
    ### 🌤️ **Weather-Dependent Business Support**
    - Special seasonality adjustments for weather-sensitive industries
    - Enhanced quarterly pattern recognition
    - Optimized for retail, tourism, agriculture, and outdoor services
    
    ### 🎯 **Advanced Scenario Planning**
    - **Optimistic Scenario**: Upper confidence forecast
    - **Realistic Scenario**: Most likely outcome
    - **Pessimistic Scenario**: Lower confidence forecast
    - 80% confidence intervals for robust planning
    - Uses historical forecast errors for accuracy
    
    ### 📊 **Comprehensive Model Analytics**
    - **MAE** (Mean Absolute Error): Average prediction error
    - **RMSE** (Root Mean Square Error): Penalizes large errors
    - **MAPE** (Mean Absolute Percentage Error): Percentage-based accuracy
    - **R²** (R-squared): Explained variance measure
    - Automatic best model selection per time series
    - Performance comparison dashboard
    
    ### 🎨 **Enhanced Visualizations**
    - Confidence bands showing forecast uncertainty
    - Scenario planning with optimistic/pessimistic bounds
    - Interactive charts with improved styling
    - Multi-sheet Excel exports with all analysis data
    """)
    
    # Show sample demo data structure
    st.markdown("### 📝 Sample Demo Data Structure")
    sample_data = pd.DataFrame({
        'Date': ['2022-01-01', '2022-02-01', '2022-03-01'],
        'Country': ['Netherlands', 'Netherlands', 'Germany'],
        'Component': ['Salaries', 'Rent', 'Salaries'],
        'Amount': [125000.50, 15000.00, 98000.75],
        'Currency': ['EUR', 'EUR', 'EUR'],
        'EmployeeCount': [45, None, 38]
    })
    st.dataframe(sample_data, use_container_width=True)



