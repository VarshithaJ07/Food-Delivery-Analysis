import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Page config
st.set_page_config(page_title="🍕 Food Delivery Analytics", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("delivery_data.csv")

df = load_data()

# Train prediction model
X = df[["delivery_time_mins", "distance_km"]]
y = df["rating"]
model = LinearRegression().fit(X, y)

# =========================================
# Sidebar Filters
# =========================================
st.sidebar.header("🔍 Filters")
cuisine_filter = st.sidebar.multiselect(
    "Select cuisines:",
    options=df["cuisine_type"].unique(),
    default=df["cuisine_type"].unique()
)

rating_filter = st.sidebar.slider(
    "Minimum rating:",
    min_value=1,
    max_value=5,
    value=1
)

weather_filter = st.sidebar.multiselect(
    "Weather conditions:",
    options=df["weather_condition"].unique(),
    default=df["weather_condition"].unique()
)

# Apply filters
filtered_df = df[
    (df["cuisine_type"].isin(cuisine_filter)) &
    (df["rating"] >= rating_filter) &
    (df["weather_condition"].isin(weather_filter))
]

# =========================================
# Main Dashboard
# =========================================
st.title("🍔 Food Delivery Performance Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["📍 Map", "📊 Trends", "🔮 Predict", "📈 Raw Data"])

with tab1:  # Map tab
    st.header("Restaurant Locations")
    
    # Create map centered on Bangalore
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
    
    # Add markers with color coding by rating
    for _, row in filtered_df.iterrows():
        color = (
            "green" if row["rating"] >= 4 
            else "orange" if row["rating"] >= 3 
            else "red"
        )
        folium.Marker(
            [row["latitude"], row["longitude"]],
            tooltip=f"{row['restaurant']} | {row['rating']}/5",
            popup=f"""
            <b>Cuisine:</b> {row['cuisine_type']}<br>
            <b>Distance:</b> {row['distance_km']}km<br>
            <b>Time:</b> {row['delivery_time_mins']} mins<br>
            <b>Weather:</b> {row['weather_condition']}
            """,
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    folium_static(m, width=1000, height=500)

with tab2:  # Trends tab
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Delivery Time vs Rating")
        fig1 = px.scatter(
            filtered_df,
            x="delivery_time_mins",
            y="rating",
            color="cuisine_type",
            trendline="lowess"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Avg Rating by Cuisine")
        fig2 = px.box(
            filtered_df,
            x="cuisine_type",
            y="rating",
            color="weather_condition"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Weather Impact")
    fig3 = px.violin(
        filtered_df,
        x="weather_condition",
        y="delivery_time_mins",
        box=True
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:  # Predict tab
    st.header("Rating Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        time = st.slider(
            "Expected delivery time (minutes):",
            min_value=10,
            max_value=120,
            value=30
        )
    
    with col2:
        distance = st.slider(
            "Distance (km):",
            min_value=1,
            max_value=20,
            value=5
        )
    
    predicted_rating = model.predict([[time, distance]])[0]
    st.metric(
        "Predicted Rating", 
        f"{predicted_rating:.1f}/5",
        delta=f"{(predicted_rating-3):.1f} vs average"  # Compared to baseline 3
    )
    
    # Show model coefficients
    st.caption(f"Model equation: Rating = {model.intercept_:.2f} + {model.coef_[0]:.2f}*Time + {model.coef_[1]:.2f}*Distance")

with tab4:  # Raw data tab
    st.header("Raw Data")
    st.dataframe(filtered_df, hide_index=True)
    
    # Download button
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_delivery_data.csv",
        mime="text/csv"
    )

# =========================================
# Footer
# =========================================
st.divider()
st.caption("Data Science Project | Created with Streamlit")