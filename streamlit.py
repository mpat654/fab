import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from datetime import datetime
import seaborn as sns

# Define brand colors
NAVY = "#222f45"
GREY_BLUE = "#7a828f"
LIGHT_GREY = "#e8e8e7"
WHITE = "#ffffff"
PINK = "#ec6f8f"
LIGHT_PINK = "#f4a9bc"
BLUE = "#8ecae4"
LIGHT_BLUE = "#bbdfef"

# Set page config
st.set_page_config(
    page_title="FDA AI/ML submissions dashboard", page_icon="📊", layout="wide"
)

# Custom CSS
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {WHITE};
        }}
        .main {{
            background-color: {WHITE};
        }}
        h1, h2, h3 {{
            color: {NAVY};
        }}
        .stMarkdown {{
            color: {GREY_BLUE};
        }}
        .stSidebar {{
            background-color: {LIGHT_GREY};
        }}
        .stMetric {{
            background-color: {LIGHT_BLUE};
            border-radius: 5px;
            padding: 10px;
        }}
        .stMetric label {{
            color: {NAVY};
        }}
        .stMetric .value {{
            color: {NAVY};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background-color: {LIGHT_GREY};
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {LIGHT_BLUE};
            color: {NAVY};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {BLUE};
        }}
        .stButton > button {{
            background-color: {PINK};
            color: {WHITE};
        }}
        .stButton > button:hover {{
            background-color: {LIGHT_PINK};
        }}
    </style>
""",
    unsafe_allow_html=True,
)

# Title and description
st.title("FDA AI/ML submissions dashboard")
st.markdown("""
This dashboard visualizes FDA AI/ML device submissions data, showing trends in medical specialties, 
device types, and submission patterns over time.

The dashboard allows you to explore:
- How many AI/ML medical devices are submitted each year
- Which medical specialties are using AI/ML technology the most
- What types of devices are being developed with AI/ML
- Detailed data about each submission

To do:
- Add DIGA
- Add DTX
- Add ability to filter by year range for device names
- Implement K submission search functionality
- Reference DTX and DIGA papers
- Add more detailed DTx search capabilities 
- Extract evidence details and company information from submission papers
- Integrate data from DTx Alliance product library: https://dtxalliance.org/understanding-dtx/product-library/
- Cross-reference with Guidea's FDA-cleared SaMD list: https://guidea.com/insights/the-first-500-fda-cleared-samd
- Reference: https://orthogonal.io/insights/samd/help-us-build-an-authoritative-list-of-samd-cleared-by-the-fda/
""")

# Add link to original analysis
st.markdown("""
You can also view the [original analysis](device_submissions.html) that was used to create this dashboard.
""")


# Set default matplotlib style to match branding
# Fix: Use sns.set_theme() instead of plt.style.use("seaborn")
sns.set_theme()
plt.rcParams["axes.facecolor"] = WHITE
plt.rcParams["figure.facecolor"] = WHITE
plt.rcParams["axes.edgecolor"] = NAVY
plt.rcParams["axes.labelcolor"] = NAVY
plt.rcParams["text.color"] = NAVY
plt.rcParams["xtick.color"] = NAVY
plt.rcParams["ytick.color"] = NAVY
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[PINK, BLUE, NAVY, GREY_BLUE, LIGHT_PINK]
)


# Load Data
@st.cache_data
def load_data():
    """Load and prepare data for the dashboard"""
    try:
        # Load device classification data
        with open("data/device-classification-0001-of-0001.json", "r") as f:
            classification_data = json.load(f)

        # Convert JSON results to dataframe
        df_class = pd.json_normalize(classification_data["results"])

        # Create refined classification dataframe
        df_class_refined = df_class[
            [
                "product_code",
                "medical_specialty_description",
                "device_class",
                "device_name",
                "definition",
            ]
        ]

        # Load AI devices data
        ai_devices = pd.read_csv("data/ai_devices_with_class.csv", encoding="latin1")

        # Convert date column to datetime
        ai_devices["Date of Final Decision"] = pd.to_datetime(
            ai_devices["Date of Final Decision"]
        )

        return df_class_refined, ai_devices

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


# Load data
df_class_refined, ai_devices = load_data()

# Check if data loaded successfully
if df_class_refined is None or ai_devices is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Sidebar for filters
st.sidebar.header("Filters")
st.sidebar.markdown("""
Use these filters to focus on specific time periods or medical specialties. 
The dashboard will update automatically when you change these filters.
""")

# Year range filter
current_year = datetime.now().year
min_year = ai_devices["Date of Final Decision"].dt.year.min()
max_year = ai_devices["Date of Final Decision"].dt.year.max()

year_range = st.sidebar.slider(
    "Select year range",
    min_value=int(min_year),
    max_value=int(max_year),
    value=(int(min_year), int(max_year)),
    help="Drag the sliders to select a specific time period. This will filter all charts to show data only from your selected years.",
)

# Filter by medical specialty
all_specialties = sorted(ai_devices["medical_specialty_description"].unique())
selected_specialties = st.sidebar.multiselect(
    "Select medical specialties",
    options=all_specialties,
    default=[],
    help="Choose one or more medical specialties to focus on. Leave empty to see all specialties.",
)

# Apply filters
filtered_data = ai_devices.copy()

# Apply year filter
filtered_data = filtered_data[
    (filtered_data["Date of Final Decision"].dt.year >= year_range[0])
    & (filtered_data["Date of Final Decision"].dt.year <= year_range[1])
]

# Apply specialty filter if any selected
if selected_specialties:
    filtered_data = filtered_data[
        filtered_data["medical_specialty_description"].isin(selected_specialties)
    ]

# Display basic stats
st.header("Overview")
st.markdown(
    "These numbers provide a quick summary of the AI/ML devices in your selected time period and specialties:"
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Total submissions",
        len(filtered_data),
        help="The total number of AI/ML device submissions to the FDA in the selected time period",
    )
with col2:
    st.metric(
        "Unique device types",
        len(filtered_data["device_name"].unique()),
        help="The number of different types of AI/ML devices submitted",
    )
with col3:
    st.metric(
        "Medical specialties",
        len(filtered_data["medical_specialty_description"].unique()),
        help="The number of different medical fields using AI/ML devices",
    )

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(
    ["Yearly trends", "Medical specialties", "Device types", "Data explorer"]
)

# Tab 1: Yearly trends
with tab1:
    st.subheader("Submissions by year")
    st.markdown("""
    This chart shows how many AI/ML devices were submitted to the FDA each year. 
    An upward trend indicates growing adoption of AI/ML in medical devices.
    """)

    # Extract year and count submissions per year
    yearly_counts = (
        filtered_data["Date of Final Decision"].dt.year.value_counts().sort_index()
    )

    # Create bar chart
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    yearly_counts.plot(kind="bar", ax=ax1, color=BLUE)
    ax1.set_title("Number of AI/ML device submissions by year")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of submissions")
    ax1.set_xticks(range(len(yearly_counts.index)))
    ax1.set_xticklabels(yearly_counts.index, rotation=0)
    plt.tight_layout()

    st.pyplot(fig1)

    # Year-over-year growth
    if len(yearly_counts) > 1:
        st.subheader("Year-over-year growth")
        st.markdown("""
        This line chart shows the percentage change in submissions from one year to the next.
        """)

        yoy_growth = yearly_counts.pct_change() * 100

        fig_growth, ax_growth = plt.subplots(figsize=(10, 5))
        yoy_growth.plot(kind="line", marker="o", ax=ax_growth, color=PINK)
        ax_growth.set_title("Year-over-year growth in submissions (%)")
        ax_growth.set_xlabel("Year")
        ax_growth.set_ylabel("Growth (%)")
        ax_growth.axhline(y=0, color=GREY_BLUE, linestyle="-", alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig_growth)

# Tab 2: Medical specialties
with tab2:
    st.subheader("Medical specialties distribution")
    st.markdown("""
    This section shows which medical fields are using AI/ML technology the most.
    Use the slider below to control how many specialties you want to see in the chart.
    """)

    # Top N specialties slider
    top_n_specialties = st.slider(
        "Show top N specialties",
        5,
        20,
        10,
        help="Move the slider to show more or fewer medical specialties in the chart below",
    )

    # Create bar plot of medical specialties
    specialty_counts = (
        filtered_data["medical_specialty_description"]
        .value_counts()
        .head(top_n_specialties)
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    specialty_counts.plot(kind="barh", ax=ax2, color=BLUE)
    ax2.set_title(f"Top {top_n_specialties} medical specialties for AI/ML devices")
    ax2.set_xlabel("Number of submissions")
    ax2.set_ylabel("Medical specialty")
    plt.tight_layout()

    st.pyplot(fig2)

    # Medical specialties over time
    st.subheader("Medical specialties trend over time")
    st.markdown("""
    This line chart tracks how the top 5 medical specialties have evolved over time.
    """)

    # Get top 5 specialties
    top_specialties = (
        filtered_data["medical_specialty_description"]
        .value_counts()
        .head(5)
        .index.tolist()
    )

    # Prepare data for line chart
    specialty_by_year = pd.crosstab(
        filtered_data["Date of Final Decision"].dt.year,
        filtered_data["medical_specialty_description"],
    )

    # Filter for top specialties
    if not specialty_by_year.empty and all(
        specialty in specialty_by_year.columns for specialty in top_specialties
    ):
        specialty_by_year = specialty_by_year[top_specialties]

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        specialty_by_year.plot(kind="line", marker="o", ax=ax3)
        ax3.set_title("Top 5 medical specialties trend over time")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Number of submissions")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        st.pyplot(fig3)

# Tab 3: Device types
with tab3:
    st.subheader("Device types analysis")
    st.markdown("""
    This section shows the most common types of medical devices that use AI/ML technology.
    Use the slider below to control how many device types you want to see in the chart.
    """)

    # Top N device types slider
    top_n_devices = st.slider(
        "Show top N device types",
        5,
        20,
        10,
        help="Move the slider to show more or fewer device types in the chart below",
    )

    # Create bar plot of device types
    device_counts = filtered_data["device_name"].value_counts().head(top_n_devices)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    device_counts.plot(kind="barh", ax=ax4, color=BLUE)
    ax4.set_title(f"Top {top_n_devices} AI/ML device types")
    ax4.set_xlabel("Number of submissions")
    ax4.set_ylabel("Device type")
    plt.tight_layout()

    st.pyplot(fig4)

    # Devices accounting for 80% of submissions
    st.subheader("Device types accounting for 80% of submissions")
    st.markdown("""
    This analysis shows which device types make up the majority (80%) of all AI/ML submissions.
    """)

    # Calculate percentages
    device_percentages = (device_counts / device_counts.sum()) * 100
    cumulative_percentages = device_percentages.cumsum()
    devices_80_percent = cumulative_percentages[cumulative_percentages <= 80]

    if not devices_80_percent.empty:
        st.write(
            f"The following {len(devices_80_percent)} device types account for {devices_80_percent.iloc[-1]:.1f}% of submissions:"
        )

        # Create dataframe for display
        devices_80_df = pd.DataFrame(
            {
                "Device type": device_percentages[devices_80_percent.index].index,
                "Percentage": device_percentages[devices_80_percent.index].values,
                "Cumulative percentage": devices_80_percent.values,
            }
        )

        st.dataframe(
            devices_80_df.style.format(
                {"Percentage": "{:.1f}%", "Cumulative percentage": "{:.1f}%"}
            )
        )
    else:
        st.write("No device types found that make up 80% of submissions.")

    # Device class distribution
    st.subheader("Device class distribution")
    st.markdown("""
    - Class I: Low risk devices with minimal regulatory controls
    - Class II: Medium risk devices with more controls
    - Class III: High risk devices with the strictest controls
    
    This chart shows how AI/ML devices are distributed across these classes.
    """)

    device_class_counts = filtered_data["device_class"].value_counts()

    # Create bar chart instead of pie chart for better visualization of imbalanced data
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    device_class_counts.plot(kind="bar", ax=ax5, color=BLUE)

    # Add value labels on top of each bar
    for i, v in enumerate(device_class_counts):
        ax5.text(i, v, str(v), ha="center", va="bottom")

    ax5.set_title("Distribution by device class")
    ax5.set_xlabel("Device class")
    ax5.set_ylabel("Number of devices")
    plt.tight_layout()

    st.pyplot(fig5)

    # Add a text explanation of the distribution
    total_devices = device_class_counts.sum()
    for device_class, count in device_class_counts.items():
        percentage = (count / total_devices) * 100
        st.write(f"**{device_class}**: {count} devices ({percentage:.1f}%)")

# Tab 4: Data explorer
with tab4:
    st.subheader("Data explorer")
    st.markdown("""
    This section allows you to explore the raw data behind the visualizations:
    - The device classification data table shows details about different types of medical devices
    - The AI/ML device submissions table shows specific AI/ML submissions to the FDA
    - You can download the filtered data as a CSV file for further analysis
    """)

    # Display classification data
    st.write("### Device classification data")
    st.dataframe(df_class_refined.head(100))

    # Display AI devices data
    st.write("### AI/ML device submissions")
    st.dataframe(filtered_data)

    # Allow CSV download
    st.download_button(
        label="Download filtered data as CSV",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ai_ml_devices.csv",
        mime="text/csv",
        help="Click to download the current filtered dataset as a CSV file",
    )

# Footer
st.markdown("---")
st.markdown(
    f'<p style="color: {GREY_BLUE};">*FDA AI/ML submissions dashboard - Prova Health 2025*</p>',
    unsafe_allow_html=True,
)
