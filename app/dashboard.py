import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="SD Airbnb Pricing", page_icon="🏠", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

@st.cache_data(ttl=3600)
def load_data():
    forecast  = pd.read_csv(os.path.join(DATA_DIR, "forecast.csv"), parse_dates=["date"])
    by_room   = pd.read_csv(os.path.join(DATA_DIR, "forecast_by_room.csv"), parse_dates=["date"])
    listings  = pd.read_csv(os.path.join(DATA_DIR, "listings_clean.csv"))
    return forecast, by_room, listings

forecast, by_room, listings = load_data()
future = forecast[forecast["date"] > pd.Timestamp.now()].copy()

# ── Header ────────────────────────────────────────────────────────────────
st.title("🏠 San Diego Airbnb — Dynamic Pricing Engine")
st.caption(f"Powered by Meta Prophet  •  {len(listings):,} active listings  •  Forecast through {forecast['date'].max().strftime('%b %Y')}")

# ── KPIs ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

peak     = future.loc[future["predicted_price"].idxmax()]
low      = future.loc[future["predicted_price"].idxmin()]
summer   = future[future["month"].isin([6,7,8])]["predicted_price"].mean()
winter   = future[future["month"].isin([12,1,2])]["predicted_price"].mean()
swing    = round((summer - winter) / winter * 100, 1)

k1.metric("📅 Peak Night",     peak["date"].strftime("%b %d"),  f"${peak['predicted_price']:.0f}/night")
k2.metric("📉 Lowest Night",   low["date"].strftime("%b %d"),   f"${low['predicted_price']:.0f}/night")
k3.metric("☀️ Summer Avg",     f"${summer:.0f}/night")
k4.metric("❄️ Winter Avg",     f"${winter:.0f}/night")
k5.metric("📊 Seasonal Swing", f"{swing}%",                     "summer vs winter")

st.divider()

# ── Row 1: Forecast line + seasonal bar ──────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Price Forecast — Next 12 Months")
    plot_fc = future[future["date"] <= pd.Timestamp.now() + pd.DateOffset(months=12)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_fc["date"], y=plot_fc["price_upper"],
        fill=None, mode="lines", line=dict(width=0),
        showlegend=False, name="Upper bound"
    ))
    fig.add_trace(go.Scatter(
        x=plot_fc["date"], y=plot_fc["price_lower"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(255,100,100,0.15)", name="Confidence band"
    ))
    fig.add_trace(go.Scatter(
        x=plot_fc["date"], y=plot_fc["predicted_price"],
        mode="lines", line=dict(color="#E63946", width=2),
        name="Predicted price"
    ))

    # Mark Comic-Con
    comic_con = plot_fc[plot_fc["event"] == "Comic-Con Season"]
    if not comic_con.empty:
        peak_cc = comic_con.loc[comic_con["predicted_price"].idxmax()]
        fig.add_annotation(
            x=peak_cc["date"], y=peak_cc["predicted_price"],
            text="🎭 Comic-Con", showarrow=True,
            arrowhead=2, ax=0, ay=-40, font=dict(size=11)
        )

    fig.update_layout(
        height=380, yaxis_title="Predicted Nightly Rate ($)",
        xaxis_title="", hovermode="x unified",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Average Price by Month")
    monthly = (future.groupby("month_name")
               .agg(avg=("predicted_price","mean"),
                    month_num=("month","first"))
               .reset_index()
               .sort_values("month_num"))

    colors = ["#E63946" if v == monthly["avg"].max()
              else "#00B4D8" if v == monthly["avg"].min()
              else "#2DC653" for v in monthly["avg"]]

    fig2 = go.Figure(go.Bar(
        x=monthly["month_name"], y=monthly["avg"],
        marker_color=colors, text=monthly["avg"].apply(lambda v: f"${v:.0f}"),
        textposition="outside"
    ))
    fig2.update_layout(height=380, yaxis_title="Avg Nightly Rate ($)",
                       xaxis_title="", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Room type forecast + Neighbourhood map ────────────────────────
col3, col4 = st.columns([2, 3])

with col3:
    st.subheader("Price by Room Type")
    room_future = by_room[by_room["date"] > pd.Timestamp.now()].copy()
    room_monthly = (room_future.groupby(["room_type","month_name","month"])
                    ["predicted_price"].mean().reset_index()
                    .sort_values("month"))

    fig3 = px.line(
        room_monthly, x="month_name", y="predicted_price",
        color="room_type", markers=True,
        labels={"predicted_price":"Avg Price ($)","month_name":"","room_type":"Room Type"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig3.update_layout(height=340, hovermode="x unified",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Listing Prices by Neighbourhood")
    neigh = (listings.groupby("neighbourhood_cleansed")
             .agg(avg_price=("price_clean","mean"),
                  count=("id","count"),
                  lat=("latitude","mean"),
                  lon=("longitude","mean"))
             .reset_index()
             .rename(columns={"neighbourhood_cleansed":"neighbourhood"}))
    neigh = neigh[neigh["count"] >= 5]

    fig4 = px.scatter_mapbox(
        neigh,
        lat="lat", lon="lon",
        color="avg_price", size="count",
        hover_name="neighbourhood",
        hover_data={"avg_price": True, "count": True, "lat": False, "lon": False},
        color_continuous_scale="RdYlGn_r",
        size_max=20, zoom=10,
        center={"lat": 32.72, "lon": -117.15},
        mapbox_style="carto-positron",
        labels={"avg_price":"Avg Price ($)","count":"Listings"}
    )
    fig4.update_layout(height=340, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Event calendar + listings distribution ────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Upcoming Price Peaks — Event Calendar")
    events_df = (future[future["event"] != "No Major Event"]
                 .groupby(["event","month_name","month"])
                 ["predicted_price"].max().reset_index()
                 .sort_values("month"))
    fig5 = px.bar(
        events_df, x="month_name", y="predicted_price",
        color="event", text="predicted_price",
        labels={"predicted_price":"Peak Price ($)","month_name":"","event":"Event"},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig5.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig5.update_layout(height=320, showlegend=True,
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("Listing Price Distribution")
    fig6 = px.histogram(
        listings, x="price_clean", nbins=60,
        color="room_type",
        labels={"price_clean":"Nightly Rate ($)","room_type":"Room Type"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        range_x=[0, 800]
    )
    fig6.update_layout(height=320, barmode="overlay",
                       legend=dict(orientation="h", y=1.05))
    fig6.update_traces(opacity=0.7)
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── Neighbourhood deep-dive table ─────────────────────────────────────────
st.subheader("🔍 Neighbourhood Breakdown")

col_s, col_f1, col_f2 = st.columns([3, 1, 1])
with col_s:
    search = st.text_input("Search neighbourhood", placeholder="e.g. La Jolla, Mission, Downtown...",
                           label_visibility="collapsed")
with col_f1:
    room_filter = st.selectbox("Room type", ["All"] + sorted(listings["room_type"].dropna().unique()),
                               label_visibility="collapsed")
with col_f2:
    sort_by = st.selectbox("Sort by", ["avg_price","count","avg_reviews"],
                           label_visibility="collapsed")

neigh_detail = (listings.groupby(["neighbourhood_cleansed","room_type"])
                .agg(avg_price=("price_clean","mean"),
                     median_price=("price_clean","median"),
                     count=("id","count"),
                     avg_reviews=("review_scores_rating","mean"),
                     avg_availability=("availability_365","mean"))
                .reset_index()
                .rename(columns={"neighbourhood_cleansed":"neighbourhood"}))

neigh_detail = neigh_detail[neigh_detail["count"] >= 3]

if search:
    neigh_detail = neigh_detail[neigh_detail["neighbourhood"].str.contains(search, case=False)]
if room_filter != "All":
    neigh_detail = neigh_detail[neigh_detail["room_type"] == room_filter]

neigh_detail = neigh_detail.sort_values(sort_by, ascending=False).reset_index(drop=True)
for col in ["avg_price","median_price","avg_reviews","avg_availability"]:
    neigh_detail[col] = neigh_detail[col].round(1)

st.caption(f"Showing {len(neigh_detail)} neighbourhood / room-type combinations")
st.dataframe(neigh_detail, use_container_width=True, height=400)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Market Summary")
    st.markdown(f"**Listings:** {len(listings):,}")
    st.markdown(f"**Neighbourhoods:** {listings['neighbourhood_cleansed'].nunique()}")
    st.markdown(f"**Room types:** {listings['room_type'].nunique()}")
    st.markdown(f"**Avg price:** ${listings['price_clean'].mean():.0f}/night")
    st.markdown(f"**Median price:** ${listings['price_clean'].median():.0f}/night")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("- 🔮 Meta Prophet")
    st.markdown("- 📅 Trained 2022–2025")
    st.markdown("- 🎭 6 SD events modeled")
    st.markdown("---")
    st.markdown("**Data Source**")
    st.markdown("- [Inside Airbnb](http://insideairbnb.com)")
    st.markdown("- Updated monthly")
    st.markdown("---")
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()
