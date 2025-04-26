import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Page config
st.set_page_config(page_title="🏏 IPL 2025 Dashboard", layout="wide", page_icon="🏆")
st.markdown("""
    <h1 style='text-align: center; font-size: 3em;'>🔥 IPL 2025 Interactive Dashboard 🔥</h1>
    <p style='text-align: center; font-size: 1.2em;'>Track players, predict matches, compare stats and relive historic moments</p>
""", unsafe_allow_html=True)

# Load data
df = pd.read_csv("cricket_data_2025.csv")
matches = pd.read_csv("matches.csv")
deliveries_df = pd.read_csv("compressed_delivery.csv.gz")

# Team colors and logos
team_colors = {
    'Chennai Super Kings': '#f1c40f',
    'Royal Challengers Bangalore': '#c0392b',
    'Mumbai Indians': '#2980b9',
    'Sunrisers Hyderabad': '#e67e22',
    'Kolkata Knight Riders': '#8e44ad',
    'Delhi Capitals': '#3498db',
    'Punjab Kings': '#e74c3c',
    'Rajasthan Royals': '#ff69b4',
    'Lucknow Super Giants': '#5dade2',
    'Gujarat Titans': '#34495e'
}
logo_path = "team_logos"
team_logos = {team: os.path.join(logo_path, f"{team.replace(' ', '_')}.png") for team in team_colors}

# Clean data
df.replace("No stats", pd.NA, inplace=True)
df.dropna(subset=["Player_Name"], inplace=True)
numerical_cols = ["Matches_Batted", "Not_Outs", "Runs_Scored", "Balls_Faced", "Batting_Average", "Batting_Strike_Rate",
                  "Centuries", "Matches_Bowled", "Balls_Bowled", "Runs_Conceded", "Wickets_Taken", "Bowling_Average",
                  "Economy_Rate", "Bowling_Strike_Rate", "Four_Wicket_Hauls", "Five_Wicket_Hauls"]
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Sidebar
st.sidebar.title("🏏 Filters & Features")
selected_view = st.sidebar.radio("Choose View", ["Player Analysis", "Match Insights", "Batsman vs Bowler", "Team Stats", "Toss Predictor", "Predict Match Winner"])

# Player Analysis
if selected_view == "Player Analysis":
    st.header("🔎 Player Performance Explorer")
    player_type = st.selectbox("Select Player Type", ["All", "Batsmen", "Bowlers"])

    if player_type == "Batsmen":
        filtered_df = df[df["Runs_Scored"] > 0]
    elif player_type == "Bowlers":
        filtered_df = df[df["Wickets_Taken"] > 0]
    else:
        filtered_df = df.copy()

    selected_players = st.multiselect("Select Players", options=filtered_df["Player_Name"].unique())
    if selected_players:
        filtered_df = filtered_df[filtered_df["Player_Name"].isin(selected_players)]

    if not filtered_df.empty:
        st.subheader("📊 Top Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Most Runs", filtered_df.sort_values(by="Runs_Scored", ascending=False).iloc[0]["Runs_Scored"])
        with col2:
            st.metric("Best Strike Rate", round(filtered_df["Batting_Strike_Rate"].max(), 2))
        with col3:
            st.metric("Most Wickets", filtered_df.sort_values(by="Wickets_Taken", ascending=False).iloc[0]["Wickets_Taken"])
        with col4:
            st.metric("Best Economy", round(filtered_df["Economy_Rate"].min(), 2))

        st.subheader("📈 Top Batsmen")
        top_batsmen = filtered_df.sort_values(by="Runs_Scored", ascending=False).head(10)
        fig1 = px.bar(top_batsmen, x="Runs_Scored", y="Player_Name", orientation="h",
                      title="Top 10 Batsmen", text="Runs_Scored", color="Runs_Scored", color_continuous_scale="Viridis")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📈 Top Bowlers")
        top_bowlers = filtered_df.sort_values(by="Wickets_Taken", ascending=False).head(10)
        fig2 = px.bar(top_bowlers, x="Wickets_Taken", y="Player_Name", orientation="h",
                      title="Top 10 Bowlers", text="Wickets_Taken", color="Wickets_Taken", color_continuous_scale="Cividis")
        st.plotly_chart(fig2, use_container_width=True)

# Match Insights
elif selected_view == "Match Insights":
    st.header("📊 IPL Match Highlights & Venue Map")

    st.subheader("🎞️ Select a Match to View Highlights")
    match_id = st.selectbox("Choose Match ID", matches["id"].unique())
    match_info = matches[matches["id"] == match_id].iloc[0]
    st.markdown(f"**Match**: {match_info['team1']} vs {match_info['team2']}")
    st.markdown(f"**Winner**: 🏆 {match_info['winner']}")
    st.markdown(f"**Venue**: {match_info['venue']}, {match_info['city']}")
    st.markdown(f"**Player of the Match**: 🌟 {match_info['player_of_match']}")

    st.subheader("📈 Match Momentum")
    delivery_subset = deliveries_df[deliveries_df["match_id"] == match_id]
    momentum_df = delivery_subset.groupby(["inning", "over"]).agg({"total_runs": "sum"}).reset_index()
    fig_momentum = px.line(momentum_df, x="over", y="total_runs", color="inning",
                           labels={"total_runs": "Runs Scored", "over": "Over", "inning": "Inning"},
                           title="Run Progression per Over")
    st.plotly_chart(fig_momentum, use_container_width=True)

    st.subheader("📍 IPL Venues on Map")
    stadium_locations = {
        "Mumbai": [19.0760, 72.8777],
        "Delhi": [28.6139, 77.2090],
        "Bangalore": [12.9716, 77.5946],
        "Kolkata": [22.5726, 88.3639],
        "Chennai": [13.0827, 80.2707],
        "Hyderabad": [17.3850, 78.4867],
        "Ahmedabad": [23.0225, 72.5714],
        "Jaipur": [26.9124, 75.7873]
    }
    venues = matches[['venue', 'city']].drop_duplicates()
    venues['lat'] = venues['city'].map(lambda x: stadium_locations.get(x, [0, 0])[0])
    venues['lon'] = venues['city'].map(lambda x: stadium_locations.get(x, [0, 0])[1])
    fig_map = px.scatter_mapbox(venues, lat='lat', lon='lon', hover_name='venue', hover_data=['city'],
                                color_discrete_sequence=["fuchsia"], zoom=4, center={"lat": 21.0, "lon": 78.0}, height=400)
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

# Batsman vs Bowler
elif selected_view == "Batsman vs Bowler":
    st.header("🎯 Batsman vs Bowler Duel")
    selected_batsman = st.selectbox("Select Batsman", deliveries_df["batter"].dropna().unique())
    selected_bowler = st.selectbox("Select Bowler", deliveries_df["bowler"].dropna().unique())

    duel_df = deliveries_df[(deliveries_df["batter"] == selected_batsman) & (deliveries_df["bowler"] == selected_bowler)]

    if not duel_df.empty:
        total_balls = duel_df.shape[0]
        total_runs = duel_df["batsman_runs"].sum()
        dismissals = duel_df["player_dismissed"].eq(selected_batsman).sum()
        avg = total_runs / dismissals if dismissals > 0 else "Not out"
        total_fours = duel_df[duel_df["batsman_runs"] == 4].shape[0]
        total_sixes = duel_df[duel_df["batsman_runs"] == 6].shape[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Balls Faced", total_balls)
            st.metric("Fours", total_fours)
        with col2:
            st.metric("Runs Scored", total_runs)
            st.metric("Sixes", total_sixes)
        with col3:
            st.metric("Dismissals", dismissals)
            st.metric("Average vs Bowler", avg)

        duel_summary = duel_df.groupby("dismissal_kind")["is_wicket"].sum().reset_index()
        if not duel_summary.empty:
            fig_duel = px.pie(duel_summary, values="is_wicket", names="dismissal_kind",
                              title=f"{selected_batsman} Dismissals by {selected_bowler}")
            st.plotly_chart(fig_duel, use_container_width=True)
    else:
        st.warning("No direct face-offs found between this batsman and bowler.")

# Team Comparison
elif selected_view == "Team Stats":
    st.header("🤜🤛 Team Comparison")
    team1 = st.selectbox("Select Team 1", matches["team1"].dropna().unique(), key="team1")
    team2 = st.selectbox("Select Team 2", matches["team2"].dropna().unique(), key="team2")

    if team1 != team2:
        team1_wins = matches[matches["winner"] == team1].shape[0]
        team2_wins = matches[matches["winner"] == team2].shape[0]
        team1_poms = matches[matches["winner"] == team1]["player_of_match"].value_counts().head(1)
        team2_poms = matches[matches["winner"] == team2]["player_of_match"].value_counts().head(1)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏆 Wins", team1_wins)
            if not team1_poms.empty:
                st.metric("🌟 Top PoM", team1_poms.index[0])
        with col2:
            st.metric("🏆 Wins", team2_wins)
            if not team2_poms.empty:
                st.metric("🌟 Top PoM", team2_poms.index[0])

        st.subheader("📊 Comparison Chart")
        fig_comp = go.Figure(data=[
            go.Bar(name=team1, x=["Wins"], y=[team1_wins], marker_color=team_colors.get(team1, "gray")),
            go.Bar(name=team2, x=["Wins"], y=[team2_wins], marker_color=team_colors.get(team2, "gray"))
        ])
        fig_comp.update_layout(barmode='group', title="Team Wins Comparison")
        st.plotly_chart(fig_comp, use_container_width=True)

# Toss Predictor
elif selected_view == "Toss Predictor":
    st.header("🧠 Toss-Based Match Outcome Predictor")

    toss_winner = st.selectbox("Toss Winner", matches["team1"].dropna().unique(), key="toss_team")
    decision = st.radio("Decision After Toss", ["bat", "field"])

    filtered = matches[(matches["toss_winner"] == toss_winner) & (matches["toss_decision"] == decision)]
    win_count = (filtered["winner"] == toss_winner).sum()
    total = filtered.shape[0]
    win_rate = (win_count / total * 100) if total > 0 else 0

    st.metric("Predicted Win %", f"{win_rate:.2f}%")
    st.progress(win_rate / 100)

    # 📅 Animated Match Timelines
    st.markdown("### 📽️ Match Wins Animated by Season")
    animation_df = matches.groupby(["season", "winner"]).size().reset_index(name="Wins")
    animation_df = animation_df[animation_df["winner"].notna()]
    fig_anim = px.bar(animation_df, x="Wins", y="winner", color="winner",
                      animation_frame="season", orientation="h",
                      title="Animated: IPL Team Wins Over Seasons",
                      color_discrete_map=team_colors)
    fig_anim.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_anim, use_container_width=True)

# Predict Match Winner
elif selected_view == "Predict Match Winner":
    st.header("🔮 Predict Upcoming Match Winner")
    upcoming_schedule = pd.read_csv("2025_IPL_Schedule.csv")
    upcoming_match = st.selectbox("Select Upcoming Match", upcoming_schedule["Home Team"] + " vs " + upcoming_schedule["Away Team"])
    team1_col = upcoming_schedule["Home Team"] + " vs " + upcoming_schedule["Away Team"]
    match_row = upcoming_schedule[team1_col == upcoming_match].iloc[0]

    team1 = match_row["Home Team"]
    team2 = match_row["Away Team"]
    st.markdown(f"#### Match: {team1} vs {team2}")

    # Simple rule-based score model
    def recent_form(team):
        last_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)].sort_values("date", ascending=False).head(5)
        wins = (last_matches["winner"] == team).sum()
        return wins / len(last_matches) if len(last_matches) > 0 else 0

    team1_form = recent_form(team1)
    team2_form = recent_form(team2)

    st.subheader("📊 Team Form (Last 5 Matches)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{team1} Form", value=f"{team1_form * 100:.1f}%")
    with col2:
        st.metric(label=f"{team2} Form", value=f"{team2_form * 100:.1f}%")

    # Simple scoring: form * 0.7 + random factor (or batting avg)
    team1_score = team1_form * 0.7
    team2_score = team2_form * 0.7

        # Head-to-head record
    head_to_head = matches[((matches["team1"] == team1) & (matches["team2"] == team2)) | ((matches["team1"] == team2) & (matches["team2"] == team1))]
    head_wins_team1 = (head_to_head["winner"] == team1).sum()
    head_wins_team2 = (head_to_head["winner"] == team2).sum()
    total_head = head_wins_team1 + head_wins_team2

    st.subheader("📚 Head-to-Head Record")
    col3, col4 = st.columns(2)
    with col3:
        st.metric(f"{team1} Wins", head_wins_team1)
    with col4:
        st.metric(f"{team2} Wins", head_wins_team2)

    # Optional: include head-to-head weight
    team1_score += (head_wins_team1 / total_head if total_head > 0 else 0) * 0.3
    team2_score += (head_wins_team2 / total_head if total_head > 0 else 0) * 0.3

    winner = team1 if team1_score > team2_score else team2
    st.success(f"🏆 Predicted Winner: {winner}")

    # 🔥 Recent Top 3 Performers
    st.subheader("🔥 Recent Top Performers (Last 5 Matches)")
    recent = matches.sort_values("date", ascending=False)
    recent_players = recent.head(5)["player_of_match"]
    top_players = recent_players.value_counts().head(3).index.tolist()
    for player in top_players:
        perf = df[df["Player_Name"] == player]
        if not perf.empty:
            st.markdown(f"**{player}** - Runs: {perf['Runs_Scored'].values[0]}, Wickets: {perf['Wickets_Taken'].values[0]}")

