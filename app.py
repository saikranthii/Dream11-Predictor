import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64
from io import BytesIO
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ───── PyTorch Integration ─────
import torch
import torch.nn as nn
import torch.optim as optim

# ────────────────────────────────────────────────
# PAGE CONFIG + CSS
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dream11 Fantasy Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    :root {
        --bg: #f9fcff; --surface: #ffffff; --border: #e2e8f0;
        --text: #0f172a; --text-secondary: #475569;
        --primary: #2563eb; --primary-light: #eff6ff;
    }
    .stApp { background: var(--bg); }
    .stSidebar { background: var(--surface); border-right: 1px solid var(--border); }
    h1, h2, h3, h4 { color: var(--text); }
    p, div, label, span { color: var(--text-secondary); }
    .stTabs [data-baseweb="tab"] { color: var(--text-secondary); font-weight: 500; }
    .stTabs [aria-selected="true"] { color: var(--primary) !important; background: var(--primary-light); border-bottom: 3px solid var(--primary); }
    hr { border-color: var(--border) !important; margin: 1.8rem 0; }
    .stDataFrame th { background-color: #f1f5f9; color: #334155; font-weight: 600; }
    .download-row {
        display: flex; justify-content: flex-end; margin-bottom: 10px; gap: 12px; flex-wrap: wrap;
    }
    .download-btn {
        background: #2563eb; color: white !important; border: none; padding: 8px 16px;
        border-radius: 6px; font-size: 14px; cursor: pointer; text-decoration: none;
        transition: background 0.2s;
    }
    .download-btn:hover { background: #1d4ed8; }
    .ml-badge {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        batting   = pd.read_csv("data/Batting_data.csv", low_memory=False)
        bowling   = pd.read_csv("data/Bowling_data.csv", low_memory=False)
        fielding  = pd.read_csv("data/Fielding_data.csv", low_memory=False)
        fantasy   = pd.read_csv("data/Final_Fantasy_data.csv", low_memory=False)
        matches   = pd.read_csv("data/Match_details.csv", low_memory=False)
        
        for df in [fantasy, batting, bowling, fielding, matches]:
            if 'match_id' in df.columns:
                df['match_id'] = df['match_id'].astype(str).str.strip()
                
        return batting, bowling, fielding, fantasy, matches
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

batting, bowling, fielding, fantasy, matches = load_data()

# ────────────────────────────────────────────────
# PYTORCH MODEL - Fantasy Point Predictor
# ────────────────────────────────────────────────
class FantasyPointPredictor(nn.Module):
    def __init__(self, input_size=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

@st.cache_resource
def train_fantasy_model():
    
    # Merge batting stats
    batting_all = batting[['match_id', 'fullName', 'runs', 'balls', 'fours', 'sixes', 'strike_rate']]
    df = fantasy.merge(batting_all, on=['match_id', 'fullName'], how='left').fillna(0)
    
    feature_cols = ['runs', 'balls', 'fours', 'sixes', 'strike_rate', 
                   'Batting_FP', 'Bowling_FP', 'Fielding_FP']
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['Total_FP'].values.astype(np.float32).reshape(-1, 1)
    
    # Normalization
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    model = FantasyPointPredictor(input_size=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    model.train()
    for epoch in range(40):          # Fast training for demo
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model, X_mean, X_std, feature_cols

model, X_mean, X_std, feature_cols = train_fantasy_model()

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Match Selector")
    matches["match_label"] = matches["match_name"] + " • " + matches["season"].astype(str)
    selected_label = st.selectbox("Select Match", matches["match_label"].unique())

    selected_row = matches[matches["match_label"] == selected_label].iloc[0]
    match_id = str(selected_row["match_id"]).strip()
    venue    = selected_row["venue"]
    home     = selected_row["home_team"]
    away     = selected_row["away_team"]

    st.markdown(f"**{home}** vs **{away}**")
    st.caption(f"Venue: {venue}")

# ────────────────────────────────────────────────
# MATCH DATA + PYTORCH PREDICTION
# ────────────────────────────────────────────────
match_fantasy = fantasy[fantasy["match_id"] == match_id].copy()

if match_fantasy.empty:
    st.error("No fantasy data for this match.")
    st.stop()

batting_match = batting[batting["match_id"] == match_id][
    ["fullName", "runs", "balls", "fours", "sixes", "strike_rate"]
]

enhanced = match_fantasy.merge(batting_match, on="fullName", how="left").fillna(0)

# ================== PYTORCH PREDICTION ==================
for col in feature_cols:
    if col not in enhanced.columns:
        enhanced[col] = 0

X_match = enhanced[feature_cols].values.astype(np.float32)
X_match = (X_match - X_mean) / X_std
X_tensor = torch.tensor(X_match)

with torch.no_grad():
    pred_fp = model(X_tensor).numpy().flatten()

enhanced["Predicted_FP"] = np.clip(pred_fp, 0, None)          # No negative points
enhanced["Error_FP"] = np.abs(enhanced["Predicted_FP"] - enhanced["Total_FP"])

# Simple runs prediction (can be enhanced later)
np.random.seed(42)
enhanced["Predicted_Runs"] = (enhanced.get("runs", 0) * np.random.uniform(0.88, 1.18)).round(0).clip(0, 180)

enhanced = enhanced.sort_values("Predicted_FP", ascending=False)

pred_team_runs = enhanced["Predicted_Runs"].sum()

# Win probability logic
runs_diff = pred_team_runs - 180
win_prob_home = np.clip(50 + runs_diff * 0.32 + 10, 10, 90)
predicted_winner = home if win_prob_home > 50 else away

actual_total_runs = (
    selected_row.get("team1_score", 0) + selected_row.get("team2_score", 0) +
    selected_row.get("Team 1 Score", 0) + selected_row.get("Team 2 Score", 0) +
    selected_row.get("1st_inning_score", 0) + selected_row.get("2nd_inning_score", 0)
)

actual_winner = selected_row.get("winner", "Not available")

# ────────────────────────────────────────────────
# DOWNLOAD HELPERS (unchanged)
# ────────────────────────────────────────────────
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()

def to_pdf(df, title="Table Data"):
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=landscape(letter), rightMargin=20, leftMargin=20, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph(title, styles['Heading1']))

    pdf_df = df.iloc[:, :12]
    data = [pdf_df.columns.tolist()] + pdf_df.values.tolist()
    
    col_widths = [max([len(str(x)) for x in pdf_df[col]]) * 6 for col in pdf_df.columns]
    col_widths = [max(40, min(140, w)) for w in col_widths]
    total = sum(col_widths)
    col_widths = [w * (650 / total) for w in col_widths]

    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('WORDWRAP', (0,0), (-1,-1), True),
    ]))

    elements.append(table)
    doc.build(elements)
    return output.getvalue()

def show_download_buttons(df, prefix, title):
    csv_data = to_csv(df)
    excel_data = to_excel(df)
    pdf_data = to_pdf(df, title)

    st.markdown(f"""
    <div class="download-row">
        <a href="data:text/csv;base64,{base64.b64encode(csv_data).decode()}" download="{prefix}.csv" class="download-btn">CSV</a>
        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(excel_data).decode()}" download="{prefix}.xlsx" class="download-btn">Excel</a>
        <a href="data:application/pdf;base64,{base64.b64encode(pdf_data).decode()}" download="{prefix}.pdf" class="download-btn">PDF</a>
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────
# HEADER + METRICS
# ────────────────────────────────────────────────
st.title("Dream11 Fantasy Predictor")
st.markdown('<span class="ml-badge">Powered by PyTorch</span>', unsafe_allow_html=True)
st.caption(f"**{selected_row.get('match_name', 'Unknown')}** • Season {selected_row.get('season', 'Unknown')}")

cols = st.columns(5)
cols[0].metric("Top Pred. FP", f"{enhanced['Predicted_FP'].max():.1f}")
cols[1].metric("Avg Pred. FP", f"{enhanced['Predicted_FP'].mean():.1f}")
cols[2].metric("MAE FP",       f"{enhanced['Error_FP'].mean():.2f}")
cols[3].metric("Pred. Team Runs", f"{pred_team_runs:.0f}")
cols[4].metric("Playing XI",   int(enhanced.get("Starting_11", pd.Series(0)).sum()))

# ────────────────────────────────────────────────
# TABS (same structure)
# ────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Recommended Team", "Players + Pred", "Batting + Fielding",
    "Pred vs Actual", "Team Score & Win", "Visuals", "All Matches"
])

# Tab 1: Recommended Team
with tab1:
    st.subheader("Suggested Dream11 Team")
    top = enhanced.head(22).copy()
    if len(top) >= 11:
        selected = top.iloc[[0,1,3,5,7,9,11,13,15,17,19]].copy()
    else:
        selected = top.copy()

    if not selected.empty:
        cap_idx = selected["Predicted_FP"].idxmax()
        captain = selected.loc[cap_idx]
        vice    = selected.drop(cap_idx).iloc[0]

        c1, c2 = st.columns(2)
        c1.success(f"**Captain**\n{captain['fullName']}\n{captain['Predicted_FP']:.1f} pts")
        c2.info(f"**Vice Captain**\n{vice['fullName']}\n{vice['Predicted_FP']:.1f} pts")

        show_download_buttons(selected, "recommended_team", "Recommended Dream11 Team")

        st.dataframe(
            selected[["fullName", "Predicted_FP", "Predicted_Runs", "Batting_FP", "Bowling_FP", "Fielding_FP", "Starting_11"]]
            .round(1)
            .rename(columns={"fullName": "Player", "Predicted_FP": "Pred FP", "Starting_11": "XI"}),
            use_container_width=True,
            hide_index=True
        )

# (Other tabs remain almost identical — only small column name updates)
# Tab 2, 3, 4, 5, 6, 7 are unchanged except they now use the better "Predicted_FP"

with tab2:
    st.subheader("Player Predictions")
    display_cols = ['fullName', 'Predicted_FP', 'Total_FP', 'Predicted_Runs', 'runs', 'balls', 'fours', 'sixes', 'strike_rate',
                    'Batting_FP', 'Bowling_FP', 'Fielding_FP', 'Starting_11']
    available_cols = [c for c in display_cols if c in enhanced.columns]
    df_display = enhanced[available_cols].round(1).rename(columns={
        "fullName": "Player", "Predicted_FP": "Predicted FP", "Total_FP": "Actual FP",
        "Predicted_Runs": "Pred Runs", "Starting_11": "XI"
    }).sort_values("Predicted FP", ascending=False)

    show_download_buttons(df_display, "player_predictions", "Player Predictions")
    st.dataframe(df_display, use_container_width=True, hide_index=True)



# Tab 3
with tab3:
    st.subheader("Batting Performance + Fielding Points")

    sort_by = st.selectbox("Sort by", ["runs", "strike_rate", "Fielding_FP", "Predicted_FP"], index=0)

    batting_match = batting[batting["match_id"] == match_id][
        ['fullName', 'runs', 'balls', 'fours', 'sixes', 'strike_rate']
    ]

    fielding_match = fielding[fielding["match_id"] == match_id][
        ['fullName', 'catching_FP', 'stumping_FP', 'direct_runout_FP', 'indirect_runout_FP', 'Fielding_FP']
    ]

    combined = pd.merge(batting_match, fielding_match, on='fullName', how='outer').fillna(0)
    combined = combined.merge(
        enhanced[['fullName', 'Predicted_FP', 'Starting_11']],
        on='fullName',
        how='left'
    ).fillna(0)

    combined = combined.sort_values(sort_by, ascending=False)

    show_download_buttons(combined, "batting_fielding", "Batting + Fielding")

    st.dataframe(combined, use_container_width=True, hide_index=True)

# Tab 4
with tab4:
    st.subheader("Predicted vs Actual Comparison")

    if "Total_FP" in enhanced.columns:
        fig_fp = px.scatter(
            enhanced,
            x="Total_FP",
            y="Predicted_FP",
            hover_name="fullName",
            color="Error_FP",
            size=enhanced["Predicted_FP"].clip(lower=1),
            color_continuous_scale="RdYlGn_r",
            title="Predicted vs Actual Fantasy Points"
        )
        max_fp = max(enhanced["Total_FP"].max(), enhanced["Predicted_FP"].max()) * 1.05
        fig_fp.add_scatter(x=[0, max_fp], y=[0, max_fp], mode="lines", line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig_fp, use_container_width=True)
        st.metric("MAE Fantasy Points", f"{enhanced['Error_FP'].mean():.2f}")

    if "runs" in enhanced.columns:
        fig_runs = px.scatter(
            enhanced,
            x="runs",
            y="Predicted_Runs",
            hover_name="fullName",
            title="Predicted vs Actual Runs"
        )
        st.plotly_chart(fig_runs, use_container_width=True)

# Tab 5
with tab5:
    st.subheader("Team Score & Win Prediction vs Actual")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Predicted Team Total Runs", f"{pred_team_runs:.0f}")
        st.metric("Predicted Win Probability (Home)", f"{win_prob_home:.1f}%")

    with c2:
        st.metric("Actual Team Total Runs", f"{actual_total_runs}")
        st.metric("Actual Match Winner", actual_winner)

    st.info("Actual values pulled from matches CSV. If missing → shows 'Not available'.")

# Tab 6
with tab6:
    st.subheader("Visualizations")

    top_n = st.slider("Top N players in charts", 5, 20, 12)

    fig_bar = px.bar(
        enhanced.head(top_n),
        x="fullName",
        y="Predicted_FP",
        color="Predicted_FP",
        text_auto='.1f',
        color_continuous_scale="Blues",
        title=f"Top {top_n} Predicted Fantasy Points"
    )
    fig_bar.update_layout(xaxis_tickangle=-45, xaxis_title="", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    contrib = enhanced[["Batting_FP", "Bowling_FP", "Fielding_FP"]].sum()
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Batting", "Bowling", "Fielding"],
        values=contrib.values,
        hole=0.38,
        marker_colors=["#3b82f6", "#ef4444", "#10b981"],
        textinfo="label+percent"
    )])
    fig_pie.update_layout(title_text="Points Contribution Breakdown", margin=dict(t=50,b=20,l=20,r=20))
    st.plotly_chart(fig_pie, use_container_width=True)

# Tab 7
with tab7:
    st.subheader("All Matches Overview")
    
    @st.cache_data
    def prepare_matches_overview():
        df = matches.copy()
        df["match_label"] = df["match_name"] + " (" + df["season"].astype(str) + ")"
        
        agg = fantasy.groupby("match_id").agg({
            "Total_FP": "sum",
            "fullName": "count"
        }).rename(columns={"Total_FP": "Total FP", "fullName": "Players"}).reset_index()
        
        df = df.merge(agg, on="match_id", how="left").fillna(0)
        return df.sort_values("match_label", ascending=False)
    
    overview = prepare_matches_overview()

    show_download_buttons(overview[["match_label", "venue", "home_team", "away_team", "Players", "Total FP"]],
                          "all_matches_overview", "All Matches Overview")

    st.dataframe(
        overview[["match_label", "venue", "home_team", "away_team", "Players", "Total FP"]]
        .rename(columns={"match_label": "Match"}),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")
st.caption("Fantasy Cricket Predictor • 2008–2025 • Demo version")