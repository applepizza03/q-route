
import textwrap
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Q-Route: Quantum Logistics Optimizer",
    page_icon="🚆",
    layout="wide"
)


# =========================================================
# Helper
# =========================================================
def html_block(html: str):
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# =========================================================
# Global CSS
# =========================================================
html_block(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 20% 10%, rgba(104, 95, 255, 0.22), transparent 28%),
            radial-gradient(circle at 80% 16%, rgba(0, 188, 212, 0.20), transparent 30%),
            linear-gradient(135deg, #eef4ff 0%, #f7f8ff 55%, #edf7ff 100%);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1220px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #101c3f !important;
    }

    p, div, span, label {
        color: #17213f !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f4f7ff 100%);
        border-right: 2px solid #dbe4ff;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #101c3f !important;
    }

    input, textarea {
        color: #111827 !important;
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }

    div[data-baseweb="select"] span {
        color: #111827 !important;
    }

    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] * {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    ul[role="listbox"] {
        background-color: #ffffff !important;
    }

    ul[role="listbox"] li {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    ul[role="listbox"] li:hover {
        background-color: #eaf0ff !important;
        color: #111827 !important;
    }

    .stSlider label {
        color: #101c3f !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #6d5dfc 0%, #00bcd4 100%);
        color: white !important;
        border: none;
        border-radius: 14px;
        padding: 0.78rem 1rem;
        font-weight: 900;
        letter-spacing: 0.2px;
        box-shadow: 0 8px 22px rgba(93, 88, 255, 0.25);
        transition: 0.12s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 11px 26px rgba(93, 88, 255, 0.34);
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe4ff;
        padding: 16px;
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(30, 41, 59, 0.08);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #101c3f !important;
    }

    div[data-testid="stDataFrame"] {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #dbe4ff;
    }

    div[data-testid="stAlert"] {
        background: #eaf3ff;
        border: 1px solid #c6d8ff;
        border-radius: 14px;
    }

    div[data-testid="stAlert"] div,
    div[data-testid="stAlert"] p {
        color: #101c3f !important;
    }

    details {
        background-color: #ffffff !important;
        border: 1px solid #dbe4ff !important;
        border-radius: 14px !important;
    }

    details summary,
    details div,
    details p,
    details li {
        color: #101c3f !important;
    }

    code {
        color: #111827 !important;
    }

    .panel {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #dbe4ff;
        border-radius: 26px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 16px 36px rgba(30, 41, 59, 0.10);
    }

    .panel h1 {
        color: #101c3f !important;
        font-size: 2.7rem;
        margin: 0 0 12px 0;
        letter-spacing: -0.8px;
    }

    .panel h2 {
        color: #101c3f !important;
        font-size: 2rem;
        margin-top: 0;
    }

    .panel p {
        color: #334155 !important;
        font-size: 1.04rem;
        line-height: 1.8;
        margin: 0.5rem 0;
    }

    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #6d5dfc, #00bcd4);
        color: white !important;
        padding: 9px 16px;
        border-radius: 999px;
        font-weight: 900;
        margin-bottom: 16px;
        box-shadow: 0 8px 22px rgba(93, 88, 255, 0.25);
    }

    .mission-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        margin: 20px 0;
    }

    .mission-card {
        background: #ffffff;
        border: 1px solid #dbe4ff;
        border-radius: 20px;
        padding: 22px;
        min-height: 180px;
        box-shadow: 0 10px 24px rgba(30, 41, 59, 0.08);
    }

    .mission-card h3 {
        color: #101c3f !important;
        margin: 0 0 12px 0;
        font-size: 1.25rem;
    }

    .mission-card p {
        color: #475569 !important;
        font-size: 0.98rem;
        line-height: 1.65;
    }

    .app-header {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #dbe4ff;
        border-radius: 24px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 14px 32px rgba(30, 41, 59, 0.10);
    }

    .app-header h1 {
        color: #101c3f !important;
        margin: 0 0 8px 0;
        font-size: 2.25rem;
    }

    .app-header p {
        color: #475569 !important;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
    }

    .winner-card {
        background: linear-gradient(135deg, #ffffff, #eef4ff);
        border: 1px solid #c8d4ff;
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 14px 34px rgba(87, 70, 255, 0.14);
    }

    .winner-card h2,
    .winner-card h3,
    .winner-card p,
    .winner-card b {
        color: #101c3f !important;
    }

    .winner-card h2 {
        margin-top: 0;
    }

    @media (max-width: 900px) {
        .mission-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """
)


# =========================================================
# City Coordinates
# =========================================================
CITY_COORDS = {
    "Seoul": {"lat": 37.5665, "lon": 126.9780},
    "Busan": {"lat": 35.1796, "lon": 129.0756},
    "Shanghai": {"lat": 31.2304, "lon": 121.4737},
    "Tokyo": {"lat": 35.6762, "lon": 139.6503},
    "Singapore": {"lat": 1.3521, "lon": 103.8198},
    "Taipei": {"lat": 25.0330, "lon": 121.5654},
    "Vancouver": {"lat": 49.2827, "lon": -123.1207},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
    "New York": {"lat": 40.7128, "lon": -74.0060},
    "Hamburg": {"lat": 53.5511, "lon": 9.9937},
    "Rotterdam": {"lat": 51.9244, "lon": 4.4777},
    "Dubai": {"lat": 25.2048, "lon": 55.2708},
    "Sydney": {"lat": -33.8688, "lon": 151.2093},
}

CITY_OPTIONS = list(CITY_COORDS.keys())


# =========================================================
# Route Data
# =========================================================
def load_routes(origin, destination):
    candidate_hubs = [
        "Shanghai", "Tokyo", "Singapore", "Taipei",
        "Busan", "Vancouver", "Dubai", "Rotterdam", "Hamburg"
    ]

    hubs = [city for city in candidate_hubs if city not in [origin, destination]]

    while len(hubs) < 6:
        hubs.append("Singapore")

    route_specs = [
        {
            "route_id": "R1",
            "stops": [origin, hubs[0], destination],
            "mode": "Sea + Truck",
            "cost": 1200,
            "time": 8,
            "carbon": 900,
            "risk": 0.30,
        },
        {
            "route_id": "R2",
            "stops": [origin, hubs[1], destination],
            "mode": "Air + Truck",
            "cost": 1500,
            "time": 6,
            "carbon": 1100,
            "risk": 0.20,
        },
        {
            "route_id": "R3",
            "stops": [origin, hubs[2], destination],
            "mode": "Sea",
            "cost": 1000,
            "time": 11,
            "carbon": 700,
            "risk": 0.40,
        },
        {
            "route_id": "R4",
            "stops": [origin, hubs[3], hubs[4], destination],
            "mode": "Sea + Rail",
            "cost": 1350,
            "time": 9,
            "carbon": 850,
            "risk": 0.25,
        },
        {
            "route_id": "R5",
            "stops": [origin, hubs[5], destination],
            "mode": "Air",
            "cost": 1450,
            "time": 7,
            "carbon": 980,
            "risk": 0.22,
        },
    ]

    rows = []
    for r in route_specs:
        row = r.copy()
        row["path"] = " → ".join(row["stops"])
        rows.append(row)

    return pd.DataFrame(rows)


SCENARIOS = {
    "평상시": {
        "description": "일반적인 글로벌 물류 상황입니다. 비용, 시간, 탄소배출, 리스크가 기본값으로 계산됩니다.",
        "effects": {}
    },
    "항구 파업": {
        "description": "주요 항구 파업으로 해상 운송 시간이 증가하고 리스크가 커집니다.",
        "effects": {
            "Sea": {"time": 1.35, "risk": 1.25},
            "Sea + Truck": {"time": 1.30, "risk": 1.20},
            "Sea + Rail": {"time": 1.25, "risk": 1.15},
        }
    },
    "유가 급등": {
        "description": "유가 급등으로 항공 및 트럭 운송 비용이 증가합니다.",
        "effects": {
            "Air": {"cost": 1.30},
            "Air + Truck": {"cost": 1.35},
            "Sea + Truck": {"cost": 1.20},
        }
    },
    "탄소세 강화": {
        "description": "탄소세 강화로 탄소 배출량이 큰 경로의 부담이 증가합니다.",
        "effects": {
            "Air": {"carbon": 1.35, "cost": 1.15},
            "Air + Truck": {"carbon": 1.40, "cost": 1.20},
        }
    },
    "긴급 배송": {
        "description": "긴급 배송 상황입니다. 시간이 오래 걸리는 경로에 더 큰 페널티가 붙습니다.",
        "effects": {
            "Sea": {"time": 1.45},
            "Sea + Rail": {"time": 1.25},
            "Sea + Truck": {"time": 1.20},
        }
    },
    "지정학적 리스크 상승": {
        "description": "특정 지역의 지정학적 리스크가 상승하여 일부 경로의 안정성이 낮아집니다.",
        "effects": {
            "Sea + Truck": {"risk": 1.35},
            "Sea": {"risk": 1.30},
        }
    },
}


def apply_scenario(df, scenario_name):
    df = df.copy()
    effects = SCENARIOS[scenario_name]["effects"]

    for idx, row in df.iterrows():
        mode = row["mode"]
        if mode in effects:
            for col, multiplier in effects[mode].items():
                df.loc[idx, col] = df.loc[idx, col] * multiplier

    df["cost"] = df["cost"].round(0).astype(int)
    df["time"] = df["time"].round(1)
    df["carbon"] = df["carbon"].round(0).astype(int)
    df["risk"] = df["risk"].round(3)

    return df


# =========================================================
# Score Calculation
# =========================================================
def normalize(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


def compute_route_scores(df, w_cost, w_time, w_carbon, w_risk):
    df = df.copy()

    df["cost_norm"] = normalize(df["cost"])
    df["time_norm"] = normalize(df["time"])
    df["carbon_norm"] = normalize(df["carbon"])
    df["risk_norm"] = normalize(df["risk"])

    df["score"] = (
        w_cost * df["cost_norm"]
        + w_time * df["time_norm"]
        + w_carbon * df["carbon_norm"]
        + w_risk * df["risk_norm"]
    )

    return df


# =========================================================
# Map Visualization
# =========================================================
def make_route_map(route_row, title="Selected Logistics Route"):
    cities = route_row["stops"]

    lats = [CITY_COORDS[city]["lat"] for city in cities]
    lons = [CITY_COORDS[city]["lon"] for city in cities]

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="lines",
            line=dict(width=4),
            name="Route Path",
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers+text",
            text=cities,
            textposition="top center",
            marker=dict(size=10),
            name="Stops"
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[lats[0]],
            lon=[lons[0]],
            mode="markers+text",
            text=["START"],
            textposition="bottom center",
            marker=dict(size=16, symbol="circle"),
            name="Start"
        )
    )

    fig.add_trace(
        go.Scattergeo(
            lat=[lats[-1]],
            lon=[lons[-1]],
            mode="markers+text",
            text=["DESTINATION"],
            textposition="bottom center",
            marker=dict(size=17, symbol="star"),
            name="Destination"
        )
    )

    fig.update_layout(
        title=title,
        geo=dict(
            projection_type="natural earth",
            showland=True,
            showcountries=True,
            showocean=True,
            showcoastlines=True,
            landcolor="#eef2ff",
            oceancolor="#dbeafe",
            countrycolor="#94a3b8",
            coastlinecolor="#64748b",
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_color="#101c3f",
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="center",
            x=0.5
        )
    )

    return fig


# =========================================================
# QUBO and QAOA
# =========================================================
def qubo_energy(bitstring, scores, penalty):
    x = np.array([int(b) for b in bitstring])
    objective = np.dot(scores, x)
    constraint = penalty * (np.sum(x) - 1) ** 2
    return objective + constraint


def build_qubo_dict(scores, penalty):
    n = len(scores)
    qubo = {}

    for i in range(n):
        qubo[(i, i)] = scores[i] - penalty

    for i in range(n):
        for j in range(i + 1, n):
            qubo[(i, j)] = 2 * penalty

    return qubo


def apply_cost_unitary(qc, gamma, qubo):
    for (i, j), coeff in qubo.items():
        if i == j:
            qc.rz(2 * gamma * coeff, i)
        else:
            qc.cx(i, j)
            qc.rz(2 * gamma * coeff, j)
            qc.cx(i, j)


def apply_mixer_unitary(qc, beta, n_qubits):
    for i in range(n_qubits):
        qc.rx(2 * beta, i)


def create_qaoa_circuit(gammas, betas, qubo, n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)

    for i in range(n_qubits):
        qc.h(i)

    for gamma, beta in zip(gammas, betas):
        apply_cost_unitary(qc, gamma, qubo)
        apply_mixer_unitary(qc, beta, n_qubits)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_qaoa_sampling(scores, penalty=2.0, p=1, shots=512, maxiter=20):
    n_qubits = len(scores)
    qubo = build_qubo_dict(scores, penalty)
    simulator = AerSimulator()

    def objective(params):
        gammas = params[:p]
        betas = params[p:]

        qc = create_qaoa_circuit(gammas, betas, qubo, n_qubits)
        tqc = transpile(qc, simulator)
        result = simulator.run(tqc, shots=shots).result()
        counts = result.get_counts()

        expected_energy = 0
        total_counts = sum(counts.values())

        for bitstring, count in counts.items():
            bitstring = bitstring[::-1]
            energy = qubo_energy(bitstring, scores, penalty)
            expected_energy += energy * count / total_counts

        return expected_energy

    initial_params = np.random.uniform(0, np.pi, 2 * p)

    opt_result = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": maxiter}
    )

    best_params = opt_result.x
    gammas = best_params[:p]
    betas = best_params[p:]

    final_qc = create_qaoa_circuit(gammas, betas, qubo, n_qubits)
    tqc = transpile(final_qc, simulator)
    final_result = simulator.run(tqc, shots=shots).result()
    counts = final_result.get_counts()

    decoded = []

    for bitstring, count in counts.items():
        bitstring_fixed = bitstring[::-1]
        energy = qubo_energy(bitstring_fixed, scores, penalty)
        selected_indices = [i for i, bit in enumerate(bitstring_fixed) if bit == "1"]

        decoded.append({
            "bitstring": bitstring_fixed,
            "count": count,
            "probability": count / shots,
            "energy": energy,
            "selected_indices": selected_indices,
            "valid": len(selected_indices) == 1
        })

    decoded_df = pd.DataFrame(decoded)

    decoded_df = decoded_df.sort_values(
        ["valid", "probability", "energy"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    valid_df = decoded_df[decoded_df["valid"] == True]

    if len(valid_df) > 0:
        best_row = valid_df.sort_values(
            ["probability", "energy"],
            ascending=[False, True]
        ).iloc[0]
    else:
        best_row = decoded_df.sort_values(
            ["probability", "energy"],
            ascending=[False, True]
        ).iloc[0]

    return best_row, decoded_df, final_qc, opt_result.fun


def classical_optimizer(df):
    return df.sort_values("score").iloc[0]


def mission_scores(best_route):
    cost_score = 100 * (1 - best_route["cost_norm"])
    speed_score = 100 * (1 - best_route["time_norm"])
    eco_score = 100 * (1 - best_route["carbon_norm"])
    stability_score = 100 * (1 - best_route["risk_norm"])

    total = np.mean([cost_score, speed_score, eco_score, stability_score])

    return {
        "Cost": cost_score,
        "Speed": speed_score,
        "Eco": eco_score,
        "Stability": stability_score,
        "Total": total
    }


# =========================================================
# Page Flow
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "start"


def go_to_start():
    st.session_state.page = "start"


def go_to_intro():
    st.session_state.page = "intro"


def go_to_app():
    st.session_state.page = "app"


# =========================================================
# Start Screen
# =========================================================
def render_start_screen():
    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {
                margin: 0;
                background: transparent;
                font-family: Arial, sans-serif;
            }

            .start-screen {
                height: 720px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                background:
                    radial-gradient(circle at 20% 20%, rgba(109, 93, 252, 0.18), transparent 30%),
                    radial-gradient(circle at 80% 30%, rgba(0, 188, 212, 0.18), transparent 35%),
                    linear-gradient(135deg, #f7f8ff 0%, #eef4ff 100%);
                border-radius: 32px;
                border: 1px solid #dbe4ff;
                box-shadow: 0 18px 44px rgba(30, 41, 59, 0.12);
                padding: 48px;
                overflow: hidden;
                position: relative;
            }

            .mission-badge {
                display: inline-block;
                background: linear-gradient(90deg, #6d5dfc, #00bcd4);
                color: white;
                padding: 10px 18px;
                border-radius: 999px;
                font-weight: 800;
                margin-bottom: 18px;
                box-shadow: 0 8px 22px rgba(109, 93, 252, 0.25);
            }

            .start-title {
                font-size: 4.2rem;
                font-weight: 900;
                color: #102047;
                margin-bottom: 0.5rem;
                letter-spacing: -1px;
            }

            .start-subtitle {
                font-size: 1.2rem;
                color: #334155;
                max-width: 860px;
                line-height: 1.8;
                margin-bottom: 30px;
                font-weight: 600;
            }

            .rail-scene {
                width: 100%;
                max-width: 900px;
                height: 190px;
                position: relative;
                margin: 18px auto 20px auto;
            }

            .rail-track {
                position: absolute;
                left: 0;
                right: 0;
                bottom: 42px;
                height: 8px;
                background: repeating-linear-gradient(
                    90deg,
                    #334155 0px,
                    #334155 34px,
                    transparent 34px,
                    transparent 52px
                );
                border-radius: 999px;
            }

            .rail-track::after {
                content: "";
                position: absolute;
                left: 0;
                right: 0;
                top: 18px;
                height: 4px;
                background: #64748b;
                border-radius: 999px;
            }

            .train {
                position: absolute;
                bottom: 68px;
                left: -300px;
                display: flex;
                align-items: end;
                gap: 10px;
                animation: trainMove 7s linear infinite;
            }

            @keyframes trainMove {
                0% {
                    transform: translateX(-160px);
                }
                100% {
                    transform: translateX(1200px);
                }
            }

            .engine {
                width: 150px;
                height: 74px;
                background: linear-gradient(135deg, #6d5dfc, #00bcd4);
                border-radius: 22px 30px 14px 14px;
                position: relative;
                box-shadow: 0 12px 24px rgba(30, 41, 59, 0.22);
            }

            .engine::before {
                content: "⚛";
                position: absolute;
                left: 18px;
                top: 12px;
                font-size: 30px;
                color: white;
            }

            .engine::after {
                content: "";
                position: absolute;
                right: 20px;
                top: 18px;
                width: 34px;
                height: 24px;
                background: #e0f7ff;
                border-radius: 8px;
            }

            .wagon {
                width: 118px;
                height: 58px;
                border-radius: 14px;
                position: relative;
                box-shadow: 0 10px 20px rgba(30, 41, 59, 0.18);
            }

            .wagon1 {
                background: linear-gradient(135deg, #ffb703, #fb8500);
            }

            .wagon2 {
                background: linear-gradient(135deg, #80ed99, #38b000);
            }

            .wagon3 {
                background: linear-gradient(135deg, #ffafcc, #b5179e);
            }

            .cargo-label {
                position: absolute;
                inset: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
                font-weight: 900;
                font-size: 0.92rem;
            }

            .wheel {
                position: absolute;
                bottom: -15px;
                width: 24px;
                height: 24px;
                background: #111827;
                border: 4px solid #94a3b8;
                border-radius: 50%;
            }

            .wheel.left {
                left: 22px;
            }

            .wheel.right {
                right: 22px;
            }

            .smoke {
                position: absolute;
                left: 20px;
                top: -36px;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                background: rgba(203, 213, 225, 0.65);
                animation: smokeRise 2.8s ease-in-out infinite;
            }

            .smoke2 {
                left: 45px;
                top: -54px;
                width: 36px;
                height: 36px;
                animation-delay: 0.7s;
            }

            .smoke3 {
                left: 75px;
                top: -72px;
                width: 44px;
                height: 44px;
                animation-delay: 1.3s;
            }

            @keyframes smokeRise {
                0% {
                    opacity: 0;
                    transform: translateY(15px) scale(0.8);
                }
                35% {
                    opacity: 0.75;
                }
                100% {
                    opacity: 0;
                    transform: translateY(-28px) scale(1.2);
                }
            }

            .caption {
                color: #475569;
                font-size: 0.95rem;
                font-weight: 700;
                background: rgba(255, 255, 255, 0.80);
                border: 1px solid #dbe4ff;
                border-radius: 14px;
                padding: 10px 16px;
                box-shadow: 0 8px 20px rgba(30, 41, 59, 0.08);
            }
        </style>
        </head>

        <body>
            <div class="start-screen">
                <div class="mission-badge">MISSION 01 · QUANTUM LOGISTICS</div>

                <div class="start-title">🚆 Q-Route</div>

                <div class="start-subtitle">
                    글로벌 물류 경로를 양자 알고리즘으로 최적화하는 인터랙티브 공급망 미션입니다.<br>
                    비용·시간·탄소배출·리스크를 동시에 고려해 가장 적합한 운송 경로를 찾아보세요.
                </div>

                <div class="rail-scene">
                    <div class="train">
                        <div class="engine">
                            <div class="smoke"></div>
                            <div class="smoke smoke2"></div>
                            <div class="smoke smoke3"></div>
                            <div class="wheel left"></div>
                            <div class="wheel right"></div>
                        </div>

                        <div class="wagon wagon1">
                            <div class="cargo-label">COST</div>
                            <div class="wheel left"></div>
                            <div class="wheel right"></div>
                        </div>

                        <div class="wagon wagon2">
                            <div class="cargo-label">TIME</div>
                            <div class="wheel left"></div>
                            <div class="wheel right"></div>
                        </div>

                        <div class="wagon wagon3">
                            <div class="cargo-label">RISK</div>
                            <div class="wheel left"></div>
                            <div class="wheel right"></div>
                        </div>
                    </div>

                    <div class="rail-track"></div>
                </div>

                <div class="caption">Supply Chain Control Tower · Ready for Quantum Dispatch</div>
            </div>
        </body>
        </html>
        """,
        height=740,
        scrolling=False
    )

    st.button(
        "🎮 미션 브리핑 시작하기",
        on_click=go_to_intro,
        use_container_width=True
    )


# =========================================================
# Intro Screen
# =========================================================
def render_intro_screen():
    html_block(
        """
        <div class="panel">
            <div class="badge">MISSION BRIEFING</div>
            <h1>📦 글로벌 공급망 경로를 최적화하라</h1>
            <p>
            Q-Route는 글로벌 물류 상황에서 여러 운송 경로 후보를 비교하고,
            비용·배송 시간·탄소 배출량·리스크를 함께 고려해 최적 경로를 추천하는
            양자 기반 물류 최적화 데모입니다.
            </p>
            <p>
            기존의 챗봇형 서비스가 조언이나 설명을 제공하는 데 그친다면,
            이 앱은 사용자가 직접 조건을 설정하고 실제 최적화 문제를 구성한 뒤
            QAOA 양자 알고리즘으로 경로 선택 결과를 계산한다는 점에서 차별화됩니다.
            </p>
        </div>
        """
    )

    html_block(
        """
        <div class="mission-grid">
            <div class="mission-card">
                <h3>1️⃣ 물류 상황 설정</h3>
                <p>
                출발지와 도착지를 선택하고, 항구 파업·유가 급등·탄소세 강화·긴급 배송 같은
                글로벌 이벤트를 선택합니다. 이벤트에 따라 경로별 비용, 시간, 탄소, 리스크 값이 달라집니다.
                </p>
            </div>
            <div class="mission-card">
                <h3>2️⃣ 의사결정 기준 조정</h3>
                <p>
                비용을 줄이는 것이 중요한지, 빠른 배송이 중요한지, 탄소 배출 저감이나 안정성이 중요한지
                미션 가중치로 직접 조정합니다.
                </p>
            </div>
            <div class="mission-card">
                <h3>3️⃣ 양자 디스패치 실행</h3>
                <p>
                각 경로 후보를 qubit에 대응시키고, QUBO 형태의 목적함수를 구성한 뒤,
                QAOA가 가능한 경로 선택 상태를 탐색합니다.
                </p>
            </div>
        </div>
        """
    )

    html_block(
        """
        <div class="panel">
            <div class="badge">QUANTUM OPTIMIZATION</div>
            <h2>⚛️ QAOA는 이 앱에서 무엇을 하나요?</h2>
            <p>
            이 앱은 경로 선택 문제를 0과 1의 이진 변수로 표현합니다.
            예를 들어 <b>01000</b>이라는 측정 결과는 두 번째 경로가 선택되었다는 뜻입니다.
            </p>
            <p>
            QAOA는 여러 경로 선택 상태를 양자 회로에서 탐색하고,
            비용·시간·탄소배출·리스크를 종합한 QUBO 에너지가 낮은 경로가
            더 높은 확률로 측정되도록 파라미터를 조정합니다.
            </p>
            <p>
            즉, Q-Route는 단순히 예쁜 대시보드를 보여주는 것이 아니라,
            실제 물류 문제를 수학적 최적화 문제로 바꾸고 양자 알고리즘으로 해를 탐색하는
            계산형 의사결정 플랫폼입니다.
            </p>
        </div>
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.button(
            "⬅️ 시작 화면으로 돌아가기",
            on_click=go_to_start,
            use_container_width=True
        )

    with c2:
        st.button(
            "⚛️ 디스패치 센터 입장하기",
            on_click=go_to_app,
            use_container_width=True
        )


# =========================================================
# Main App Screen
# =========================================================
def render_app_screen():
    html_block(
        """
        <div class="app-header">
            <div class="badge">DISPATCH CENTER</div>
            <h1>⚛️ Quantum Dispatch Center</h1>
            <p>
            왼쪽 Mission Control에서 물류 상황과 최적화 기준을 설정한 뒤,
            QAOA 기반 양자 디스패치를 실행해 최적의 물류 경로를 찾아보세요.
            </p>
        </div>
        """
    )

    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        st.button(
            "📦 미션 브리핑 다시 보기",
            on_click=go_to_intro,
            use_container_width=True
        )

    with top_col2:
        st.button(
            "🏠 시작 화면으로 돌아가기",
            on_click=go_to_start,
            use_container_width=True
        )

    st.divider()

    st.sidebar.title("🎮 Mission Control")

    origin = st.sidebar.selectbox(
        "출발지",
        CITY_OPTIONS,
        index=CITY_OPTIONS.index("Seoul")
    )

    destination_options = [city for city in CITY_OPTIONS if city != origin]

    default_destination = "Los Angeles"
    if default_destination not in destination_options:
        default_destination = destination_options[0]

    destination = st.sidebar.selectbox(
        "도착지",
        destination_options,
        index=destination_options.index(default_destination)
    )

    scenario_name = st.sidebar.selectbox(
        "글로벌 이벤트 카드",
        list(SCENARIOS.keys())
    )

    st.sidebar.info(SCENARIOS[scenario_name]["description"])

    priority = st.sidebar.selectbox(
        "최적화 미션 유형",
        [
            "균형형 미션",
            "비용 최소화",
            "최단 시간 배송",
            "친환경 배송",
            "안전성 우선",
            "직접 설정"
        ]
    )

    if priority == "균형형 미션":
        w_cost, w_time, w_carbon, w_risk = 0.25, 0.25, 0.25, 0.25
    elif priority == "비용 최소화":
        w_cost, w_time, w_carbon, w_risk = 0.60, 0.15, 0.15, 0.10
    elif priority == "최단 시간 배송":
        w_cost, w_time, w_carbon, w_risk = 0.15, 0.60, 0.15, 0.10
    elif priority == "친환경 배송":
        w_cost, w_time, w_carbon, w_risk = 0.15, 0.15, 0.60, 0.10
    elif priority == "안전성 우선":
        w_cost, w_time, w_carbon, w_risk = 0.15, 0.15, 0.10, 0.60
    else:
        w_cost, w_time, w_carbon, w_risk = 0.25, 0.25, 0.25, 0.25

    st.sidebar.markdown("### 미션 가중치")

    w_cost = st.sidebar.slider("비용 중요도", 0.0, 1.0, w_cost, 0.05)
    w_time = st.sidebar.slider("시간 중요도", 0.0, 1.0, w_time, 0.05)
    w_carbon = st.sidebar.slider("탄소배출 중요도", 0.0, 1.0, w_carbon, 0.05)
    w_risk = st.sidebar.slider("리스크 중요도", 0.0, 1.0, w_risk, 0.05)

    total_w = w_cost + w_time + w_carbon + w_risk

    if total_w == 0:
        st.error("최소 하나 이상의 가중치는 0보다 커야 합니다.")
        st.stop()

    w_cost /= total_w
    w_time /= total_w
    w_carbon /= total_w
    w_risk /= total_w

    st.sidebar.markdown("### 양자 알고리즘 설정")
    qaoa_p = st.sidebar.slider("QAOA 회로 깊이 p", 1, 3, 1)
    shots = st.sidebar.select_slider("측정 횟수 Shots", options=[256, 512, 1024, 2048], value=512)
    maxiter = st.sidebar.slider("파라미터 최적화 반복 횟수", 10, 100, 20, 10)
    penalty = st.sidebar.slider("제약조건 페널티", 0.5, 5.0, 2.0, 0.5)

    base_routes = load_routes(origin, destination)
    routes = apply_scenario(base_routes, scenario_name)
    routes = compute_route_scores(routes, w_cost, w_time, w_carbon, w_risk)

    classical_best = classical_optimizer(routes)

    run_button = st.button("⚛️ 양자 디스패치 실행하기", use_container_width=True)

    if run_button:
        with st.spinner("QAOA 양자 회로가 가능한 경로 상태를 탐색하는 중입니다..."):
            scores = routes["score"].to_numpy()
            q_best, q_samples, q_circuit, expected_energy = run_qaoa_sampling(
                scores=scores,
                penalty=penalty,
                p=qaoa_p,
                shots=shots,
                maxiter=maxiter
            )

        selected_indices = q_best["selected_indices"]

        if len(selected_indices) == 1:
            q_selected_idx = selected_indices[0]
        else:
            q_selected_idx = int(routes["score"].idxmin())

        quantum_best = routes.iloc[q_selected_idx]
        scores_dict = mission_scores(quantum_best)

        if quantum_best["route_id"] == classical_best["route_id"]:
            st.success("✅ QAOA가 고전 최적화와 동일한 최적 경로를 찾았습니다.")
        else:
            st.warning("⚠️ QAOA 샘플링 결과가 고전 최적화와 다른 경로를 선택했습니다. 이는 QAOA의 확률적 탐색 특성 때문입니다.")

        html_block(
            f"""
            <div class="winner-card">
                <h2>🏆 양자 디스패치 결과</h2>
                <h3>{quantum_best["route_id"]}: {quantum_best["path"]}</h3>
                <p>
                운송 방식: <b>{quantum_best["mode"]}</b><br>
                측정된 bitstring: <b>{q_best["bitstring"]}</b><br>
                샘플링 확률: <b>{q_best["probability"]:.2%}</b><br>
                QUBO 에너지: <b>{q_best["energy"]:.4f}</b>
                </p>
                <p>
                위 bitstring은 QAOA 회로를 측정했을 때 얻어진 경로 선택 결과입니다.
                이 앱에서는 제약조건을 만족하는 valid bitstring 중 가장 높은 확률로 측정된 결과를
                Quantum Route로 선택합니다. QUBO 에너지가 낮을수록 비용·시간·탄소배출·리스크를
                종합적으로 고려했을 때 더 좋은 경로입니다.
                </p>
            </div>
            """
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("💰 비용 점수", f"{scores_dict['Cost']:.0f}")
        col2.metric("⚡ 속도 점수", f"{scores_dict['Speed']:.0f}")
        col3.metric("🌱 친환경 점수", f"{scores_dict['Eco']:.0f}")
        col4.metric("🛡 안정성 점수", f"{scores_dict['Stability']:.0f}")
        col5.metric("🎯 종합 미션 점수", f"{scores_dict['Total']:.0f}")

        st.divider()

        st.subheader("🗺️ 선택된 물류 경로 시각화")

        route_map = make_route_map(
            quantum_best,
            title=f"Quantum Selected Route: {quantum_best['route_id']} | {quantum_best['path']}"
        )
        st.plotly_chart(route_map, use_container_width=True)

        st.divider()

        st.subheader("⚔️ 경로 대결: 고전 최적화 vs 양자 최적화")

        battle_col1, battle_col2 = st.columns(2)

        with battle_col1:
            st.markdown("### 고전 최적화 방식")
            st.write(f"**{classical_best['route_id']}**")
            st.write(classical_best["path"])
            st.write(f"목적함수 점수: **{classical_best['score']:.4f}**")
            st.caption("모든 경로 후보의 점수를 직접 계산한 뒤 가장 낮은 점수를 가진 경로를 선택합니다.")

        with battle_col2:
            st.markdown("### QAOA 양자 최적화 방식")
            st.write(f"**{quantum_best['route_id']}**")
            st.write(quantum_best["path"])
            st.write(f"목적함수 점수: **{quantum_best['score']:.4f}**")
            st.caption("QUBO 문제를 양자 회로로 변환한 뒤, 측정 결과 중 제약조건을 만족하면서 가장 높은 확률로 나온 bitstring을 선택합니다.")

        st.subheader("🗺️ 경로 지도 비교")

        map_col1, map_col2 = st.columns(2)

        with map_col1:
            classical_map = make_route_map(
                classical_best,
                title=f"Classical Route: {classical_best['route_id']}"
            )
            st.plotly_chart(classical_map, use_container_width=True)

        with map_col2:
            quantum_map = make_route_map(
                quantum_best,
                title=f"Quantum Route: {quantum_best['route_id']}"
            )
            st.plotly_chart(quantum_map, use_container_width=True)

        st.divider()

        st.subheader("📦 물류 경로 후보")

        display_df = routes[
            ["route_id", "path", "mode", "cost", "time", "carbon", "risk", "score"]
        ].copy()

        display_df.columns = [
            "경로 ID", "경로", "운송 방식", "비용", "시간", "탄소배출량", "리스크", "목적함수 점수"
        ]
        display_df["목적함수 점수"] = display_df["목적함수 점수"].round(4)

        st.dataframe(display_df, use_container_width=True)

        st.subheader("📊 미션 대시보드")

        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                routes,
                x="route_id",
                y="score",
                hover_data=["path", "cost", "time", "carbon", "risk"],
                title="경로별 목적함수 점수"
            )
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font_color="#101c3f"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            q_plot = q_samples.copy()
            q_plot["label"] = q_plot["bitstring"] + " | " + q_plot["valid"].map({True: "valid", False: "invalid"})

            fig_q = px.bar(
                q_plot.head(12),
                x="label",
                y="probability",
                title="QAOA 측정 결과 분포"
            )
            fig_q.update_layout(
                xaxis_tickangle=-45,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font_color="#101c3f"
            )
            st.plotly_chart(fig_q, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            radar_df = pd.DataFrame({
                "metric": ["비용", "속도", "친환경", "안정성"],
                "score": [
                    scores_dict["Cost"],
                    scores_dict["Speed"],
                    scores_dict["Eco"],
                    scores_dict["Stability"]
                ]
            })

            fig_radar = px.line_polar(
                radar_df,
                r="score",
                theta="metric",
                line_close=True,
                title="추천 경로의 미션 능력치"
            )
            fig_radar.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font_color="#101c3f"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with c4:
            comparison = routes.melt(
                id_vars=["route_id"],
                value_vars=["cost_norm", "time_norm", "carbon_norm", "risk_norm"],
                var_name="metric",
                value_name="normalized_value"
            )

            metric_map = {
                "cost_norm": "비용",
                "time_norm": "시간",
                "carbon_norm": "탄소배출",
                "risk_norm": "리스크"
            }

            comparison["metric"] = comparison["metric"].map(metric_map)

            fig_compare = px.bar(
                comparison,
                x="route_id",
                y="normalized_value",
                color="metric",
                barmode="group",
                title="경로별 정규화 지표 비교"
            )
            fig_compare.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font_color="#101c3f"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        with st.expander("⚛️ QAOA 회로 보기"):
            st.text(q_circuit.draw(output="text"))

        with st.expander("🧠 QAOA 계산 방식 다시 보기"):
            st.markdown(
                """
                이 앱은 물류 경로 선택 문제를 양자 알고리즘이 처리할 수 있는 형태로 바꿉니다.

                #### 1. 경로를 이진 변수로 표현
                각 경로 후보는 하나의 이진 변수로 표현됩니다.

                - x₁ = 1이면 1번 경로 선택
                - x₂ = 1이면 2번 경로 선택
                - xᵢ = 0이면 해당 경로는 선택하지 않음

                예를 들어 `01000`이라는 bitstring은 두 번째 경로만 선택했다는 의미입니다.

                #### 2. 목적함수 구성
                각 경로의 점수는 비용, 시간, 탄소배출량, 리스크를 가중합하여 계산합니다.

                `score = 비용가중치 × 비용 + 시간가중치 × 시간 + 탄소가중치 × 탄소배출량 + 리스크가중치 × 리스크`

                #### 3. 제약조건 추가
                물류 경로는 여러 개를 동시에 선택하는 것이 아니라, 하나의 최종 경로를 선택해야 합니다.
                따라서 다음 제약조건을 추가합니다.

                `x₁ + x₂ + x₃ + x₄ + x₅ = 1`

                #### 4. QUBO 문제로 변환
                위 목적함수와 제약조건을 합쳐 다음과 같은 QUBO 문제로 만듭니다.

                `minimize Σ scoreᵢ xᵢ + penalty × (Σxᵢ - 1)²`

                #### 5. Quantum Route 선택 방식
                이 앱에서는 QAOA 측정 결과 중 제약조건을 만족하는 valid bitstring을 먼저 찾고,
                그중 가장 높은 확률로 측정된 bitstring을 Quantum Route로 선택합니다.
                """
            )

    else:
        st.info("왼쪽 Mission Control에서 물류 상황과 최적화 기준을 설정한 뒤, **양자 디스패치 실행하기** 버튼을 눌러주세요.")

        st.subheader("현재 물류 경로 후보")

        preview_df = routes[
            ["route_id", "path", "mode", "cost", "time", "carbon", "risk", "score"]
        ].copy()

        preview_df.columns = [
            "경로 ID", "경로", "운송 방식", "비용", "시간", "탄소배출량", "리스크", "목적함수 점수"
        ]
        preview_df["목적함수 점수"] = preview_df["목적함수 점수"].round(4)

        st.dataframe(preview_df, use_container_width=True)

        fig_preview = px.bar(
            routes,
            x="route_id",
            y="score",
            title="현재 조건에서의 경로별 목적함수 점수"
        )
        fig_preview.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font_color="#101c3f"
        )
        st.plotly_chart(fig_preview, use_container_width=True)

        st.subheader("🗺️ 현재 조건에서의 예상 최적 경로")

        preview_best = classical_optimizer(routes)

        preview_map = make_route_map(
            preview_best,
            title=f"Preview Best Route: {preview_best['route_id']} | {preview_best['path']}"
        )
        st.plotly_chart(preview_map, use_container_width=True)


# =========================================================
# Page Router
# =========================================================
if st.session_state.page == "start":
    render_start_screen()

elif st.session_state.page == "intro":
    render_intro_screen()

elif st.session_state.page == "app":
    render_app_screen()
