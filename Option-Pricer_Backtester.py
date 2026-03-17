"""
Options Pricer — Streamlit
Compatible GitHub Codespaces / navigateur
Lancer avec : streamlit run Option-Pricer_Backtester.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Pricer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { 
        background-color: #000000 !important; 
    }
    .block-container { 
        padding-top: 1rem !important;
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"] { 
        background-color: #0a0a0a !important; 
    }
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: #000000 !important;
    }
    h1 { 
        color: #4a9eff !important; 
        font-family: monospace !important; 
        font-weight: bold !important;
        font-size: 1.73rem !important;
        text-shadow: 0 0 8px rgba(74, 158, 255, 0.4);
    }
    h2, h3 { 
        color: #4a9eff !important; 
        font-family: monospace !important; 
        font-weight: normal !important;
        font-size: 1.16rem !important;
    }
    p, span, div, label, .stMarkdown { 
        color: #ffffff !important; 
        font-weight: normal !important;
        font-size: 0.72rem !important;
    }
    input[type="number"],
    input[type="text"],
    .stNumberInput input,
    .stTextInput input {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 2px solid #4a9eff !important;
        font-weight: normal !important;
    }
    .stSelectbox > div > div,
    select {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border: 2px solid #4a9eff !important;
        font-weight: normal !important;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox ul,
    .stSelectbox li,
    [role="listbox"],
    [role="option"] {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
    }
    .stSelectbox li:hover,
    [role="option"]:hover {
        background-color: #1a1a1a !important;
        color: #4a9eff !important;
    }
    div[data-testid="metric-container"] {
        background-color: #000000 !important;
        border: 2px solid #4a9eff !important;
        border-radius: 10px !important;
        padding: 13px !important;
        box-shadow: 0 0 15px rgba(74, 158, 255, 0.3) !important;
    }
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] label p {
        color: #4a9eff !important;
        font-weight: normal !important;
        font-size: 0.72rem !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricValue"] p {
        color: #ffffff !important;
        font-weight: normal !important;
        font-size: 1.3rem !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #4a9eff !important;
        font-weight: normal !important;
    }
    .stAlert, .stSuccess, .stWarning {
        background-color: #0a0a0a !important;
        border: 2px solid #4a9eff !important;
        color: #ffffff !important;
    }
    .stAlert p, .stSuccess p, .stWarning p {
        color: #ffffff !important;
        font-weight: normal !important;
    }
    .dataframe {
        font-size: 0.72rem !important;
        font-family: monospace !important;
        background-color: #000000 !important;
        border: 2px solid #4a9eff !important;
    }
    .dataframe th {
        background-color: #0a0a0a !important;
        color: #4a9eff !important;
        font-weight: normal !important;
        border: 1px solid #4a9eff !important;
        padding: 6px !important;
    }
    .dataframe td {
        color: #ffffff !important;
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        padding: 6px !important;
        font-weight: normal !important;
    }
    .author-link { 
        color: #888888 !important; 
        font-size: 0.72rem; 
        font-family: monospace; 
        margin-top: -10px;
        margin-bottom: 15px;
        font-weight: normal !important;
    }
    .author-link a {
        color: #4a9eff !important;
        text-decoration: none;
        font-weight: normal !important;
    }
    .author-link a:hover {
        color: #60a5fa !important;
        text-decoration: underline;
        text-shadow: 0 0 5px rgba(74, 158, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ─── STYLE MATPLOTLIB ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#000000",
    "axes.facecolor":    "#0a0a0a",
    "axes.edgecolor":    "#4a9eff",
    "axes.labelcolor":   "#4a9eff",
    "text.color":        "#e5e7eb",
    "xtick.color":       "#e5e7eb",
    "ytick.color":       "#e5e7eb",
    "grid.color":        "#1e2a38",
    "grid.linewidth":    0.35,
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "font.weight":       "normal",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
})

BG     = "#000000"
PANEL  = "#0a0a0a"
BORDER = "#4a9eff"
ACCENT = "#4a9eff"
GREEN  = "#10b981"
RED    = "#ef4444"
YELLOW = "#f59e0b"
PURPLE = "#8b5cf6"
CYAN   = "#06b6d4"
ORANGE = "#f97316"
GRAY   = "#6b7280"
TEXT   = "#e5e7eb"
TITLE  = "#4a9eff"

# ─── BLACK-SCHOLES ────────────────────────────────────────────────────────────

def bs(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10:
        return max(S-K, 0) if opt=="call" else max(K-S, 0)
    if sigma <= 1e-10:
        return max(S*np.exp(-q*T)-K*np.exp(-r*T), 0) if opt=="call" \
               else max(K*np.exp(-r*T)-S*np.exp(-q*T), 0)
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def greeks(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return {k: 0.0 for k in ["delta","gamma","vega","theta","rho"]}
    d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.pdf(d1)
    if opt == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*norm.cdf(d2)
                 + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)
                 - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = np.exp(-q*T)*nd1 / (S*sigma*np.sqrt(T))
    vega  = S*np.exp(-q*T)*nd1*np.sqrt(T) / 100
    return {"delta":delta, "gamma":gamma, "vega":vega, "theta":theta, "rho":rho}

def prob_itm(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return 1.0 if (opt=="call" and S>K) or (opt=="put" and S<K) else 0.0
    d2 = (np.log(S/K)+(r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d2) if opt=="call" else norm.cdf(-d2)

# ─── IMPLIED VOLATILITY ───────────────────────────────────────────────────────

def implied_volatility(market_price, S, K, T, r, q=0.0, opt="call"):
    if T <= 1e-10:
        return np.nan
    intrinsic = max(S-K, 0) if opt=="call" else max(K-S, 0)
    if market_price < intrinsic * 0.99:
        return np.nan
    def objective(sigma):
        try:
            return bs(S, K, T, r, sigma, q, opt) - market_price
        except:
            return 1e10
    try:
        iv = brentq(objective, 0.001, 5.0, maxiter=100)
        return iv
    except:
        return np.nan

# ─── MONTE CARLO ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def monte_carlo_pricer_cached(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed):
    return monte_carlo_pricer(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed)

def monte_carlo_pricer(S, K, T, r, sigma, q=0.0, opt="call", n_sims=100000, n_steps=252, antithetic=True, seed=42):
    np.random.seed(seed)
    dt = T / n_steps
    max_paths_to_store = min(1000, n_sims)
    batch_size = min(n_sims, 100000)
    n_batches = int(np.ceil(n_sims / batch_size))
    all_payoffs = []
    sample_paths = None
    for batch in range(n_batches):
        n_paths_batch = min(batch_size, n_sims - batch * batch_size)
        n_paths = n_paths_batch // 2 if antithetic else n_paths_batch
        Z = np.random.standard_normal((n_paths, n_steps))
        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)
        drift = (r - q - 0.5*sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        log_returns = drift + diffusion * Z
        log_price_paths = np.log(S) + np.cumsum(log_returns, axis=1)
        S_T = np.exp(log_price_paths[:, -1])
        if batch == 0:
            sample_paths = S_T[:max_paths_to_store].copy()
        if opt == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        all_payoffs.append(payoffs)
        del Z, log_returns, log_price_paths, S_T, payoffs
    all_payoffs = np.concatenate(all_payoffs)
    price = np.exp(-r*T) * np.mean(all_payoffs)
    std_error = np.exp(-r*T) * np.std(all_payoffs) / np.sqrt(len(all_payoffs))
    g_bs = greeks(S, K, T, r, sigma, q, opt)
    return {
        "price": price, "std_error": std_error,
        "delta": g_bs["delta"], "gamma": g_bs["gamma"],
        "vega": g_bs["vega"], "theta": g_bs["theta"],
        "rho": g_bs["rho"], "paths": sample_paths
    }

# ─── BACKTESTING ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def backtest_strategy_cached(strategy, S0, K, T, r, sigma, q, n_days, n_sims):
    return backtest_strategy(strategy, S0, K, T, r, sigma, q, n_days, n_sims)

def backtest_strategy(strategy, S0, K, T, r, sigma, q, n_days, n_sims=1000):
    np.random.seed(42)
    dt = T / n_days
    Z = np.random.standard_normal((n_sims, n_days))
    drift = (r - q - 0.5*sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * Z
    log_price_paths = np.log(S0) + np.cumsum(log_returns, axis=1)
    S_final = np.exp(log_price_paths[:, -1])
    results = []
    for S_end in S_final:
        if strategy == "long_call":
            entry = bs(S0, K, T, r, sigma, q, "call")
            pnl = max(S_end - K, 0) - entry
        elif strategy == "long_put":
            entry = bs(S0, K, T, r, sigma, q, "put")
            pnl = max(K - S_end, 0) - entry
        elif strategy == "covered_call":
            call_entry = bs(S0, K, T, r, sigma, q, "call")
            pnl = (S_end - S0) + call_entry - max(S_end - K, 0)
        elif strategy == "protective_put":
            put_entry = bs(S0, K, T, r, sigma, q, "put")
            pnl = (S_end - S0) - put_entry + max(K - S_end, 0)
        elif strategy == "straddle":
            call_entry = bs(S0, K, T, r, sigma, q, "call")
            put_entry  = bs(S0, K, T, r, sigma, q, "put")
            pnl = max(S_end-K, 0) + max(K-S_end, 0) - call_entry - put_entry
        elif strategy == "strangle":
            K_call = K * 1.05; K_put = K * 0.95
            call_entry = bs(S0, K_call, T, r, sigma, q, "call")
            put_entry  = bs(S0, K_put,  T, r, sigma, q, "put")
            pnl = max(S_end-K_call, 0) + max(K_put-S_end, 0) - call_entry - put_entry
        results.append({"final_spot": S_end, "pnl": pnl, "return_pct": (pnl/S0)*100})
    return pd.DataFrame(results)

# ─── HELPERS PLOT ─────────────────────────────────────────────────────────────

def sty(ax, title, xl, yl):
    """Style épuré : traits fins, grille discrète, typographie légère."""
    ax.set_title(title, color=TITLE, fontsize=8.5, pad=7, fontweight="normal", loc="left")
    ax.set_xlabel(xl, color="#8eafc2", fontsize=7.5, fontweight="normal", labelpad=6)
    ax.set_ylabel(yl, color="#8eafc2", fontsize=7.5, fontweight="normal", labelpad=6)
    ax.grid(True, alpha=0.18, linewidth=0.3, linestyle="--")
    ax.tick_params(labelsize=7, colors="#9ca3af", width=0.5, length=3, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#2a4a6b")
    ax.set_axisbelow(True)

def annotate_be(ax, be_val, ymin, label_color=GREEN):
    """Annotation Break-Even en bas du graphique."""
    ax.annotate(
        f"BE  ${be_val:.2f}",
        xy=(be_val, ymin),
        xytext=(be_val, ymin),
        fontsize=6.5,
        color=label_color,
        fontfamily="monospace",
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#000000",
            edgecolor=label_color,
            linewidth=0.5,
            alpha=0.85
        )
    )

def annotate_vline(ax, x_val, label, color, ymin, ymax):
    """Annotation verticale fine avec label en bas."""
    ax.axvline(x_val, color=color, lw=0.7, linestyle="--", alpha=0.75)
    ax.annotate(
        label,
        xy=(x_val, ymin + (ymax - ymin) * 0.03),
        fontsize=6.2,
        color=color,
        fontfamily="monospace",
        ha="center",
        va="bottom",
        rotation=90,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="#000000",
            edgecolor=color,
            linewidth=0.4,
            alpha=0.8
        )
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ OPTIONS PRICER")
    st.markdown("---")
    st.markdown("### Mode")
    mode = st.selectbox("Choisir le mode", ["Pricing", "Implied Volatility", "Backtesting"])
    if mode == "Pricing":
        st.markdown("---")
        st.markdown("### Méthode de pricing")
        pricing_method = st.selectbox("Modèle", ["Black-Scholes", "Monte Carlo"])
    st.markdown("---")
    st.markdown("### Paramètres")
    S     = st.number_input("Spot S ($)", value=100.0, step=1.0)
    K     = st.number_input("Strike K ($)", value=100.0, step=1.0)
    T_day = st.number_input("Maturité (jours)", value=30, step=1, min_value=1)
    r     = st.number_input("Taux sans risque r (%)", value=5.0, step=0.1) / 100
    sigma = st.number_input("Volatilité σ (%)", value=20.0, step=0.5) / 100
    q     = st.number_input("Dividend yield q (%)", value=0.0, step=0.1) / 100
    if mode == "Pricing":
        prem = st.number_input("Prime payée ($) [opt.]", value=0.0, step=0.01)
    opt = st.radio("Type d'option", ["call", "put"], horizontal=True)
    if mode == "Pricing" and pricing_method == "Monte Carlo":
        st.markdown("---")
        st.markdown("### Paramètres Monte Carlo")
        n_sims   = st.selectbox("Simulations", [10000, 50000, 100000, 250000], index=2)
        n_steps  = st.selectbox("Pas de temps", [50, 100, 252], index=2)
        antithetic = st.checkbox("Variables antithétiques", value=True)
        seed     = st.number_input("Seed", value=42, step=1)
    elif mode == "Implied Volatility":
        st.markdown("---")
        st.markdown("### Prix de marché")
        market_price = st.number_input("Prix observé ($)", value=5.0, step=0.01, min_value=0.01)
    elif mode == "Backtesting":
        st.markdown("---")
        st.markdown("### Paramètres Backtest")
        strategy = st.selectbox(
            "Stratégie",
            ["long_call", "long_put", "covered_call", "protective_put", "straddle", "strangle"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        backtest_days  = st.slider("Horizon (jours)", 1, min(365, T_day), min(T_day, 30))
        n_simulations  = st.selectbox("Simulations", [100, 500, 1000, 2000], index=2)
    st.markdown("---")
    run = st.button("⚡ RUN", use_container_width=True, type="primary")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if mode == "Pricing":
    st.markdown(f"# Options Pricer — {pricing_method}")
elif mode == "Implied Volatility":
    st.markdown("# Implied Volatility Calibrator")
else:
    st.markdown("# Strategy Backtester")

st.markdown(
    '<div class="author-link">by <a href="https://www.linkedin.com/in/arthurcotten/" target="_blank">Arthur Cotten</a> • '
    '<a href="https://github.com/arthurcotten" target="_blank">@arthurcotten</a></div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_app, tab_en, tab_fr = st.tabs(["◈ App", "📖 Guide — EN", "📖 Guide — FR"])

with tab_en:
    st.markdown("## Options Pricer — User Guide")
    st.markdown("""
**Three modes are available from the sidebar.**

---

### 1. Pricing
Price a call or put option using either Black-Scholes or Monte Carlo simulation.

**Inputs:**
- **S** — current spot price of the underlying
- **K** — strike price
- **T** — time to maturity in days
- **r** — risk-free rate (%)
- **σ** — implied or historical volatility (%)
- **q** — continuous dividend yield (%)
- **Premium paid** *(optional)* — if filled, the break-even recalculates from your actual entry cost

**Outputs:**
| Metric | Meaning |
|---|---|
| Price | Theoretical fair value of the option |
| Break-even | Spot at which P&L = 0 at expiry |
| Prob ITM | Risk-neutral probability of finishing in-the-money |
| Time Value | Price minus intrinsic value |
| Moneyness | ITM / ATM / OTM |

**Greeks:**
| Greek | Interpretation |
|---|---|
| Delta | P&L change per $1 move in the underlying |
| Gamma | Rate of change of Delta |
| Vega | P&L change per +1% vol move |
| Theta | Daily time decay (negative for long options) |
| Rho | Sensitivity to interest rate changes |

**P&L chart:** green zone = profit territory, red zone = loss. The **BE** label shows your exact break-even level. The **K** label marks the strike.

---

### 2. Implied Volatility
Back-solve the volatility implied by a market-observed option price.

**How to use:** enter the market price of the option, adjust the other parameters to match the contract, and the tool calibrates IV using Brent's root-finding method.

**Volatility Skew chart** — IV across strikes (calls in cyan, puts in purple). A downward slope to the left of ATM is typical (put skew / fear premium).

**Term Structure chart** — IV across maturities for a fixed strike. A flat or inverted curve can signal short-term stress.

---

### 3. Backtesting
Simulate a strategy across thousands of Monte Carlo price paths to evaluate its statistical distribution.

**Available strategies:**
| Strategy | Description |
|---|---|
| Long Call | Buy a call, bet on upside |
| Long Put | Buy a put, bet on downside |
| Covered Call | Hold stock, sell a call — cap upside, collect premium |
| Protective Put | Hold stock, buy a put — insure against downside |
| Straddle | Buy call + put at same strike — bet on high volatility |
| Strangle | Buy OTM call + OTM put — same idea, cheaper, wider range needed |

**Reading the results:**
- **Win Rate** — % of paths finishing with positive P&L
- **Sharpe** — risk-adjusted return (annualised)
- **Distribution chart** — green bars = winning scenarios, red = losing
- **P&L vs Spot chart** — each dot is one simulation; the BE line shows where you break even
- **Percentile table** — worst 5%, median, best 5%, etc.

---
> Volatility and rates are assumed constant. This tool is for educational and analytical purposes.
""")

with tab_fr:
    st.markdown("## Options Pricer — Guide d'utilisation")
    st.markdown("""
**Trois modes disponibles depuis la barre latérale.**

---

### 1. Pricing
Pricer un call ou un put via Black-Scholes ou simulation Monte Carlo.

**Paramètres d'entrée :**
- **S** — prix spot actuel du sous-jacent
- **K** — prix d'exercice (strike)
- **T** — maturité en jours
- **r** — taux sans risque (%)
- **σ** — volatilité implicite ou historique (%)
- **q** — dividende continu (%)
- **Prime payée** *(optionnel)* — si renseigné, le break-even est recalculé à partir de votre coût d'entrée réel

**Résultats :**
| Métrique | Signification |
|---|---|
| Prix | Valeur théorique de l'option |
| Break-even | Spot auquel le P&L = 0 à maturité |
| Prob ITM | Probabilité risque-neutre de finir dans la monnaie |
| Time Value | Prix moins la valeur intrinsèque |
| Moneyness | ITM / ATM / OTM |

**Greeks :**
| Greek | Interprétation |
|---|---|
| Delta | Variation du P&L pour +1$ sur le sous-jacent |
| Gamma | Vitesse de variation du Delta |
| Vega | Variation du P&L pour +1% de volatilité |
| Theta | Perte de valeur quotidienne (négatif pour une option longue) |
| Rho | Sensibilité aux variations de taux |

**Graphique P&L :** zone verte = territoire de gain, rouge = perte. Le label **BE** indique votre break-even exact. Le label **K** marque le strike.

---

### 2. Volatilité Implicite
Retrouver la volatilité implicite à partir du prix de marché d'une option.

**Utilisation :** entrez le prix observé sur le marché, ajustez les autres paramètres pour correspondre au contrat, et l'outil calibre l'IV par la méthode de Brent.

**Graphique Skew** — IV en fonction des strikes (calls en cyan, puts en violet). Une pente descendante à gauche de l'ATM est typique (skew put / prime de crainte).

**Graphique Term Structure** — IV en fonction des maturités pour un strike fixe. Une courbe plate ou inversée peut signaler un stress à court terme.

---

### 3. Backtesting
Simuler une stratégie sur plusieurs milliers de chemins Monte Carlo pour analyser sa distribution statistique.

**Stratégies disponibles :**
| Stratégie | Description |
|---|---|
| Long Call | Achat d'un call — pari haussier |
| Long Put | Achat d'un put — pari baissier |
| Covered Call | Action longue + vente d'un call — plafonne le gain, encaisse la prime |
| Protective Put | Action longue + achat d'un put — protection contre la baisse |
| Straddle | Achat call + put au même strike — pari sur forte volatilité |
| Strangle | Achat call OTM + put OTM — même logique, moins cher, besoin de plus de mouvement |

**Lecture des résultats :**
- **Win Rate** — % de chemins terminant avec un P&L positif
- **Sharpe** — rendement ajusté du risque (annualisé)
- **Graphique distribution** — barres vertes = scénarios gagnants, rouges = perdants
- **Graphique P&L vs Spot** — chaque point est une simulation ; la ligne BE indique le seuil de rentabilité
- **Table percentiles** — pire 5%, médiane, meilleur 5%, etc.

---
> La volatilité et les taux sont supposés constants. Cet outil est destiné à des fins pédagogiques et analytiques.
""")

with tab_app:

    # ═══ MODE: PRICING ════════════════════════════════════════════════════════════

    if mode == "Pricing":
        if run or True:
            T = T_day / 365
            if pricing_method == "Black-Scholes":
                price = bs(S, K, T, r, sigma, q, opt)
                g = greeks(S, K, T, r, sigma, q, opt)
                std_error = None
                mc_paths  = None
            else:
                with st.spinner('Calcul Monte Carlo...'):
                    try:
                        mc_result = monte_carlo_pricer_cached(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed)
                        price     = mc_result["price"]
                        std_error = mc_result["std_error"]
                        g         = {k: mc_result[k] for k in ["delta","gamma","vega","theta","rho"]}
                        mc_paths  = mc_result["paths"]
                    except Exception as e:
                        st.error(f"❌ Erreur Monte Carlo: {str(e)}")
                        st.stop()

            prob    = prob_itm(S, K, T, r, sigma, q, opt)
            cost    = prem if prem > 0 else price
            be      = (K + cost) if opt == "call" else (K - cost)
            intrin  = max(S-K, 0) if opt == "call" else max(K-S, 0)
            tv      = price - intrin
            moneyness = S / K
            mon_lbl = ("ATM" if abs(moneyness-1) < 0.01
                       else "ITM" if (opt=="call" and moneyness>1) or (opt=="put" and moneyness<1)
                       else "OTM")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Prix", f"${price:.4f}")
            if std_error is not None:
                c2.metric("Std Error", f"${std_error:.4f}")
            else:
                c2.metric("Break-even", f"${be:.2f}")
            c3.metric("Prob ITM", f"{prob*100:.1f}%")
            c4.metric("Time Value", f"${tv:.4f}")
            c5.metric("Moneyness", mon_lbl)
            c6.metric("Intrinsèque", f"${intrin:.4f}")

            st.markdown("---")
            st.markdown("### Greeks")
            gc1, gc2, gc3, gc4, gc5 = st.columns(5)
            gc1.metric("Delta", f"{g['delta']:+.5f}")
            gc2.metric("Gamma", f"{g['gamma']:.6f}")
            gc3.metric("Vega",  f"{g['vega']:.5f}")
            gc4.metric("Theta", f"{g['theta']:+.5f}")
            gc5.metric("Rho",   f"{g['rho']:+.5f}")

            st.markdown("---")
            alerts = []
            if T < 7/365:              alerts.append("⚠️ Maturité < 7 jours")
            if abs(g["delta"]) < 0.10: alerts.append("⚠️ Delta très faible")
            if tv < 0.005:             alerts.append("⚠️ Time value nulle")
            if prob < 0.15:            alerts.append("⚠️ Prob ITM < 15%")
            for a in alerts:
                st.warning(a)
            if not alerts:
                st.success("✓ OK")

            S_range = np.linspace(S * 0.7, S * 1.3, 300)
            col1, col2 = st.columns([2, 1])

            with col1:
                fig1, ax = plt.subplots(figsize=(5.6, 3.2), facecolor=BG)
                ax.set_facecolor(PANEL)
                pnl = (np.maximum(S_range-K, 0) - cost if opt == "call"
                       else np.maximum(K-S_range, 0) - cost)
                ymin_val, ymax_val = pnl.min(), pnl.max()
                y_pad = (ymax_val - ymin_val) * 0.12
                ax.fill_between(S_range, pnl, 0, where=pnl >= 0, alpha=0.12, color=GREEN, zorder=1)
                ax.fill_between(S_range, pnl, 0, where=pnl <  0, alpha=0.12, color=RED,   zorder=1)
                ax.plot(S_range, pnl, color=ACCENT, lw=1.2, zorder=3)
                ax.axhline(0, color=GRAY, lw=0.5, alpha=0.5, zorder=2)
                annotate_vline(ax, K,  f"K  ${K:.0f}",    YELLOW,     ymin_val - y_pad, ymax_val)
                annotate_vline(ax, be, f"BE  ${be:.2f}",  GREEN,      ymin_val - y_pad, ymax_val)
                annotate_vline(ax, S,  f"S  ${S:.0f}",    "#9ca3af",  ymin_val - y_pad, ymax_val)
                ax.set_ylim(ymin_val - y_pad * 1.8, ymax_val + y_pad)
                annotate_be(ax, be, ymin_val - y_pad * 1.6)
                sty(ax, f"P&L  ·  {opt.upper()}", "Spot ($)", "P&L ($)")
                fig1.tight_layout(pad=1.2)
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

            with col2:
                if pricing_method == "Monte Carlo" and mc_paths is not None:
                    fig2, ax = plt.subplots(figsize=(3.1, 3.2), facecolor=BG)
                    ax.set_facecolor(PANEL)
                    ax.hist(mc_paths, bins=34, color=CYAN, alpha=0.6, edgecolor="none", linewidth=0)
                    ymax_h = ax.get_ylim()[1]
                    annotate_vline(ax, K, f"K  ${K:.0f}", YELLOW, 0, ymax_h)
                    sty(ax, "Distribution  S(T)", "Prix terminal ($)", "Freq")
                    fig2.tight_layout(pad=1.2)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

    # ═══ MODE: IMPLIED VOLATILITY ═════════════════════════════════════════════

    elif mode == "Implied Volatility":
        if run or True:
            T = T_day / 365
            with st.spinner('Calibration...'):
                iv = implied_volatility(market_price, S, K, T, r, q, opt)

            if np.isnan(iv):
                st.error("❌ Impossible de calibrer l'IV")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Vol Implicite",  f"{iv*100:.2f}%")
                col2.metric("Prix marché",    f"${market_price:.4f}")
                col3.metric("Prix théorique", f"${bs(S, K, T, r, iv, q, opt):.4f}")
                g_iv = greeks(S, K, T, r, iv, q, opt)
                col4.metric("Vega", f"{g_iv['vega']:.5f}")

                st.markdown("---")
                st.markdown("### Volatility Skew")
                col_skew, col_term = st.columns(2)

                with col_skew:
                    strikes    = np.linspace(S * 0.7, S * 1.3, 20)
                    iv_calls, iv_puts = [], []
                    for strike in strikes:
                        cp = bs(S, strike, T, r, iv, q, "call")
                        pp = bs(S, strike, T, r, iv, q, "put")
                        iv_calls.append(implied_volatility(cp, S, strike, T, r, q, "call"))
                        iv_puts.append( implied_volatility(pp, S, strike, T, r, q, "put"))

                    fig_skew, ax = plt.subplots(figsize=(4.8, 3.4), facecolor=BG)
                    ax.set_facecolor(PANEL)
                    valid_c = [(k/S, v*100) for k, v in zip(strikes, iv_calls) if v and not np.isnan(v)]
                    valid_p = [(k/S, v*100) for k, v in zip(strikes, iv_puts)  if v and not np.isnan(v)]
                    if valid_c:
                        xc, yc = zip(*valid_c)
                        ax.plot(xc, yc, color=CYAN,   lw=1.1, marker='o', markersize=3.2, label='Calls', markeredgewidth=0)
                    if valid_p:
                        xp, yp = zip(*valid_p)
                        ax.plot(xp, yp, color=PURPLE, lw=1.1, marker='s', markersize=3.2, label='Puts',  markeredgewidth=0)
                    ax.axvline(1.0,     color=GRAY,   lw=0.5, linestyle=":", alpha=0.6, label="ATM")
                    ax.axhline(iv*100,  color=ACCENT, lw=0.5, linestyle="--", alpha=0.6)
                    ax.legend(fontsize=7, facecolor=PANEL, edgecolor="#2a4a6b", labelcolor=TEXT, framealpha=0.8)
                    sty(ax, "Skew  ·  Calls vs Puts", "Moneyness (K/S)", "IV (%)")
                    fig_skew.tight_layout(pad=1.2)
                    st.pyplot(fig_skew, use_container_width=True)
                    plt.close(fig_skew)

                with col_term:
                    maturities   = np.linspace(max(T, 7/365), min(T*3, 1.0), 12)
                    term_iv_list = []
                    for mat in maturities:
                        cp   = bs(S, K, mat, r, iv, q, "call")
                        iv_c = implied_volatility(cp, S, K, mat, r, q, "call")
                        term_iv_list.append(iv_c)

                    fig_term, ax = plt.subplots(figsize=(4.8, 3.4), facecolor=BG)
                    ax.set_facecolor(PANEL)
                    valid_t = [(m*365, v*100) for m, v in zip(maturities, term_iv_list) if v and not np.isnan(v)]
                    if valid_t:
                        xt, yt = zip(*valid_t)
                        ax.plot(xt, yt, color=CYAN, lw=1.1, marker='o', markersize=3.2, markeredgewidth=0)
                    ax.axvline(T*365,  color=GRAY,   lw=0.5, linestyle=":", alpha=0.6)
                    ax.axhline(iv*100, color=ACCENT, lw=0.5, linestyle="--", alpha=0.6)
                    sty(ax, "Term Structure", "Maturité (jours)", "IV (%)")
                    fig_term.tight_layout(pad=1.2)
                    st.pyplot(fig_term, use_container_width=True)
                    plt.close(fig_term)

    # ═══ MODE: BACKTESTING ════════════════════════════════════════════════════

    elif mode == "Backtesting":
        if run or True:
            T = T_day / 365
            with st.spinner('Backtesting...'):
                try:
                    results_df = backtest_strategy_cached(strategy, S, K, T, r, sigma, q, backtest_days, n_simulations)
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
                    st.stop()

            mean_pnl   = results_df['pnl'].mean()
            median_pnl = results_df['pnl'].median()
            std_pnl    = results_df['pnl'].std()
            win_rate   = (results_df['pnl'] > 0).sum() / len(results_df) * 100
            max_gain   = results_df['pnl'].max()
            max_loss   = results_df['pnl'].min()
            sharpe     = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0

            st.markdown("### Performance")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("P&L Moyen",  f"${mean_pnl:.2f}")
            c2.metric("P&L Médian", f"${median_pnl:.2f}")
            c3.metric("Win Rate",   f"{win_rate:.1f}%")
            c4.metric("Max Gain",   f"${max_gain:.2f}")
            c5.metric("Max Loss",   f"${max_loss:.2f}")
            c6.metric("Sharpe",     f"{sharpe:.3f}")

            st.markdown("---")
            col_hist, col_scatter = st.columns(2)

            with col_hist:
                fig_hist, ax = plt.subplots(figsize=(4.8, 3.4), facecolor=BG)
                ax.set_facecolor(PANEL)
                pnl_arr = results_df['pnl'].values
                bins    = np.linspace(pnl_arr.min(), pnl_arr.max(), 44)
                ax.hist(pnl_arr[pnl_arr >= 0], bins=bins, color=GREEN, alpha=0.55, edgecolor="none")
                ax.hist(pnl_arr[pnl_arr <  0], bins=bins, color=RED,   alpha=0.55, edgecolor="none")
                ymax_h = ax.get_ylim()[1]
                ax.axvline(mean_pnl, color=ACCENT, lw=0.8, linestyle="--", alpha=0.9, label=f"Moy  ${mean_pnl:.2f}")
                ax.axvline(0, color=GRAY, lw=0.5, linestyle="-", alpha=0.6, label="BE  $0.00")
                ax.set_ylim(0, ymax_h * 1.12)
                ax.annotate(
                    "BE  $0.00",
                    xy=(0, ymax_h * 0.02),
                    fontsize=6.2, color=GRAY, fontfamily="monospace",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", edgecolor=GRAY, linewidth=0.4, alpha=0.85)
                )
                ax.legend(fontsize=7, facecolor=PANEL, edgecolor="#2a4a6b", labelcolor=TEXT, framealpha=0.8)
                sty(ax, f"Distribution P&L  ·  {strategy.replace('_', ' ').title()}", "P&L ($)", "Freq")
                fig_hist.tight_layout(pad=1.2)
                st.pyplot(fig_hist, use_container_width=True)
                plt.close(fig_hist)

            with col_scatter:
                fig_spot, ax = plt.subplots(figsize=(4.8, 3.4), facecolor=BG)
                ax.set_facecolor(PANEL)
                spots = results_df['final_spot'].values
                pnls  = results_df['pnl'].values
                ax.scatter(spots[pnls >= 0], pnls[pnls >= 0], alpha=0.45, s=8, color=GREEN,  edgecolors="none", zorder=3)
                ax.scatter(spots[pnls <  0], pnls[pnls <  0], alpha=0.45, s=8, color=PURPLE, edgecolors="none", zorder=3)
                ymin_s, ymax_s = pnls.min(), pnls.max()
                y_pad_s = (ymax_s - ymin_s) * 0.12
                ax.axhline(0, color=GRAY, lw=0.5, linestyle="-", alpha=0.55, zorder=2)
                annotate_vline(ax, S, f"S0  ${S:.0f}", YELLOW,     ymin_s - y_pad_s, ymax_s)
                annotate_vline(ax, K, f"K  ${K:.0f}",  "#9ca3af",  ymin_s - y_pad_s, ymax_s)
                ax.set_ylim(ymin_s - y_pad_s * 1.8, ymax_s + y_pad_s)
                ax.annotate(
                    "BE  $0.00",
                    xy=(spots.min() + (spots.max()-spots.min())*0.03, 0),
                    fontsize=6.2, color=GRAY, fontfamily="monospace",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", edgecolor=GRAY, linewidth=0.4, alpha=0.85)
                )
                ax.legend(
                    handles=[
                        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=GREEN,  markersize=5, label='Gain',  linewidth=0),
                        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=PURPLE, markersize=5, label='Perte', linewidth=0),
                        plt.Line2D([0],[0], color=YELLOW, lw=0.8, linestyle='--', label='S0'),
                    ],
                    fontsize=7, facecolor=PANEL, edgecolor="#2a4a6b", labelcolor=TEXT, framealpha=0.8
                )
                sty(ax, "P&L vs Spot Final", "Spot final ($)", "P&L ($)")
                fig_spot.tight_layout(pad=1.2)
                st.pyplot(fig_spot, use_container_width=True)
                plt.close(fig_spot)

            st.markdown("---")
            st.markdown("### Percentiles")
            percentiles = [5, 25, 50, 75, 95]
            pct_data = {
                "Percentile": [f"{p}%" for p in percentiles],
                "P&L ($)":    [f"${results_df['pnl'].quantile(p/100):.2f}" for p in percentiles],
                "Return (%)": [f"{results_df['return_pct'].quantile(p/100):.2f}%" for p in percentiles]
            }
            st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)
