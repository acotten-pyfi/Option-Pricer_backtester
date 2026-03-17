"""
Options Pricer — Streamlit
Run with: streamlit run Option-Pricer_Backtester.py
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
    .main { background-color: #000000 !important; }
    .block-container { padding-top: 1rem !important; background-color: #000000 !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; }
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] { background-color: #000000 !important; }
    h1 { color: #4a9eff !important; font-family: monospace !important; font-weight: bold !important;
         font-size: 1.73rem !important; text-shadow: 0 0 8px rgba(74,158,255,0.4); }
    h2, h3 { color: #4a9eff !important; font-family: monospace !important; font-weight: normal !important; font-size: 1.16rem !important; }
    p, span, div, label, .stMarkdown { color: #ffffff !important; font-weight: normal !important; font-size: 0.72rem !important; }
    input[type="number"], input[type="text"], .stNumberInput input, .stTextInput input {
        background-color: #0a0a0a !important; color: #ffffff !important; border: 2px solid #4a9eff !important; }
    .stSelectbox > div > div, select {
        background-color: #0a0a0a !important; color: #ffffff !important; border: 2px solid #4a9eff !important; }
    .stSelectbox div[data-baseweb="select"] > div, .stSelectbox ul, .stSelectbox li,
    [role="listbox"], [role="option"] { background-color: #0a0a0a !important; color: #ffffff !important; }
    .stSelectbox li:hover, [role="option"]:hover { background-color: #1a1a1a !important; color: #4a9eff !important; }
    div[data-testid="metric-container"] {
        background-color: #000000 !important; border: 2px solid #4a9eff !important;
        border-radius: 10px !important; padding: 13px !important; box-shadow: 0 0 15px rgba(74,158,255,0.3) !important; }
    div[data-testid="metric-container"] label, div[data-testid="metric-container"] label p {
        color: #4a9eff !important; font-weight: normal !important; font-size: 0.72rem !important; }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div, div[data-testid="stMetricValue"] p {
        color: #ffffff !important; font-weight: normal !important; font-size: 1.3rem !important; }
    div[data-testid="stMetricDelta"] { color: #4a9eff !important; }
    .stAlert, .stSuccess, .stWarning { background-color: #0a0a0a !important; border: 2px solid #4a9eff !important; }
    .stAlert p, .stSuccess p, .stWarning p { color: #ffffff !important; }
    .dataframe { font-size: 0.72rem !important; font-family: monospace !important;
                 background-color: #000000 !important; border: 2px solid #4a9eff !important; }
    .dataframe th { background-color: #0a0a0a !important; color: #4a9eff !important;
                    border: 1px solid #4a9eff !important; padding: 6px !important; }
    .dataframe td { color: #ffffff !important; background-color: #000000 !important;
                    border: 1px solid #333333 !important; padding: 6px !important; }
    .author-link { color: #888888 !important; font-size: 0.72rem; font-family: monospace; margin-top: -10px; margin-bottom: 15px; }
    .author-link a { color: #4a9eff !important; text-decoration: none; }
    .author-link a:hover { color: #60a5fa !important; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB STYLE ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#000000", "axes.facecolor": "#0a0a0a",
    "axes.edgecolor": "#4a9eff",   "axes.labelcolor": "#4a9eff",
    "text.color": "#e5e7eb",       "xtick.color": "#e5e7eb",
    "ytick.color": "#e5e7eb",      "grid.color": "#1e2a38",
    "grid.linewidth": 0.35,        "grid.alpha": 0.5,
    "font.family": "monospace",    "font.weight": "normal",
    "xtick.major.width": 0.5,      "ytick.major.width": 0.5,
})

BG, PANEL, BORDER = "#000000", "#0a0a0a", "#4a9eff"
ACCENT, GREEN, RED = "#4a9eff", "#10b981", "#ef4444"
YELLOW, PURPLE, CYAN = "#f59e0b", "#8b5cf6", "#06b6d4"
GRAY, TEXT, TITLE = "#6b7280", "#e5e7eb", "#4a9eff"

# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

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
    d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    nd1 = norm.pdf(d1)
    if opt == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)) / 365
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-(S*np.exp(-q*T)*nd1*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)) / 365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
    gamma = np.exp(-q*T)*nd1 / (S*sigma*np.sqrt(T))
    vega  = S*np.exp(-q*T)*nd1*np.sqrt(T) / 100
    return {"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}

def prob_itm(S, K, T, r, sigma, q=0.0, opt="call"):
    if T <= 1e-10 or sigma <= 1e-10:
        return 1.0 if (opt=="call" and S>K) or (opt=="put" and S<K) else 0.0
    d2 = (np.log(S/K)+(r-q-0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d2) if opt=="call" else norm.cdf(-d2)

def implied_volatility(market_price, S, K, T, r, q=0.0, opt="call"):
    if T <= 1e-10: return np.nan
    intrinsic = max(S-K,0) if opt=="call" else max(K-S,0)
    if market_price < intrinsic*0.99: return np.nan
    try:
        return brentq(lambda sig: bs(S,K,T,r,sig,q,opt)-market_price, 0.001, 5.0, maxiter=100)
    except:
        return np.nan

@st.cache_data(ttl=300)
def monte_carlo_pricer_cached(S, K, T, r, sigma, q, opt, n_sims, n_steps, antithetic, seed):
    np.random.seed(seed)
    dt = T / n_steps
    batch_size = min(n_sims, 100000)
    all_payoffs, sample_paths = [], None
    for batch in range(int(np.ceil(n_sims/batch_size))):
        nb = min(batch_size, n_sims-batch*batch_size)
        n  = nb//2 if antithetic else nb
        Z  = np.random.standard_normal((n, n_steps))
        if antithetic: Z = np.concatenate([Z, -Z], axis=0)
        S_T = S * np.exp(np.sum((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
        if batch == 0: sample_paths = S_T[:min(1000,n_sims)].copy()
        all_payoffs.append(np.maximum(S_T-K,0) if opt=="call" else np.maximum(K-S_T,0))
        del Z, S_T
    all_payoffs = np.concatenate(all_payoffs)
    price = np.exp(-r*T)*np.mean(all_payoffs)
    se    = np.exp(-r*T)*np.std(all_payoffs)/np.sqrt(len(all_payoffs))
    g     = greeks(S,K,T,r,sigma,q,opt)
    return {"price":price,"std_error":se,"paths":sample_paths,**g}

@st.cache_data(ttl=300)
def backtest_strategy_cached(strategy, S0, K, T, r, sigma, q, n_days, n_sims):
    np.random.seed(42)
    dt   = T / n_days
    Z    = np.random.standard_normal((n_sims, n_days))
    S_f  = S0 * np.exp(np.sum((r-q-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
    rows = []
    for Se in S_f:
        if   strategy=="long_call":       pnl = max(Se-K,0) - bs(S0,K,T,r,sigma,q,"call")
        elif strategy=="long_put":        pnl = max(K-Se,0) - bs(S0,K,T,r,sigma,q,"put")
        elif strategy=="covered_call":    pnl = (Se-S0)+bs(S0,K,T,r,sigma,q,"call")-max(Se-K,0)
        elif strategy=="protective_put":  pnl = (Se-S0)-bs(S0,K,T,r,sigma,q,"put")+max(K-Se,0)
        elif strategy=="straddle":        pnl = max(Se-K,0)+max(K-Se,0)-bs(S0,K,T,r,sigma,q,"call")-bs(S0,K,T,r,sigma,q,"put")
        elif strategy=="strangle":
            Kc,Kp = K*1.05, K*0.95
            pnl   = max(Se-Kc,0)+max(Kp-Se,0)-bs(S0,Kc,T,r,sigma,q,"call")-bs(S0,Kp,T,r,sigma,q,"put")
        rows.append({"final_spot":Se,"pnl":pnl,"return_pct":(pnl/S0)*100})
    return pd.DataFrame(rows)

# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────

def sty(ax, title, xl, yl):
    ax.set_title(title, color=TITLE, fontsize=8.5, pad=7, fontweight="normal", loc="left")
    ax.set_xlabel(xl, color="#8eafc2", fontsize=7.5, labelpad=6)
    ax.set_ylabel(yl, color="#8eafc2", fontsize=7.5, labelpad=6)
    ax.grid(True, alpha=0.18, linewidth=0.3, linestyle="--")
    ax.tick_params(labelsize=7, colors="#9ca3af", width=0.5, length=3, pad=4)
    for sp in ax.spines.values(): sp.set_linewidth(0.6); sp.set_edgecolor("#2a4a6b")
    ax.set_axisbelow(True)

def annotate_be(ax, val, ymin, color=GREEN):
    ax.annotate(f"BE  ${val:.2f}", xy=(val, ymin), fontsize=6.5, color=color,
                fontfamily="monospace", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#000000", edgecolor=color, linewidth=0.5, alpha=0.85))

def vline(ax, x, label, color, ymin, ymax):
    ax.axvline(x, color=color, lw=0.7, linestyle="--", alpha=0.75)
    ax.annotate(label, xy=(x, ymin+(ymax-ymin)*0.03), fontsize=6.2, color=color,
                fontfamily="monospace", ha="center", va="bottom", rotation=90,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", edgecolor=color, linewidth=0.4, alpha=0.8))

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ OPTIONS PRICER")
    st.markdown("---")
    st.markdown("### Mode")
    mode = st.selectbox("Select mode", ["Pricing", "Implied Volatility", "Backtesting"])

    # defaults
    pricing_method = "Black-Scholes"
    market_price   = 5.0
    strategy       = "long_call"
    backtest_days  = 30
    n_simulations  = 1000
    prem           = 0.0
    n_sims         = 100000
    n_steps        = 252
    antithetic     = True
    seed           = 42

    if mode == "Pricing":
        st.markdown("---")
        st.markdown("### Pricing model")
        pricing_method = st.selectbox("Model", ["Black-Scholes", "Monte Carlo"])

    st.markdown("---")
    st.markdown("### Parameters")
    S     = st.number_input("Spot S ($)",            value=100.0, step=1.0)
    K     = st.number_input("Strike K ($)",          value=100.0, step=1.0)
    T_day = st.number_input("Maturity (days)",       value=30,    step=1,   min_value=1)
    r     = st.number_input("Risk-free rate r (%)",  value=5.0,   step=0.1) / 100
    sigma = st.number_input("Volatility σ (%)",      value=20.0,  step=0.5) / 100
    q     = st.number_input("Dividend yield q (%)",  value=0.0,   step=0.1) / 100
    opt   = st.radio("Option type", ["call", "put"], horizontal=True)

    if mode == "Pricing":
        prem = st.number_input("Premium paid ($) [opt.]", value=0.0, step=0.01)
        if pricing_method == "Monte Carlo":
            st.markdown("---")
            st.markdown("### Monte Carlo settings")
            n_sims     = st.selectbox("Simulations", [10000, 50000, 100000, 250000], index=2)
            n_steps    = st.selectbox("Time steps",  [50, 100, 252], index=2)
            antithetic = st.checkbox("Antithetic variates", value=True)
            seed       = st.number_input("Seed", value=42, step=1)

    elif mode == "Implied Volatility":
        st.markdown("---")
        st.markdown("### Market price")
        market_price = st.number_input("Observed price ($)", value=5.0, step=0.01, min_value=0.01)

    elif mode == "Backtesting":
        st.markdown("---")
        st.markdown("### Backtest settings")
        strategy = st.selectbox("Strategy",
            ["long_call","long_put","covered_call","protective_put","straddle","strangle"],
            format_func=lambda x: x.replace('_',' ').title())
        backtest_days = st.slider("Horizon (days)", 1, min(365,T_day), min(T_day,30))
        n_simulations = st.selectbox("Simulations", [100, 500, 1000, 2000], index=2)

    st.markdown("---")
    run = st.button("⚡ RUN", use_container_width=True, type="primary")

# ─── HEADER ───────────────────────────────────────────────────────────────────
title_map = {"Pricing": f"Options Pricer — {pricing_method}",
             "Implied Volatility": "Implied Volatility Calibrator",
             "Backtesting": "Strategy Backtester"}
st.markdown(f"# {title_map[mode]}")
st.markdown('<div class="author-link">by <a href="https://www.linkedin.com/in/arthurcotten/" target="_blank">Arthur Cotten</a> • '
            '<a href="https://github.com/arthurcotten" target="_blank">@arthurcotten</a></div>', unsafe_allow_html=True)
st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_app, tab_en, tab_fr = st.tabs(["◈  App", "📖  Guide — EN", "📖  Guide — FR"])

# ══════════════════════════════════════════════════════════════════════════════
with tab_app:
    T = T_day / 365

    # ── PRICING ───────────────────────────────────────────────────────────────
    if mode == "Pricing":
        if pricing_method == "Black-Scholes":
            price = bs(S,K,T,r,sigma,q,opt); g = greeks(S,K,T,r,sigma,q,opt)
            std_error = None; mc_paths = None
        else:
            with st.spinner("Running Monte Carlo..."):
                try:
                    mc = monte_carlo_pricer_cached(S,K,T,r,sigma,q,opt,n_sims,n_steps,antithetic,int(seed))
                    price=mc["price"]; std_error=mc["std_error"]; mc_paths=mc["paths"]
                    g = {k:mc[k] for k in ["delta","gamma","vega","theta","rho"]}
                except Exception as e:
                    st.error(f"Monte Carlo error: {e}"); st.stop()

        prob  = prob_itm(S,K,T,r,sigma,q,opt)
        cost  = prem if prem>0 else price
        be    = (K+cost) if opt=="call" else (K-cost)
        intr  = max(S-K,0) if opt=="call" else max(K-S,0)
        tv    = price - intr
        mon   = S/K
        mlbl  = "ATM" if abs(mon-1)<0.01 else ("ITM" if (opt=="call" and mon>1) or (opt=="put" and mon<1) else "OTM")

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Price",      f"${price:.4f}")
        c2.metric("Std Error" if std_error else "Break-even",
                  f"${std_error:.4f}" if std_error else f"${be:.2f}")
        c3.metric("Prob ITM",   f"{prob*100:.1f}%")
        c4.metric("Time Value", f"${tv:.4f}")
        c5.metric("Moneyness",  mlbl)
        c6.metric("Intrinsic",  f"${intr:.4f}")

        st.markdown("---")
        st.markdown("### Greeks")
        g1,g2,g3,g4,g5 = st.columns(5)
        g1.metric("Delta", f"{g['delta']:+.5f}"); g2.metric("Gamma", f"{g['gamma']:.6f}")
        g3.metric("Vega",  f"{g['vega']:.5f}");  g4.metric("Theta", f"{g['theta']:+.5f}")
        g5.metric("Rho",   f"{g['rho']:+.5f}")

        st.markdown("---")
        alr = []
        if T<7/365:               alr.append("⚠️ Maturity < 7 days")
        if abs(g["delta"])<0.10:  alr.append("⚠️ Delta very low")
        if tv<0.005:              alr.append("⚠️ Time value near zero")
        if prob<0.15:             alr.append("⚠️ Prob ITM < 15%")
        for a in alr: st.warning(a)
        if not alr: st.success("✓ OK")

        Sr = np.linspace(S*0.7, S*1.3, 300)
        col1, col2 = st.columns([2,1])
        with col1:
            fig, ax = plt.subplots(figsize=(5.6,3.2), facecolor=BG); ax.set_facecolor(PANEL)
            pnl = (np.maximum(Sr-K,0)-cost if opt=="call" else np.maximum(K-Sr,0)-cost)
            ym, yM = pnl.min(), pnl.max(); yp = (yM-ym)*0.12
            ax.fill_between(Sr,pnl,0,where=pnl>=0,alpha=0.12,color=GREEN,zorder=1)
            ax.fill_between(Sr,pnl,0,where=pnl<0, alpha=0.12,color=RED,  zorder=1)
            ax.plot(Sr,pnl,color=ACCENT,lw=1.2,zorder=3)
            ax.axhline(0,color=GRAY,lw=0.5,alpha=0.5,zorder=2)
            vline(ax,K, f"K  ${K:.0f}",   YELLOW,    ym-yp,yM)
            vline(ax,be,f"BE  ${be:.2f}", GREEN,     ym-yp,yM)
            vline(ax,S, f"S  ${S:.0f}",   "#9ca3af", ym-yp,yM)
            ax.set_ylim(ym-yp*1.8, yM+yp); annotate_be(ax,be,ym-yp*1.6)
            sty(ax,f"P&L  ·  {opt.upper()}","Spot ($)","P&L ($)")
            fig.tight_layout(pad=1.2); st.pyplot(fig,use_container_width=True); plt.close(fig)
        with col2:
            if pricing_method=="Monte Carlo" and mc_paths is not None:
                fig2,ax2 = plt.subplots(figsize=(3.1,3.2),facecolor=BG); ax2.set_facecolor(PANEL)
                ax2.hist(mc_paths,bins=34,color=CYAN,alpha=0.6,edgecolor="none")
                yh = ax2.get_ylim()[1]; vline(ax2,K,f"K  ${K:.0f}",YELLOW,0,yh)
                sty(ax2,"Distribution  S(T)","Terminal price ($)","Freq")
                fig2.tight_layout(pad=1.2); st.pyplot(fig2,use_container_width=True); plt.close(fig2)

    # ── IMPLIED VOLATILITY ────────────────────────────────────────────────────
    elif mode == "Implied Volatility":
        with st.spinner("Calibrating..."):
            iv = implied_volatility(market_price,S,K,T,r,q,opt)
        if np.isnan(iv):
            st.error("❌ Cannot calibrate IV — check inputs")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Implied Vol",       f"{iv*100:.2f}%")
            c2.metric("Market price",      f"${market_price:.4f}")
            c3.metric("Theoretical price", f"${bs(S,K,T,r,iv,q,opt):.4f}")
            c4.metric("Vega",              f"{greeks(S,K,T,r,iv,q,opt)['vega']:.5f}")

            st.markdown("---"); st.markdown("### Volatility Surface")
            cs, ct = st.columns(2)
            with cs:
                stks = np.linspace(S*0.7,S*1.3,20)
                ivc  = [implied_volatility(bs(S,k,T,r,iv,q,"call"),S,k,T,r,q,"call") for k in stks]
                ivp  = [implied_volatility(bs(S,k,T,r,iv,q,"put"), S,k,T,r,q,"put")  for k in stks]
                fig_s,ax = plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
                vc = [(k/S,v*100) for k,v in zip(stks,ivc) if v and not np.isnan(v)]
                vp = [(k/S,v*100) for k,v in zip(stks,ivp) if v and not np.isnan(v)]
                if vc: xc,yc=zip(*vc); ax.plot(xc,yc,color=CYAN,  lw=1.1,marker='o',markersize=3.2,label='Calls',markeredgewidth=0)
                if vp: xp,yp=zip(*vp); ax.plot(xp,yp,color=PURPLE,lw=1.1,marker='s',markersize=3.2,label='Puts', markeredgewidth=0)
                ax.axvline(1.0,   color=GRAY,  lw=0.5,linestyle=":",alpha=0.6,label="ATM")
                ax.axhline(iv*100,color=ACCENT,lw=0.5,linestyle="--",alpha=0.6)
                ax.legend(fontsize=7,facecolor=PANEL,edgecolor="#2a4a6b",labelcolor=TEXT,framealpha=0.8)
                sty(ax,"Skew  ·  Calls vs Puts","Moneyness (K/S)","IV (%)")
                fig_s.tight_layout(pad=1.2); st.pyplot(fig_s,use_container_width=True); plt.close(fig_s)
            with ct:
                mats = np.linspace(max(T,7/365),min(T*3,1.0),12)
                ivt  = [implied_volatility(bs(S,K,m,r,iv,q,"call"),S,K,m,r,q,"call") for m in mats]
                fig_t,ax = plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
                vt = [(m*365,v*100) for m,v in zip(mats,ivt) if v and not np.isnan(v)]
                if vt: xt,yt=zip(*vt); ax.plot(xt,yt,color=CYAN,lw=1.1,marker='o',markersize=3.2,markeredgewidth=0)
                ax.axvline(T*365, color=GRAY,  lw=0.5,linestyle=":",alpha=0.6)
                ax.axhline(iv*100,color=ACCENT,lw=0.5,linestyle="--",alpha=0.6)
                sty(ax,"Term Structure","Maturity (days)","IV (%)")
                fig_t.tight_layout(pad=1.2); st.pyplot(fig_t,use_container_width=True); plt.close(fig_t)

    # ── BACKTESTING ───────────────────────────────────────────────────────────
    elif mode == "Backtesting":
        with st.spinner("Running backtest..."):
            try:
                df = backtest_strategy_cached(strategy,S,K,T,r,sigma,q,backtest_days,n_simulations)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

        mp  = df['pnl'].mean(); med = df['pnl'].median(); sdp = df['pnl'].std()
        wr  = (df['pnl']>0).sum()/len(df)*100
        mg  = df['pnl'].max(); ml  = df['pnl'].min()
        shr = (mp/sdp*np.sqrt(252)) if sdp>0 else 0

        st.markdown("### Performance")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Avg P&L",    f"${mp:.2f}");  c2.metric("Median P&L", f"${med:.2f}")
        c3.metric("Win Rate",   f"{wr:.1f}%");  c4.metric("Max Gain",   f"${mg:.2f}")
        c5.metric("Max Loss",   f"${ml:.2f}");  c6.metric("Sharpe",     f"{shr:.3f}")

        st.markdown("---")
        ch, cs2 = st.columns(2)
        with ch:
            fig_h,ax = plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
            pa  = df['pnl'].values
            bns = np.linspace(pa.min(),pa.max(),44)
            ax.hist(pa[pa>=0],bins=bns,color=GREEN,alpha=0.55,edgecolor="none")
            ax.hist(pa[pa<0], bins=bns,color=RED,  alpha=0.55,edgecolor="none")
            yh2 = ax.get_ylim()[1]
            ax.axvline(mp,color=ACCENT,lw=0.8,linestyle="--",alpha=0.9,label=f"Avg  ${mp:.2f}")
            ax.axvline(0, color=GRAY,  lw=0.5,alpha=0.6,label="BE  $0.00")
            ax.set_ylim(0,yh2*1.12)
            ax.annotate("BE  $0.00",xy=(0,yh2*0.02),fontsize=6.2,color=GRAY,fontfamily="monospace",
                        ha="center",va="bottom",bbox=dict(boxstyle="round,pad=0.2",facecolor="#000000",edgecolor=GRAY,linewidth=0.4,alpha=0.85))
            ax.legend(fontsize=7,facecolor=PANEL,edgecolor="#2a4a6b",labelcolor=TEXT,framealpha=0.8)
            sty(ax,f"P&L Distribution  ·  {strategy.replace('_',' ').title()}","P&L ($)","Freq")
            fig_h.tight_layout(pad=1.2); st.pyplot(fig_h,use_container_width=True); plt.close(fig_h)

        with cs2:
            fig_sc,ax = plt.subplots(figsize=(4.8,3.4),facecolor=BG); ax.set_facecolor(PANEL)
            sp  = df['final_spot'].values; pn = df['pnl'].values
            ax.scatter(sp[pn>=0],pn[pn>=0],alpha=0.45,s=8,color=GREEN, edgecolors="none",zorder=3)
            ax.scatter(sp[pn<0], pn[pn<0], alpha=0.45,s=8,color=PURPLE,edgecolors="none",zorder=3)
            ys,yS = pn.min(),pn.max(); yps = (yS-ys)*0.12
            ax.axhline(0,color=GRAY,lw=0.5,alpha=0.55,zorder=2)
            vline(ax,S,f"S0  ${S:.0f}",YELLOW,   ys-yps,yS)
            vline(ax,K,f"K  ${K:.0f}", "#9ca3af",ys-yps,yS)
            ax.set_ylim(ys-yps*1.8,yS+yps)
            ax.annotate("BE  $0.00",xy=(sp.min()+(sp.max()-sp.min())*0.03,0),fontsize=6.2,color=GRAY,
                        fontfamily="monospace",ha="left",va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2",facecolor="#000000",edgecolor=GRAY,linewidth=0.4,alpha=0.85))
            ax.legend(handles=[
                plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=GREEN, markersize=5,label='Gain',linewidth=0),
                plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=PURPLE,markersize=5,label='Loss',linewidth=0),
                plt.Line2D([0],[0],color=YELLOW,lw=0.8,linestyle='--',label='S0'),
            ],fontsize=7,facecolor=PANEL,edgecolor="#2a4a6b",labelcolor=TEXT,framealpha=0.8)
            sty(ax,"P&L vs Final Spot","Final spot ($)","P&L ($)")
            fig_sc.tight_layout(pad=1.2); st.pyplot(fig_sc,use_container_width=True); plt.close(fig_sc)

        st.markdown("---"); st.markdown("### Percentiles")
        pcts = [5,25,50,75,95]
        st.dataframe(pd.DataFrame({
            "Percentile": [f"{p}%" for p in pcts],
            "P&L ($)":    [f"${df['pnl'].quantile(p/100):.2f}" for p in pcts],
            "Return (%)": [f"{df['return_pct'].quantile(p/100):.2f}%" for p in pcts],
        }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_en:
    st.markdown("## Options Pricer — User Guide")
    st.markdown("""
Three modes are available from the sidebar.

---

### 1. Pricing
Price a vanilla call or put using Black-Scholes or Monte Carlo.

**Inputs**

| Parameter | Description |
|---|---|
| S | Current spot price of the underlying |
| K | Strike price |
| T | Time to maturity in days |
| r | Risk-free rate (%) |
| σ | Volatility — implied or historical (%) |
| q | Continuous dividend yield (%) |
| Premium paid | Optional — if filled, break-even recalculates from your actual entry cost |

**Output metrics**

| Metric | Meaning |
|---|---|
| Price | Theoretical fair value |
| Break-even | Spot at which P&L = 0 at expiry |
| Prob ITM | Risk-neutral probability of finishing in-the-money |
| Time Value | Price minus intrinsic value |
| Moneyness | ITM / ATM / OTM |

**Greeks**

| Greek | Interpretation |
|---|---|
| Delta | P&L change per $1 move in the underlying |
| Gamma | Rate of change of Delta |
| Vega | P&L change per +1% move in volatility |
| Theta | Daily time decay (negative for long options) |
| Rho | Sensitivity to interest rate moves |

**P&L chart** — green = profit zone, red = loss zone. BE marks your exact break-even. K marks the strike, S marks the current spot.

---

### 2. Implied Volatility
Back-solve the volatility implied by a market-observed option price (Brent's method).

Enter the market price, match the other parameters to the contract, and the tool returns the IV.

**Skew chart** — IV across strikes. A downward slope to the left of ATM is the typical put skew.

**Term Structure chart** — IV across maturities for a fixed strike. Inversion can signal short-term stress.

---

### 3. Backtesting
Simulate a strategy across Monte Carlo paths and analyse the statistical P&L distribution.

**Strategies**

| Strategy | Logic |
|---|---|
| Long Call | Buy a call — bullish bet |
| Long Put | Buy a put — bearish bet |
| Covered Call | Long stock + short call — collect premium, cap upside |
| Protective Put | Long stock + long put — insure downside |
| Straddle | Long call + long put at same strike — bet on high volatility |
| Strangle | OTM call + OTM put — same idea, cheaper, needs a bigger move |

**Reading results**

| Metric | Meaning |
|---|---|
| Win Rate | % of paths finishing with positive P&L |
| Sharpe | Annualised risk-adjusted return |
| Distribution chart | Green bars = winning scenarios, red = losing |
| P&L vs Spot chart | Each dot is one simulation; BE line shows breakeven |
| Percentile table | Worst 5%, median, best 5%, etc. |

---
> Volatility and rates are assumed constant. For educational and analytical use only.
""")

# ══════════════════════════════════════════════════════════════════════════════
with tab_fr:
    st.markdown("## Options Pricer — Guide d'utilisation")
    st.markdown("""
Trois modes disponibles depuis la barre latérale.

---

### 1. Pricing
Pricer un call ou un put via Black-Scholes ou Monte Carlo.

**Paramètres d'entrée**

| Paramètre | Description |
|---|---|
| S | Prix spot actuel du sous-jacent |
| K | Strike |
| T | Maturité en jours |
| r | Taux sans risque (%) |
| σ | Volatilité implicite ou historique (%) |
| q | Dividende continu (%) |
| Premium paid | Optionnel — recalcule le break-even à partir de votre coût d'entrée réel |

**Métriques de sortie**

| Métrique | Signification |
|---|---|
| Price | Valeur théorique de l'option |
| Break-even | Spot auquel le P&L = 0 à maturité |
| Prob ITM | Probabilité risque-neutre de finir dans la monnaie |
| Time Value | Prix moins valeur intrinsèque |
| Moneyness | ITM / ATM / OTM |

**Greeks**

| Greek | Interprétation |
|---|---|
| Delta | Variation du P&L pour +1$ sur le sous-jacent |
| Gamma | Vitesse de variation du Delta |
| Vega | Variation du P&L pour +1% de volatilité |
| Theta | Perte de valeur quotidienne (négatif en position longue) |
| Rho | Sensibilité aux variations de taux |

**Graphique P&L** — zone verte = profit, rouge = perte. BE indique votre break-even exact. K marque le strike, S le spot actuel.

---

### 2. Volatilité Implicite
Retrouver la volatilité implicite à partir d'un prix de marché observé (méthode de Brent).

Entrez le prix de marché, ajustez les autres paramètres pour correspondre au contrat, et l'outil calibre l'IV.

**Graphique Skew** — IV en fonction des strikes. Une pente descendante à gauche de l'ATM est typique (skew put).

**Graphique Term Structure** — IV en fonction des maturités pour un strike fixe. Une inversion peut signaler un stress à court terme.

---

### 3. Backtesting
Simuler une stratégie sur des chemins Monte Carlo et analyser la distribution statistique du P&L.

**Stratégies**

| Stratégie | Logique |
|---|---|
| Long Call | Achat d'un call — pari haussier |
| Long Put | Achat d'un put — pari baissier |
| Covered Call | Action longue + vente d'un call — encaisser la prime, plafonner le gain |
| Protective Put | Action longue + achat d'un put — protection contre la baisse |
| Straddle | Call + put au même strike — pari sur forte volatilité |
| Strangle | Call OTM + put OTM — même logique, moins cher, besoin de plus de mouvement |

**Lecture des résultats**

| Métrique | Signification |
|---|---|
| Win Rate | % de chemins avec un P&L positif |
| Sharpe | Rendement ajusté du risque (annualisé) |
| Distribution chart | Barres vertes = scénarios gagnants, rouges = perdants |
| P&L vs Spot chart | Chaque point est une simulation ; la ligne BE indique le seuil de rentabilité |
| Percentile table | Pire 5%, médiane, meilleur 5%, etc. |

---
> La volatilité et les taux sont supposés constants. Outil à vocation pédagogique et analytique.
""")
