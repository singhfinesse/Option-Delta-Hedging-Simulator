import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------- Parameters (change these!) ----------
S0, K, T = 100.0, 100.0, 0.25      # spot, strike, 3 months
r, sigma = 0.02, 0.20              # interest rate, volatility
COST = 0.001                       # transaction cost: 0.1% of traded value
N_PATHS = 20000
SEED = 42

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_delta(S, K, tau, r, sigma):
    tau = np.maximum(tau, 1e-8)
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return norm.cdf(d1)

def simulate_paths(n_steps, rng):
    dt = T/n_steps
    Z = rng.standard_normal((N_PATHS, n_steps))
    logret = (r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    return S0*np.exp(np.concatenate([np.zeros((N_PATHS,1)), np.cumsum(logret,axis=1)],axis=1))

def hedge_pnl(S, n_steps, cost):
    """Sell 1 call, delta-hedge at each step. Returns P&L per path at maturity."""
    dt = T/n_steps
    cash = np.full(S.shape[0], bs_call(S0,K,T,r,sigma))   # premium received
    delta = bs_delta(S[:,0],K,T,r,sigma)
    cash -= delta*S[:,0] + cost*np.abs(delta)*S[:,0]      # buy initial hedge
    for i in range(1, n_steps):
        cash *= np.exp(r*dt)
        new = bs_delta(S[:,i],K,T-i*dt,r,sigma)
        trade = new - delta
        cash -= trade*S[:,i] + cost*np.abs(trade)*S[:,i]
        delta = new
    cash *= np.exp(r*dt)
    payoff = np.maximum(S[:,-1]-K,0)
    cash += delta*S[:,-1] - cost*np.abs(delta)*S[:,-1]    # unwind hedge
    return cash - payoff

def cvar(x, a=0.95):
    q = np.quantile(x, 1-a)
    return -x[x<=q].mean()

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    freqs = [4, 12, 63, 252]   # ~ weekly-ish(4=every 3wk), monthly, daily-ish
    labels = {4:"~Every 3 weeks",12:"Weekly",63:"Daily",252:"4x per day"}
    rows=[]
    for cost in [0.0, COST]:
        for n in freqs:
            S = simulate_paths(n, np.random.default_rng(SEED))
            p = hedge_pnl(S, n, cost)
            rows.append((cost,n,p.mean(),p.std(),cvar(p)))
            print(f"cost={cost:.4f} steps={n:4d} mean={p.mean():7.3f} std={p.std():6.3f} CVaR95={cvar(p):6.3f}")
    import json; json.dump(rows, open("results.json","w"))

    # Plot 1: risk vs hedge frequency
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for cost,c in [(0.0,"tab:blue"),(COST,"tab:red")]:
        sub=[x for x in rows if x[0]==cost]
        ax[0].plot([x[1] for x in sub],[x[3] for x in sub],"o-",color=c,label=f"cost={cost*100:.1f}%")
        ax[1].plot([x[1] for x in sub],[x[2] for x in sub],"o-",color=c,label=f"cost={cost*100:.1f}%")
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
    ax[0].set_title("Risk (std of P&L) falls as you hedge more"); ax[0].set_xlabel("Hedges over 3 months"); ax[0].set_ylabel("Std of P&L ($)")
    ax[1].set_title("But average P&L drops with costs"); ax[1].set_xlabel("Hedges over 3 months"); ax[1].set_ylabel("Mean P&L ($)")
    ax[0].legend(); ax[1].legend(); ax[1].axhline(0,color="gray",lw=.8)
    plt.tight_layout(); plt.savefig("plot1_risk_vs_frequency.png",dpi=150); plt.close()

    # Plot 2: P&L distributions
    fig,ax=plt.subplots(figsize=(8,4.5))
    for n,c in [(12,"tab:orange"),(252,"tab:green")]:
        S=simulate_paths(n,np.random.default_rng(SEED)); p=hedge_pnl(S,n,COST)
        ax.hist(p,bins=80,alpha=.55,color=c,label=f"{n} hedges (CVaR95={cvar(p):.2f})")
    ax.set_title("Hedging P&L distribution with 0.1% transaction cost"); ax.set_xlabel("P&L per option sold ($)"); ax.legend()
    plt.tight_layout(); plt.savefig("plot2_pnl_distribution.png",dpi=150); plt.close()

    # Plot 3: sample paths
    S=simulate_paths(63,np.random.default_rng(SEED))
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(S[:30].T,lw=.8); ax.axhline(K,color="k",ls="--",lw=1,label="Strike")
    ax.set_title("30 simulated stock price paths (3 months)"); ax.set_xlabel("Trading day"); ax.set_ylabel("Price ($)"); ax.legend()
    plt.tight_layout(); plt.savefig("plot3_sample_paths.png",dpi=150); plt.close()
