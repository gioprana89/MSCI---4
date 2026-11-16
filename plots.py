"""Publication-oriented, data-derived figures for Gate C."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_engine import EVENT_DATE, STATES


def plot_all(r: dict, outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    files=[]
    daily=r["daily"].sort_values("Tanggal")
    def save(fig,name):
        fp=out/name
        fig.savefig(fp,dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(fig)
        files.append(fp)
    # Figure 1: main cohort state percentages across 153 dates
    fig,ax=plt.subplots(figsize=(11,4.8))
    shares=[100*daily["main_"+s].to_numpy()/daily.main_n.replace(0,np.nan) for s in STATES]
    ax.stackplot(daily.Tanggal,shares,labels=["Two-sided","Offer-only","Bid-only","Neither"],alpha=.78)
    ax.axvline(EVENT_DATE,ls="--",lw=1.1,label="16 Oct event")
    ax.set_ylim(0,100); ax.set_ylabel("Share of main cohort (%)")
    ax.set_title("Figure 1. Daily distribution of reported quote states")
    ax.legend(loc="upper left",ncol=5,fontsize=8)
    fig.autofmt_xdate();save(fig,"Figure_1_Daily_Quote_States.png")
    # Figure 2: actual 15->16 4x4 transitions, no illustrative counts
    mat=r["event_matrix"].set_index("from_state")[STATES]
    fig,ax=plt.subplots(figsize=(7.6,5.0))
    img=ax.imshow(mat.to_numpy(),aspect="auto",cmap="Blues")
    for i in range(4):
        for j in range(4):
            val=int(mat.iloc[i,j]); ax.text(j,i,str(val),ha="center",va="center",fontsize=11,
                                        color="white" if val>mat.to_numpy().max()*.55 else "black")
    ax.set_xticks(range(4),["Two-sided","Offer-only","Bid-only","Neither"],rotation=20)
    ax.set_yticks(range(4),["Two-sided","Offer-only","Bid-only","Neither"])
    ax.set_ylabel("15 Oct state"); ax.set_xlabel("16 Oct state")
    ax.set_title("Figure 2. Matched quote-state transition, 15 to 16 Oct 2026")
    fig.colorbar(img,ax=ax,label="Stocks")
    save(fig,"Figure_2_Transition_Matrix.png")
    # Figure 3: empirical recovery after offer-only observed Oct 16
    fig,ax=plt.subplots(figsize=(8,4.6))
    for label in ["full","main"]:
        z=r["recovery_curve"].query("cohort == @label")
        ax.step(z.h,z.pct_restored,where="post",marker="o",ms=3,label=label.title())
    ax.set_xlabel("Trading sessions after 16 October")
    ax.set_ylabel("Restored to recorded two-sided at least once (%)")
    ax.set_ylim(0,103);ax.set_title("Figure 3. Interday quote restoration (right-censor-aware counts)")
    ax.legend();ax.grid(alpha=.25)
    save(fig,"Figure_3_Quote_Restoration.png")
    # Figure 4: coverage + conditional spread must have distinct axes/panels
    sub=r["processed"].loc[r["processed"].sample_main].copy()
    d=sub.groupby("Tanggal").agg(n=("Kode Saham","size"),valid=("valid_quoted_spread","sum"),
         spread=("relative_spread","mean")).reset_index()
    fig,(a1,a2)=plt.subplots(2,1,figsize=(10,6.3),sharex=True)
    a1.plot(d.Tanggal,100*d.valid/d.n);a1.set_ylabel("Valid quote coverage (%)")
    a2.plot(d.Tanggal,100*d.spread);a2.set_ylabel("Conditional spread (%)")
    a2.set_xlabel("Trading date")
    for ax in (a1,a2):ax.axvline(EVENT_DATE,ls="--",lw=1.1);ax.grid(alpha=.2)
    fig.suptitle("Figure 4. Extensive coverage vs intensive quoted spread")
    fig.autofmt_xdate();save(fig,"Figure_4_Coverage_vs_Spread.png")
    # Figure 5: coefficient plot (if estimates valid)
    reg=r["regressions"]
    z=reg.loc[reg.status.eq("OK") & reg.model_id.isin(["M1_OFFER_LPM_FE","M2_DOWNSIDE_LPM_FE"])].copy() if "status" in reg else pd.DataFrame()
    fig,ax=plt.subplots(figsize=(9,5))
    if len(z):
        z=z.iloc[:16]
        labels=(z.model_id.str.replace("M1_OFFER_LPM_FE","M1",regex=False)
                .str.replace("M2_DOWNSIDE_LPM_FE","M2",regex=False)+": "+z.term)
        ypos=np.arange(len(z))[::-1]
        ax.errorbar(z.coefficient,ypos,xerr=[z.coefficient-z.ci_95_low,z.ci_95_high-z.coefficient],
                    fmt="o",capsize=3)
        ax.set_yticks(ypos,labels,fontsize=8);ax.axvline(0,ls="--",color="grey",alpha=.7)
        ax.set_xlabel("Coefficient (decimal probability units)")
    else:
        ax.text(.5,.5,"Regressions not run / unavailable",ha="center",va="center",transform=ax.transAxes)
    ax.set_title("Figure 5. Two-way fixed-effects estimates with 95% confidence intervals")
    ax.grid(axis="x",alpha=.15)
    save(fig,"Figure_5_FE_Coefficients.png")
    # Figure 6: true out-of-sample score comparisons
    f=r["forecast"]
    fig,ax=plt.subplots(figsize=(8,4.3))
    z=f.loc[f.status.eq("OK")] if "status" in f else pd.DataFrame()
    if len(z):
        names=sorted(z.fold.unique()); w=.37;xx=np.arange(len(names))
        for k,model in enumerate(["baseline","baseline_plus_quotes"]):
            ss=z.loc[z.model==model].set_index("fold")
            ax.bar(xx+(k-.5)*w,[ss.loc[n,"brier_score"] if n in ss.index else np.nan for n in names],w,
                   label=model)
        ax.set_xticks(xx,names);ax.legend();ax.set_ylabel("Brier score (lower is better)")
    else:
        ax.text(.5,.5,"Forecasts not run / unavailable",ha="center",va="center",transform=ax.transAxes)
    ax.set_title("Figure 6. Time-ordered out-of-sample forecast comparison")
    save(fig,"Figure_6_OOS_Brier.png")
    return files
