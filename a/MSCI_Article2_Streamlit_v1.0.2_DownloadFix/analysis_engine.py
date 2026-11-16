"""Research engine for MSCI Article 2 / Gate C.

All quantitative inputs are derived solely from the uploaded IDX daily stock summary.
The reported daily quote states do not identify order cancellation or intraday book dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, Logit, add_constant
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

PROTOCOL = "Article 2 Research Design Protocol v1.0 + Gate B conditional semantics"
VERSION = "GateC-1.0.1-AUDITED"
EVENT_DATE = pd.Timestamp("2026-10-16")
MAIN_END = pd.Timestamp("2026-11-10")
TRAIN_END = pd.Timestamp("2026-09-30")
STATES = ["two_sided", "offer_only", "bid_only", "neither"]
SOURCE_FIELDS = [
 "Tanggal", "Kode Saham", "File Sumber", "No", "Nama Perusahaan", "Remarks",
 "Sebelumnya", "Open Price", "Tanggal Perdagangan Terakhir", "First Trade",
 "Tertinggi", "Terendah", "Penutupan", "Selisih", "Volume", "Nilai", "Frekuensi",
 "Index Individual", "Offer", "Offer Volume", "Bid", "Bid Volume", "Listed Shares",
 "Tradeble Shares", "Weight For Index", "Foreign Sell", "Foreign Buy",
 "Non Regular Volume", "Non Regular Value", "Non Regular Frequency"
]
NUMERIC = ["Sebelumnya", "Open Price", "First Trade", "Tertinggi", "Terendah", "Penutupan",
           "Selisih", "Volume", "Nilai", "Frekuensi", "Index Individual", "Offer", "Offer Volume",
           "Bid", "Bid Volume", "Listed Shares", "Tradeble Shares", "Weight For Index",
           "Foreign Sell", "Foreign Buy", "Non Regular Volume", "Non Regular Value", "Non Regular Frequency"]
METADATA_WARNING = (
 "CONDITIONAL IDX METADATA: daily Bid/Offer fields are reported quotes; exact snapshot timing, "
 "zero-value semantics, quote-volume units and board mapping remain unverified. "
 "Never infer intraday cancellations, an empty book throughout the session, or causal MSCI effects."
)


def load_source(path_or_bytes):
    """Return unmodified-column-order data frame and original source SHA-256."""
    if isinstance(path_or_bytes, (str, Path)):
        raw = Path(path_or_bytes).read_bytes()
    else:
        raw = path_or_bytes.read() if hasattr(path_or_bytes, "read") else bytes(path_or_bytes)
    frame = pd.read_excel(BytesIO(raw), sheet_name="Data", dtype={"Kode Saham": "string"})
    missing = set(SOURCE_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError("Kolom wajib hilang: " + ", ".join(sorted(missing)))
    if list(frame.columns) != SOURCE_FIELDS:
        raise ValueError("Urutan/daftar 30 kolom berbeda dengan protokol; verifikasi file sumber terlebih dahulu.")
    return frame, hashlib.sha256(raw).hexdigest()


def prepare(raw: pd.DataFrame, activity_threshold=20):
    """Keep raw separately; create full stock-date processed data with reproducible flags."""
    if activity_threshold not in (20, 40, 60):
        raise ValueError("Activity threshold harus 20, 40, atau 60 sesi.")
    x = raw.copy(deep=True)
    x["Tanggal"] = pd.to_datetime(x["Tanggal"], errors="coerce").dt.normalize()
    if x["Tanggal"].isna().any() or x["Kode Saham"].isna().any():
        raise ValueError("Tanggal/Kode Saham kosong: perlu tindakan audit eksplisit.")
    if x.duplicated(["Kode Saham", "Tanggal"]).any():
        raise ValueError("Duplikasi (Kode Saham, Tanggal) terdeteksi; tidak boleh dihapus otomatis.")
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    dates = pd.DatetimeIndex(x["Tanggal"].drop_duplicates().sort_values())
    ix = {pd.Timestamp(d): k for k, d in enumerate(dates)}
    x["date_idx"] = x["Tanggal"].map(ix).astype("int32")
    b, a = x["Bid"], x["Offer"]
    valid_signs = b.notna() & a.notna() & (b >= 0) & (a >= 0)
    x["quote_state"] = np.select(
        [valid_signs & (b>0) & (a>0), valid_signs & (b==0) & (a>0),
         valid_signs & (b>0) & (a==0), valid_signs & (b==0) & (a==0)],
        STATES, default="invalid_or_missing")
    x["two_sided"] = (x.quote_state=="two_sided").astype("int8")
    x["offer_only"] = (x.quote_state=="offer_only").astype("int8")
    x["valid_quoted_spread"] = (b>0) & (a>=b) & valid_signs
    x["relative_spread"] = np.where(x.valid_quoted_spread,
        (a-b)/((a+b)/2.0), np.nan)
    bv, av = x["Bid Volume"], x["Offer Volume"]
    x["valid_qdi"] = x.valid_quoted_spread & (bv>0) & (av>0)
    x["quote_depth_imbalance"] = np.where(x.valid_qdi, (bv-av)/(bv+av), np.nan)
    x["log_bid_depth"] = np.where(bv>=0, np.log1p(bv), np.nan)
    x["log_offer_depth"] = np.where(av>=0, np.log1p(av), np.nan)
    x["return_t"] = np.where(x["Sebelumnya"]>0,
        (x["Penutupan"]-x["Sebelumnya"])/x["Sebelumnya"], np.nan)
    x["abs_return"] = x["return_t"].abs()
    x["log_value"] = np.where(x["Nilai"]>=0, np.log1p(x["Nilai"]), np.nan)
    x["log_frequency"] = np.where(x["Frekuensi"]>=0, np.log1p(x["Frekuensi"]), np.nan)
    x["turnover"] = np.where(x["Listed Shares"]>0,
        x["Volume"]/x["Listed Shares"], np.nan)
    sz = x["Penutupan"]*x["Listed Shares"]
    x["log_size"] = np.where(sz>0, np.log(sz), np.nan)
    x["quote_price_volume_mismatch"] = ((b.eq(0) & bv.ne(0)) | (b.gt(0) & bv.eq(0)) |
                                        (a.eq(0) & av.ne(0)) | (a.gt(0) & av.eq(0))).fillna(True)
    x["crossed_quote"] = (b>a) & (b>0) & (a>0)
    x["foreign_qty_gt_volume"] = ((x["Foreign Buy"]+x["Foreign Sell"])>x["Volume"])
    # Stock-specific sorted lag/lead, only accept adjacent exchange-wide session indices.
    x = x.sort_values(["Kode Saham", "date_idx"], kind="stable")
    g = x.groupby("Kode Saham", sort=False)
    lagix, nextix = g.date_idx.shift(1), g.date_idx.shift(-1)
    has_lag, has_next = x.date_idx.eq(lagix+1), x.date_idx.eq(nextix-1)
    x["prior_close"] = g["Penutupan"].shift(1).where(has_lag)
    x["reference_mismatch"] = has_lag & x["Sebelumnya"].ne(x["prior_close"])
    x["lag_return"] = g.return_t.shift(1).where(has_lag)
    x["lag_log_value"] = g.log_value.shift(1).where(has_lag)
    x["lag_quote_state"] = g.quote_state.shift(1).where(has_lag)
    x["next_return"] = g.return_t.shift(-1).where(has_next)
    x["next_date"] = g.Tanggal.shift(-1).where(has_next)
    x["downside_5pct"] = pd.Series(np.where(x.next_return.notna(),
        (x.next_return <= -.05).astype(float), np.nan), index=x.index)
    x["next_session_valid"] = has_next
    x["event_day"] = x.Tanggal.eq(EVENT_DATE)
    x["main_period"] = "other"
    pre = dates[dates<EVENT_DATE]
    post = dates[(dates>EVENT_DATE) & (dates<=MAIN_END)]
    if len(pre)<60 or len(post)<17:
        raise ValueError("Data tidak memenuhi kalender 60 sesi pra dan 17 sesi pasca sampai 10 Nov 2026.")
    pre17, post17 = pre[-17:], post[:17]
    x.loc[x.Tanggal.isin(pre17), "main_period"] = "pre17"
    x.loc[x.Tanggal.isin(post17), "main_period"] = "post17"
    x.loc[x.event_day, "main_period"] = "event"
    x["pre60"] = x.Tanggal.isin(pre[-60:])
    exposure = x.loc[x.pre60].groupby("Kode Saham")["Volume"].apply(lambda s: int((s>0).sum()))
    event_codes = set(x.loc[x.event_day, "Kode Saham"])
    selected = {t: set(exposure[exposure>=t].index).intersection(event_codes) for t in (20,40,60)}
    x["pre60_active_sessions"] = x["Kode Saham"].map(exposure).fillna(0).astype("int16")
    for t in (20, 40, 60):
        x[f"sample_activity_{t}"] = x["Kode Saham"].isin(selected[t])
    x["sample_main"] = x[f"sample_activity_{activity_threshold}"]
    x["sample_spread"] = x.sample_main & x.valid_quoted_spread
    x["sample_prediction"] = (x.sample_main & (x.Tanggal<=MAIN_END) & x.next_date.le(MAIN_END) & x.next_session_valid &
            x.return_t.notna() & x.log_value.notna() & x.log_frequency.notna())
    x["sample_exclusion_reason"] = np.where(~x["Kode Saham"].isin(event_codes), "No event-day record",
        np.where(x.pre60_active_sessions<activity_threshold,"Below pre-event activity threshold","Eligible"))
    x["metadata_validation_status"] = "CONDITIONAL: IDX quote zero/timing/units unverified"
    # Restore row order of input, maintaining one-to-one mapping.
    x = x.sort_index(kind="stable")
    x.attrs["pre17_dates"] = [str(d.date()) for d in pre17]
    x.attrs["post17_dates"] = [str(d.date()) for d in post17]
    x.attrs["pre60_dates"] = [str(d.date()) for d in pre[-60:]]
    return x


def audit(raw: pd.DataFrame, x: pd.DataFrame, source_hash: str):
    stats_rows = [
        ("source_rows", len(raw)), ("source_columns", len(raw.columns)),
        ("distinct_stock_codes", raw["Kode Saham"].nunique()),
        ("distinct_trading_dates", x["Tanggal"].nunique()),
        ("duplicate_stock_dates", int(raw.duplicated(["Kode Saham", "Tanggal"]).sum())),
        ("zero_open_price", int((x["Open Price"]==0).sum())),
        ("zero_traded_volume", int((x["Volume"]==0).sum())),
        ("zero_bid_price", int((x["Bid"]==0).sum())),
        ("zero_offer_price", int((x["Offer"]==0).sum())),
        ("zero_bid_volume", int((x["Bid Volume"]==0).sum())),
        ("zero_offer_volume", int((x["Offer Volume"]==0).sum())),
        ("crossed_quoted_prices", int(x.crossed_quote.sum())),
        ("quote_price_volume_mismatch", int(x.quote_price_volume_mismatch.sum())),
        ("reference_price_mismatch", int(x.reference_mismatch.sum())),
        ("foreign_buy_sell_qty_gt_volume", int(x.foreign_qty_gt_volume.sum())),
        ("number_main_cohort", int(x.loc[x.sample_main, "Kode Saham"].nunique())),
        ("number_cohort_20", int(x.loc[x.sample_activity_20, "Kode Saham"].nunique())),
        ("number_cohort_40", int(x.loc[x.sample_activity_40, "Kode Saham"].nunique())),
        ("number_cohort_60", int(x.loc[x.sample_activity_60, "Kode Saham"].nunique())),
        ("source_sha256", source_hash),
        ("metadata_validation_status", METADATA_WARNING),
    ]
    count_states = x.quote_state.value_counts()
    for q in STATES:
        stats_rows.append(("quote_state_"+q, int(count_states.get(q,0))))
    return pd.DataFrame(stats_rows, columns=["metric", "value"])


def coverage_daily(x):
    cols = ["Tanggal", "quote_state", "Kode Saham"]
    c = x.groupby(["Tanggal","quote_state"],observed=True).size().unstack(fill_value=0)
    for state in STATES:
        if state not in c.columns: c[state]=0
    c = c[STATES]
    c["n_observed"] = x.groupby("Tanggal").size()
    c["two_sided_pct"] = 100*c["two_sided"]/c.n_observed
    c["offer_only_pct"] = 100*c["offer_only"]/c.n_observed
    c["nonzero_trade_volume"] = x.groupby("Tanggal")["Volume"].apply(lambda s:int((s>0).sum()))
    m = x.loc[x.sample_main].groupby(["Tanggal","quote_state"]).size().unstack(fill_value=0)
    for state in STATES:
        if state not in m.columns: m[state]=0
        c["main_"+state] = m[state].reindex(c.index,fill_value=0)
    c["main_n"] = x.loc[x.sample_main].groupby("Tanggal").size().reindex(c.index,fill_value=0)
    c["main_offer_only_pct"] = np.where(c.main_n>0,100*c.main_offer_only/c.main_n,np.nan)
    c["main_two_sided_pct"] = np.where(c.main_n>0,100*c.main_two_sided/c.main_n,np.nan)
    return c.reset_index()


def transition_pairs(x, restrict_dates=None):
    d = x[["Kode Saham","Tanggal","date_idx","quote_state","sample_main"]].copy()
    right = d[["Kode Saham","date_idx","quote_state"]].copy()
    right["date_idx"] -= 1
    right = right.rename(columns={"quote_state":"next_state"})
    paired = d.merge(right, on=["Kode Saham","date_idx"], how="inner", validate="one_to_one")
    if restrict_dates is not None:
        paired=paired[paired.Tanggal.isin(restrict_dates)]
    paired["next_date"] = paired.Tanggal.map(lambda _: pd.NaT)
    date_map=x[["date_idx","Tanggal"]].drop_duplicates().set_index("date_idx")["Tanggal"].to_dict()
    paired["next_date"] = (paired.date_idx+1).map(date_map)
    return paired


def transition_matrix(pairs, date=None, cohort_only=False):
    p=pairs
    if date is not None: p=p.loc[p.Tanggal.eq(pd.Timestamp(date))]
    if cohort_only: p=p.loc[p.sample_main]
    counts=pd.crosstab(p.quote_state,p.next_state).reindex(index=STATES,columns=STATES,fill_value=0)
    out=counts.reset_index().rename(columns={"quote_state":"from_state"})
    out["n_from"] = counts.sum(axis=1).values
    for s in STATES:
        out["pct_to_"+s] = np.where(out.n_from>0, 100*out[s]/out.n_from,np.nan)
    return out


def transition_daily(pairs, cohort_only=False):
    p = pairs.loc[pairs.sample_main] if cohort_only else pairs
    g=p.groupby("Tanggal")
    a=g.size().rename("paired_n")
    for from_s,to_s,name in [("two_sided","offer_only","two_to_offer"),("offer_only","two_sided","offer_to_two"),
                             ("two_sided","two_sided","two_to_two")]:
        q=p.loc[(p.quote_state==from_s)&(p.next_state==to_s)].groupby("Tanggal").size()
        a=a.to_frame() if isinstance(a,pd.Series) and a.name=="paired_n" else a
        a[name]=q
    a=a.fillna(0)
    a["pct_two_to_offer_of_pairs"]=100*a.two_to_offer/a.paired_n
    return a.reset_index()


def recovery(x, cutoff=MAIN_END):
    """Event-day offer_only cohort; restoration is first subsequent recorded two_sided."""
    ev=x.loc[x.Tanggal.eq(EVENT_DATE) & x.quote_state.eq("offer_only"),["Kode Saham","sample_main"]].copy()
    post_days=sorted(x.loc[(x.Tanggal>EVENT_DATE)&(x.Tanggal<=cutoff),"Tanggal"].unique())
    tmap={d:i+1 for i,d in enumerate(post_days)}
    post=x.loc[(x.Tanggal>EVENT_DATE)&(x.Tanggal<=cutoff)&x["Kode Saham"].isin(ev["Kode Saham"]),
               ["Kode Saham","Tanggal","quote_state"]].copy()
    post["h"] = post.Tanggal.map(tmap)
    recovered=post.loc[post.quote_state.eq("two_sided")].groupby("Kode Saham")["h"].min().rename("first_restored_h")
    last_obs=post.groupby("Kode Saham")["h"].max().rename("last_observed_h")
    table=ev.merge(recovered, how="left", left_on="Kode Saham",right_index=True)
    table=table.merge(last_obs, how="left",left_on="Kode Saham",right_index=True)
    table["restored"] = table.first_restored_h.notna()
    table["right_censored"] = ~table.restored
    # Observations on subsequent sessions may be missing; exact follow-up coverage is kept.
    table["followup_days"] = table.last_observed_h.fillna(0).astype(int)
    stats_rows=[]
    for cohort, sub in [("Full event offer-only",table),("Main 842 eligible",table[table.sample_main])]:
        z={"cohort":cohort,"n_event_offer_only":len(sub),"restored_by_cutoff":int(sub.restored.sum()),
           "right_censored":int(sub.right_censored.sum()),"cutoff":str(cutoff.date())}
        for h in (1,2,5,10,17,20):
            z[f"restored_within_{h}_sessions"]=int((sub.first_restored_h<=h).sum())
            z[f"restored_pct_{h}_sessions"] = 100*int((sub.first_restored_h<=h).sum())/len(sub) if len(sub) else np.nan
        stats_rows.append(z)
    curve=[]
    for h in range(0,len(post_days)+1):
        for label, sub in [("full",table),("main",table[table.sample_main])]:
            n=len(sub); rec=int((sub.first_restored_h<=h).sum())
            curve.append({"cohort":label,"h":h,"n_initial":n,"n_restored":rec,
                          "pct_restored":100*rec/n if n else np.nan})
    return table,pd.DataFrame(stats_rows),pd.DataFrame(curve)


def prepost_summary(x):
    sub=x.loc[x.sample_main & x.main_period.isin(["pre17","post17"])]
    rows=[]
    for variable in ["two_sided","offer_only","relative_spread","quote_depth_imbalance",
                     "return_t","abs_return","log_value","log_frequency","turnover"]:
        means = sub.groupby(["Kode Saham","main_period"])[variable].mean().unstack()
        valid=means.dropna(subset=["pre17","post17"])
        diff=valid.post17-valid.pre17
        rows.append({"metric":variable,"n_pairs":len(diff),"before_mean":valid.pre17.mean(),
                     "after_mean":valid.post17.mean(),"after_minus_before":diff.mean(),
                     "sd_stock_level_delta":diff.std(ddof=1),
                     "t_p_naive":stats.ttest_1samp(diff,0).pvalue if len(diff)>1 and diff.std()>0 else np.nan,
                     "inference_warning":"Naive stock-paired p; not clustered for common dated shocks"})
    return pd.DataFrame(rows)


def twoway_demean(y: pd.Series, f: pd.Series, d: pd.Series, max_iter=70, tol=1e-9):
    z = y.astype(float).copy()
    for k in range(max_iter):
        old=z.to_numpy(copy=True)
        z -= z.groupby(f, sort=False).transform("mean")
        z -= z.groupby(d, sort=False).transform("mean")
        if np.max(np.abs(z.to_numpy()-old))<tol:
            break
    return z


def fit_two_way_fe(data: pd.DataFrame, outcome: str, predictors: list[str], model_id: str,
                   date_end=MAIN_END, sample_flag="sample_main"):
    """Two-way fixed effects linear model with 2-way cluster SE and exact sample flags."""
    keep=["Kode Saham","Tanggal",outcome]+predictors
    s=data.loc[data[sample_flag] & data.Tanggal.le(date_end),keep].dropna().copy()
    if len(s)<300 or s.Tanggal.nunique()<12 or s["Kode Saham"].nunique()<30:
        return pd.DataFrame([{"model_id":model_id,"term":"FAILED","status":"insufficient eligible observations"}])
    y=twoway_demean(s[outcome],s["Kode Saham"],s["Tanggal"])
    X=pd.DataFrame({p:twoway_demean(s[p],s["Kode Saham"],s["Tanggal"]) for p in predictors})
    keepcols=[p for p in predictors if (X[p].std()>1e-10)]
    if not keepcols:
        return pd.DataFrame([{"model_id":model_id,"term":"FAILED","status":"no within variation"}])
    X=X[keepcols]
    try:
        res=OLS(y.to_numpy(dtype=float),X.to_numpy(dtype=float)).fit()
        firm_code=pd.factorize(s["Kode Saham"],sort=True)[0].astype("int64")
        date_code=pd.factorize(s["Tanggal"],sort=True)[0].astype("int64")
        cov=cov_cluster_2groups(res,firm_code,date_code,use_correction=True)[0]
        diag=np.diag(cov)
        se=np.sqrt(np.where(diag>=0,diag,np.nan))
        df=min(s.Tanggal.nunique(),s["Kode Saham"].nunique())-1
        tval=np.divide(res.params,se,out=np.full(len(se),np.nan),where=np.isfinite(se)&(se>0))
        pval=2*stats.t.sf(np.abs(tval),df)
        crit=stats.t.ppf(.975,df)
        rows=[]
        for j,name in enumerate(keepcols):
            rows.append({"model_id":model_id,"outcome":outcome,"term":name,
             "coefficient":float(res.params[j]),"clustered_se":float(se[j]) if np.isfinite(se[j]) else np.nan,
             "t":float(tval[j]),"p_value":float(pval[j]),
             "ci_95_low":float(res.params[j]-crit*se[j]),"ci_95_high":float(res.params[j]+crit*se[j]),
             "n_obs":len(s),"n_firms":s["Kode Saham"].nunique(),"n_dates":s.Tanggal.nunique(),
             "within_r2":float(res.rsquared),"df_inference":df,
             "covariance":"Two-way firm/date clustered; small-sample correction",
             "status":"OK" if np.isfinite(se[j]) else "UNSTABLE COVARIANCE",
             "estimand":"Associational; firm and date fixed effects"})
        for p in predictors:
            if p not in keepcols:
                rows.append({"model_id":model_id,"outcome":outcome,"term":p,"status":"Dropped: zero within variation",
                             "n_obs":len(s),"n_firms":s["Kode Saham"].nunique(),"n_dates":s.Tanggal.nunique()})
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame([{"model_id":model_id,"term":"FAILED","status":f"fit exception: {type(e).__name__}: {str(e)[:180]}"}])


def run_regressions(x):
    dat=x.copy()
    for s in ["offer_only","bid_only","neither"]:
        dat["prev_"+s] = (dat.lag_quote_state==s).astype(float).where(dat.lag_quote_state.notna())
        dat["now_"+s] = (dat.quote_state==s).astype(float)
    dat["current_return"] = dat.return_t
    dat["current_abs_return"] = dat.abs_return
    model1=fit_two_way_fe(dat,"offer_only",["lag_return","lag_log_value","prev_offer_only","prev_bid_only","prev_neither"],"M1_OFFER_LPM_FE")
    model2=fit_two_way_fe(dat,"downside_5pct",["now_offer_only","now_bid_only","now_neither",
                         "current_return","log_value","log_frequency"],"M2_DOWNSIDE_LPM_FE",sample_flag="sample_prediction")
    base=fit_two_way_fe(dat,"downside_5pct",["current_return","log_value","log_frequency"],
                        "M2_BASE_DOWNSIDE_FE",sample_flag="sample_prediction")
    conditional=fit_two_way_fe(dat,"relative_spread",["lag_return","lag_log_value"],
                        "M3_CONDITIONAL_SPREAD_FE",sample_flag="sample_spread")
    recovery_reg=fit_recovery_hc3(dat)
    return pd.concat([model1,model2,base,conditional,recovery_reg],ignore_index=True)


def fit_recovery_hc3(x):
    """Restoration on first post-event session regressed on predetermined exposures.

    One cross-sectional observation per event-day offer-only main-cohort stock. The
    principal model is intentionally distinct from the panel FE specifications.
    """
    base=x.loc[(x.Tanggal<EVENT_DATE)&(x.Tanggal>=pd.Timestamp("2026-09-18"))]
    pre=base.groupby("Kode Saham").agg(pre_log_value=("log_value","mean"),
        pre_log_size=("log_size","median"),pre_log_price=("Penutupan","median"))
    pre["pre_log_price"]=np.where(pre.pre_log_price>0,np.log(pre.pre_log_price),np.nan)
    rec,_,_=recovery(x)
    d=rec.loc[rec.sample_main].merge(pre,left_on="Kode Saham",right_index=True,how="left")
    d["recovered_h1"]=(d.first_restored_h==1).astype(int)
    preds=["pre_log_value","pre_log_size","pre_log_price"]
    d=d.dropna(subset=preds)
    if len(d)<50 or d.recovered_h1.nunique()<2:
        return pd.DataFrame([{"model_id":"M4_RECOVERY_LOGIT_HC3","term":"FAILED","status":"insufficient recovery variation"}])
    zz=d[preds].std(ddof=0)
    valid=[c for c in preds if zz[c]>1e-10]
    design=add_constant((d[valid]-d[valid].mean())/zz[valid],has_constant="add")
    try:
        fit=Logit(d.recovered_h1,design).fit(disp=0,maxiter=100,cov_type="HC3")
        results=[]
        for term in design.columns:
            coef=float(fit.params[term]); se=float(fit.bse[term]);lo,hi=fit.conf_int().loc[term]
            results.append({"model_id":"M4_RECOVERY_LOGIT_HC3","outcome":"first-session two-sided restoration",
                 "term":term,"coefficient":coef,"clustered_se":se,"t":coef/se,
                 "p_value":float(fit.pvalues[term]),"ci_95_low":float(lo),"ci_95_high":float(hi),
                 "odds_ratio":float(np.exp(coef)),"n_obs":len(d),"n_firms":d["Kode Saham"].nunique(),
                 "n_dates":1,"within_r2":float(fit.prsquared),"covariance":"HC3 heteroskedastic robust",
                 "status":"OK","estimand":"Cross-sectional first-session restoration; pre-event standardized predictors"})
        return pd.DataFrame(results)
    except Exception as e:
        return pd.DataFrame([{"model_id":"M4_RECOVERY_LOGIT_HC3","term":"FAILED","status":f"logit fit failed: {type(e).__name__}: {str(e)[:180]}"}])


def predict_walk_forward(x):
    """Expanding monthly time split, only labels with known next trading day in same split."""
    data=x.loc[x.sample_prediction & x.Tanggal.le(MAIN_END)].copy()
    data["qdi_missing"] = data.quote_depth_imbalance.isna().astype(int)
    for state in ("offer_only","bid_only","neither"):
        data["q_"+state] = data.quote_state.eq(state).astype(int)
    base=["return_t","log_value","log_frequency","log_size"]
    extended=base+["q_offer_only","q_bid_only","q_neither","quote_depth_imbalance","qdi_missing"]
    folds=[("October2026",pd.Timestamp("2026-09-30"),pd.Timestamp("2026-10-01"),pd.Timestamp("2026-10-30")),
           ("November2026",pd.Timestamp("2026-10-30"),pd.Timestamp("2026-11-02"),MAIN_END)]
    rows=[]; preds=[]
    for fold,trainend,teststart,testend in folds:
        train=data.loc[(data.Tanggal<=trainend)&(data.next_date<=trainend)&data.downside_5pct.notna()]
        test=data.loc[(data.Tanggal>=teststart)&(data.Tanggal<=testend)&(data.next_date<=testend)&data.downside_5pct.notna()]
        if len(train)<100 or len(test)<50 or train.downside_5pct.nunique()<2:
            rows.append({"fold":fold,"model":"FAILED","status":"not enough rows or one-class train"}); continue
        for name,cols in [("baseline",base),("baseline_plus_quotes",extended)]:
            pipeline=Pipeline([("fill",SimpleImputer(strategy="median",add_indicator=False)),
                               ("scale",StandardScaler()),
                               ("model",LogisticRegression(max_iter=120,C=1.0,solver="liblinear"))])
            pipeline.fit(train[cols],train.downside_5pct.astype(int))
            proba=pipeline.predict_proba(test[cols])[:,1]
            y=test.downside_5pct.to_numpy(dtype=int)
            metric={"fold":fold,"model":name,"train_start":str(train.Tanggal.min().date()),
                    "train_end":str(trainend.date()),"test_start":str(teststart.date()),"test_end":str(testend.date()),
                    "n_train":len(train),"n_test":len(test),"test_events":int(y.sum()),
                    "event_rate":float(y.mean()),"brier_score":float(brier_score_loss(y,proba)),
                    "average_precision":float(average_precision_score(y,proba)) if y.sum()>0 else np.nan,
                    "roc_auc":float(roc_auc_score(y,proba)) if np.unique(y).size==2 else np.nan,
                    "log_loss":float(log_loss(y,proba,labels=[0,1])),"status":"OK"}
            rows.append(metric)
            z=test[["Kode Saham","Tanggal","next_date","downside_5pct"]].copy()
            z["fold"]=fold; z["model"]=name; z["predicted_probability"]=proba
            preds.append(z)
    return pd.DataFrame(rows), (pd.concat(preds,ignore_index=True) if preds else pd.DataFrame())


def robustness(x, pairs):
    rows=[]
    for thr in (20,40,60):
        eligible=x.loc[x[f"sample_activity_{thr}"] & x.Tanggal.eq(EVENT_DATE)]
        event_n=len(eligible)
        previous=x.loc[x[f"sample_activity_{thr}"] & x.Tanggal.eq(pd.Timestamp("2026-10-15"))]
        pair=pairs.loc[(pairs.Tanggal==pd.Timestamp("2026-10-15")) & pairs["Kode Saham"].isin(eligible["Kode Saham"]) ]
        rows.append({"active_session_threshold":thr,"n_event_stocks":event_n,
                     "offer_only_event":int((eligible.quote_state=="offer_only").sum()),
                     "offer_only_event_pct":100*(eligible.quote_state=="offer_only").mean(),
                     "two_sided_event":int((eligible.quote_state=="two_sided").sum()),
                     "two_to_offer_transition":int(((pair.quote_state=="two_sided")&(pair.next_state=="offer_only")).sum()),
                     "pre_day_two_sided":int((previous.quote_state=="two_sided").sum()),
                     "descriptor":"not a causal event effect"})
    return pd.DataFrame(rows)


def variable_dictionary():
    rows=[
        ("quote_state","Bid, Offer","four-state classification on nonnegative quoted prices; invalid separate","categorical","daily reported; not intraday cancellations"),
        ("two_sided","quote_state","1 if bid and offer positive","binary","valid Bid/Offer record"),
        ("offer_only","quote_state","1 if bid zero and offer positive","binary","zero coding unverified"),
        ("relative_spread","Bid, Offer","(Offer-Bid)/((Offer+Bid)/2)","ratio","Offer>=Bid>0; not execution cost"),
        ("quote_depth_imbalance","Bid Volume, Offer Volume","(Bid Volume - Offer Volume)/(Bid Volume + Offer Volume)","ratio [-1,1]","only positive two-sided volumes; unit assumption"),
        ("return_t","Penutupan, Sebelumnya","(Penutupan-Sebelumnya)/Sebelumnya","decimal","Sebelumnya>0; stale risk when Volume=0"),
        ("next_return","return_t","return at next exchange-wide session if stock observed","decimal","no calendar-day shifting"),
        ("downside_5pct","next_return","1 if next_return<=-0.05 else 0","binary","valid next return"),
        ("turnover","Volume, Listed Shares","Volume/Listed Shares","ratio","not verified free-float turnover"),
        ("log_value","Nilai","ln(1+Nilai)","log IDR","Nilai>=0"),
        ("log_frequency","Frekuensi","ln(1+Frekuensi)","log count","Frekuensi>=0"),
        ("log_size","Penutupan, Listed Shares","ln(Penutupan*Listed Shares)","log nominal capitalization","positive source fields"),
        ("lag_quote_state","quote_state","previous global trading session state if observed","categorical","missing adjacent stock date stays NA"),
        ("sample_main","Volume, Tanggal, Kode Saham",">= threshold of 60 pre-event active trading sessions and event-day record","boolean","threshold chosen ex ante"),
        ("reference_mismatch","Sebelumnya, prior close","Sebelumnya differs from prior recorded consecutive-session close","boolean","source value not overwritten"),
        ("metadata_validation_status","IDX external field dictionary","conditional / unresolved","text","no official timing, zero semantics or units")
    ]
    return pd.DataFrame(rows,columns=["variable","source","formula_or_rule","unit","validity_and_limits"])


def methods_table():
    entries=[
        ("T1","Quote state incidence","count_s(t)/number_of_observed_stocks_t","pandas.groupby / crosstab","No inferential p-values for market-wide event"),
        ("T2","Transition matrix","count(from_state=s, next_state=s')","pandas merge adjacent global trading sessions","Missing next-day record excluded from matched denominator"),
        ("T3","Interday restoration","min h>=1 where quote_state==two_sided after event offer_only","groupby + right censor","Not intraday replenishment"),
        ("T4","Conditional spread","(Offer-Bid)/((Offer+Bid)/2); valid quotes only","pandas/numpy","Selection/availability reported separately"),
        ("T5","Quote depth imbalance","(BidVolume-OfferVolume)/(BidVolume+OfferVolume)","pandas/numpy","Only both positive; common-unit assumption"),
        ("M1","Offer-only linear probability two-way fixed effects","offer_only_t~lag_return+lag_log_value+lag_state_dummies+firmFE+dateFE","statsmodels OLS; iterative demeaning","2-way clustered SE; associational"),
        ("M2","Next-day downside linear probability two-way fixed effects","downside_(t+1)~quote_state_dummies+return_t+log_value+log_frequency+firmFE+dateFE","statsmodels OLS; iterative demeaning","next adjacent IDX session; associational"),
        ("M3","Conditional spread two-way fixed effects","relative_spread_t~lag_return+lag_log_value+firmFE+dateFE","statsmodels OLS; valid spreads only","2-way clustered SE; selection caveat"),
        ("M4","First-session recovery logit","Pr(restored_h1)~z(pre_log_value)+z(pre_log_size)+z(pre_log_price)","statsmodels Logit(HC3)","Cross-sectional and associational; odds ratios"),
        ("M5","Walk-forward regularized logistic","baseline features versus baseline+quote_state+QDI","sklearn LogisticRegression pipeline","Split in calendar order, train-label cutoff enforced"),
        ("R1","Activity sensitivity","pre60_volume>0 on >=20, >=40, >=60 days","pandas masks","Pre-specified sample robustness"),
        ("P1","Proportion uncertainty","description; no automatic independent-stock inference","none","Date common shock precludes naive firm-independent event p")
    ]
    return pd.DataFrame(entries,columns=["spec_id","method","equation_or_rule","python_implementation","inference_and_limits"])


def all_results(raw, sha, threshold=20, advanced=True):
    x=prepare(raw,threshold)
    daily=coverage_daily(x)
    pairs=transition_pairs(x)
    eventmatrix=transition_matrix(pairs,"2026-10-15")
    eventmatrix_main=transition_matrix(pairs,"2026-10-15",True)
    transdaily=transition_daily(pairs)
    rec,rectable,reccurve=recovery(x)
    reg=run_regressions(x) if advanced else pd.DataFrame([{"status":"NOT RUN: advanced=False"}])
    forecast,pred=predict_walk_forward(x) if advanced else (pd.DataFrame([{"status":"NOT RUN: advanced=False"}]),pd.DataFrame())
    return {"raw":raw,"processed":x,"sha":sha,"audit":audit(raw,x,sha),"daily":daily,"pairs":pairs,
     "event_matrix":eventmatrix,"event_matrix_main":eventmatrix_main,"transition_daily":transdaily,
     "recovery_stock":rec,"recovery_table":rectable,"recovery_curve":reccurve,
     "prepost":prepost_summary(x),"regressions":reg,"forecast":forecast,"prediction_rows":pred,
     "robustness":robustness(x,pairs),"dictionary":variable_dictionary(),"methods":methods_table(),
     "params":{"version":VERSION,"protocol":PROTOCOL,"activity_threshold":threshold,"main_end":str(MAIN_END.date()),
       "event_date":str(EVENT_DATE.date()),"metadata_warning":METADATA_WARNING,"advanced":advanced}}
