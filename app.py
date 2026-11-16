"""Streamlit research dashboard — Gate C, Article 2."""
from __future__ import annotations
import io
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analysis_engine import (load_source, all_results, METADATA_WARNING, EVENT_DATE,
                             MAIN_END, VERSION, STATES)
from exporter import research_zip, SHEET_NAMES
from download_support import (CHUNK_BYTES, part_count, read_part, part_filename,
                              sha256_file, windows_reassemble_batch)

st.set_page_config(page_title="MSCI Article 2 · Quote States",page_icon="📊",layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.25rem;}
h1,h2,h3{color:#123247;}div[data-testid="stMetric"]{background:#f1f5f8;border-radius:10px;padding:12px;}
</style>
""",unsafe_allow_html=True)
st.title("When Recorded Bids Are Absent")
st.caption("Gate C · Research Dashboard · Quote states, interday restoration, and next-session downside risk · "+VERSION)
st.warning(METADATA_WARNING,icon="⚠️")

with st.sidebar:
    st.header("Research configuration")
    uploaded=st.file_uploader("Upload Ringkasan_Saham_Gabungan.xlsx",type=["xlsx"])
    threshold=st.selectbox("Pre-event activity threshold (60 sessions)",[20,40,60],index=0)
    advanced=st.checkbox("Run regressions and time-ordered predictions",value=True)
    st.caption("Default cohort: ≥20 active sessions before 16 October; main post ends 10 November 2026.")
    st.info("No numerical information is added from external sources. Models are associational, not causal.")
    compute=st.button("Run full analysis",type="primary",use_container_width=True)

if compute:
    if uploaded is None:
        st.error("Unggah file Excel sumber terlebih dahulu.")
    else:
        with st.spinner("Loading and validating the original 30 fields..."):
            raw, sha=load_source(io.BytesIO(uploaded.getvalue()))
        with st.spinner("Preparing all rows, quote states, transitions, FE models and forecasts..."):
            result=all_results(raw,sha,threshold=threshold,advanced=advanced)
        # Prevent downloads from an earlier analysis run being reused accidentally.
        for k in ("export_file", "export_excel"):
            st.session_state.pop(k, None)
        st.session_state["research_result"]=result
        st.session_state["research_filename"]=uploaded.name
        st.success("Calculation completed. Please inspect the audit and model statuses.")

if "research_result" not in st.session_state:
    st.info("Upload the raw IDX Excel file, then click **Run full analysis**. The dashboard will expose eight research tabs, including Excel export.")
    st.stop()
r=st.session_state["research_result"]
x=r["processed"]
raw=r["raw"]

m1,m2,m3,m4=st.columns(4)
m1.metric("Source observations",f"{len(raw):,}")
m2.metric("Stock codes",f"{raw['Kode Saham'].nunique():,}")
m3.metric("Trading sessions",f"{x.Tanggal.nunique():,}")
m4.metric("Main eligible stocks",f"{x.loc[x.sample_main,'Kode Saham'].nunique():,}")
tabs=st.tabs(["1 · Dataset Explorer","2 · Data Audit & Sampling","3 · Quote Availability",
              "4 · Transitions & Recovery","5 · Liquidity Margins","6 · Statistical Models",
              "7 · Robustness & Prediction","8 · Export Results"])

with tabs[0]:
    st.subheader("All original data and model-ready observations")
    view=st.radio("Dataset",["Raw source — original 30 columns","Processed — all features"],horizontal=True)
    source=raw if view.startswith("Raw") else x
    c1,c2,c3=st.columns([2,2,2])
    code_query=c1.text_input("Filter stock code (contains)","").strip().upper()
    dates=sorted(x.Tanggal.unique())
    min_date,max_date=pd.Timestamp(dates[0]).date(),pd.Timestamp(dates[-1]).date()
    selected=c2.date_input("Date range",(min_date,max_date),min_value=min_date,max_value=max_date)
    nrows=c3.selectbox("Rows displayed",[50,200,1000,5000],index=1)
    mask=pd.Series(True,index=source.index)
    if code_query: mask &= source["Kode Saham"].astype(str).str.contains(code_query,regex=False)
    if isinstance(selected,tuple) and len(selected)==2:
        mask &= pd.to_datetime(source.Tanggal).between(pd.Timestamp(selected[0]),pd.Timestamp(selected[1]))
    if not view.startswith("Raw"):
        state_filter=st.multiselect("Quote-state filter",STATES,default=STATES)
        mask &= source.quote_state.isin(state_filter)
    st.write(f"Matching observations: **{int(mask.sum()):,}**. Showing first {nrows:,} rows; full records are in the export workbook.")
    st.dataframe(source.loc[mask].head(nrows),use_container_width=True,height=460)
    st.caption("The source workbook is preserved in its original 30-column order. Only the analysis dataset has additional features.")

with tabs[1]:
    st.subheader("Data provenance, validation and sample attrition")
    st.dataframe(r["audit"],use_container_width=True,hide_index=True)
    f=x.loc[x.event_day]
    cohort=pd.DataFrame({"threshold":[20,40,60],"stocks":[f[f[f"sample_activity_{n}"]]["Kode Saham"].nunique() for n in (20,40,60)]})
    st.plotly_chart(px.bar(cohort,x="threshold",y="stocks",title="Event-day cohort size by activity threshold"),use_container_width=True)
    st.markdown("**Integrity checks:** unchanged source hash; stock–date uniqueness; reference mismatches remain flagged; no automatic removal of zero-volume or zero-bid observations.")
    st.code(r["sha"],language=None)

with tabs[2]:
    st.subheader("Reported quote-state incidence (all trading days)")
    daily=r["daily"]
    cohort=st.radio("Denominator",["Fixed main cohort","Observed full universe"],horizontal=True)
    if cohort.startswith("Fixed"):
        graph=pd.DataFrame({"date":daily.Tanggal,**{state:100*daily["main_"+state]/daily.main_n for state in STATES}})
        st.caption("Cohort denominator: fixed selected 842 stocks on covered sessions; displays recorded states.")
    else:
        graph=pd.DataFrame({"date":daily.Tanggal,**{state:100*daily[state]/daily.n_observed for state in STATES}})
        st.caption("Full universe denominator varies with stock-date availability.")
    long=graph.melt("date",var_name="state",value_name="percentage")
    fig=px.area(long,x="date",y="percentage",color="state",title="Share of daily reported quote states")
    fig.add_vline(x=str(EVENT_DATE.date()),line_dash="dash")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(daily,use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("Four-state transitions and restoration")
    st.markdown("**15 → 16 October 2026:** matched stocks, not inferred order cancellations.")
    cohort_only=st.checkbox("Use main eligible cohort for event transition",value=False)
    mat=r["event_matrix_main"] if cohort_only else r["event_matrix"]
    st.dataframe(mat,use_container_width=True,hide_index=True)
    fig=px.imshow(mat.set_index("from_state")[STATES],text_auto=True,aspect="auto",
                  title="Matched transition counts · 15 to 16 October")
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Interday restoration: stocks initially reported offer-only on 16 October**")
    st.dataframe(r["recovery_table"],use_container_width=True,hide_index=True)
    curve=r["recovery_curve"]
    st.plotly_chart(px.line(curve,x="h",y="pct_restored",color="cohort",markers=True,
               title="First subsequent two-sided quotation (cumulative %)"),use_container_width=True)
    st.caption("At-risk and right-censor counts are retained in recovery stock-level records. Quote restoration is observed between trading sessions, not intraday.")

with tabs[4]:
    st.subheader("Extensive versus intensive margins")
    st.dataframe(r["prepost"],use_container_width=True,hide_index=True)
    f=x.loc[x.sample_main].groupby("Tanggal").agg(n=("Kode Saham","size"),valid=("valid_quoted_spread","sum"),
                  conditional_spread=("relative_spread","mean"),conditional_qdi=("quote_depth_imbalance","mean")).reset_index()
    f["coverage_pct"]=100*f.valid/f.n
    f["spread_pct"]=100*f.conditional_spread
    st.plotly_chart(px.line(f,x="Tanggal",y="coverage_pct",title="Proportion with valid quoted spread (%)"),use_container_width=True)
    st.plotly_chart(px.line(f,x="Tanggal",y="spread_pct",title="Quoted spread conditional on valid two-sided quotes (%)"),use_container_width=True)
    st.caption("Offer-only states have undefined spread and QDI, not zero. Any mean spread is conditional on valid quotes and must be presented with its coverage rate.")

with tabs[5]:
    st.subheader("Panel models: firm and date fixed effects")
    if not advanced: st.info("Advanced models disabled. Re-run with the sidebar checkbox enabled.")
    st.dataframe(r["regressions"],use_container_width=True,hide_index=True)
    st.caption("All coefficient estimates are associations; two-way clustered firm/date SE use finite-cluster inference. Models without an estimable result must remain explicitly labeled.")
    st.markdown("**Registered model specifications**")
    st.dataframe(r["methods"],use_container_width=True,hide_index=True)

with tabs[6]:
    st.subheader("Robustness, data limitations, and out-of-sample tests")
    st.dataframe(r["robustness"],use_container_width=True,hide_index=True)
    st.markdown("**Walk-forward model performance**")
    st.dataframe(r["forecast"],use_container_width=True,hide_index=True)
    if "status" in r["forecast"]:
        good=r["forecast"].loc[r["forecast"].status.eq("OK")]
        if len(good):
            st.plotly_chart(px.bar(good,x="fold",y="brier_score",color="model",barmode="group",
                     title="Out-of-sample Brier score (lower is better)"),use_container_width=True)
    st.caption("Training cutoff requires every training label's next trading session to occur no later than training end. No hyperparameter optimization on test folds.")

with tabs[7]:
    st.subheader("Audit-ready Excel and publication figures")
    st.write(f"Workbook: **{len(SHEET_NAMES)} sheets**, all {len(raw):,} source observations, processed values, model outputs and embedded figures.")
    st.dataframe(pd.DataFrame({"sheet": SHEET_NAMES}), use_container_width=True, hide_index=True)
    st.warning("Full export files can exceed 90 MB. If one-click download is blocked, use the 12 MB fallback parts below.")
    if st.button("Build full Excel + Figures ZIP", type="primary", use_container_width=True):
        try:
            with st.spinner("Building all 17 worksheets, 6 figures and run manifest..."):
                outdir = Path(tempfile.mkdtemp(prefix="msci_article2_"))
                p, excel, figures = research_zip(r, outdir / "MSCI_Article2_Research_Output.zip", outdir)
            if not p.exists() or not excel.exists():
                raise FileNotFoundError("Export routine did not write both expected files")
            st.session_state["export_file"] = str(p)
            st.session_state["export_excel"] = str(excel)
            st.success(f"Export complete: ZIP {p.stat().st_size/2**20:.1f} MB, Excel {excel.stat().st_size/2**20:.1f} MB, {len(figures)} figures.")
        except Exception as exc:
            st.session_state.pop("export_file", None)
            st.session_state.pop("export_excel", None)
            st.error(f"Export failed: {type(exc).__name__}: {exc}")
            st.stop()

    zip_path = Path(st.session_state.get("export_file", "__file_not_built__"))
    excel_path = Path(st.session_state.get("export_excel", "__file_not_built__"))
    if not zip_path.is_file() or not excel_path.is_file():
        st.info("Click **Build full Excel + Figures ZIP** above first. When finished, the download controls appear below.")
    else:
        st.success(f"Ready on server: ZIP {zip_path.stat().st_size / 2**20:.1f} MB and Excel {excel_path.stat().st_size / 2**20:.1f} MB.")
        st.markdown("**Step 1 — Test browser download (small file)**")
        test_bytes = ("MSCI Article 2 download diagnostic OK.\n" * 30).encode("utf-8")
        st.download_button("Test download (1 KB)", data=test_bytes,
                           file_name="MSCI_Download_Test.txt", mime="text/plain",
                           use_container_width=True, key="gatec_test_download")
        st.caption("If even the 1 KB test fails, check the browser Downloads list (Ctrl+J) and site download permissions. If the test works but large files do not, try fallback parts.")

        st.markdown("**Step 2 — Direct download (one large file)**")
        chosen_direct = st.radio("Select full export", ["ZIP package", "Excel workbook"],
                                 key="gatec_direct_choice", horizontal=True)
        chosen_path = zip_path if chosen_direct == "ZIP package" else excel_path
        large_mime = "application/zip" if chosen_direct == "ZIP package" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        st.info(f"Direct file size: {chosen_path.stat().st_size / 2**20:.1f} MB. Click 'Enable large-file download' to load this one file into Streamlit memory.")
        if st.checkbox("Enable large-file download (high memory usage)", value=False,
                       key="gatec_large_download_enabled"):
            with chosen_path.open("rb") as f:
                st.download_button(f"Download {chosen_direct} — {chosen_path.stat().st_size / 2**20:.1f} MB",
                                   data=f.read(), file_name=chosen_path.name, mime=large_mime,
                                   use_container_width=True, key="gatec_large_download")

        st.markdown("**Step 3 — Reliable fallback: download parts (12 MB each)**")
        st.caption("The fallback does not change the Excel/ZIP content. Download every part and the small reconstruction script into the SAME Windows folder.")
        fallback_type = st.radio("Which file should be split?", ["ZIP package", "Excel workbook"],
                                 key="gatec_fallback_choice", horizontal=True)
        fallback_path = zip_path if fallback_type == "ZIP package" else excel_path
        n_parts = part_count(fallback_path, CHUNK_BYTES)
        part_number = st.selectbox("Choose part to download", list(range(1, n_parts+1)),
                                   format_func=lambda i: f"Part {i:02d} of {n_parts:02d}",
                                   key="gatec_part_number")
        this_part = read_part(fallback_path, part_number-1, CHUNK_BYTES)
        st.download_button(f"Download part {part_number}/{n_parts} ({len(this_part)/2**20:.1f} MB)",
                           data=this_part, file_name=part_filename(fallback_path, part_number-1),
                           mime="application/octet-stream", use_container_width=True,
                           key="gatec_download_part")
        with st.expander("Instructions to recombine the parts on Windows"):
            st.markdown("1. Download parts **1 through " + str(n_parts) + "** using the selector above.\n"
                        "2. Download the .bat script below.\n"
                        "3. Put the downloaded parts and .bat in the same folder.\n"
                        "4. Double-click the .bat. It will combine all parts into the original file, then display an SHA-256 hash for comparison.\n"
                        "5. Open the resulting .zip or .xlsx as usual. Keep the window open to inspect any error.")
        if st.button("Prepare Windows reassembly instructions", key="gatec_prepare_hash"):
            st.session_state["gatec_hash_path"] = str(fallback_path)
            st.session_state["gatec_hash"] = sha256_file(fallback_path)
        if st.session_state.get("gatec_hash_path") == str(fallback_path):
            digest = st.session_state["gatec_hash"]
            batch = windows_reassemble_batch(fallback_path, digest, CHUNK_BYTES)
            st.download_button("Download reassembly script (.bat, small file)",
                               data=batch, file_name=f"REASSEMBLE_{fallback_path.name}.bat",
                               mime="application/octet-stream", use_container_width=True,
                               key="gatec_download_batch")
            st.code(f"Expected SHA-256: {digest}")
            st.caption("This script uses the built-in Windows copy and certutil commands. Run only a script downloaded from your own application.")
        else:
            st.caption("Click 'Prepare Windows reassembly instructions' to generate the small reconstruction script and file integrity hash.")
