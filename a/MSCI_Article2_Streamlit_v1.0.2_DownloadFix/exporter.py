"""Audit-ready XLSX and research package ZIP exporting. 

Uses streaming Excel row writing to remain feasible for >11 million cells.
Raw source fields, order and values are preserved logically; original file SHA retained.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, date
import hashlib, io, json, math, platform, zipfile
import numpy as np
import pandas as pd
import xlsxwriter
from analysis_engine import VERSION, METADATA_WARNING, SOURCE_FIELDS, STATES
from plots import plot_all

NAVY="#123247"; TEAL="#087E80"; LIGHT="#EAF1F4"; TEXT="#16313F"
SHEET_NAMES=["00_Readme","01_Source_Raw","02_Data_Processed","03_Sample_Flags","04_Data_Audit",
 "05_Variable_Dictionary","06_Formulas_Models","07_Daily_Quote_States","08_Quote_Transitions",
 "09_Pre_Post","10_TwoPart_Liquidity","11_Regression_Results","12_Predictive_Results",
 "13_Robustness","14_Figures","15_Results_Index","16_Analysis_Log"]

def xlsx_value(v):
    if v is None or v is pd.NaT or v is pd.NA: return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, (pd.Timestamp,datetime,np.datetime64)):
        d=pd.Timestamp(v);return None if pd.isna(d) else d.to_pydatetime()
    if isinstance(v, (date,)):return datetime.combine(v,datetime.min.time())
    if isinstance(v,str):return v[:32767]
    if pd.isna(v):return None
    return str(v)


def formats(wb):
    return {
      "head":wb.add_format({"bold":True,"font_color":"#FFFFFF","bg_color":NAVY,"valign":"vcenter","bottom":2,"bottom_color":TEAL,"text_wrap":True}),
      "title":wb.add_format({"bold":True,"font_color":NAVY,"font_size":15,"bottom":1,"bottom_color":TEAL}),
      "sub":wb.add_format({"bold":True,"font_color":TEAL,"bg_color":LIGHT}),
      "date":wb.add_format({"num_format":"yyyy-mm-dd"}),
      "num":wb.add_format({"num_format":"#,##0.0000"}),
      "integer":wb.add_format({"num_format":"#,##0"}),
      "note":wb.add_format({"font_color":"#9A4D00","bg_color":"#FFF1D6","text_wrap":True,"valign":"top"}),
      "text":wb.add_format({"font_color":TEXT,"valign":"top"}),
    }


def new_sheet(wb,name,header_style=True):
    ws=wb.add_worksheet(name)
    ws.freeze_panes(1,2 if name in ("01_Source_Raw","02_Data_Processed") else 0)
    ws.set_default_row(16)
    ws.set_tab_color(TEAL if name.startswith(("00","07","08","14")) else NAVY)
    return ws


def dataframe(ws, df: pd.DataFrame, fmt, start=0, limit=None, filterable=True):
    """Stream values to a worksheet, with defined header row and reasonable column widths."""
    if df is None or df.empty:
        ws.write(start,0,"NO OBSERVATIONS / MODEL NOT RUN",fmt["note"])
        return start+2
    frame=df.head(limit) if limit else df
    headers=[str(c) for c in frame.columns]
    ws.set_row(start,26)
    ws.write_row(start,0,headers,fmt["head"])
    for ci,c in enumerate(headers):
        # do not use Excel autofit on massive tables
        width=min(30,max(13,len(c)+2))
        if c in ("metadata_validation_status","inference_warning","status","equation_or_rule","formula_or_rule","validity_and_limits"):
            width=42
        ws.set_column(ci,ci,width)
    for i, row in enumerate(frame.itertuples(index=False,name=None), start+1):
        vals=[xlsx_value(v) for v in row]
        try:
            ws.write_row(i,0,vals)
        except (TypeError,ValueError):
            ws.write_row(i,0,[str(v)[:32767] if v is not None else None for v in vals])
    if filterable and len(frame)>1:
        ws.autofilter(start,0,start+len(frame),len(headers)-1)
    return start+len(frame)+2


def block(ws,title,frame,fmt,row):
    ws.write(row,0,title,fmt["title"])
    return dataframe(ws,frame,fmt,start=row+2,filterable=False)+2


def export_excel(r:dict, outpath, figure_paths=None):
    """Export ALL original source and processed rows. Typically tens of MB."""
    outpath=Path(outpath);outpath.parent.mkdir(parents=True,exist_ok=True)
    wb=xlsxwriter.Workbook(str(outpath), {"constant_memory":True,"default_date_format":"yyyy-mm-dd", "nan_inf_to_errors":False,
                                           "strings_to_formulas":False,"strings_to_urls":False})
    fmt=formats(wb)
    def sheet(name):return new_sheet(wb,name)
    # 00 - warnings and executive navigation
    ws=sheet("00_Readme")
    info=pd.DataFrame([
        ("Research topic","When Recorded Bids Are Absent — MSCI Article 2"),
        ("Analysis engine",VERSION),("Source file SHA-256",r["sha"]),
        ("Source records",len(r["raw"])),("Code count",r["processed"]["Kode Saham"].nunique()),
        ("Trading sessions",r["processed"]["Tanggal"].nunique()),
        ("Condition",METADATA_WARNING),
        ("Causal attribution","NO. One-country single event; association only."),
        ("How to interpret outputs","See 05_Variable_Dictionary, 06_Formulas_Models, and 16_Analysis_Log"),
        ("Full data guarantee","Raw original 30 columns; processed all records; row-level sample flags and matched transitions."),
        ("Regression status","Check 11_Regression_Results; failure is explicitly labeled."),
        ("Prediction status","Check 12_Predictive_Results; chronological training/validation only."),
    ],columns=["item","details"])
    dataframe(ws,info,fmt)
    ws.set_column("B:B",105)
    ws.write(17,0,"ALL 17 SHEETS",fmt["sub"])
    for j,name in enumerate(SHEET_NAMES,18):
        ws.write_url(j,0,"internal:'"+name+"'!A1",string=name)
    # 01 Raw 147k rows x 30 original columns
    print("WRITING 01_Source_Raw",flush=True)
    ws=sheet("01_Source_Raw")
    dataframe(ws,r["raw"][SOURCE_FIELDS],fmt)
    # 02 full processed (source 30 columns followed by feature columns)
    print("WRITING 02_Data_Processed",flush=True)
    ws=sheet("02_Data_Processed")
    dataframe(ws,r["processed"],fmt)
    # 03 flags
    print("WRITING 03_Sample_Flags",flush=True)
    ws=sheet("03_Sample_Flags")
    flagcols=["Tanggal","Kode Saham","date_idx","quote_state","pre60_active_sessions",
              "sample_activity_20","sample_activity_40","sample_activity_60","sample_main","sample_spread",
              "sample_prediction","sample_exclusion_reason","next_session_valid","reference_mismatch",
              "crossed_quote","quote_price_volume_mismatch","foreign_qty_gt_volume"]
    dataframe(ws,r["processed"][flagcols],fmt)
    # 04 audit
    ws=sheet("04_Data_Audit")
    block(ws,"RAW DATA AUDIT",r["audit"],fmt,0)
    daily_quality=r["daily"][["Tanggal","n_observed","nonzero_trade_volume","two_sided","offer_only","bid_only","neither","two_sided_pct","offer_only_pct"]]
    block(ws,"PER-SESSION COVERAGE",daily_quality,fmt,len(r["audit"])+8)
    # 05 dictionary
    ws=sheet("05_Variable_Dictionary")
    block(ws,"RAW SOURCE COLUMNS — ALL 30 FIELDS",pd.DataFrame({"original_column":SOURCE_FIELDS}),fmt,0)
    block(ws,"GENERATED FEATURES",r["dictionary"],fmt,36)
    # 06 formulas
    ws=sheet("06_Formulas_Models")
    block(ws,"STATISTICAL FORMULAS / MODEL SPECIFICATIONS",r["methods"],fmt,0)
    # Excel formula demonstration, not used as the analytic estimator
    row=17
    ws.write(row,0,"AUDIT FORMULA EXAMPLE — illustrative cell references",fmt["title"])
    for j,h in enumerate(["Bid","Offer","Bid Volume","Offer Volume","Relative spread (Excel formula)","QDI (Excel formula)","Quote State (Excel formula)"]):
        ws.write(row+2,j,h,fmt["head"])
    ws.write_row(row+3,0,[0,110,0,500])
    rr=row+4
    ws.write_formula(row+3,4,f'=IF(AND(A{rr}>0,B{rr}>=A{rr}),(B{rr}-A{rr})/((A{rr}+B{rr})/2),"")')
    ws.write_formula(row+3,5,f'=IF(AND(A{rr}>0,B{rr}>0,C{rr}>0,D{rr}>0),(C{rr}-D{rr})/(C{rr}+D{rr}),"")')
    ws.write_formula(row+3,6,f'=IF(AND(A{rr}>0,B{rr}>0),"two_sided",IF(AND(A{rr}=0,B{rr}>0),"offer_only",IF(AND(A{rr}>0,B{rr}=0),"bid_only","neither")))')
    ws.write(row+5,0,"All 147,044 derived observation values are calculated by versioned Python; demo Excel formulas are NOT replacements for model estimation.",fmt["note"])
    # 07 daily
    ws=sheet("07_Daily_Quote_States");dataframe(ws,r["daily"],fmt)
    # 08 matrices and pairwise details
    print("WRITING 08_Quote_Transitions",flush=True)
    ws=sheet("08_Quote_Transitions")
    q=block(ws,"15 OCT TO 16 OCT — FULL UNIVERSE",r["event_matrix"],fmt,0)
    q=block(ws,"15 OCT TO 16 OCT — MAIN COHORT",r["event_matrix_main"],fmt,q)
    q=block(ws,"DAILY TRANSITIONS",r["transition_daily"],fmt,q)
    q=block(ws,"RECOVERY SUMMARY",r["recovery_table"],fmt,q)
    q=block(ws,"RECOVERY CURVE",r["recovery_curve"],fmt,q)
    q=block(ws,"RECOVERY STOCK-LEVEL DETAIL",r["recovery_stock"],fmt,q)
    paircols=["Kode Saham","Tanggal","date_idx","quote_state","next_date","next_state","sample_main"]
    block(ws,"ALL MATCHED ADJACENT-SESSION TRANSITIONS",r["pairs"][paircols],fmt,q)
    # 09 pre/post
    ws=sheet("09_Pre_Post");dataframe(ws,r["prepost"],fmt)
    # 10 extensive/intensive
    xx=r["processed"].loc[r["processed"].sample_main]
    parts=[]
    for period in ("pre17","event","post17"):
        d=xx[xx.main_period.eq(period)]
        parts.append({"period":period,"n_obs":len(d),"n_valid_two_sided":int(d.valid_quoted_spread.sum()),
              "two_sided_coverage_pct":100*d.two_sided.mean() if len(d) else np.nan,
              "offer_only_pct":100*d.offer_only.mean() if len(d) else np.nan,
              "mean_spread_pct_conditional":100*d.relative_spread.mean(),
              "median_spread_pct_conditional":100*d.relative_spread.median(),
              "mean_qdi_conditional":d.quote_depth_imbalance.mean(),
              "n_qdi_valid":int(d.valid_qdi.sum())})
    ws=sheet("10_TwoPart_Liquidity")
    block(ws,"STAGE 1: AVAILABILITY; STAGE 2: CONDITIONAL SPREAD/DEPTH",pd.DataFrame(parts),fmt,0)
    ws.write(10,0,"IMPORTANT: spread and QDI are undefined for one-sided states; no zeros are imputed.",fmt["note"])
    # 11 regression
    ws=sheet("11_Regression_Results");dataframe(ws,r["regressions"],fmt)
    # 12 prediction metrics and predictions
    print("WRITING 12_Predictive_Results",flush=True)
    ws=sheet("12_Predictive_Results")
    q=block(ws,"WALK-FORWARD PERFORMANCE (BASELINE VS + QUOTES)",r["forecast"],fmt,0)
    if not r["prediction_rows"].empty:
        block(ws,"INDIVIDUAL OUT-OF-SAMPLE PREDICTIONS",r["prediction_rows"],fmt,q)
    # 13 robustness
    ws=sheet("13_Robustness")
    q=block(ws,"ACTIVITY THRESHOLD SENSITIVITY",r["robustness"],fmt,0)
    block(ws,"PER-SESSION TWO-SIDED TO OFFER-ONLY TRANSITION CHECK",r["transition_daily"],fmt,q)
    # 14 6 images
    print("WRITING 14_Figures",flush=True)
    ws=sheet("14_Figures")
    ws.set_column("A:A",23); ws.set_column("B:M",14)
    if figure_paths:
        for i,fp in enumerate(figure_paths):
            top=i*29
            ws.write(top,0,Path(fp).stem.replace("_"," "),fmt["title"])
            ws.insert_image(top+2,0,str(fp),{"x_scale":.58,"y_scale":.58,"object_position":1})
    # 15 result index
    index=pd.DataFrame([
        ("Table 1","Sample Selection and Data Quality Assessment","04_Data_Audit;03_Sample_Flags","Main"),
        ("Table 2","Descriptive Statistics of Quotation States","07_Daily_Quote_States","Main"),
        ("Table 3","Quote-State Transitions around MSCI Consultation","08_Quote_Transitions","Main"),
        ("Table 4","Recovery of Two-Sided Quotations","08_Quote_Transitions","Main"),
        ("Table 5","Extensive and Intensive Liquidity Margins","10_TwoPart_Liquidity","Main"),
        ("Table 6","Panel Determinants of Offer-Only States","11_Regression_Results","Main"),
        ("Table 7","Next-Session Downside Return Risk","11_Regression_Results;12_Predictive_Results","Main"),
        ("Table 8","Robustness and Sensitivities","13_Robustness","Main"),
        *[("Figure "+str(i),Path(p).stem,"14_Figures","Main") for i,p in enumerate(figure_paths or [],1)]
    ],columns=["manuscript_id","item_title","source_sheet","priority"])
    ws=sheet("15_Results_Index");dataframe(ws,index,fmt)
    # 16 analysis log
    keys={"analysis_version":VERSION,"source_sha256":r["sha"],"source_n":len(r["raw"]),
          "event_date":r["params"]["event_date"],"main_end":r["params"]["main_end"],
          "activity_threshold":r["params"]["activity_threshold"],
          "metadata_validation_status":"CONDITIONAL / written field dictionary absent",
          "python_version":platform.python_version(),"pandas_version":pd.__version__,
          "numpy_version":np.__version__,"xlsxwriter_version":xlsxwriter.__version__,
          "created_utc":datetime.utcnow().isoformat(timespec="seconds"),
          "limits":"Reported daily quotes, noncausal associations. Researcher's single IDX dataset only."}
    print("WRITING 16_Analysis_Log",flush=True)
    ws=sheet("16_Analysis_Log")
    dataframe(ws,pd.DataFrame(keys.items(),columns=["key","value"]),fmt)
    wb.set_properties({"title":"MSCI Article 2 — Reproducible IDX Quote State Analysis",
                       "subject":"Gate C / audit-ready statistical analysis",
                       "comments":METADATA_WARNING,
                       "author":"Research analysis workflow"})
    print("FINALIZING XLSX ZIP",flush=True)
    wb.close()
    print("XLSX READY",str(outpath),flush=True)
    return outpath


def research_zip(r,output_zip,work_dir):
    work_dir=Path(work_dir);work_dir.mkdir(parents=True,exist_ok=True)
    figures=plot_all(r,work_dir/"figures")
    excel=export_excel(r,work_dir/"MSCI_Article2_Complete_Analysis.xlsx",figure_paths=figures)
    manifest={"analysis_version":VERSION,"source_hash":r["sha"],"raw_records":len(r["raw"]),
              "source_columns":len(SOURCE_FIELDS),"required_sheets":SHEET_NAMES,
              "figures":[p.name for p in figures],"model_status":r["regressions"].status.value_counts(dropna=False).to_dict(),
              "quote_metadata_validation":"CONDITIONAL / unresolved field-level verification",
              "analysis_params":r["params"]}
    manifest_fp=work_dir/"analysis_manifest.json";manifest_fp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    output_zip=Path(output_zip)
    with zipfile.ZipFile(output_zip,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        z.write(excel,arcname=excel.name)
        z.write(manifest_fp,arcname=manifest_fp.name)
        for f in figures:z.write(f,arcname="figures/"+f.name)
    return output_zip,excel,figures
