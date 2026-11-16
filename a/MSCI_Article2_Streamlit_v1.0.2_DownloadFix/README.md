# MSCI Article 2 — Gate C Streamlit Research Application

**Research question:** When recorded bids are absent, do one-sided daily quotations reveal an extensive-margin liquidity condition that ordinary spread analyses cannot capture, and are quote states associated with next-session downside risk?

**Research input:** Exactly the user's 2026 IDX `Ringkasan_Saham_Gabungan.xlsx`, one tab `Data`, 30 original columns. No new quantitative dataset is downloaded. Public institutional sources from Gates A/B establish context only.

## Windows: easiest method

1. Install **Python 3.11 or 3.12** from python.org (select *Add Python to PATH*).
2. Extract this package into a folder. Double-click `START_WINDOWS.bat`. The first launch installs dependencies.
3. The browser opens the application. Upload **Ringkasan_Saham_Gabungan.xlsx** using the left sidebar.
4. Choose the pre-event activity threshold (default **20**) and check **Run regressions and time-ordered predictions**.
5. Click **Run full analysis**. Inspect all eight tabs and model statuses.
6. In tab **8 · Export Results**, click **Build full Excel + Figures ZIP** and then Download.

Alternatively from Terminal / macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## What is implemented

- All 147,044 original observations and original 30 source fields preserved in `01_Source_Raw`.
- The complete 147,044-row processed panel with quote state, bid/offer validations, return/lag/lead, exposure, and eligibility flags.
- Three pre-event activity screens (20/40/60 active sessions), 842/833/758 stocks expected for the original source.
- Four status classifications: `two_sided`, `offer_only`, `bid_only`, `neither`; invalid or missing source quote entries have an explicit fifth audit category.
- Trading-session-aware 4×4 stock-state transition matrices (including 15→16 October) and first interday two-sided restoration.
- Quoted-spread availability separated from conditional spread; QDI only when both quote sides and quoted sizes are valid.
- Main matched 17+17 dates around 16 October, full series April–Nov, and extended information in audit dataset.
- Firm and date fixed effects in two research LPMs (two-way firm/date clustered SE, finite degrees of freedom), plus baseline model.
- Chronological expanding monthly logistic prediction in October and November: returns/activity baseline versus +quote-state/QDI, with label-date leakage checks.
- 17 worksheet workbook, six 300-DPI data-derived figures embedded in Excel and exported as separate PNGs, and JSON manifest.

## Important caveats

- **The 2026 IDX field-level documentation is unresolved:** unknown precise capture time for Bid/Offer, the operational meaning of zero, units of Bid Volume/Offer Volume, and board/auction flags. These are **reported daily quote states**, not observed lifetime of all orders or intraday cancellations.
- Return is `(Penutupan - Sebelumnya) / Sebelumnya`; the source has 23 previous-price discrepancies vs sequential closes, flagged but not overwritten.
- Zero volume and zero prices are not overwritten. Undefined spread and QDI are **missing**, never set to zero. `Tradeble Shares` is retained as originally spelled.
- Firm/date FE absorb common market shocks and therefore do **not** identify a causal MSCI effect. The 19–20 October ARB change and November index review remain potential confounders.
- Model failure, nonfinite standard error and insufficient class representation are reported as failure/NA, not forced into a significant result.
- The forecast tests incremental association and uses same-day reported quote data; its practical realtime tradability remains subject to timestamp validation.
- The 17+17 sample and main end date (10 Nov) are primary. Some advanced robustness (e.g. alternate price bands with verified board metadata, controlled survival hazard, and true intraday resiliency) cannot be identified from the available source and are deliberately **not claimed**.

## Workbook contents

`00_Readme`, `01_Source_Raw`, `02_Data_Processed`, `03_Sample_Flags`, `04_Data_Audit`, `05_Variable_Dictionary`, `06_Formulas_Models`, `07_Daily_Quote_States`, `08_Quote_Transitions`, `09_Pre_Post`, `10_TwoPart_Liquidity`, `11_Regression_Results`, `12_Predictive_Results`, `13_Robustness`, `14_Figures`, `15_Results_Index`, `16_Analysis_Log`.

## Research provenance

Version: GateC-1.0.0. Builds on `Article_2_Research_Design_Protocol_v1.0.docx` and `Article_2_Gate_B_Literature_Quote_Validation_v1.0.docx`. The source SHA-256 and registered specifications are exported. Related Manuscript 1 is distinct: no event-study CAR replication or causal index claims.

## Reproducing the smoke tests

From the project directory:

```bash
python -m unittest discover -s tests -v
```

The tests expect the original source at `../Ringkasan_Saham_Gabungan.xlsx` or set environment variable `MSCI_SOURCE_FILE` to the full path. Where no data file is found, tests are skipped.

## Gate D correction (v1.0.1)
Sample prediction now requires next_date<=10 Nov 2026. This excludes 842 t=10 Nov cases whose next-day outcomes are 11 Nov; corrected M2 estimates are provided separately in Gate D audit. Regenerate 17-sheet workbook from this version before manuscript finalization.

## Gate D v1.0.2 — reliable large downloads (16 November 2026)

The old Export tab showed two large download buttons immediately after building the workbook. This could force Streamlit to read ~100 MB ZIP and ~100 MB XLSX into application memory together, and downloads could silently fail if the server or browser was resource constrained. This update retains the original 17-sheet XLSX content, including all source observations and analysis results, and changes only the delivery workflow:

1. Upload `Ringkasan_Saham_Gabungan.xlsx`, run full analysis, and enter tab 8.
2. Click **Build full Excel + Figures ZIP** and wait until **Ready on server** appears.
3. Click **Test download (1 KB)**. If this fails, check Edge/Chrome Downloads (Ctrl+J), site permissions, or the hosting server.
4. If that works, select **ZIP package** or **Excel workbook**, enable large-file download, and use the direct button.
5. If full-file download fails, use **Reliable fallback**. Download each of the sequential 12 MiB parts by choosing Part 01, Part 02, ... in the drop-down; click **Prepare Windows reassembly instructions**; download the small `.bat` file; put all parts and `.bat` in the same folder; double-click the `.bat` to restore exactly the original file. Confirm the printed SHA256 hash matches the expected value in the app.

The fallback neither omits analyses nor changes values. It streams only one ~12 MiB part into Streamlit memory per button interaction, instead of both ~100 MiB files. Re-run the application after replacing the previous code folder; the old built exports are in the previous Streamlit server session and may need rebuilding.
