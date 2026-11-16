Gate D targeted correction, 2026-11-16. All original 30 source columns and 17-sheet export are unchanged.
Change: sample_prediction requires next_date <= 2026-11-10; this removes 842 predictor rows on 10 Nov where targets are on 11 Nov.
Original M2: beta_offer_only=0.254901616, N=125921; independently corrected: beta=0.253884, N=125079 (model sample).
Gate B caveat unresolved: quote snapshot time, zero-code meaning, volume unit, board/auction.
Run Streamlit and rebuild full Excel for the authoritative canonical workbook.
