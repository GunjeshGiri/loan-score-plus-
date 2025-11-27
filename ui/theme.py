CSS = """
:root{
  --bg:#f6f9fc;
  --card:#ffffff;
  --muted:#6b7280;
  --accent:#0ea5a4;
  --accent-2:#0369a1;
}

body { background: var(--bg); }

.card{
  background: var(--card);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 18px rgba(16,24,40,0.06);
}

.header-row{display:flex;align-items:center;justify-content:space-between}
.score-card{display:flex;gap:18px;align-items:center}
.small-muted{color:var(--muted);font-size:12px}
.gauge-wrap{display:flex;align-items:center;gap:12px}
"""

def inject_css():
    import streamlit as st
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
