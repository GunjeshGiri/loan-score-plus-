import streamlit as st
import math

def card(title, content_func, width=100):
    st.markdown(f"<div class='card'> <h3>{title}</h3>", unsafe_allow_html=True)
    content_func()
    st.markdown("</div>", unsafe_allow_html=True)

def render_score_gauge(score, subtitle="Credit Score"):
    score = max(0, min(100, score))
    if score < 40:
        color = "#ef4444"
    elif score < 70:
        color = "#f59e0b"
    else:
        color = "#10b981"

    angle = (score/100)*180
    svg = f"""
    <svg width='220' height='120' viewBox='0 0 220 120'>
      <defs>
        <linearGradient id='g1' x1='0' x2='1'>
          <stop offset='0%' stop-color='#f97316'/>
          <stop offset='100%' stop-color='{color}'/>
        </linearGradient>
      </defs>
      <g transform='translate(110,110)'>
        <path d='M -100 0 A 100 100 0 0 1 100 0' fill='none' stroke='#e6e9ef' stroke-width='20'/>
        <path d='M -100 0 A 100 100 0 0 1 100 0' fill='none' 
              stroke='url(#g1)' stroke-width='20' 
              stroke-dasharray='{(angle/180)*628} 628' stroke-linecap='round'/>
        <line x1='0' y1='0' x2='{100*math.cos(math.radians(180-angle)):.2f}' 
              y2='{-100*math.sin(math.radians(180-angle)):.2f}' 
              stroke='#0f172a' stroke-width='3'/>
        <circle cx='0' cy='0' r='4' fill='#0f172a'/>
      </g>
      <text x='110' y='95' font-size='18' text-anchor='middle' fill='#0f172a'>{score}</text>
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)

def small_kv(k, v):
    st.markdown(f"<div><strong>{k}</strong>: <span class='small-muted'>{v}</span></div>", unsafe_allow_html=True)
