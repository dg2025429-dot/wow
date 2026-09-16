import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

APP_TITLE = "선수 유형 나누기"
APP_ICON = "⚽"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/eafc25_top100.csv"
RANDOM_STATE = 42

ABILITY_MAP = {
    "pace": "속도",
    "shooting": "슈팅",
    "passing": "패스",
    "dribbling": "드리블",
    "defending": "수비",
    "physic": "몸싸움",
}
ABILITIES = list(ABILITY_MAP.values())
CLUSTER_MARKS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳"]

FORWARD = {"ST", "CF", "LW", "RW", "LF", "RF"}
MIDFIELDER = {"CAM", "CM", "CDM", "LM", "RM", "LAM", "RAM"}
DEFENDER = {"CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB"}

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("EA FC 25 상위 100명 선수의 능력치를 k-평균으로 묶어 봅니다.")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df = df.rename(columns=ABILITY_MAP)
    df["첫포지션"] = df["positions"].astype(str).str.split(",").str[0].str.strip().str.upper()
    df["포지션군"] = df["첫포지션"].map(position_group)
    return df


def position_group(pos):
    if pos in FORWARD:
        return "공격수"
    if pos in MIDFIELDER:
        return "미드필더"
    if pos in DEFENDER:
        return "수비수"
    if pos == "GK":
        return "골키퍼"
    return "기타"


df = load_data()

# ── 설정 ────────────────────────────────────────────────────────────────
st.subheader("묶기 설정")
col_left, col_right = st.columns([3, 2])

with col_left:
    features = st.multiselect(
        "묶는 데 사용할 능력치 (2개 이상)",
        options=ABILITIES,
        default=ABILITIES,
    )
with col_right:
    n_clusters = st.slider("묶음 수", min_value=2, max_value=6, value=3, step=1)

if len(features) < 2:
    st.warning("능력치를 2개 이상 골라 주세요.")
    st.stop()

# ── k-평균 ──────────────────────────────────────────────────────────────
X = StandardScaler().fit_transform(df[features])
labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
df["_label"] = labels

order = (
    df.groupby("_label")["슈팅"].mean().sort_values(ascending=False).index.tolist()
)
label_to_mark = {label: CLUSTER_MARKS[i] for i, label in enumerate(order)}
df["묶음"] = df["_label"].map(label_to_mark)
mark_order = [CLUSTER_MARKS[i] for i in range(n_clusters)]
df["묶음"] = pd.Categorical(df["묶음"], categories=mark_order, ordered=True)

color_map = {
    mark: color
    for mark, color in zip(mark_order, px.colors.qualitative.Set1)
}

# ── 2차원 산점도 ─────────────────────────────────────────────────────────
st.subheader("2차원 산점도")
c1, c2 = st.columns(2)
with c1:
    x2 = st.selectbox("가로축", features, index=0, key="x2")
with c2:
    y2 = st.selectbox("세로축", features, index=1 if len(features) > 1 else 0, key="y2")

fig2 = px.scatter(
    df,
    x=x2,
    y=y2,
    color="묶음",
    hover_name="name_ko",
    hover_data={"club": True, "overall": True, "묶음": False},
    category_orders={"묶음": mark_order},
    color_discrete_map=color_map,
)
fig2.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
st.plotly_chart(fig2, use_container_width=True)

# ── 3차원 산점도 ─────────────────────────────────────────────────────────
st.subheader("3차원 산점도")
if len(features) < 3:
    st.info("3차원 산점도를 그리려면 능력치를 3개 이상 골라 주세요.")
else:
    c3, c4, c5 = st.columns(3)
    with c3:
        x3 = st.selectbox("x축", features, index=0, key="x3")
    with c4:
        y3 = st.selectbox("y축", features, index=1, key="y3")
    with c5:
        z3 = st.selectbox("z축", features, index=2, key="z3")

    fig3 = px.scatter_3d(
        df,
        x=x3,
        y=y3,
        z=z3,
        color="묶음",
        hover_name="name_ko",
        category_orders={"묶음": mark_order},
        color_discrete_map=color_map,
    )
    fig3.update_traces(marker=dict(size=3))
    fig3.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("마우스로 끌어서 돌려 볼 수 있습니다.")

# ── 묶음별 요약 ──────────────────────────────────────────────────────────
st.subheader("묶음별 인원과 능력치 평균")
summary = df.groupby("묶음", observed=True).agg(
    인원=("name_ko", "count"),
    **{a: (a, "mean") for a in ABILITIES},
)
st.dataframe(summary.round(1), use_container_width=True)

# ── 묶음별 대표 선수 ─────────────────────────────────────────────────────
st.subheader("묶음별 종합 능력치 상위 5명")
cols = st.columns(n_clusters)
for col, mark in zip(cols, mark_order):
    top5 = (
        df[df["묶음"] == mark]
        .sort_values("overall", ascending=False)
        .head(5)[["name_ko", "overall"]]
        .reset_index(drop=True)
    )
    top5.columns = ["이름", "종합"]
    top5.index = range(1, len(top5) + 1)
    with col:
        st.markdown(f"**{mark} 묶음**")
        st.dataframe(top5, use_container_width=True)

# ── 교차표 ──────────────────────────────────────────────────────────────
st.subheader("묶음 × 포지션 교차표")
st.caption("포지션은 묶는 데 사용하지 않고, 결과를 해석할 때만 씁니다.")
cross = pd.crosstab(df["묶음"], df["포지션군"])
pos_order = [p for p in ["공격수", "미드필더", "수비수", "골키퍼", "기타"] if p in cross.columns]
cross = cross[pos_order]
st.dataframe(cross, use_container_width=True)

fig_heat = px.imshow(
    cross,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="포지션", y="묶음", color="인원"),
    aspect="auto",
)
fig_heat.update_layout(height=380, coloraxis_showscale=False)
st.plotly_chart(fig_heat, use_container_width=True)
