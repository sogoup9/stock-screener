import re
import time
import warnings

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="飆股選股器 V2", page_icon="🚀", layout="wide")

# =========================
# 官方資料來源
# =========================
TWSE = "https://openapi.twse.com.tw/v1"
TPEX = "https://www.tpex.org.tw/openapi/v1"

URL = {
    "listed_company": f"{TWSE}/opendata/t187ap03_L",
    "otc_company": f"{TPEX}/mopsfin_t187ap03_O",
    "listed_revenue": f"{TWSE}/opendata/t187ap05_L",
    "otc_revenue": f"{TPEX}/mopsfin_t187ap05_O",
    "listed_eps": f"{TWSE}/opendata/t187ap14_L",
    "otc_eps": f"{TPEX}/mopsfin_t187ap14_O",
    "listed_pe": f"{TWSE}/exchangeReport/BWIBBU_ALL",
    "otc_pe": f"{TPEX}/tpex_mainboard_peratio_analysis",
}


def clean_code(x):
    m = re.search(r"([1-9]\d{3})", str(x or "").strip())
    return m.group(1) if m else ""


def to_num(x):
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "--", "None", "nan", "NaN", "null"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def pick(row, names):
    for name in names:
        if name in row.index:
            return row[name]
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_json(url):
    try:
        r = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def fetch_df(url):
    data = fetch_json(url)
    return pd.DataFrame(data)


# =========================
# 股票池：上市 + 上櫃
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def load_universe():
    rows = []

    for market, url in [
        ("上市", URL["listed_company"]),
        ("上櫃", URL["otc_company"]),
    ]:
        df = fetch_df(url)
        if df.empty:
            continue

        for _, row in df.iterrows():
            code = clean_code(
                pick(row, ["公司代號", "股票代號", "Code"])
            )
            name = pick(row, ["公司名稱", "名稱", "Name"])

            if code:
                rows.append(
                    {
                        "代號": code,
                        "名稱": str(name or "").strip(),
                        "市場": market,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["代號", "名稱", "市場"])

    result = pd.DataFrame(rows)
    result = result[result["代號"].str.fullmatch(r"[1-9]\d{3}", na=False)]
    result = result.drop_duplicates(["代號", "市場"])
    return result.sort_values("代號").reset_index(drop=True)


# =========================
# 最新月營收
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_revenue():
    rows = []

    for market, url in [
        ("上市", URL["listed_revenue"]),
        ("上櫃", URL["otc_revenue"]),
    ]:
        df = fetch_df(url)
        if df.empty:
            continue

        for _, row in df.iterrows():
            code = clean_code(
                pick(row, ["公司代號", "股票代號", "Code"])
            )
            if not code:
                continue

            rows.append(
                {
                    "代號": code,
                    "市場": market,
                    "營收年月": str(
                        pick(row, ["資料年月", "DataMonth"]) or ""
                    ),
                    "月營收YoY": to_num(
                        pick(
                            row,
                            [
                                "營業收入-去年同月增減(%)",
                                "營業收入-去年同月增減(％)",
                                "去年同月增減(%)",
                            ],
                        )
                    ),
                    "累計營收YoY": to_num(
                        pick(
                            row,
                            [
                                "累計營業收入-前期比較增減(%)",
                                "累計營業收入-前期比較增減(％)",
                                "累計營收增減(%)",
                            ],
                        )
                    ),
                    "當月營收": to_num(
                        pick(
                            row,
                            [
                                "營業收入-當月營收",
                                "當月營收",
                                "當月營業收入",
                            ],
                        )
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "代號",
                "市場",
                "營收年月",
                "月營收YoY",
                "累計營收YoY",
                "當月營收",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates(["代號", "市場"])


# =========================
# 最新季度 EPS
# =========================
def quarter_num(x):
    m = re.search(r"([1-4])", str(x or ""))
    return int(m.group(1)) if m else 0


@st.cache_data(ttl=1800, show_spinner=False)
def load_eps():
    rows = []

    for market, url in [
        ("上市", URL["listed_eps"]),
        ("上櫃", URL["otc_eps"]),
    ]:
        df = fetch_df(url)
        if df.empty:
            continue

        for _, row in df.iterrows():
            code = clean_code(
                pick(row, ["公司代號", "股票代號", "Code"])
            )
            if not code:
                continue

            year = to_num(pick(row, ["年度", "Year"]))
            quarter = quarter_num(
                pick(row, ["季別", "Quarter"])
            )

            rows.append(
                {
                    "代號": code,
                    "市場": market,
                    "EPS": to_num(
                        pick(
                            row,
                            [
                                "基本每股盈餘(元)",
                                "基本每股盈餘",
                                "EPS",
                            ],
                        )
                    ),
                    "EPS年度": int(year) if pd.notna(year) else 0,
                    "EPS季別": quarter,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values(
        ["代號", "市場", "EPS年度", "EPS季別"]
    )
    return result.drop_duplicates(
        ["代號", "市場"], keep="last"
    )


# =========================
# PE
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_pe():
    rows = []

    df = fetch_df(URL["listed_pe"])
    if not df.empty:
        for _, row in df.iterrows():
            code = clean_code(
                pick(row, ["Code", "股票代號", "公司代號"])
            )
            if code:
                rows.append(
                    {
                        "代號": code,
                        "市場": "上市",
                        "PE": to_num(
                            pick(row, ["PEratio", "本益比"])
                        ),
                    }
                )

    df = fetch_df(URL["otc_pe"])
    if not df.empty:
        for _, row in df.iterrows():
            code = clean_code(
                pick(
                    row,
                    [
                        "SecuritiesCompanyCode",
                        "股票代號",
                        "代號",
                        "Code",
                    ],
                )
            )
            if code:
                rows.append(
                    {
                        "代號": code,
                        "市場": "上櫃",
                        "PE": to_num(
                            pick(
                                row,
                                [
                                    "PriceEarningRatio",
                                    "本益比",
                                    "PEratio",
                                ],
                            )
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["代號", "市場", "PE"])

    return pd.DataFrame(rows).drop_duplicates(
        ["代號", "市場"]
    )


# =========================
# 個股歷史基本面：月營收／季EPS／季本益比
# =========================
FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


@st.cache_data(ttl=3600, show_spinner=False)
def finmind_dataset(dataset, code, start_date="2020-01-01"):
    try:
        r = requests.get(
            FINMIND_URL,
            params={
                "dataset": dataset,
                "data_id": str(code),
                "start_date": start_date,
            },
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        obj = r.json()
        data = obj.get("data", []) if isinstance(obj, dict) else []
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()


def quarter_label(dt):
    return f"{dt.year} Q{((dt.month - 1) // 3) + 1}"


@st.cache_data(ttl=3600, show_spinner=False)
def load_detail_fundamentals(code):
    """
    個股詳細頁最重要的基本面：
    1. 每月營收 + 月營收年增率
    2. 每季彙總營收 + 季營收年增率
    3. 每季單季 EPS
    4. 每季本益比：取該季最後一個有資料交易日的 PER
    最新季度在最上面。
    """
    revenue = finmind_dataset(
        "TaiwanStockMonthRevenue", code, "2020-01-01"
    )
    eps = finmind_dataset(
        "TaiwanStockFinancialStatements", code, "2020-01-01"
    )
    per = finmind_dataset(
        "TaiwanStockPER", code, "2020-01-01"
    )

    # ---- 月營收 ----
    if revenue.empty:
        rev = pd.DataFrame()
    else:
        rev = revenue.copy()
        rev["date"] = pd.to_datetime(rev.get("date"), errors="coerce")
        rev["revenue"] = pd.to_numeric(rev.get("revenue"), errors="coerce")
        rev["revenue_year"] = pd.to_numeric(rev.get("revenue_year"), errors="coerce")
        rev["revenue_month"] = pd.to_numeric(rev.get("revenue_month"), errors="coerce")
        rev = rev.dropna(subset=["date", "revenue"])
        rev = rev.sort_values("date")
        # 同一月份若有重複資料，保留最後一筆。
        if "revenue_year" in rev.columns and "revenue_month" in rev.columns:
            rev = rev.drop_duplicates(["revenue_year", "revenue_month"], keep="last")
        rev["YoY%"] = (
            rev["revenue"].pct_change(12) * 100
        )
        # 優先使用資料年月重新排序，避免公告日與營收月份不同造成錯位。
        rev["年月"] = rev["revenue_year"].astype("Int64").astype(str) + "-" + rev["revenue_month"].astype("Int64").astype(str).str.zfill(2)
        rev["季度"] = rev["date"].apply(lambda x: quarter_label(x))

    # ---- 季 EPS ----
    if eps.empty:
        eps_q = pd.DataFrame()
    else:
        e = eps.copy()
        e["date"] = pd.to_datetime(e.get("date"), errors="coerce")
        e["value"] = pd.to_numeric(e.get("value"), errors="coerce")
        e = e[e.get("type", pd.Series(index=e.index)).astype(str).eq("EPS")]
        e = e.dropna(subset=["date", "value"]).sort_values("date")
        e["季度"] = e["date"].apply(lambda x: quarter_label(x))
        e = e.drop_duplicates("季度", keep="last")
        eps_q = e[["季度", "date", "value"]].rename(columns={"value": "季EPS"})

    # ---- 季本益比 ----
    if per.empty:
        per_q = pd.DataFrame()
    else:
        pe = per.copy()
        pe["date"] = pd.to_datetime(pe.get("date"), errors="coerce")
        pe["PER"] = pd.to_numeric(pe.get("PER"), errors="coerce")
        pe = pe.dropna(subset=["date"]).sort_values("date")
        pe["季度"] = pe["date"].apply(lambda x: quarter_label(x))
        # 一季取最後一個有 PE 的交易日，避免中間缺值。
        pe_q = pe.dropna(subset=["PER"]).drop_duplicates("季度", keep="last")
        per_q = pe_q[["季度", "date", "PER"]].rename(columns={"PER": "季PE"})

    # ---- 合併：以最近 12 季為主，最新在上 ----
    quarters = set()
    if not rev.empty:
        quarters.update(rev["季度"].dropna().tolist())
    if not eps_q.empty:
        quarters.update(eps_q["季度"].dropna().tolist())
    if not per_q.empty:
        quarters.update(per_q["季度"].dropna().tolist())

    if not quarters:
        return pd.DataFrame(), pd.DataFrame()

    qdf = pd.DataFrame({"季度": list(quarters)})
    qdf["排序"] = qdf["季度"].str.extract(r"(\d{4}) Q(\d)")[0].astype(int) * 10 + qdf["季度"].str.extract(r"(\d{4}) Q(\d)")[1].astype(int)

    # 每季三個月份欄位：營收與月YoY。
    if not rev.empty:
        rev2 = rev.copy()
        rev2["年份"] = pd.to_numeric(rev2["revenue_year"], errors="coerce")
        rev2["月份"] = pd.to_numeric(rev2["revenue_month"], errors="coerce")
        rev2["季排序"] = rev2["年份"] * 10 + ((rev2["月份"] - 1) // 3 + 1)
        for m in [1, 2, 3]:
            # m 是季度內第幾個月，不是月份數字。
            pass
        rows = []
        for qsort, g in rev2.groupby("季排序"):
            if pd.isna(qsort):
                continue
            year = int(qsort // 10)
            qnum = int(qsort % 10)
            row = {"季度": f"{year} Q{qnum}", "季度營收": g["revenue"].sum()}
            prev = rev2[rev2["季排序"] == qsort - 10]
            prev_sum = prev["revenue"].sum() if not prev.empty else np.nan
            row["季度營收YoY"] = ((row["季度營收"] / prev_sum) - 1) * 100 if pd.notna(prev_sum) and prev_sum != 0 else np.nan
            for month_num in range((qnum - 1) * 3 + 1, qnum * 3 + 1):
                mg = g[g["月份"] == month_num]
                if not mg.empty:
                    rr = mg.iloc[-1]
                    row[f"{month_num}月營收"] = rr["revenue"]
                    row[f"{month_num}月YoY"] = rr["YoY%"]
            rows.append(row)
        qrev = pd.DataFrame(rows)
        qdf = qdf.merge(qrev, on="季度", how="left")

    if not eps_q.empty:
        qdf = qdf.merge(eps_q[["季度", "季EPS"]], on="季度", how="left")
    if not per_q.empty:
        qdf = qdf.merge(per_q[["季度", "季PE"]], on="季度", how="left")

    qdf = qdf.sort_values("排序", ascending=False).head(12).drop(columns=["排序"])

    # 再給個股頁一張「月營收明細」：最新到舊，方便核對每個月數字。
    if rev.empty:
        monthly_detail = pd.DataFrame()
    else:
        monthly_detail = rev.sort_values("date", ascending=False).head(36).copy()
        monthly_detail["營收(千元)"] = monthly_detail["revenue"]
        monthly_detail = monthly_detail[["年月", "營收(千元)", "YoY%", "季度"]]

    return qdf, monthly_detail


# =========================
# Yahoo Finance 技術資料
# =========================
def yahoo_symbol(code, market):
    return f"{code}.TW" if market == "上市" else f"{code}.TWO"


def make_weekly(df):
    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in x.columns for c in needed):
        return pd.DataFrame()

    x = x.dropna(subset=["Close"])

    w = x.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna(subset=["Close"])

    for n in [5, 12, 20, 24]:
        w[f"MA{n}"] = w["Close"].rolling(n).mean()

    return w


@st.cache_data(ttl=900, show_spinner=False)
def load_weekly_technical(items):
    if not items:
        return pd.DataFrame()

    symbols = [yahoo_symbol(c, m) for c, m in items]

    try:
        raw = yf.download(
            symbols,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()

    rows = []

    for code, market in items:
        symbol = yahoo_symbol(code, market)

        try:
            if len(symbols) == 1:
                d = raw.copy()
            else:
                if not isinstance(raw.columns, pd.MultiIndex):
                    continue
                if symbol not in raw.columns.get_level_values(0):
                    continue
                d = raw[symbol].copy()

            w = make_weekly(d)

            if len(w) < 25:
                continue

            last = w.iloc[-1]
            prev = w.iloc[-2]

            ma12 = last["MA12"]
            prev_ma12 = prev["MA12"]
            ma24 = last["MA24"]
            close = last["Close"]

            if (
                pd.notna(ma12)
                and pd.notna(prev_ma12)
                and prev_ma12 != 0
            ):
                slope = (
                    (ma12 - prev_ma12)
                    / prev_ma12
                    * 100
                )
            else:
                slope = np.nan

            if close > ma12 > ma24:
                trend = "強勢多頭"
            elif ma12 > ma24:
                trend = "多頭"
            elif close > ma24:
                trend = "整理"
            else:
                trend = "偏弱"

            rows.append(
                {
                    "代號": code,
                    "市場": market,
                    "股價": float(close),
                    "週MA12": float(ma12),
                    "週MA24": float(ma24),
                    "週MA12斜率": float(slope)
                    if pd.notna(slope)
                    else np.nan,
                    "週趨勢": trend,
                    "技術日期": w.index[-1].strftime(
                        "%Y-%m-%d"
                    ),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


# =========================
# 70 分基本模型
# =========================
def score_revenue(x):
    if pd.isna(x):
        return 0
    if x >= 100:
        return 20
    if x >= 80:
        return 18
    if x >= 60:
        return 15
    if x >= 40:
        return 12
    if x >= 25:
        return 9
    if x >= 15:
        return 6
    if x >= 5:
        return 3
    return 0


def score_cumulative(x):
    if pd.isna(x):
        return 0
    if x >= 50:
        return 15
    if x >= 30:
        return 12
    if x >= 20:
        return 10
    if x >= 10:
        return 7
    if x >= 5:
        return 4
    return 0


def score_eps(x):
    if pd.isna(x):
        return 0
    if x > 10:
        return 10
    if x > 5:
        return 8
    if x > 3:
        return 6
    if x > 1:
        return 4
    if x > 0:
        return 2
    return 0


def score_pe(x):
    if pd.isna(x) or x <= 0:
        return 0
    if 10 <= x <= 20:
        return 10
    if x <= 30:
        return 8
    if x <= 40:
        return 5
    if x <= 50:
        return 2
    return 0


def score_slope(x):
    if pd.isna(x):
        return 0
    if x >= 3:
        return 15
    if x >= 2:
        return 13
    if x >= 1:
        return 10
    if x >= 0:
        return 5
    return 0


def add_scores(df):
    x = df.copy()

    x["月營收分"] = x["月營收YoY"].apply(score_revenue)
    x["累計營收分"] = x["累計營收YoY"].apply(score_cumulative)
    x["EPS分"] = x["EPS"].apply(score_eps)
    x["PE分"] = x["PE"].apply(score_pe)
    x["週MA12斜率分"] = x["週MA12斜率"].apply(score_slope)

    x["V2基本總分"] = (
        x["月營收分"]
        + x["累計營收分"]
        + x["EPS分"]
        + x["PE分"]
        + x["週MA12斜率分"]
    )

    # V2.1 再加入真正的 30 分型態模型
    x["型態分"] = 0
    x["V2總分"] = x["V2基本總分"]

    return x


# =========================
# 個股詳細 K 線
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def load_daily(code, market):
    try:
        df = yf.download(
            yahoo_symbol(code, market),
            period="3y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return (
            df.dropna(subset=["Close"])
            if not df.empty
            else df
        )
    except Exception:
        return pd.DataFrame()


def make_monthly(df):
    if df.empty:
        return pd.DataFrame()

    m = df.resample("ME").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna(subset=["Close"])

    for n in [5, 12, 24]:
        m[f"MA{n}"] = m["Close"].rolling(n).mean()

    return m


def add_bollinger(df):
    if df.empty:
        return df

    x = df.copy()
    x["BB_MID"] = x["Close"].rolling(20).mean()
    std = x["Close"].rolling(20).std()
    x["BB_UPPER"] = x["BB_MID"] + 2 * std
    x["BB_LOWER"] = x["BB_MID"] - 2 * std
    return x


def candle_chart(df, title, show_bb=False):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K",
            increasing_line_color="#d64545",
            increasing_fillcolor="#d64545",
            decreasing_line_color="#159957",
            decreasing_fillcolor="#159957",
        )
    )

    for ma in ["MA5", "MA12", "MA24"]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma],
                    mode="lines",
                    name=ma,
                )
            )

    if show_bb:
        for col, name in [
            ("BB_UPPER", "布林上軌"),
            ("BB_LOWER", "布林下軌"),
        ]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode="lines",
                        name=name,
                    )
                )

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=18)),
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Arial, Microsoft JhengHei, sans-serif", size=12, color="#263238"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="#edf0f2", zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor"),
        yaxis=dict(showgrid=True, gridcolor="#edf0f2", zeroline=False, fixedrange=False),
        margin=dict(l=45, r=20, t=65, b=35),
    )

    return fig


# =========================
# Google AdSense 廣告區塊
# =========================
def render_adsense(slot, height=125):
    # Streamlit 的 st.markdown 不會執行 script，因此使用 HTML component。
    # 這裡只是保留 AdSense 版位；是否填充廣告由 Google AdSense 決定。
    if slot == "4790829615":
        html = f"""
        <div style="width:100%; min-height:{height}px; display:flex; align-items:center; justify-content:center; overflow:hidden;">
          <ins class="adsbygoogle"
               style="display:block; text-align:center; width:100%;"
               data-ad-layout="in-article"
               data-ad-format="fluid"
               data-ad-client="ca-pub-7239713661981419"
               data-ad-slot="4790829615"></ins>
        </div>
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7239713661981419" crossorigin="anonymous"></script>
        <script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>
        """
    else:
        html = f"""
        <div style="width:100%; min-height:{height}px; display:flex; align-items:center; justify-content:center; overflow:hidden;">
          <ins class="adsbygoogle"
               style="display:block"
               data-ad-client="ca-pub-7239713661981419"
               data-ad-slot="{slot}"
               data-ad-format="auto"
               data-full-width-responsive="true"></ins>
        </div>
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7239713661981419" crossorigin="anonymous"></script>
        <script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>
        """
    # Streamlit 1.63+ 的 st.html 不使用 iframe，較適合正式網站的 HTML/JS 版位。
    st.html(html, unsafe_allow_javascript=True)


# =========================
# Streamlit UI
# =========================
st.title("🚀 飆股選股器 V2.0")
render_adsense("2164666279", height=135)
st.caption(
    "上市＋上櫃｜月營收＋累計營收＋EPS＋PE＋週MA12斜率｜目前 70 分基礎模型"
)

with st.sidebar:
    st.header("⚙️ 選股設定")

    scan_size = st.selectbox(
        "掃描數量",
        ["100 檔測試", "300 檔測試", "全市場"],
        index=0,
    )

    sort_field = st.selectbox(
        "排序方式",
        [
            "V2總分",
            "月營收YoY",
            "累計營收YoY",
            "週MA12斜率",
            "EPS",
            "PE",
            "代號",
        ],
    )

    high_to_low = st.checkbox("高 → 低", value=True)

    run = st.button(
        "🔄 開始掃描",
        type="primary",
        use_container_width=True,
    )

if run or "scan_result" not in st.session_state:

    with st.spinner("① 抓取上市＋上櫃官方基本面資料…"):
        universe = load_universe()
        rev = load_revenue()
        eps = load_eps()
        pe = load_pe()

    base = universe.merge(
        rev,
        on=["代號", "市場"],
        how="left",
    )
    base = base.merge(
        eps,
        on=["代號", "市場"],
        how="left",
    )
    base = base.merge(
        pe,
        on=["代號", "市場"],
        how="left",
    )

    # 先用營收/EPS 做技術資料抓取前的預篩，
    # 避免一次對全市場大量呼叫 Yahoo。
    base["初步分"] = (
        base["月營收YoY"].fillna(-999) * 0.4
        + base["累計營收YoY"].fillna(-999) * 0.2
        + base["EPS"].fillna(-999)
    )

    base = base.sort_values(
        "初步分",
        ascending=False,
    )

    if scan_size == "100 檔測試":
        selected = base.head(100)
    elif scan_size == "300 檔測試":
        selected = base.head(300)
    else:
        selected = base

    items = tuple(
        zip(
            selected["代號"].tolist(),
            selected["市場"].tolist(),
        )
    )

    with st.spinner(
        f"② 正在抓取 {len(items)} 檔週K並計算週MA12斜率…"
    ):
        technical = load_weekly_technical(items)

    result = add_scores(
        selected.merge(
            technical,
            on=["代號", "市場"],
            how="left",
        )
    )

    st.session_state["scan_result"] = result
    st.session_state["scan_time"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

result = st.session_state["scan_result"].copy()

ascending = not high_to_low
result = result.sort_values(
    sort_field,
    ascending=ascending,
    na_position="last",
).reset_index(drop=True)

result.insert(
    0,
    "排名",
    np.arange(1, len(result) + 1),
)

# =========================
# 個股模式：由 URL 唯一控制
# =========================
code_options = result["代號"].astype(str).tolist() if len(result) else []
query_stock = str(st.query_params.get("stock", "")).strip()
detail_mode = query_stock in code_options

if detail_mode:
    st.subheader("🔎 個股詳細分析")
    if st.button("← 回到飆股候選排名", key="back_to_rank"):
        st.query_params.clear()
        st.rerun()

# =========================
# 摘要
# =========================
if not detail_mode:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("股票數", f"{len(result):,}")
    c2.metric("最高分", f"{result['V2總分'].max():.0f}" if len(result) else "-")
    c3.metric("月營收 ≥100%", f"{int((result['月營收YoY'] >= 100).sum())}")
    c4.metric("週MA12上彎", f"{int((result['週MA12斜率'] >= 0).sum())}")
    st.caption(f"最後掃描：{st.session_state.get('scan_time', '-')}")

# =========================
# 排名
# =========================
if not detail_mode:
    st.subheader("📊 飆股候選排名")
    show = result[["排名","代號","名稱","股價","市場","營收年月","月營收YoY","累計營收YoY","EPS","PE","週MA12斜率","週趨勢","V2基本總分","型態分","V2總分"]].copy()
    for col in ["月營收YoY","累計營收YoY","EPS","PE","週MA12斜率"]:
        show[col] = show[col].round(2)

    # 股名本身就是連結：直接點股名進入個股詳細頁。
    def make_stock_link(row):
        code = str(row["代號"])
        name = str(row["名稱"])
        return f'<a href="?stock={code}" target="_self" style="text-decoration:none; font-weight:600; color:#1769aa;">{name}</a>'

    html_cols = ["排名","代號","名稱","股價","市場","營收年月","月營收YoY","累計營收YoY","EPS","PE","週MA12斜率","週趨勢","V2基本總分","型態分","V2總分"]
    html = [
        '<div style="overflow-x:auto; border:1px solid #e6e8eb; border-radius:12px; background:#fff;">',
        '<table style="width:100%; border-collapse:collapse; font-size:14px; white-space:nowrap;">',
        '<thead><tr style="background:#f7f8fa;">'
    ]
    for col in html_cols:
        html.append(f'<th style="padding:10px 9px; border-bottom:1px solid #e6e8eb; text-align:left; font-weight:650;">{col}</th>')
    html.append('</tr></thead><tbody>')
    for _, r in show.iterrows():
        vals = []
        for col in html_cols:
            if col == "名稱":
                val = make_stock_link(r)
            elif col in ["月營收YoY", "累計營收YoY", "週MA12斜率"] and pd.notna(r[col]):
                val = f'{r[col]:.2f}%'
            elif col in ["股價", "EPS", "PE"] and pd.notna(r[col]):
                val = f'{r[col]:.2f}'
            elif col in ["V2基本總分", "型態分", "V2總分"] and pd.notna(r[col]):
                val = f'{r[col]:.0f}'
            else:
                val = "-" if pd.isna(r[col]) else str(r[col])
            vals.append(f'<td style="padding:9px; border-bottom:1px solid #f0f1f3;">{val}</td>')
        html.append('<tr>' + ''.join(vals) + '</tr>')
    html.append('</tbody></table></div>')
    st.markdown(''.join(html), unsafe_allow_html=True)
    st.info("💡 直接點「股名」即可進入個股詳細資料。")


# =========================
# 個股詳細頁
# =========================
render_adsense("4790829615", height=125)

if detail_mode:
    def render_detail(result, selected_code):
        code_options = result["代號"].astype(str).tolist()
        if not code_options:
            return

        selected_code = str(selected_code)
        if selected_code not in code_options:
            selected_code = code_options[0]

        current_index = code_options.index(selected_code)
        chosen_from_box = st.selectbox(
            "選擇股票",
            code_options,
            index=current_index,
            format_func=lambda x: (
                f"{x} "
                f"{result.loc[result['代號'].eq(x), '名稱'].iloc[0]}"
            ),
            key="detail_stock_box",
        )

        if chosen_from_box != selected_code:
            st.query_params["stock"] = chosen_from_box
            st.rerun()

        row = result[
            result["代號"].eq(selected_code)
        ].iloc[0]

        code_value = row["代號"]
        name_value = row["名稱"]
        market_value = row["市場"]

        st.markdown(
            f"### {code_value} {name_value}〔{market_value}〕"
        )

        a, b, c, d, e, f = st.columns(6)

        a.metric(
            "今收股價",
            f"{row['股價']:.2f}"
            if pd.notna(row["股價"])
            else "-",
        )
        b.metric(
            "V2總分",
            f"{row['V2總分']:.0f}/100",
        )
        c.metric(
            "月營收YoY",
            f"{row['月營收YoY']:.2f}%"
            if pd.notna(row["月營收YoY"])
            else "-",
        )
        d.metric(
            "累計營收YoY",
            f"{row['累計營收YoY']:.2f}%"
            if pd.notna(row["累計營收YoY"])
            else "-",
        )
        e.metric(
            "EPS",
            f"{row['EPS']:.2f}"
            if pd.notna(row["EPS"])
            else "-",
        )
        f.metric(
            "PE",
            f"{row['PE']:.2f}"
            if pd.notna(row["PE"])
            else "-",
        )

        # 使用者指定：最重要的基本面放最上面，而且最新 → 舊。
        st.markdown("#### ① 季度基本面（最新 → 舊）")
        st.caption("每季列出該季 3 個月營收、各月年增率、季度營收年增率、單季 EPS 與該季末本益比。")

        with st.spinner("正在抓取歷史月營收、季度 EPS 與歷史本益比…"):
            quarterly_detail, monthly_detail = load_detail_fundamentals(code_value)

        if quarterly_detail.empty:
            st.warning("目前抓不到這檔股票的歷史基本面資料，請稍後再試。")
        else:
            qshow = quarterly_detail.copy()
            money_cols = [c for c in qshow.columns if "營收" in c]
            for col in money_cols:
                if col in qshow.columns:
                    qshow[col] = pd.to_numeric(qshow[col], errors="coerce")
            for col in ["季度營收", "1月營收", "2月營收", "3月營收", "4月營收", "5月營收", "6月營收", "7月營收", "8月營收", "9月營收", "10月營收", "11月營收", "12月營收"]:
                if col in qshow.columns:
                    qshow[col] = qshow[col].round(0)
            for col in [c for c in qshow.columns if "YoY" in c]:
                qshow[col] = qshow[col].round(2)
            for col in ["季EPS", "季PE"]:
                if col in qshow.columns:
                    qshow[col] = qshow[col].round(2)

            # 把季度內三個月整理成固定欄位，避免畫面出現不相關月份。
            fixed_cols = [
                "季度",
                "1月營收", "1月YoY", "2月營收", "2月YoY", "3月營收", "3月YoY",
                "季度營收", "季度營收YoY", "季EPS", "季PE",
            ]
            # 實際月份依季度不同，另外用較直覺的三欄顯示。
            display_rows = []
            for _, qr in quarterly_detail.iterrows():
                q = str(qr["季度"])
                # 季度格式固定為 YYYY Qn；避免 NaN 導致 int(NaN) 錯誤
                mm = re.search(r"Q([1-4])$", q)
                if not mm:
                    continue
                m = int(mm.group(1))
                base_month = (m - 1) * 3 + 1
                rr = {"季度": q}
                for idx, month_no in enumerate(range(base_month, base_month + 3), start=1):
                    rr[f"{month_no}月營收"] = qr.get(f"{month_no}月營收", np.nan)
                    rr[f"{month_no}月YoY"] = qr.get(f"{month_no}月YoY", np.nan)
                rr["季度營收"] = qr.get("季度營收", np.nan)
                rr["季度營收YoY"] = qr.get("季度營收YoY", np.nan)
                rr["季EPS"] = qr.get("季EPS", np.nan)
                rr["季PE"] = qr.get("季PE", np.nan)
                display_rows.append(rr)
            qshow = pd.DataFrame(display_rows)

            # 欄位依最新資料排序，每季由新到舊。
            for col in qshow.columns:
                if "營收" in col and "YoY" not in col:
                    qshow[col] = pd.to_numeric(qshow[col], errors="coerce").round(0)
                elif "YoY" in col:
                    qshow[col] = pd.to_numeric(qshow[col], errors="coerce").round(2)
                elif col in ["季EPS", "季PE"]:
                    qshow[col] = pd.to_numeric(qshow[col], errors="coerce").round(2)

            st.dataframe(qshow, use_container_width=True, hide_index=True)

            if not monthly_detail.empty:
                st.markdown("**最近36個月營收明細（最新 → 舊）**")
                mshow = monthly_detail.copy()
                mshow["營收(千元)"] = pd.to_numeric(mshow["營收(千元)"], errors="coerce").round(0)
                mshow["YoY%"] = pd.to_numeric(mshow["YoY%"], errors="coerce").round(2)
                st.dataframe(mshow, use_container_width=True, hide_index=True)

        daily = load_daily(
            code_value,
            market_value,
        )

        if daily.empty:
            st.warning(
                "這檔股票目前抓不到 Yahoo Finance 歷史 K 線。"
            )
        else:

            weekly = add_bollinger(
                make_weekly(daily)
            )
            monthly = make_monthly(daily)

            # 使用者指定：技術圖在下，週在上、月在下
            st.markdown("#### ② 週K技術圖")

            if not weekly.empty:

                st.plotly_chart(
                    candle_chart(
                        weekly.tail(104),
                        f"{code_value} 週K",
                        show_bb=True,
                    ),
                    use_container_width=True,
                )

                recent_w = (
                    weekly.tail(20)
                    .sort_index(ascending=False)
                    .copy()
                )

                recent_w.index = recent_w.index.strftime(
                    "%Y-%m-%d"
                )

                recent_w = recent_w[
                    [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "MA5",
                        "MA12",
                        "MA24",
                    ]
                ]

                st.markdown(
                    "**最近20週（新 → 舊）**"
                )

                st.dataframe(
                    recent_w.round(2),
                    use_container_width=True,
                )

            st.markdown("#### ③ 月K技術圖")

            if not monthly.empty:

                st.plotly_chart(
                    candle_chart(
                        monthly.tail(60),
                        f"{code_value} 月K",
                    ),
                    use_container_width=True,
                )

                recent_m = (
                    monthly.tail(24)
                    .sort_index(ascending=False)
                    .copy()
                )

                recent_m.index = recent_m.index.strftime(
                    "%Y-%m"
                )

                recent_m = recent_m[
                    [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "MA5",
                        "MA12",
                        "MA24",
                    ]
                ]

                st.markdown(
                    "**最近24個月（新 → 舊）**"
                )

                st.dataframe(
                    recent_m.round(2),
                    use_container_width=True,
                )

    render_detail(result, query_stock)

render_adsense("4790829615", height=125)

st.divider()

st.caption(
    "資料來源：TWSE / TPEX 官方 OpenAPI、Yahoo Finance。"
    " V2.0 目前為測試版；V2.1 再加入 30 分突破回測型態。"
)
