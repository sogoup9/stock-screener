import re
import time
import warnings
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="飆股獵奇-傑森", page_icon="🦅", layout="wide")

# =========================
# 網站外觀＋上方導覽列
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: #FFF7ED;
    }
    [data-testid="stHeader"] {
        background: rgba(255,247,237,0.96);
    }
    .top-nav {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        margin: 0 0 18px 0;
        background: #FFE8CC;
        border: 1px solid #F5D3AD;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(160, 100, 40, 0.08);
        overflow-x: auto;
        white-space: nowrap;
    }
    .brand-logo {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 13px;
        margin-right: 8px;
        color: #7A3E00;
        font-weight: 800;
        font-size: 17px;
        text-decoration: none;
    }
    .brand-icon {
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #FFFDF8;
        border: 1px solid #E8B77F;
        font-size: 18px;
    }
    .nav-item {
        display: inline-flex;
        align-items: center;
        padding: 8px 13px;
        border-radius: 10px;
        color: #633A1C;
        text-decoration: none;
        font-weight: 650;
    }
    .nav-item:hover {
        background: #FFF7ED;
        color: #9A4D00;
    }
    .nav-sep {
        color: #B9855A;
        font-weight: 700;
    }
    /* 左側設定欄縮窄，讓主畫面有更多空間 */
    section[data-testid="stSidebar"] {
        width: 225px !important;
        min-width: 225px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 225px !important;
    }
    .num-positive { color: #d62728 !important; font-weight: 700; }
    .num-negative { color: #159447 !important; font-weight: 700; }

    .page-card {
        background: #FFFFFF;
        border: 1px solid #F1D9C1;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(120,80,40,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page = str(st.query_params.get("page", "home")).strip().lower()
if page not in {"home", "events", "futures"}:
    page = "home"

# 品牌獨立放在左上角；導覽列只保留功能頁面。
st.markdown(
    """
    <div style="margin:2px 0 8px 2px;">
      <a class="brand-logo" href="?page=home" target="_self" style="padding-left:0;">
        <span class="brand-icon">🦅</span>
        <span>飆股獵奇-傑森</span>
      </a>
    </div>
    <div class="top-nav">
      <a class="nav-item" href="?page=home" target="_self">首頁</a>
      <span class="nav-sep">›</span>
      <a class="nav-item" href="?page=events" target="_self">近期事件（處置／除權息）</a>
      <span class="nav-sep">›</span>
      <a class="nav-item" href="?page=futures" target="_self">海期日記</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 官方資料來源
# =========================
TWSE = "https://openapi.twse.com.tw/v1"
TPEX = "https://www.tpex.org.tw/openapi/v1"
MOPS = "https://mopsov.twse.com.tw"
TWSE_STOCK_DAY = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
TPEX_TRADING_STOCK = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"

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


# TWSE／TPEx 公開資料的「產業別」可能是數字代碼。
# 只轉換已確認的交易所產業分類；未知代碼明確標示，避免誤分類。
INDUSTRY_NAMES = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業",
    "04": "紡織纖維", "05": "電機機械", "06": "電器電纜",
    "08": "玻璃陶瓷", "09": "造紙工業", "10": "鋼鐵工業",
    "11": "橡膠工業", "12": "汽車工業", "14": "建材營造",
    "15": "航運業", "16": "觀光餐旅", "17": "金融保險",
    "18": "貿易百貨", "20": "其他", "21": "化學工業",
    "22": "生技醫療業", "23": "油電燃氣業", "24": "半導體業",
    "25": "電腦及週邊設備業", "26": "光電業", "27": "通信網路業",
    "28": "電子零組件業", "29": "電子通路業", "30": "資訊服務業",
    "31": "其他電子業", "32": "文化創意業", "33": "農業科技業",
    "34": "電子商務", "35": "綠能環保", "36": "數位雲端",
    "37": "運動休閒", "38": "居家生活",
}

def industry_name(value):
    if value is None or pd.isna(value):
        return "未分類"
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "-"}:
        return "未分類"
    match = re.fullmatch(r"(\d{1,2})(?:\.0+)?", raw)
    if match:
        code = match.group(1).zfill(2)
        return INDUSTRY_NAMES.get(code, f"未分類（{code}）")
    return raw


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
            industry = pick(row, ["產業別", "產業類別", "Industry"])

            if code:
                rows.append(
                    {
                        "代號": code,
                        "名稱": str(name or "").strip(),
                        "市場": market,
                        "產業": industry_name(industry),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["代號", "名稱", "市場", "產業"])

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
                    "營收MoM": to_num(
                        pick(
                            row,
                            [
                                "營業收入-上月比較增減(%)",
                                "營業收入-上月比較增減(％)",
                                "上月比較增減(%)",
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
                "營收MoM",
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
MOPS_OPEN = "https://mopsfin.twse.com.tw/opendata"


def fetch_mops_csv(url):
    try:
        r=requests.get(url,timeout=30,headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return pd.read_csv(StringIO(r.content.decode("utf-8-sig",errors="ignore")))
    except Exception:
        return pd.DataFrame()


def quarter_label(dt):
    return f"{dt.year} Q{((dt.month - 1) // 3) + 1}"


@st.cache_data(ttl=3600, show_spinner=False)
def mops_monthly_report(market, year, month):
    """MOPS 歷史月營收靜態 CSV；sii=上市、otc=上櫃。"""
    roc_year=year-1911
    market_code="sii" if market=="上市" else "otc"
    url=f"{MOPS}/nas/t21/{market_code}/t21sc03_{roc_year}_{month}.csv"
    try:
        r=requests.get(url,timeout=30,headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return pd.read_csv(StringIO(r.content.decode("utf-8-sig",errors="ignore")))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def mops_stock_monthly_history(code, market, months=36):
    end=pd.Timestamp(datetime.now().date()).replace(day=1)
    frames=[]
    for i in range(months-1,-1,-1):
        d=end-pd.DateOffset(months=i)
        df=mops_monthly_report(market,int(d.year),int(d.month))
        if df.empty:
            continue
        code_col=next((c for c in ["公司代號","股票代號","證券代號"] if c in df.columns),None)
        if not code_col:
            continue
        codes=df[code_col].astype(str).str.extract(r"(\d{4})",expand=False)
        hit=df[codes.eq(str(code))].copy()
        if not hit.empty:
            hit["西元年月"]=f"{int(d.year):04d}-{int(d.month):02d}"
            frames.append(hit)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def mops_eps_history(code, market, quarters=12):
    """MOPS 個股合併損益表的基本每股盈餘。"""
    rows=[]
    endpoint=f"{MOPS}/mops/web/ajax_t164sb04"
    today=pd.Timestamp(datetime.now().date())
    for offset in range(quarters):
        qdate=today-pd.DateOffset(months=3*offset)
        year=int(qdate.year)-1911
        season=((int(qdate.month)-1)//3)+1
        try:
            payload={"encodeURIComponent":"1","step":"1","firstin":"1","off":"1","co_id":str(code),"year":str(year),"season":f"{season:02d}","TYPEK":"all","isnew":"false","queryName":"co_id"}
            r=requests.post(endpoint,data=payload,timeout=30,headers={"User-Agent":"Mozilla/5.0"})
            tables=pd.read_html(StringIO(r.text))
            eps_val=np.nan
            for t in tables:
                for _,rr in t.iterrows():
                    cells=[str(v).strip() for v in rr.tolist()]
                    if not any("基本每股盈餘" in c for c in cells):
                        continue
                    nums=[]
                    for v in cells:
                        n=to_num(v)
                        if pd.notna(n): nums.append(n)
                    if nums:
                        eps_val=nums[0]
                        break
                if pd.notna(eps_val): break
            rows.append({"季度":f"{qdate.year} Q{season}","季EPS":eps_val})
        except Exception:
            continue
        time.sleep(0.12)
    return pd.DataFrame(rows).drop_duplicates("季度") if rows else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_detail_fundamentals(code, market="上市"):
    rev=mops_stock_monthly_history(code,market,36)
    eps_q=mops_eps_history(code,market,12)
    if rev.empty:
        return pd.DataFrame(),pd.DataFrame()
    revenue_col=next((c for c in ["營業收入-當月營收","當月營收","當月營業收入"] if c in rev.columns),None)
    yoy_col=next((c for c in ["營業收入-去年同月增減(%)","營業收入-去年同月增減(％)","去年同月增減(%)"] if c in rev.columns),None)
    if not revenue_col:
        return pd.DataFrame(),pd.DataFrame()
    rev["revenue"]=pd.to_numeric(rev[revenue_col],errors="coerce")
    rev["YoY%"]=pd.to_numeric(rev[yoy_col],errors="coerce") if yoy_col else np.nan
    rev["date"]=pd.to_datetime(rev["西元年月"]+"-01",errors="coerce")
    rev=rev.dropna(subset=["date","revenue"]).sort_values("date")
    rev["季度"]=rev["date"].apply(quarter_label)
    rows=[]
    for q,g in rev.groupby("季度"):
        qnum=int(q[-1]); year=int(q[:4])
        row={"季度":q,"季度營收":g["revenue"].sum()}
        prev=rev[rev["季度"].eq(f"{year-1} Q{qnum}")]["revenue"]
        prev_sum=prev.sum() if not prev.empty else np.nan
        row["季度營收YoY"]=((row["季度營收"]/prev_sum)-1)*100 if pd.notna(prev_sum) and prev_sum!=0 else np.nan
        for month_no in range((qnum-1)*3+1,qnum*3+1):
            mg=g[g["date"].dt.month.eq(month_no)]
            if not mg.empty:
                rr=mg.iloc[-1]
                row[f"{month_no}月營收"]=rr["revenue"]
                row[f"{month_no}月YoY"]=rr["YoY%"]
        rows.append(row)
    qdf=pd.DataFrame(rows)
    if not eps_q.empty: qdf=qdf.merge(eps_q,on="季度",how="left")
    qdf=qdf.sort_values("季度",ascending=False).head(12)
    monthly_detail=rev.sort_values("date",ascending=False).head(36).copy()
    monthly_detail["年月"]=monthly_detail["date"].dt.strftime("%Y-%m")
    monthly_detail["營收(千元)"]=monthly_detail["revenue"]
    monthly_detail=monthly_detail[["年月","營收(千元)","YoY%","季度"]]
    return qdf,monthly_detail


# =========================
# 官方歷史行情：TWSE / TPEx
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_official_month(code, market, month_start):
    """抓單一股票單一月份官方日K；完全移除 Yahoo Finance。"""
    try:
        if market == "上市":
            r = requests.get(
                TWSE_STOCK_DAY,
                params={
                    "response": "json",
                    "date": month_start.strftime("%Y%m%d"),
                    "stockNo": str(code),
                },
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            payload = r.json()
            if payload.get("stat") != "OK":
                return pd.DataFrame()
            rows = []
            for item in payload.get("data", []):
                if len(item) < 9:
                    continue
                rows.append({
                    "DateText": item[0],
                    "Volume": to_num(item[1]),
                    "Turnover": to_num(item[2]),
                    "Open": to_num(item[3]),
                    "High": to_num(item[4]),
                    "Low": to_num(item[5]),
                    "Close": to_num(item[6]),
                })
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            def roc_date(v):
                m = re.search(r"(\d{3})/(\d{1,2})/(\d{1,2})", str(v))
                if not m:
                    return pd.NaT
                return pd.Timestamp(int(m.group(1))+1911, int(m.group(2)), int(m.group(3)))
            df.index = pd.to_datetime(df["DateText"].map(roc_date), errors="coerce")
            return df.drop(columns=["DateText"]).loc[lambda x: x.index.notna()]

        # TPEx 官方個股月資料：POST 回傳 JSON。成交量原始單位為張。
        r = requests.post(
            TPEX_TRADING_STOCK,
            data={
                "response": "json",
                "code": str(code),
                "date": month_start.strftime("%Y/%m/%d"),
            },
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        payload = r.json()
        if str(payload.get("stat", "")).lower() != "ok":
            return pd.DataFrame()
        tables = payload.get("tables") or []
        if not tables:
            return pd.DataFrame()
        rows = []
        for item in tables[0].get("data", []):
            if len(item) < 7:
                continue
            rows.append({
                "DateText": item[0],
                "Volume": to_num(item[1]) * 1000,
                "Turnover": to_num(item[2]) * 1000,
                "Open": to_num(item[3]),
                "High": to_num(item[4]),
                "Low": to_num(item[5]),
                "Close": to_num(item[6]),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        def roc_date(v):
            m = re.search(r"(\d{3})/(\d{1,2})/(\d{1,2})", str(v))
            if not m:
                return pd.NaT
            return pd.Timestamp(int(m.group(1))+1911, int(m.group(2)), int(m.group(3)))
        df.index = pd.to_datetime(df["DateText"].map(roc_date), errors="coerce")
        return df.drop(columns=["DateText"]).loc[lambda x: x.index.notna()]
    except Exception:
        return pd.DataFrame()


def month_starts_back(months=8):
    today = pd.Timestamp(datetime.now().date()).replace(day=1)
    return [today - pd.DateOffset(months=i) for i in range(months-1, -1, -1)]


@st.cache_data(ttl=1800, show_spinner=False)
def load_daily(code, market, months=26):
    frames = []
    for m in month_starts_back(months):
        df = load_official_month(code, market, pd.Timestamp(m).to_pydatetime())
        if not df.empty:
            frames.append(df)
        time.sleep(0.12)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames).sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x.dropna(subset=["Close"])


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


@st.cache_data(ttl=1800, show_spinner=False)
def load_weekly_technical(items):
    if not items:
        return pd.DataFrame()
    rows = []
    for code, market in items:
        try:
            # 7 個月約 30 週，足夠計算週 MA24 與 MA12 斜率。
            d = load_daily(code, market, months=8)
            w = make_weekly(d)
            if len(w) < 25:
                continue
            last = w.iloc[-1]
            prev = w.iloc[-2]
            ma12 = last["MA12"]
            prev_ma12 = prev["MA12"]
            ma24 = last["MA24"]
            close = last["Close"]
            slope = ((ma12-prev_ma12)/prev_ma12*100) if pd.notna(ma12) and pd.notna(prev_ma12) and prev_ma12 != 0 else np.nan
            if close > ma12 > ma24:
                trend = "強勢多頭"
            elif ma12 > ma24:
                trend = "多頭"
            elif close > ma24:
                trend = "整理"
            else:
                trend = "偏弱"
            rows.append({
                "代號": code, "市場": market, "股價": float(close),
                "週MA12": float(ma12) if pd.notna(ma12) else np.nan,
                "週MA24": float(ma24) if pd.notna(ma24) else np.nan,
                "週MA12斜率": float(slope) if pd.notna(slope) else np.nan,
                "週趨勢": trend,
                "技術日期": w.index[-1].strftime("%Y-%m-%d"),
            })
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
# 近期事件：處置／除權息
# =========================
@st.cache_data(ttl=900, show_spinner=False)
def load_twse_disposal():
    """TWSE 官方 OpenAPI：集中市場公布處置股票。"""
    url = f"{TWSE}/announcement/punish"
    data = fetch_json(url)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    rename = {
        "Number": "序號",
        "Date": "公布日期",
        "Code": "股票代號",
        "Name": "股票名稱",
        "NumberOfAnnouncement": "累計次數",
        "ReasonsOfDisposition": "處置條件",
        "DispositionPeriod": "處置起迄",
        "DispositionMeasures": "處置措施",
    }
    df = df.rename(columns=rename)
    keep = [c for c in rename.values() if c in df.columns]
    return df[keep].copy()


def _read_html_tables(url):
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return pd.read_html(StringIO(r.text))
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def load_twse_exrights():
    """TWSE 官方除權除息預告表。"""
    url = "https://www.twse.com.tw/exchangeReport/TWT48U?response=html"
    tables = _read_html_tables(url)
    if not tables:
        return pd.DataFrame()
    # 通常第一個表就是預告表；若版型變動，挑出含股票代號的表。
    table = tables[0]
    for t in tables:
        text = " ".join(map(str, t.columns)) + " " + " ".join(map(str, t.head(2).astype(str).values.flatten()))
        if "股票代號" in text and "除權除息日期" in text:
            table = t
            break
    df = table.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["".join(str(x) for x in col if str(x) != "nan").strip() for col in df.columns]
    df.columns = [str(c).replace("<br>", "").strip() for c in df.columns]
    # 常見欄位名稱整理
    mapping = {}
    for c in df.columns:
        cc = str(c)
        if "除權除息日期" in cc:
            mapping[c] = "除權息日期"
        elif cc in ("股票代號", "代號") or "股票代號" in cc:
            mapping[c] = "股票代號"
        elif cc in ("名稱", "股票名稱"):
            mapping[c] = "股票名稱"
        elif "除權息" in cc and "日期" not in cc:
            mapping[c] = "權息別"
        elif "現金股利" in cc:
            mapping[c] = "現金股利"
        elif "無償配股率" in cc:
            mapping[c] = "無償配股率"
    df = df.rename(columns=mapping)
    wanted = ["除權息日期", "股票代號", "股票名稱", "權息別", "無償配股率", "現金股利"]
    wanted = [c for c in wanted if c in df.columns]
    if not wanted:
        return pd.DataFrame()
    return df[wanted].copy()


@st.cache_data(ttl=900, show_spinner=False)
def load_tpex_disposal_html():
    """TPEx 官方處置股票頁；若官方頁面拒絕自動抓取，保留空表並提供官方查詢連結。"""
    url = "https://www.tpex.org.tw/zh-tw/announcement/mainboard/disposal.html"
    tables = _read_html_tables(url)
    if not tables:
        return pd.DataFrame()
    for t in tables:
        text = " ".join(map(str, t.columns)) + " " + " ".join(map(str, t.head(2).astype(str).values.flatten()))
        if "證券代號" in text and "處置" in text:
            return t.copy()
    return tables[0].copy() if tables else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_tpex_exrights_html():
    """TPEx 官方除權除息公告頁。"""
    url = "https://www.tpex.org.tw/zh-tw/announce/market/ex/announce.html"
    tables = _read_html_tables(url)
    if not tables:
        return pd.DataFrame()
    for t in tables:
        text = " ".join(map(str, t.columns)) + " " + " ".join(map(str, t.head(2).astype(str).values.flatten()))
        if "除權" in text or "除息" in text:
            return t.copy()
    return tables[0].copy() if tables else pd.DataFrame()


def parse_roc_date(value):
    """將民國年月日字串轉成 pandas Timestamp；支援 115/09/30、11509030、2026/09/30。"""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "-"}:
        return pd.NaT
    # 先處理含民國日期的字串；只抓第一組完整日期。
    m = re.search(r"(\d{3})[\-/年](\d{1,2})[\-/月](\d{1,2})", text)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))
        except Exception:
            return pd.NaT
    m = re.search(r"(\d{3})(\d{2})(\d{2})", text)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))
        except Exception:
            return pd.NaT
    # 西元日期
    m = re.search(r"(20\d{2})[\-/](\d{1,2})[\-/](\d{1,2})", text)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return pd.NaT
    return pd.to_datetime(text, errors="coerce")


def parse_disposition_end(value):
    """從處置起迄文字取最後一個日期，作為處置結束日。"""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    matches = re.findall(r"(?:\d{3}[\-/年]\d{1,2}[\-/月]\d{1,2}|\d{3}\d{4}|20\d{2}[\-/]\d{1,2}[\-/]\d{1,2})", text)
    if not matches:
        return pd.NaT
    return parse_roc_date(matches[-1])


def _normalize_event_code(value):
    m = re.search(r"\b(\d{4})\b", str(value or ""))
    return m.group(1) if m else clean_code(value)


def make_event_stock_link(code, name, market="listed"):
    code = _normalize_event_code(code)
    name = str(name or code)
    return f'<a href="?page=events&event_stock={code}&event_market={market}" target="_self" style="text-decoration:none;font-weight:700;color:#9A4D00;">{name}</a>'


def render_disposition_detail(code, market, event_name="", end_date=pd.NaT):
    """處置股詳細頁：日K＋布林通道＋處置結束日標記。"""
    code = _normalize_event_code(code)
    market = "otc" if str(market).lower() in {"otc", "上櫃", "tpex"} else "listed"
    title_market = "上櫃" if market == "otc" else "上市"

    if st.button("← 回到近期事件", key="back_event_detail"):
        st.query_params.clear()
        st.query_params["page"] = "events"
        st.rerun()

    st.markdown(f"### 🚨 {code} {event_name}｜處置股日線分析")
    if pd.notna(end_date):
        st.info(f"處置結束日：**{end_date.strftime('%Y-%m-%d')}**（{title_market}）")
    else:
        st.warning("目前無法從官方處置資料解析處置結束日，因此圖表不會標示結束日期。")

    daily = load_daily(code, market)
    if daily.empty:
        st.error("目前抓不到這檔股票的官方歷史日K資料，請稍後再試。")
        return

    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index, errors="coerce")
    daily = daily[daily.index.notna()].sort_index()
    daily = add_bollinger(daily)

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=daily.index,
        open=daily["Open"], high=daily["High"], low=daily["Low"], close=daily["Close"],
        name="日K",
        increasing_line_color="#d64545", increasing_fillcolor="#d64545",
        decreasing_line_color="#159957", decreasing_fillcolor="#159957",
    ))
    for col, name in [("BB_MID", "布林中軌"), ("BB_UPPER", "布林上軌"), ("BB_LOWER", "布林下軌")]:
        if col in daily.columns:
            fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name=name))

    if pd.notna(end_date):
        end_ts = pd.Timestamp(end_date).normalize()
        fig.add_vline(
            x=end_ts,
            line_width=2,
            line_dash="dash",
            line_color="#E67E22",
        )
        fig.add_annotation(
            x=end_ts,
            y=1,
            yref="paper",
            text=f"處置結束 {end_ts.strftime('%Y-%m-%d')}",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            bgcolor="#FFF0DE",
            bordercolor="#E67E22",
            borderwidth=1,
            font=dict(color="#8A4B08", size=12),
        )

    fig.update_layout(
        title=dict(text=f"{code} {event_name}｜日K＋布林通道", x=0.01, xanchor="left", font=dict(size=18)),
        height=560,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Arial, Microsoft JhengHei, sans-serif", size=12, color="#263238"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(showgrid=True, gridcolor="#edf0f2", showspikes=True, spikemode="across"),
        yaxis=dict(showgrid=True, gridcolor="#edf0f2"),
        margin=dict(l=45, r=20, t=70, b=35),
    )
    st.plotly_chart(fig, use_container_width=True)

    recent = daily.tail(60).copy().sort_index(ascending=False)
    recent.index = recent.index.strftime("%Y-%m-%d")
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "BB_MID", "BB_UPPER", "BB_LOWER"] if c in recent.columns]
    st.markdown("**最近60個交易日（新 → 舊）**")
    st.dataframe(recent[cols].round(2), use_container_width=True)


def render_disposition_table(df, market, title):
    if df.empty:
        st.warning(f"目前無法直接取得 {title} 處置資料，請使用下方官方查詢。")
        return
    x = df.copy()
    # 常見欄位標準化
    code_col = next((c for c in x.columns if c in ["股票代號", "證券代號", "代號"] or "代號" in str(c)), None)
    name_col = next((c for c in x.columns if c in ["股票名稱", "證券名稱", "名稱"] or "名稱" in str(c)), None)
    period_col = next((c for c in x.columns if c in ["處置起迄", "處置期間", "處置日期"] or "處置" in str(c) and ("期" in str(c) or "迄" in str(c))), None)
    if code_col is None:
        st.dataframe(x, use_container_width=True, hide_index=True)
        return
    # 只保留有四碼股票代號的資料
    x["_code"] = x[code_col].map(_normalize_event_code)
    x = x[x["_code"].str.len().eq(4)].copy()
    if x.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return
    html_cols = [c for c in x.columns if c != "_code"]
    st.markdown(f"**{title}**")
    html = ['<div style="overflow-x:auto;border:1px solid #ead9c8;border-radius:12px;background:#fff;">', '<table style="width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap;">', '<thead><tr style="background:#fff1e2;">']
    for c in html_cols:
        html.append(f'<th style="padding:10px 9px;border-bottom:1px solid #ead9c8;text-align:left;">{c}</th>')
    html.append('</tr></thead><tbody>')
    for _, r in x.iterrows():
        code = r["_code"]
        name = r[name_col] if name_col else code
        vals=[]
        for c in html_cols:
            val = r[c]
            if c == name_col:
                val = make_event_stock_link(code, name, market)
            elif pd.isna(val):
                val = "-"
            else:
                val = str(val)
            vals.append(f'<td style="padding:9px;border-bottom:1px solid #f3eee8;">{val}</td>')
        html.append('<tr>' + ''.join(vals) + '</tr>')
    html.append('</tbody></table></div>')
    st.markdown(''.join(html), unsafe_allow_html=True)
    st.info("💡 點擊處置股票名稱，可查看日K、布林通道與處置結束日。")


def render_events_page():
    event_stock = str(st.query_params.get("event_stock", "")).strip()
    event_market = str(st.query_params.get("event_market", "listed")).strip().lower()

    if event_stock:
        twse = load_twse_disposal()
        tpex = load_tpex_disposal_html()
        source = twse if event_market == "listed" else tpex
        event_name = event_stock
        end_date = pd.NaT
        if not source.empty:
            code_col = next((c for c in source.columns if c in ["股票代號", "證券代號", "代號"] or "代號" in str(c)), None)
            name_col = next((c for c in source.columns if c in ["股票名稱", "證券名稱", "名稱"] or "名稱" in str(c)), None)
            period_col = next((c for c in source.columns if c in ["處置起迄", "處置期間", "處置日期"] or ("處置" in str(c) and ("期" in str(c) or "迄" in str(c)))), None)
            if code_col:
                hit = source[source[code_col].map(_normalize_event_code).eq(_normalize_event_code(event_stock))]
                if not hit.empty:
                    if name_col:
                        event_name = str(hit.iloc[0][name_col])
                    if period_col:
                        end_date = parse_disposition_end(hit.iloc[0][period_col])
        render_disposition_detail(event_stock, event_market, event_name, end_date)
        return

    st.title("📅 近期事件")
    st.caption("處置／除權息｜基本面／公告資料以 MOPS 為主；行情資料採 TWSE／TPEx 官方歷史行情。")

    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"本頁資料查詢時間：{today}")

    tab1, tab2 = st.tabs(["🚨 處置股票", "💰 除權息"])

    with tab1:
        st.markdown("### 🚨 近期處置股票")
        st.info("處置資料用來提醒交易撮合方式、預收款券等限制，實際規則以交易所公告為準。")

        twse = load_twse_disposal()
        if not twse.empty and "公布日期" in twse.columns:
            twse["公布日期"] = twse["公布日期"].astype(str).str.replace(r"^(\d{3})(\d{2})(\d{2})$", r"\1/\2/\3", regex=True)
        render_disposition_table(twse, "listed", "上市｜TWSE")

        tpex = load_tpex_disposal_html()
        render_disposition_table(tpex, "otc", "上櫃｜TPEx")
        st.link_button("🔗 開啟 TPEx 官方處置股票查詢", "https://www.tpex.org.tw/zh-tw/announcement/mainboard/disposal.html", use_container_width=True)
        st.link_button("🔗 開啟 TWSE 官方處置股票查詢", "https://www.twse.com.tw/rwd/zh/announcement/punish?response=html", use_container_width=True)

    with tab2:
        st.markdown("### 💰 近期除權息")
        st.info("除權息日期、股利及配股資料以交易所最新公告為準。")

        twse_ex = load_twse_exrights()
        if not twse_ex.empty:
            # 只保留未來約 60 天與近期 14 天，讓頁面不會塞滿歷史資料。
            if "除權息日期" in twse_ex.columns:
                raw = twse_ex["除權息日期"].astype(str)
                # 交易所日期常為民國年；將 115年10月08日 轉成可排序日期。
                def roc_to_date(v):
                    m = re.search(r"(\d{3})年(\d{1,2})月(\d{1,2})日", v)
                    if not m:
                        m = re.search(r"(\d{3})/(\d{1,2})/(\d{1,2})", v)
                    if not m:
                        return pd.NaT
                    return pd.Timestamp(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))
                dt = raw.map(roc_to_date)
                start = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=14)
                end = pd.Timestamp(datetime.now().date()) + pd.Timedelta(days=60)
                mask = dt.between(start, end)
                view = twse_ex.loc[mask].copy()
                view.insert(0, "排序日期", dt.loc[mask].dt.strftime("%Y-%m-%d"))
                view = view.sort_values("排序日期")
                view = view.drop(columns=["排序日期"])
            else:
                view = twse_ex.copy()
            st.markdown("**上市｜TWSE**")
            st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            st.warning("目前無法直接取得 TWSE 除權息預告資料，請使用官方查詢。")

        tpex_ex = load_tpex_exrights_html()
        st.markdown("**上櫃｜TPEx**")
        if not tpex_ex.empty:
            st.dataframe(tpex_ex, use_container_width=True, hide_index=True)
        else:
            st.info("TPEx 官方除權息頁面可能限制自動抓取，因此這裡提供官方查詢入口。")
        st.link_button("🔗 開啟 TWSE 官方除權除息預告表", "https://www.twse.com.tw/exchangeReport/TWT48U?response=html", use_container_width=True)
        st.link_button("🔗 開啟 TPEx 官方除權除息公告", "https://www.tpex.org.tw/zh-tw/announce/market/ex/announce.html", use_container_width=True)


def render_futures_diary():
    st.title("📖 海期日記")
    st.caption("這裡會是傑森自己的海期操盤日記。")
    st.markdown(
        """
        <div class="page-card">
          <h3 style="margin-top:0;color:#7A3E00;">📝 操盤日記</h3>
          <p>之後可以在這裡發表你的海期交易紀錄，例如：</p>
          <ul>
            <li>交易日期與商品（NQ／MNQ 等）</li>
            <li>進場理由、出場理由</li>
            <li>當時使用的 SURF／均線／背離訊號</li>
            <li>停損、停利與實際結果</li>
            <li>當天盤後檢討與下一次改善事項</li>
          </ul>
          <p style="margin-bottom:0;color:#8A6A52;">目前先建立頁面；之後可以再加上「新增日記、編輯、日期搜尋、標籤、圖片」以及永久儲存功能。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("📌 目前這一頁先作為日記入口；尚未啟用永久儲存，所以不要把正式日記只存放在這個測試頁面。")


# =========================
# 導覽頁面分流
# =========================
if page == "events":
    render_events_page()
    st.divider()
    st.caption("資料來源：TWSE／TPEx 官方網站與公開資料。")
    st.stop()

if page == "futures":
    render_futures_diary()
    st.divider()
    st.caption("飆股獵奇-傑森｜海期日記預備頁")
    st.stop()

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
        ["20 檔快速掃描", "100 檔測試", "300 檔測試", "全市場"],
        index=0,
    )

    sort_field = st.selectbox(
        "排序方式",
        [
            "V2總分",
            "營收YoY",
            "營收MoM",
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

    # 技術資料抓取前，先依使用者指定的營收優先順序預篩。
    base = base.sort_values(
        ["月營收YoY", "營收MoM", "累計營收YoY"],
        ascending=[False, False, False],
        na_position="last",
    )

    if scan_size == "20 檔快速掃描":
        selected = base.head(20)
    elif scan_size == "100 檔測試":
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
        f"② 正在抓取 {len(items)} 檔官方週K並計算週MA12斜率…"
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
if sort_field == "營收YoY":
    result = result.sort_values(
        ["月營收YoY", "營收MoM", "累計營收YoY"],
        ascending=[ascending, ascending, ascending],
        na_position="last",
    )
else:
    sort_col = "月營收YoY" if sort_field == "營收YoY" else sort_field
    result = result.sort_values(
        sort_col,
        ascending=ascending,
        na_position="last",
    )
result = result.reset_index(drop=True)

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
    c3.metric("營收YoY ≥100%", f"{int((result['月營收YoY'] >= 100).sum())}")
    c4.metric("週MA12上彎", f"{int((result['週MA12斜率'] >= 0).sum())}")
    st.caption(f"最後掃描：{st.session_state.get('scan_time', '-')}")

# =========================
# 排名
# =========================
if not detail_mode:
    st.subheader("📊 飆股候選排名")
    st.caption("營收YoY＝單月年增率｜營收MoM＝單月月增率｜累計營收YoY＝今年累計營收年增率")
    show = result[["排名","代號","名稱","產業","股價","市場","營收年月","月營收YoY","營收MoM","累計營收YoY","EPS","PE","週MA12斜率","週趨勢","V2基本總分","型態分","V2總分"]].copy()
    for col in ["月營收YoY","營收MoM","累計營收YoY","EPS","PE","週MA12斜率"]:
        show[col] = show[col].round(2)

    # 股名本身就是連結：直接點股名進入個股詳細頁。
    def make_stock_link(row):
        code = str(row["代號"])
        name = str(row["名稱"])
        return f'<a href="?page=home&stock={code}" target="_self" style="text-decoration:none; font-weight:600; color:#9A4D00;">{name}</a>'

    html_cols = ["排名","代號","名稱","產業","股價","市場","營收年月","月營收YoY","營收MoM","累計營收YoY","EPS","PE","週MA12斜率","週趨勢","V2基本總分","型態分","V2總分"]
    html = [
        '<div translate="no" lang="zh-Hant" style="overflow-x:auto; border:1px solid #e6e8eb; border-radius:12px; background:#fff;">',
        '<table style="width:100%; border-collapse:collapse; font-size:14px; white-space:nowrap;">',
        '<thead><tr style="background:#f7f8fa;">'
    ]
    header_names = {
        "產業": "主要產業",
        "營收年月": "營收年月",
        "月營收YoY": "營收YoY（年增率）",
        "營收MoM": "營收MoM（月增率）",
        "累計營收YoY": "累計營收YoY（累計年增率）",
        "PE": "本益比PE",
    }
    for col in html_cols:
        header = header_names.get(col, col)
        html.append(f'<th style="padding:10px 9px; border-bottom:1px solid #e6e8eb; text-align:left; font-weight:650;">{header}</th>')
    html.append('</tr></thead><tbody>')
    for _, r in show.iterrows():
        vals = []
        for col in html_cols:
            if col == "名稱":
                val = make_stock_link(r)
            elif col in ["月營收YoY", "營收MoM", "累計營收YoY", "週MA12斜率"] and pd.notna(r[col]):
                cls = "num-positive" if float(r[col]) > 0 else ("num-negative" if float(r[col]) < 0 else "")
                val = f'<span class="{cls}">{r[col]:.2f}%</span>' if cls else f'{r[col]:.2f}%'
            elif col in ["股價", "EPS", "PE"] and pd.notna(r[col]):
                cls = "num-positive" if float(r[col]) > 0 else ("num-negative" if float(r[col]) < 0 else "")
                val = f'<span class="{cls}">{r[col]:.2f}</span>' if cls else f'{r[col]:.2f}'
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
        codes = result["代號"].astype(str).tolist()
        if not codes:
            return
        selected_code = str(selected_code)
        if selected_code not in codes:
            selected_code = codes[0]
        chosen = st.selectbox(
            "選擇股票", codes, index=codes.index(selected_code),
            format_func=lambda x: f"{x} {result.loc[result['代號'].astype(str).eq(x), '名稱'].iloc[0]}",
            key="detail_stock_box",
        )
        if chosen != selected_code:
            st.query_params["stock"] = chosen
            st.rerun()

        stock = result[result["代號"].astype(str).eq(selected_code)].iloc[0]
        st.markdown(f"### {stock['代號']} {stock['名稱']}〔{stock['市場']}〕")
        st.caption("上方顯示營收年增率、月增率、季增率、EPS 與本益比；下方保留週K、月K線圖。正數紅色、負數綠色。")

        with st.spinner("讀取各季及各月基本面資料…"):
            quarterly, monthly = load_detail_fundamentals(stock["代號"], stock["市場"])

        def growth_style(v):
            if pd.isna(v):
                return ""
            if v > 0:
                return "color: #d62728; font-weight: 700"
            if v < 0:
                return "color: #159447; font-weight: 700"
            return "color: #333333"

        if quarterly.empty and monthly.empty:
            st.warning("目前沒有取得這檔股票的歷史基本面資料，請稍後再試。")
            # 即使基本面暫時無資料，也繼續顯示下方K線。

        if not quarterly.empty:
            st.markdown("#### 各季營收成長率、EPS、本益比（最新在上）")
            q = quarterly.copy()
            q["季度營收"] = pd.to_numeric(q.get("季度營收"), errors="coerce")
            q = q.sort_values("季度")
            # 季增率為本季營收相對上一季（QoQ），與月增率 MoM 不同。
            q["季增率QoQ(%)"] = q["季度營收"].pct_change(fill_method=None) * 100
            q = q.sort_values("季度", ascending=False)
            qview = pd.DataFrame({
                "季度": q["季度"],
                "年增率YoY(%)": pd.to_numeric(q.get("季度營收YoY"), errors="coerce"),
                "季增率QoQ(%)": q["季增率QoQ(%)"],
                "EPS(元)": pd.to_numeric(q.get("季EPS", pd.Series(index=q.index, dtype=float)), errors="coerce"),
                "本益比PE": pd.Series(float("nan"), index=q.index),
            })
            # 最新 PE 是目前市場本益比，不可當作每一季的歷史 PE。
            if pd.notna(stock.get("PE", float("nan"))) and not qview.empty:
                qview.loc[qview.index[0], "本益比PE"] = float(stock["PE"])
            numeric = [c for c in qview.columns if c != "季度"]
            st.dataframe(qview.style.format({c: "{:.2f}" for c in numeric}, na_rep="—").map(growth_style, subset=numeric),
                         use_container_width=True, hide_index=True)
            st.caption("本益比只在最新一季列顯示目前的 PE（非該季歷史 PE）；其餘季度沒有可靠歷史值時留白。季度未結束時，季增率可能不具可比性。")

        if not monthly.empty:
            st.markdown("#### 各月營收年增率、月增率（最新在上）")
            m = monthly.copy()
            m["營收(千元)"] = pd.to_numeric(m["營收(千元)"], errors="coerce")
            m = m.sort_values("年月")
            m["月增率MoM(%)"] = m["營收(千元)"].pct_change(fill_method=None) * 100
            mview = pd.DataFrame({
                "年月": m["年月"],
                "年增率YoY(%)": pd.to_numeric(m["YoY%"], errors="coerce"),
                "月增率MoM(%)": m["月增率MoM(%)"],
            }).sort_values("年月", ascending=False)
            numeric = ["年增率YoY(%)", "月增率MoM(%)"]
            st.dataframe(mview.style.format({c: "{:.2f}" for c in numeric}, na_rep="—").map(growth_style, subset=numeric),
                         use_container_width=True, hide_index=True)
            st.caption("EPS 是季度財報數字，本益比是行情估值，因此不將季度 EPS 或最新 PE 假裝成每月數據。")

        # 下方保留原有週K、月K線圖；其他技術明細表不顯示。
        with st.spinner("讀取官方歷史行情及K線…"):
            daily = load_daily(str(stock["代號"]), str(stock["市場"]))

        if daily.empty:
            st.warning("目前抓不到這檔股票的官方歷史K線資料，請稍後再試。")
        else:
            daily = daily.copy()
            daily.index = pd.to_datetime(daily.index, errors="coerce")
            daily = daily[daily.index.notna()].sort_index()
            weekly = add_bollinger(make_weekly(daily))
            monthly_k = make_monthly(daily)

            st.markdown("#### 週K技術圖")
            if not weekly.empty:
                st.plotly_chart(
                    candle_chart(weekly.tail(104), f"{stock['代號']} 週K", show_bb=True),
                    use_container_width=True,
                )
            else:
                st.info("目前沒有足夠的週K資料。")

            st.markdown("#### 月K技術圖")
            if not monthly_k.empty:
                st.plotly_chart(
                    candle_chart(monthly_k.tail(60), f"{stock['代號']} 月K"),
                    use_container_width=True,
                )
            else:
                st.info("目前沒有足夠的月K資料。")

    render_detail(result, query_stock)

render_adsense("4790829615", height=125)

st.divider()

st.caption(
    "資料來源：公開資訊觀測站（MOPS）＋TWSE／TPEx 官方交易資料。"
    " 基本面／事件以 MOPS 為主，K 線使用交易所官方歷史行情；已移除 Yahoo Finance 依賴。"
)
