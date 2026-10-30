import os

import folium
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Dashboard E-Commerce", layout="wide")

# ---------------------------------------------------------------- style
HIGHLIGHT = "#2a78d6"
SECOND = "#eb6834"
MUTED = "#c9c8c2"
TEXT = "#52514e"
GRID = "#e6e5e0"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

sns.set_theme(style="white")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})

DAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
TIME_BINS = [-1, 4, 10, 14, 17, 23]
TIME_LABELS = ["Dini hari", "Pagi", "Siang", "Sore", "Malam"]
TIME_RANGES = ["00.00–04.59", "05.00–10.59", "11.00–14.59", "15.00–17.59", "18.00–23.59"]
SEGMENT_ORDER = ["Best Customers", "At Risk", "Potential Loyalist", "New Customers", "Hibernating", "Lost"]
PAYMENT_TYPE_LABELS = {
    "credit_card": "Kartu kredit",
    "boleto": "Boleto",
    "voucher": "Voucher",
    "debit_card": "Kartu debit",
}
STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí",
    "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul",
    "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
    "SE": "Sergipe", "TO": "Tocantins",
}


def format_number(value, decimals=0):
    text = f"{np.round(value, decimals):,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


thousands_formatter = mticker.FuncFormatter(lambda value, _: format_number(value))


def style_axis(ax, grid_axis):
    ax.grid(axis=grid_axis, color=GRID)
    ax.set_axisbelow(True)
    if grid_axis == "y":
        # Memberi ruang di atas bar supaya label angka tidak terpotong judul.
        ax.margins(y=0.12)


def show(fig):
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- data helpers
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "main_data.csv")
    return pd.read_csv(path, parse_dates=["order_purchase_timestamp"])


def create_customer_orders_df(delivered_df):
    customer_orders_df = delivered_df.groupby("customer_unique_id").agg(
        jumlah_pesanan=("order_id", "nunique"),
        total_belanja=("total_payment", "sum"),
    )
    customer_orders_df["kategori_pelanggan"] = pd.cut(
        customer_orders_df["jumlah_pesanan"], bins=[0, 1, np.inf], labels=["Sekali beli", "Belanja ulang"]
    )
    return customer_orders_df


def quartile_score(series, labels):
    try:
        return pd.qcut(series, q=4, labels=labels).astype(int)
    except ValueError:
        # Rentang tanggal yang sempit bisa membuat batas kuartil kembar, jadi dibagi berdasarkan urutan.
        return pd.qcut(series.rank(method="first"), q=4, labels=labels).astype(int)


def create_rfm_df(delivered_df, reference_date):
    rfm_df = delivered_df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("total_payment", "sum"),
    )
    rfm_df["recency"] = (reference_date - rfm_df["last_purchase"]).dt.days
    rfm_df["r_score"] = quartile_score(rfm_df["recency"], labels=[4, 3, 2, 1])
    rfm_df["m_score"] = quartile_score(rfm_df["monetary"], labels=[1, 2, 3, 4])
    rfm_df["f_score"] = pd.cut(rfm_df["frequency"], bins=[0, 1, 2, np.inf], labels=[1, 2, 3]).astype(int)

    is_repeat = rfm_df["f_score"] >= 2
    is_recent = rfm_df["r_score"] >= 3
    is_high_value = rfm_df["m_score"] >= 3
    rfm_df["segment"] = pd.Categorical(
        np.select(
            [
                is_repeat & is_recent,
                is_repeat & ~is_recent,
                ~is_repeat & is_recent & is_high_value,
                ~is_repeat & is_recent & ~is_high_value,
                ~is_repeat & ~is_recent & is_high_value,
            ],
            SEGMENT_ORDER[:5],
            default="Lost",
        ),
        categories=SEGMENT_ORDER,
        ordered=True,
    )
    return rfm_df[["recency", "frequency", "monetary", "r_score", "f_score", "m_score", "segment"]]


def create_segment_summary_df(rfm_df):
    segment_summary_df = rfm_df.groupby("segment", observed=False).agg(
        jumlah_pelanggan=("recency", "size"),
        rata_rata_recency=("recency", "mean"),
        rata_rata_frequency=("frequency", "mean"),
        rata_rata_monetary=("monetary", "mean"),
        total_monetary=("monetary", "sum"),
    )
    segment_summary_df["persen_pelanggan"] = segment_summary_df["jumlah_pelanggan"] / segment_summary_df["jumlah_pelanggan"].sum() * 100
    segment_summary_df["persen_pendapatan"] = segment_summary_df["total_monetary"] / segment_summary_df["total_monetary"].sum() * 100
    return segment_summary_df.fillna(0)


def create_customer_state_df(delivered_df):
    # Pelanggan yang tercatat di beberapa negara bagian memakai lokasi dari pesanan terakhirnya.
    return (
        delivered_df.sort_values("order_purchase_timestamp")
        .drop_duplicates("customer_unique_id", keep="last")
        .set_index("customer_unique_id")[["customer_state", "lat", "lng"]]
    )


def create_state_segment_df(rfm_df, customer_state_df):
    rfm_state_df = rfm_df[["segment"]].join(customer_state_df)
    state_segment_df = rfm_state_df.groupby("customer_state").agg(
        jumlah_pelanggan=("segment", "size"),
        at_risk=("segment", lambda s: (s == "At Risk").sum()),
        lost=("segment", lambda s: (s == "Lost").sum()),
        lat=("lat", "first"),
        lng=("lng", "first"),
    )
    state_segment_df["at_risk_lost"] = state_segment_df["at_risk"] + state_segment_df["lost"]
    state_segment_df["persen_at_risk_lost"] = state_segment_df["at_risk_lost"] / state_segment_df["jumlah_pelanggan"] * 100
    return state_segment_df.sort_values("at_risk_lost", ascending=False)


def create_purchase_time_df(orders_df):
    purchase_time_df = orders_df[["order_id", "order_purchase_timestamp"]].copy()
    purchase_time_df["hari"] = pd.Categorical(
        purchase_time_df["order_purchase_timestamp"].dt.dayofweek.map(dict(enumerate(DAY_NAMES))),
        categories=DAY_NAMES,
        ordered=True,
    )
    purchase_time_df["jam"] = purchase_time_df["order_purchase_timestamp"].dt.hour
    purchase_time_df["kelompok_waktu"] = pd.cut(purchase_time_df["jam"], bins=TIME_BINS, labels=TIME_LABELS)
    return purchase_time_df


# ---------------------------------------------------------------- sidebar & filter
all_df = load_data()
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.header("Filter")
    date_range = st.date_input(
        "Rentang tanggal pembelian",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    st.caption("Data tersedia untuk pembelian Januari 2017 – Agustus 2018.")

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir pada filter di sidebar.")
    st.stop()

start_date, end_date = date_range
reference_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)

orders_df = all_df[
    (all_df["order_purchase_timestamp"] >= pd.Timestamp(start_date))
    & (all_df["order_purchase_timestamp"] < reference_date)
]
delivered_df = orders_df[orders_df["order_status"] == "delivered"]

st.title("Dashboard Analisis E-Commerce")
st.caption(
    f"Periode {start_date.strftime('%d-%m-%Y')} s.d. {end_date.strftime('%d-%m-%Y')}. "
    "Angka pelanggan, pembayaran, RFM, dan wilayah memakai pesanan berstatus delivered; "
    "waktu belanja memakai pesanan dengan semua status."
)

if delivered_df["customer_unique_id"].nunique() < 4:
    st.warning("Data pada rentang tanggal ini terlalu sedikit. Silakan perlebar rentang tanggal.")
    st.stop()

customer_orders_df = create_customer_orders_df(delivered_df)
rfm_df = create_rfm_df(delivered_df, reference_date)
segment_summary_df = create_segment_summary_df(rfm_df)

total_customers = len(customer_orders_df)
repeat_customers = (customer_orders_df["jumlah_pesanan"] > 1).sum()
repeat_rate = repeat_customers / total_customers * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pesanan delivered", format_number(delivered_df["order_id"].nunique()))
col2.metric("Pelanggan", format_number(total_customers))
col3.metric("Repeat rate", f"{format_number(repeat_rate, 2)}%")
total_revenue = delivered_df["total_payment"].sum()
col4.metric(
    "Total pendapatan",
    f"{format_number(total_revenue / 1_000_000, 2)} juta" if total_revenue >= 1_000_000 else format_number(total_revenue, 2),
)

tab_retention, tab_payment, tab_time, tab_region = st.tabs(
    ["Retensi & RFM", "Pembayaran", "Waktu Belanja", "Wilayah"]
)

# ---------------------------------------------------------------- Pertanyaan 1
with tab_retention:
    st.subheader("Pembelian ulang")
    st.markdown(
        f"Dari **{format_number(total_customers)}** pelanggan, hanya **{format_number(repeat_customers)}** "
        f"yang belanja lebih dari sekali (repeat rate **{format_number(repeat_rate, 2)}%**)."
    )

    order_bins = pd.cut(
        customer_orders_df["jumlah_pesanan"], bins=[0, 1, 2, 3, np.inf], labels=["1x", "2x", "3x", "4x atau lebih"]
    )
    order_bins_df = customer_orders_df.groupby(order_bins, observed=False).size().rename("jumlah_pelanggan").to_frame()
    order_bins_df["persen"] = order_bins_df["jumlah_pelanggan"] / order_bins_df["jumlah_pelanggan"].sum() * 100

    customer_type_df = customer_orders_df.groupby("kategori_pelanggan", observed=False).agg(
        rata_rata_belanja=("total_belanja", "mean"),
        total_belanja=("total_belanja", "sum"),
    ).fillna(0)
    customer_type_df["persen_pendapatan"] = customer_type_df["total_belanja"] / customer_type_df["total_belanja"].sum() * 100

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(order_bins_df.index.astype(str), order_bins_df["jumlah_pelanggan"], color=[HIGHLIGHT, MUTED, MUTED, MUTED], width=0.6)
        ax.bar_label(
            bars,
            labels=[f"{format_number(n)} ({format_number(p, 2)}%)" for n, p in zip(order_bins_df["jumlah_pelanggan"], order_bins_df["persen"])],
            padding=3,
            color=TEXT,
        )
        ax.set_title("Jumlah pelanggan berdasarkan jumlah pesanan")
        ax.set_xlabel("Jumlah pesanan per pelanggan")
        ax.set_ylabel("Jumlah pelanggan")
        ax.yaxis.set_major_formatter(thousands_formatter)
        style_axis(ax, "y")
        show(fig)

    with right:
        fig, axes = plt.subplots(1, 2, figsize=(7, 4.5))
        labels = customer_type_df.index.astype(str)
        bars = axes[0].bar(labels, customer_type_df["rata_rata_belanja"], color=[MUTED, HIGHLIGHT], width=0.55)
        axes[0].bar_label(bars, labels=[format_number(v, 2) for v in customer_type_df["rata_rata_belanja"]], padding=3, color=TEXT)
        axes[0].set_title("Rata-rata belanja\nper pelanggan")
        style_axis(axes[0], "y")
        bars = axes[1].bar(labels, customer_type_df["persen_pendapatan"], color=[MUTED, HIGHLIGHT], width=0.55)
        axes[1].bar_label(bars, labels=[f"{format_number(v, 1)}%" for v in customer_type_df["persen_pendapatan"]], padding=3, color=TEXT)
        axes[1].set_title("Persen dari\ntotal pendapatan")
        axes[1].set_ylim(0, 105)
        style_axis(axes[1], "y")
        show(fig)

    st.subheader("Segmentasi RFM")
    st.markdown(
        f"Recency dihitung sampai tanggal acuan **{reference_date.strftime('%d-%m-%Y')}** (sehari setelah tanggal akhir filter). "
        "Skor R dan M dibagi berdasarkan kuartil, sedangkan skor F: 1x = 1, 2x = 2, 3x atau lebih = 3."
    )

    segment_plot_df = segment_summary_df.iloc[::-1]
    top_revenue_segments = segment_summary_df["persen_pendapatan"].nlargest(2).index
    segment_colors = [HIGHLIGHT if segment in top_revenue_segments else MUTED for segment in segment_plot_df.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bars = axes[0].barh(segment_plot_df.index.astype(str), segment_plot_df["persen_pelanggan"], color=segment_colors, height=0.6)
    axes[0].bar_label(
        bars,
        labels=[f"{format_number(n)} ({format_number(p, 1)}%)" for n, p in zip(segment_plot_df["jumlah_pelanggan"], segment_plot_df["persen_pelanggan"])],
        padding=3,
        color=TEXT,
    )
    axes[0].set_title("Persen pelanggan per segmen")
    axes[0].set_xlim(0, max(segment_plot_df["persen_pelanggan"].max() * 1.4, 1))
    bars = axes[1].barh(segment_plot_df.index.astype(str), segment_plot_df["persen_pendapatan"], color=segment_colors, height=0.6)
    axes[1].bar_label(bars, labels=[f"{format_number(v, 1)}%" for v in segment_plot_df["persen_pendapatan"]], padding=3, color=TEXT)
    axes[1].set_title("Persen pendapatan per segmen")
    axes[1].set_xlim(0, max(segment_plot_df["persen_pendapatan"].max() * 1.3, 1))
    for ax in axes:
        style_axis(ax, "x")
    show(fig)
    st.caption("Biru: dua segmen dengan kontribusi pendapatan terbesar.")

    st.dataframe(
        segment_summary_df.rename(columns={
            "jumlah_pelanggan": "Jumlah pelanggan",
            "rata_rata_recency": "Rata-rata recency (hari)",
            "rata_rata_frequency": "Rata-rata frequency",
            "rata_rata_monetary": "Rata-rata monetary",
            "total_monetary": "Total monetary",
            "persen_pelanggan": "Persen pelanggan",
            "persen_pendapatan": "Persen pendapatan",
        }).round(2),
    )

# ---------------------------------------------------------------- Pertanyaan 2
with tab_payment:
    st.subheader("Metode pembayaran")
    payment_method_df = delivered_df.groupby("main_payment_type").agg(jumlah_pesanan=("order_id", "nunique"))
    payment_method_df["persen_pesanan"] = payment_method_df["jumlah_pesanan"] / payment_method_df["jumlah_pesanan"].sum() * 100
    payment_method_df = payment_method_df.sort_values("persen_pesanan")

    credit_card_df = delivered_df[delivered_df["credit_card_group"].notna()]
    credit_card_group_df = credit_card_df.groupby("credit_card_group").agg(
        jumlah_pesanan=("order_id", "nunique"),
        rata_rata_nilai=("total_payment", "mean"),
        median_nilai=("total_payment", "median"),
    ).reindex(["Kartu kredit 1x", "Cicilan"]).fillna(0)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        top_method = payment_method_df["persen_pesanan"].idxmax()
        bars = ax.barh(
            payment_method_df.index.map(PAYMENT_TYPE_LABELS),
            payment_method_df["persen_pesanan"],
            color=[HIGHLIGHT if method == top_method else MUTED for method in payment_method_df.index],
            height=0.6,
        )
        ax.bar_label(bars, labels=[f"{format_number(v, 2)}%" for v in payment_method_df["persen_pesanan"]], padding=3, color=TEXT)
        ax.set_title("Persen pesanan per metode pembayaran utama")
        ax.set_xlim(0, 100)
        style_axis(ax, "x")
        show(fig)

    with right:
        x = np.arange(len(credit_card_group_df))
        width = 0.36
        fig, ax = plt.subplots(figsize=(7, 4.5))
        mean_bars = ax.bar(x - width / 2 - 0.01, credit_card_group_df["rata_rata_nilai"], width, color=HIGHLIGHT, label="Rata-rata")
        median_bars = ax.bar(x + width / 2 + 0.01, credit_card_group_df["median_nilai"], width, color=SECOND, label="Median")
        for container in [mean_bars, median_bars]:
            ax.bar_label(container, labels=[format_number(v, 2) for v in container.datavalues], padding=3, color=TEXT)
        ax.set_xticks(x, credit_card_group_df.index)
        ax.set_title("Nilai transaksi: cicilan vs kartu kredit 1x")
        ax.set_ylabel("Nilai transaksi per pesanan")
        ax.legend(frameon=False, loc="upper left")
        style_axis(ax, "y")
        show(fig)

    st.subheader("Cicilan kartu kredit")
    installments_df = (
        credit_card_df.groupby(credit_card_df["credit_card_installments"].astype(int))
        .agg(rata_rata_nilai=("total_payment", "mean"), median_nilai=("total_payment", "median"))
        .reindex(range(1, 11))
        .dropna()
    )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(installments_df.index, installments_df["rata_rata_nilai"], color=HIGHLIGHT, linewidth=2, marker="o", markersize=7, label="Rata-rata")
        ax.plot(installments_df.index, installments_df["median_nilai"], color=SECOND, linewidth=2, marker="o", markersize=7, label="Median")
        ax.set_xticks(range(1, 11))
        ax.set_title("Nilai transaksi per jumlah cicilan (1x–10x)")
        ax.set_xlabel("Jumlah cicilan")
        ax.set_ylabel("Nilai transaksi per pesanan")
        ax.legend(frameon=False)
        style_axis(ax, "y")
        show(fig)
        st.caption("Hanya cicilan 1x–10x yang ditampilkan karena cicilan 11x ke atas jumlah pesanannya sangat sedikit, sehingga rata-ratanya tidak stabil.")

    with right:
        value_bins = pd.qcut(credit_card_df["total_payment"], q=10, duplicates="drop")
        installment_share_df = credit_card_df.groupby(value_bins, observed=True).agg(
            persen_cicilan=("credit_card_group", lambda s: (s == "Cicilan").mean() * 100)
        )
        share_labels = [
            f"{format_number(interval.left, 2)} – {format_number(interval.right, 2)}"
            for interval in installment_share_df.index
        ]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(share_labels[::-1], installment_share_df["persen_cicilan"].iloc[::-1], color=HIGHLIGHT, height=0.6)
        ax.bar_label(bars, labels=[f"{format_number(v, 1)}%" for v in installment_share_df["persen_cicilan"].iloc[::-1]], padding=3, color=TEXT)
        ax.set_title("Persen pesanan kartu kredit yang mencicil")
        ax.set_xlabel("Persen pesanan")
        ax.set_ylabel("Rentang nilai transaksi")
        ax.set_xlim(0, 100)
        style_axis(ax, "x")
        show(fig)

# ---------------------------------------------------------------- Pertanyaan 3
with tab_time:
    purchase_time_df = create_purchase_time_df(orders_df)
    orders_by_day_df = purchase_time_df.groupby("hari", observed=False).agg(jumlah_pesanan=("order_id", "nunique"))
    orders_by_hour_df = purchase_time_df.groupby("jam").agg(jumlah_pesanan=("order_id", "nunique")).reindex(range(24), fill_value=0)
    day_hour_df = purchase_time_df.pivot_table(
        index="hari", columns="jam", values="order_id", aggfunc="nunique", observed=False
    ).reindex(columns=range(24)).fillna(0)

    hours_per_group = pd.cut(pd.Series(range(24)), bins=TIME_BINS, labels=TIME_LABELS).value_counts()
    orders_by_time_group_df = purchase_time_df.groupby("kelompok_waktu", observed=False).agg(jumlah_pesanan=("order_id", "nunique"))
    orders_by_time_group_df["rata_rata_pesanan_per_jam"] = orders_by_time_group_df["jumlah_pesanan"] / hours_per_group

    busiest_day = orders_by_day_df["jumlah_pesanan"].idxmax()
    busiest_hour = orders_by_hour_df["jumlah_pesanan"].idxmax()
    st.markdown(
        f"Hari paling ramai: **{busiest_day}**. Jam paling ramai: **pukul {busiest_hour:02d}.00–{busiest_hour:02d}.59**. "
        f"Total pesanan (semua status): **{format_number(orders_df['order_id'].nunique())}**."
    )

    left, right = st.columns(2)
    with left:
        top_days = orders_by_day_df["jumlah_pesanan"].nlargest(2).index
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(
            orders_by_day_df.index.astype(str),
            orders_by_day_df["jumlah_pesanan"],
            color=[HIGHLIGHT if day in top_days else MUTED for day in orders_by_day_df.index],
            width=0.6,
        )
        ax.bar_label(bars, labels=[format_number(v) for v in orders_by_day_df["jumlah_pesanan"]], padding=3, color=TEXT)
        ax.set_title("Jumlah pesanan per hari")
        ax.yaxis.set_major_formatter(thousands_formatter)
        style_axis(ax, "y")
        show(fig)

    with right:
        top_groups = orders_by_time_group_df["rata_rata_pesanan_per_jam"].nlargest(2).index
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(
            range(len(orders_by_time_group_df)),
            orders_by_time_group_df["rata_rata_pesanan_per_jam"],
            color=[HIGHLIGHT if group in top_groups else MUTED for group in orders_by_time_group_df.index],
            width=0.6,
        )
        ax.bar_label(bars, labels=[format_number(v) for v in orders_by_time_group_df["rata_rata_pesanan_per_jam"]], padding=3, color=TEXT)
        ax.set_xticks(range(len(TIME_LABELS)), [f"{group}\n{hours}" for group, hours in zip(TIME_LABELS, TIME_RANGES)])
        ax.set_title("Rata-rata pesanan per jam per kelompok waktu")
        ax.yaxis.set_major_formatter(thousands_formatter)
        style_axis(ax, "y")
        show(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(
        orders_by_hour_df.index,
        orders_by_hour_df["jumlah_pesanan"],
        color=[HIGHLIGHT if hour == busiest_hour else MUTED for hour in orders_by_hour_df.index],
        width=0.75,
    )
    ax.set_xticks(range(24), [f"{hour:02d}" for hour in range(24)])
    ax.set_title("Jumlah pesanan per jam")
    ax.set_xlabel("Jam pembelian")
    ax.yaxis.set_major_formatter(thousands_formatter)
    style_axis(ax, "y")
    show(fig)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    sns.heatmap(
        day_hour_df,
        cmap=BLUE_RAMP,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Jumlah pesanan", "format": thousands_formatter},
        ax=ax,
    )
    ax.set_title("Jumlah pesanan per hari dan jam")
    ax.set_xlabel("Jam pembelian")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    show(fig)

# ---------------------------------------------------------------- Pertanyaan 4
with tab_region:
    customer_state_df = create_customer_state_df(delivered_df)
    state_segment_df = create_state_segment_df(rfm_df, customer_state_df)
    total_at_risk_lost = state_segment_df["at_risk_lost"].sum()
    top_state = state_segment_df.index[0]

    st.subheader("Sebaran pelanggan At Risk dan Lost")
    st.markdown(
        f"Total pelanggan At Risk + Lost: **{format_number(total_at_risk_lost)}**. "
        f"Terbanyak di **{STATE_NAMES.get(top_state, top_state)} ({top_state})** dengan "
        f"**{format_number(state_segment_df.loc[top_state, 'at_risk_lost'])}** pelanggan "
        f"({format_number(state_segment_df.loc[top_state, 'at_risk_lost'] / total_at_risk_lost * 100, 2)}% dari total At Risk + Lost)."
    )

    state_map = folium.Map(location=[-14.5, -52.0], zoom_start=4, tiles="OpenStreetMap")
    max_at_risk_lost = max(state_segment_df["at_risk_lost"].max(), 1)
    for state, row in state_segment_df.iterrows():
        popup_html = (
            f"<b>{STATE_NAMES.get(state, state)} ({state})</b><br>"
            f"At Risk: {format_number(row['at_risk'])}<br>"
            f"Lost: {format_number(row['lost'])}<br>"
            f"At Risk + Lost: {format_number(row['at_risk_lost'])} "
            f"({format_number(row['persen_at_risk_lost'], 1)}% dari {format_number(row['jumlah_pelanggan'])} pelanggan)"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=3 + 37 * np.sqrt(row["at_risk_lost"] / max_at_risk_lost),
            color="white",
            weight=1,
            fill=True,
            fill_color=HIGHLIGHT,
            fill_opacity=0.65,
            tooltip=f"{state}: {format_number(row['at_risk_lost'])} pelanggan At Risk + Lost",
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(state_map)

    left, right = st.columns([3, 2])
    with left:
        st.iframe(state_map.get_root().render(), height=520)
        st.caption("Ukuran lingkaran sebanding dengan jumlah pelanggan At Risk + Lost. Klik lingkaran untuk melihat detail.")

    with right:
        top_states_df = state_segment_df.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 6.5))
        ax.barh(top_states_df.index, top_states_df["lost"], color=HIGHLIGHT, height=0.6, label="Lost", edgecolor="white", linewidth=2)
        ax.barh(top_states_df.index, top_states_df["at_risk"], left=top_states_df["lost"], color=SECOND, height=0.6, label="At Risk", edgecolor="white", linewidth=2)
        for y, total in enumerate(top_states_df["at_risk_lost"]):
            ax.text(total, y, f"  {format_number(total)}", va="center", color=TEXT)
        ax.set_title("10 negara bagian teratas")
        ax.set_xlabel("Jumlah pelanggan")
        ax.set_xlim(0, top_states_df["at_risk_lost"].max() * 1.25)
        ax.xaxis.set_major_formatter(thousands_formatter)
        ax.legend(frameon=False, loc="lower right")
        style_axis(ax, "x")
        show(fig)

    st.dataframe(
        state_segment_df.assign(nama_negara_bagian=state_segment_df.index.map(STATE_NAMES))[
            ["nama_negara_bagian", "jumlah_pelanggan", "at_risk", "lost", "at_risk_lost", "persen_at_risk_lost"]
        ].rename(columns={
            "nama_negara_bagian": "Negara bagian",
            "jumlah_pelanggan": "Jumlah pelanggan",
            "at_risk": "At Risk",
            "lost": "Lost",
            "at_risk_lost": "At Risk + Lost",
            "persen_at_risk_lost": "Persen At Risk + Lost",
        }).round(2),
    )

st.caption("Sumber data: E-Commerce Public Dataset · Dibuat oleh Nathanael Kevin Ivanco")
