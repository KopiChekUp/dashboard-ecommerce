# Dashboard Analisis E-Commerce

Proyek analisis data E-Commerce Public Dataset untuk menjawab empat pertanyaan bisnis periode Januari 2017 – Agustus 2018:

1. Berapa persentase pelanggan yang melakukan pembelian ulang, dan bagaimana segmentasi pelanggan berdasarkan RFM untuk program retensi?
2. Metode pembayaran apa yang paling banyak digunakan, dan apakah pembayaran cicilan kartu kredit memiliki rata-rata nilai transaksi lebih tinggi dibanding kartu kredit 1x?
3. Pada hari dan jam berapa pelanggan paling banyak melakukan pembelian?
4. Di negara bagian mana pelanggan segmen At Risk dan Lost paling banyak?

## Struktur Folder

```
submission
├───dashboard
│   ├───main_data.csv
│   └───dashboard.py
├───data
│   └───(9 file CSV E-Commerce Public Dataset)
├───notebook.ipynb
├───README.md
├───requirements.txt
└───url.txt
```

## Setup Environment - Anaconda

```
conda create --name ecommerce-ds python=3.12
conda activate ecommerce-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Di macOS atau Linux, aktifkan environment dengan `source .venv/bin/activate`.

Gunakan Python 3.11 atau yang lebih baru, karena `pandas==3.0.1` tidak mendukung versi Python yang lebih lama.

## Menjalankan Notebook

Buka `notebook.ipynb` dengan Jupyter Notebook, JupyterLab, atau VS Code, lalu jalankan semua cell dari atas. Cell terakhir notebook akan membuat ulang `dashboard/main_data.csv`.

## Menjalankan Dashboard Streamlit

Jalankan perintah berikut dari folder `submission`:

```
streamlit run dashboard/dashboard.py
```

Dashboard akan terbuka di browser pada alamat `http://localhost:8501`.
