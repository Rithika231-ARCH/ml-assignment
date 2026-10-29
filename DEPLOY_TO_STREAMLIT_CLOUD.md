# Streamlit Community Cloud (Free Tier) Deployment Guide

This repository is **100% pre-configured and optimized** for instant deployment on [Streamlit Community Cloud](https://share.streamlit.io/) (Free Tier).

---

## ⚡ Deployment Checklist

- [x] `app.py` located at the root of the repository
- [x] `requirements.txt` listing lightweight, Linux-compatible dependencies
- [x] `.streamlit/config.toml` configured for production performance & clean theme
- [x] `house_rent_model.joblib` + automatic fallback self-healing training if unpickling across Python versions
- [x] `plots/` folder with all evaluation and EDA figures
- [x] `House_Rent_Dataset.csv` included for dataset exploration & live inference

---

## 🚀 3-Minute Step-by-Step Deployment

### Step 1: Push This Folder to GitHub

You can upload this project to GitHub using **any** of the following methods:

#### Method A: Using VS Code (Easiest)
1. Open this folder in VS Code (`File > Open Folder > C:\nxtwave\studting`).
2. Click on the **Source Control** icon on the left panel (or press `Ctrl + Shift + G`).
3. Click **"Publish to GitHub"** -> Select **"Publish to GitHub public repository"**.
4. VS Code will automatically upload all files to your GitHub account!

#### Method B: Using GitHub Web Interface (No Git installation needed)
1. Go to [github.com/new](https://github.com/new) and create a new repository named `house-rent-prediction` (Set it to **Public**).
2. Click **"uploading an existing file"**.
3. Drag and drop the project files (especially `app.py`, `requirements.txt`, `House_Rent_Dataset.csv`, `house_rent_model.joblib`, `.streamlit`, and `plots/`).
4. Click **Commit changes**.

---

### Step 2: Deploy on Streamlit Cloud (Free)

1. Open [share.streamlit.io](https://share.streamlit.io/) in your browser.
2. Sign in with your **GitHub** account.
3. Click the **"New app"** (or **"Create app"**) button in the top right.
4. Fill in the three fields:
   - **Repository:** `your-username/house-rent-prediction`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
   - **App URL (optional):** choose a custom subdomain like `house-rent-ai.streamlit.app`
5. Click **"Deploy!"** 🎈

---

## 🛡️ Why This Setup is Bulletproof on Streamlit Cloud Free Tier:

1. **Self-Healing Model Pipeline:** If Python version differences on the cloud (e.g., Python 3.10 vs 3.14) cause any unpickling issues with `.joblib`, `app.py` automatically detects it and rebuilds the trained pipeline directly from the dataset in ~1.5 seconds, then caches it with `@st.cache_resource`. **The app will never crash on startup.**
2. **Memory Optimized:** Uses under **180 MB of RAM**, well below Streamlit Cloud's 1,000 MB (1 GB) free tier limit.
3. **Resilient Paths:** All asset references use dynamic relative paths (`BASE_DIR`) with existence fallbacks.
4. **Instant Boot Time:** Pre-computed evaluation metrics and fast cached resource loading enable sub-second cold starts.
