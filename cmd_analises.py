# -*- coding: utf-8 -*-
"""
CMD_Analises_Bot
----------------
Versão consolidada SEM YAML:
- IA principal: Openai (via OpenAI SDK).
- Gemini opcional (reserva).
- Vigia a pasta Downloads 24h/dia.
- Sempre que cai um CSV NOVO:
    * analisa
    * gera relatório HTML
    * manda e-mail
    * mostra notificação Windows
- Uma vez por dia gera relatório diário agregado.
"""
from pathlib import Path
from PIL import Image, ImageOps
import pytesseract
# se o Tesseract estiver em outro caminho, ajuste aqui
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\elymar.santos_anagam\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


import os
import glob
import json
import base64
import time
import argparse
import re
import sys
import webbrowser
import logging
import smtplib
import ssl
from logging.handlers import RotatingFileHandler
from email.message import EmailMessage
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from typing import List, Dict, Any, Optional


import httpx
import pandas as pd
import numpy as np
import matplotlib
import base64
import io
import pandas as pd  # Para pd.to_datetime
import re
import base64
from io import BytesIO
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from jinja2 import Template
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import certifi

# backend sem interface gráfica, sem GUI
matplotlib.use("Agg")  # usar API orientada a objetos, sem pyplot

# garante certificados SSL válidos para requests/SDKs
os.environ["SSL_CERT_FILE"] = certifi.where()

# Notificação Windows (popup)
try:
    from plyer import notification
    HAVE_NOTIFY = True
except Exception:
    HAVE_NOTIFY = False

# OpenAI (SDK) – usado para falar com a API do Openai (base_url customizada)
try:
    from openai import OpenAI
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

# =====================================================================
# 🔑 CHAVE DA PERPLEXITY / OPENAI (API KEY)
# =====================================================================
# Correção: Definindo a string diretamente para evitar erros de leitura do Windows


# Gemini (opcional)
try:
    from google import genai
    HAVE_GEMINI = True
except Exception:
    HAVE_GEMINI = False

# =====================================================================
# CÉREBRO ANALÍTICO
# =====================================================================

# =====================================================================
# BLOCO OLHO VIVO – ÍNDICE DE RELATÓRIOS PARA O PAINEL
# =====================================================================

import json
from typing import List, Dict, Any, Optional
from datetime import date
import os

BASEDIR = os.path.dirname(__file__)
REPORTSDIR = os.path.join(BASEDIR, "reports")
REPORTSINDEXPATH = os.path.join(REPORTSDIR, "reportsindex.json")  # ← SEM _

def ensurereportsdir() -> None:
    """Cria pasta reports se não existir."""
    os.makedirs(REPORTSDIR, exist_ok=True)

def loadreportsindex() -> List[Dict[str, Any]]:
    """Carrega reportsindex.json ou retorna lista vazia."""
    ensurereportsdir()
    if not os.path.isfile(REPORTSINDEXPATH):
        return []
    try:
        with open(REPORTSINDEXPATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erro lendo index: {e}")
        return []

def savereportsindex(data: List[Dict[str, Any]]) -> None:
    """Salva reportsindex.json com confirmação."""
    ensurereportsdir()
    try:
        with open(REPORTSINDEXPATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Index salvo: {len(data)} entradas → {REPORTSINDEXPATH}")
    except Exception as e:
        print(f"❌ Erro salvando index: {e}")

def appendreportindex(
    tipo: str,
    htmlpath: str,
    alertscriticos: int,
    refdate: Optional[date] = None,
) -> None:
    """Adiciona relatório ao index para frontend."""
    ensurereportsdir()
    data = loadreportsindex()
    
    refdate = refdate or date.today()
    datestr = refdate.isoformat()
    fname = os.path.basename(htmlpath)
    rel_path = f"/reports/{fname}"

    entry = {
        "tipo": tipo,
        "date": datestr,
        "file": rel_path,
        "label": fname,
        "alertsCritical": int(alertscriticos or 0),
    }

    data.append(entry)
    savereportsindex(data)
    print(f"➕ Adicionado: {entry['label']} ({entry['tipo']})")

def map_proctype_to_menu_tipo(pt: str) -> str:
    pt = (pt or "").lower()
    if pt.startswith("saque"):
        return "saques"
    if pt.startswith("dep"):
        return "depositos"
    if "rollback" in pt or "freespin" in pt:
        return "rollbacks"
    if "cadastro" in pt or "kyc" in pt or "onboarding" in pt:
        return "cadastro"
    if "trans" in pt:
        return "transaction"
    return "analises"

# =====================================================================
# FIM BLOCO OLHO VIVO
# =====================================================================


# ========================================================================
# 🔧 CONFIGURAÇÃO CENTRAL (SEM YAML)
# ========================================================================

CONFIG: Dict[str, Any] = {
    "input_folder": r"C:\Resumo Erros",
    "output_folder": r"C:\Olho Vivo\backend\reports",
    "file_glob": "*.csv",
    "max_rows_for_1h_calc": 200000,
    "ai_sample_rows": 800,
    "alerts": {
        "rejected_rate_threshold": 0.15,
        "pending_rate_threshold": 0.30,
        "spike_zscore": 3.0,
        "hour_spike_ratio": 2.5,
        "top_processor_share_threshold": 0.55,
    },
    "charts": {
        "enabled": True,
        "top_n_categories": 10,
        "skip_histograms": True,
        "max_charts_per_file": 50,
    },
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 465,
        "use_tls": True,
        "username": "elymar.santos@anagaming.com.br",
        "from": "elymar.santos@anagaming.com.br",
        "recipients": [
            "elymar.santos@anagaming.com.br",
            "matheus.novaes@anagaming.com.br",
        ],
        "subject_prefix": "[CMD Análises]",
    },
"openai_primary": {
    "enabled": True,
    "model": "gpt-4.1",   # modelo forte para análise
    "max_output_tokens": 700,
    "base_url": "https://api.openai.com/v1",
},

"openai_automation": {
    "enabled": True,
    "model": "gpt-4.1-mini",   # modelo barato para tarefas leves
},

    "daily_report": {
        "enabled": True,
        "hour": 18,
    },
}

OPENAI_API_KEY = OPENAI_API_KEY

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_SSL = os.getenv("SMTP_SSL", "true").lower() in ("1", "true", "yes")

SMTP_USER = os.getenv("SMTP_USER", "elymar.santos@anagaming.com.br")
SMTP_PASS = os.getenv("SMTP_PASS", "amek ncuj oatc eogi")

EMAIL_TO = os.getenv(
    "EMAIL_TO",
    "elymar.santos@anagaming.com.br,matheus.novaes@anagaming.com.br",
)

EMAIL_SUBJECT_PREFIX = os.getenv(
    "EMAIL_SUBJECT_PREFIX",
    "[CMD Análises] ",
)

ALWAYS_RECIPIENTS = [
    "elymar.santos@anagaming.com.br",
    "matheus.novaes@anagaming.com.br",
]

CRITICAL_ONLY_RECIPIENTS = [
    "nathalia.merij@anagaming.com.br",
    "elymar.fonseca@gmail.com",
]

BASELINE_PATH = "cmd_baselines.json"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def setup_logger(output_folder: str) -> logging.Logger:
    ensure_dir(output_folder)
    log_path = os.path.join(output_folder, "cmd_analises.log")

    logger = logging.getLogger("CMDAnalises")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# -----------------------------
# Utils de DataFrame / CSV
# -----------------------------

def clean_ptbr_currency(val) -> float:
    """
    Converte 'R$ 1.200,50', 'Sem Freespin', '100', etc. para float puro.
    Nova função adicionada para tratar colunas sujas de Rollback/Freespin.
    """
    if pd.isna(val):
        return 0.0
    s = str(val).lower().strip()
    
    # Lista de termos que indicam zero ou nulo
    if s in ["", "-", "sem freespin", "sem rollback", "null", "nan", "none"]:
        return 0.0
    
    try:
        # Remove R$ e espaços
        s = s.replace("r$", "").replace(" ", "")
        
        # Lógica para converter formato BR (1.000,00) para US (1000.00)
        if "," in s and "." in s:
            s = s.replace(".", "")  # Tira ponto de milhar
            s = s.replace(",", ".") # Troca vírgula por ponto decimal
        elif "," in s:
            s = s.replace(",", ".")
            
        return float(s)
    except Exception:
        return 0.0


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def fig_to_base64(fig: Figure) -> str:
    """
    VOLTAR AO PADRÃO SIMPLES:
    - Gera PNG em memória.
    - Retorna APENAS a string base64.
    - Usado direto no HTML: <img src="data:image/png;base64,{{ img_b64 }}">
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_value_counts(s: pd.Series, topn: int = 10) -> pd.Series:
    vc = s.value_counts(dropna=True)
    if topn is not None:
        vc = vc.head(topn)
    return vc


def files_modified_today(paths: List[str]) -> List[str]:
    today = date.today()
    out: List[str] = []
    for p in paths:
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(p)).date()
            if mtime == today:
                out.append(p)
        except Exception:
            continue
    return out


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_lower_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Versão mais robusta: tenta match exato e depois substring.
    Evita quebrar comportamento antigo, mas melhora acerto para colunas tipo user_id_x.
    """
    cols_map = {str(c).lower(): c for c in df.columns}
    # 1) match exato
    for cand in candidates:
        key = cand.lower()
        if key in cols_map:
            return cols_map[key]
    # 2) substring
    for cand in candidates:
        key = cand.lower()
        for c_low, original in cols_map.items():
            if key in c_low:
                return original
    return None


def coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def read_csv_smart(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    seps = [",", ";", "\t", "|"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(path, sep=sep, encoding=enc)
            except Exception as e:
                last_err = e
                continue
    raise last_err

# -----------------------------
# Utils de IMAGEM / OCR
# -----------------------------


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def is_image_path(path: str) -> bool:
    """
    Indica se o caminho aponta para uma imagem suportada (JPG/PNG).
    """
    return Path(path).suffix.lower() in IMG_EXTS


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Pré-processa prints de dashboard (fundo escuro) para melhorar o OCR.
    Simula o cuidado de um analista sênior em aumentar contraste e legibilidade.
    """
    img = img.convert("L")  # grayscale
    img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda x: 0 if x < 128 else 255, "1")
    return img


def ocr_dashboard(path: str, lang: str = "por+eng") -> str:
    """
    Extrai o texto principal do dashboard a partir de um JPG/PNG.
    """
    img = Image.open(path)
    img = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(img, lang=lang)
    return text


# -----------------------------
# Inteligência sobre o TIPO do PAINEL
# -----------------------------


def painel_from_ocr_text(txt: str) -> str:
    """
    Identifica o 'tipo de painel' só pelo texto extraído:
    Apostas e Ganhos, Rollback, Saques, Depósitos, KYC, Freespins, etc.
    É o análogo visual do guess_process_type para CSV.
    """
    low = txt.lower()

    # termos mais específicos primeiro
    if "freespin" in low or "free spin" in low or "bonus" in low:
        return "freespins"
    if "apostas e ganhos" in low or "aposta e ganh" in low or "ggr" in low:
        return "apostas_ganhos"
    if "rollback" in low:
        return "rollback"
    if "saques" in low or "saque" in low or "withdraw" in low or "cashout" in low:
        return "saques"
    if "depósitos" in low or "depositos" in low or "deposit" in low or "pixin" in low:
        return "depositos"
    if "kyc" in low or "onboarding" in low or "liveness" in low:
        return "kyc"
    return "dashboard"


# -----------------------------
# Extração de KPIs do texto OCR
# -----------------------------


_NUM_PATTERN = r"([\dR\$\.,]+)"


def _find_num(txt: str, pattern: str, default: str = "") -> str:
    m = re.search(pattern, txt, flags=re.IGNORECASE)
    if not m:
        return default
    return m.group(1).strip().replace(" ", "")


def extract_kpis_from_ocr(txt: str) -> Dict[str, Any]:
    """
    Extrai KPIs genéricos dos dashboards a partir do texto OCR.

    Retorna:
      - painel_tipo
      - janela
      - bet_acum_1h / win_acum_1h
      - status_min / status_max
      - rollback_observado / rollback_esperado (quando presente)
      - excesso_percent (quando der para calcular)
    """
    painel_tipo = painel_from_ocr_text(txt)

    # janela textual (Last X hours, Last 24 hours, etc.)
    janela = _find_num(txt, r"Last\s+([\d\s\w]+hours?)", default="Última 1 hora")

    bet_1h = _find_num(txt, rf"BET\s+ACUM.*?{_NUM_PATTERN}")
    win_1h = _find_num(txt, rf"WIN\s+ACUM.*?{_NUM_PATTERN}")

    status_min = "UP" if re.search(r"m[ií]n.*?UP", txt, flags=re.IGNORECASE) else ""
    status_max = "UP" if re.search(r"m[aá]x.*?UP", txt, flags=re.IGNORECASE) else ""

    # painéis tipo rollback observado vs esperado
    rollback_obs = _find_num(txt, rf"(?:rollback|rb)\s*observado.*?{_NUM_PATTERN}")
    rollback_exp = _find_num(txt, rf"(?:esperado|baseline).*?{_NUM_PATTERN}")

    excesso_percent: Optional[float] = None
    try:
        if rollback_obs and rollback_exp:
            def _to_float(v: str) -> float:
                v = v.replace("R$", "").replace(".", "").replace(" ", "")
                v = v.replace(",", ".")
                return float(v)

            obs = _to_float(rollback_obs)
            exp = _to_float(rollback_exp)
            if exp > 0:
                excesso_percent = (obs / exp - 1.0) * 100.0
    except Exception:
        excesso_percent = None

    return {
        "painel_tipo": painel_tipo,
        "janela": janela,
        "bet_acum_1h": bet_1h,
        "win_acum_1h": win_1h,
        "status_min": status_min,
        "status_max": status_max,
        "rollback_observado": rollback_obs,
        "rollback_esperado": rollback_exp,
        "excesso_percent": excesso_percent,
    }


def build_fake_csv_from_ocr(txt: str) -> str:
    """
    Cria um pseudo-CSV com KPIs chave lidos do painel.
    Isso permite reaproveitar a IA e o cérebro analítico como se fosse um arquivo tabular.
    """
    kpis = extract_kpis_from_ocr(txt)

    header = (
        "painel_tipo,janela,bet_acum_1h,win_acum_1h,"
        "status_min,status_max,rollback_observado,rollback_esperado,excesso_percent"
    )

    row = (
        f"{kpis.get('painel_tipo','')},"
        f"{kpis.get('janela','')},"
        f"{kpis.get('bet_acum_1h','')},"
        f"{kpis.get('win_acum_1h','')},"
        f"{kpis.get('status_min','')},"
        f"{kpis.get('status_max','')},"
        f"{kpis.get('rollback_observado','')},"
        f"{kpis.get('rollback_esperado','')},"
        f"{'' if kpis.get('excesso_percent') is None else round(kpis['excesso_percent'], 1)}"
    )

    return header + "\n" + row

# -----------------------------
# Utils de IMAGEM / OCR
# -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_image_path(path: str) -> bool:
    """
    Indica se o caminho aponta para uma imagem suportada (JPG/PNG).
    """
    return Path(path).suffix.lower() in IMG_EXTS

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Pré-processa prints de dashboard (fundo escuro) para melhorar o OCR.
    Simula o cuidado de um analista sênior em aumentar contraste e legibilidade.
    """
    img = img.convert("L")  # grayscale
    img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.point(lambda x: 0 if x < 128 else 255, "1")
    return img

def ocr_dashboard(path: str, lang: str = "por+eng") -> str:
    """
    Extrai o texto principal do dashboard a partir de um JPG/PNG.
    """
    img = Image.open(path)
    img = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def painel_from_ocr_text(txt: str) -> str:
    """
    Identifica o 'tipo de painel' só pelo texto extraído: Apostas e Ganhos, Rollback, Saques, etc.
    É o análogo visual do guess_process_type para CSV.
    """
    low = txt.lower()
    if "apostas e ganhos" in low or "aposta e ganh" in low:
        return "apostas_ganhos"
    if "rollback" in low:
        return "rollback"
    if "saques" in low or "saque" in low or "withdraw" in low:
        return "saques"
    if "depósitos" in low or "depositos" in low or "deposit" in low:
        return "depositos"
    if "kyc" in low or "onboarding" in low:
        return "kyc"
    return "dashboard"

def build_fake_csv_from_ocr(txt: str) -> str:
    """
    Cria um pseudo-CSV com KPIs chave lidos do painel.
    Isso permite reaproveitar a IA e o cérebro analítico como se fosse um arquivo tabular.
    """
    import re

    def find_num(pattern: str, default: str = "") -> str:
        m = re.search(pattern, txt, flags=re.IGNORECASE)
        if not m:
            return default
        return m.group(1).strip().replace(" ", "")

    # bet/win acumulados em 1h, presentes na maioria dos seus dashboards 7K
    bet_1h = find_num(r"BET ACUM.*?([\dR\$\.,]+)")
    win_1h = find_num(r"WIN ACUM.*?([\dR\$\.,]+)")

    # status mínimo/máximo, se aparecerem como 'Mín Apostas: UP' etc.
    status_min = "UP" if re.search(r"m[ií]n.*?UP", txt, flags=re.IGNORECASE) else ""
    status_max = "UP" if re.search(r"m[aá]x.*?UP", txt, flags=re.IGNORECASE) else ""

    # janela textual (Last 1 hour, Last 24 hours, etc.)
    janela = find_num(r"Last\s+([\d\s\w]+hours?)", default="Última 1 hora")

    painel = painel_from_ocr_text(txt)

    header = "painel,janela,bet_acum_1h,win_acum_1h,status_min,status_max"
    row = f"{painel},{janela},{bet_1h},{win_1h},{status_min},{status_max}"

    return header + "\n" + row


# -----------------------------
# Inteligência sobre o TIPO do arquivo
# -----------------------------


def infer_process_type_from_name(filename: str) -> str:
    name = os.path.basename(filename).lower()

    if "rollback" in name or "roll-back" in name:
        return "rollback"

    if any(kw in name for kw in ["freespin", "freespins", "free-spin", "fs_", "fs-", "bonus"]):
        return "freespins"

    if any(kw in name for kw in ["saque", "saques", "withdraw", "cashout", "cash-out", "pix_out"]):
        return "saques"

    if any(kw in name for kw in ["deposit", "depósito", "deposito", "dep_", "charges", "charge", "pix_in"]):
        return "depositos"

    if any(kw in name for kw in ["kyc", "liveness", "onboarding", "documento", "identidade"]):
        return "kyc"

    return "analises"


def process_type_label(tipo: str) -> str:
    mapping = {
        "rollback": "Relatório sobre rollback",
        "freespins": "Relatório sobre freespins/bonus",
        "saques": "Relatório sobre saques",
        "depositos": "Relatório sobre depósitos",
        "kyc": "Relatório sobre KYC",
        "analises": "Relatório de análises gerais",
        "geral": "Relatório de análises gerais",
        "bonus/freespin": "Relatório sobre freespins/bonus",
    }
    return mapping.get(tipo, "Relatório de análises gerais")


def guess_process_type(filename: str, df: pd.DataFrame) -> str:
    tipo = infer_process_type_from_name(filename)
    if tipo != "analises":
        return tipo

    name = filename.lower()
    cols_join = " ".join([str(c).lower() for c in df.columns])

    kyc_keys = [
        "kyc", "liveness", "onboarding", "status_kyc", "kyc_status",
        "kyc_processos", "document"
    ]
    if any(k in name for k in kyc_keys) or any(k in cols_join for k in kyc_keys):
        return "kyc"

    if any(k in name for k in ["withdraw", "saque", "cashout", "pix_out"]) \
       or any(k in cols_join for k in ["withdraw", "saque"]):
        return "saques"

    if any(k in name for k in ["charge", "deposit", "depósito", "deposito", "pix_in"]) \
       or any(k in cols_join for k in ["charge", "deposit"]):
        return "depositos"

    if "rollback" in name or "rollback" in cols_join or "casino_transaction" in cols_join:
        return "rollback"

    if any(kw in name for kw in ["freespin", "freespins", "free_spin", "bonus"]) \
       or any(kw in cols_join for kw in ["freespin", "bonus"]):
        return "freespins"

    return "analises"

# -----------------------------
# Baselines
# -----------------------------


def load_baselines() -> Dict[str, Any]:
    if not os.path.exists(BASELINE_PATH):
        return {}
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_baselines(baselines: Dict[str, Any]) -> None:
    try:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baselines, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# -----------------------------
# Alertas e Score (pontos críticos)
# -----------------------------


@dataclass
class Alert:
    level: str   # green / yellow / red
    title: str
    detail: str
    file: str
    column: str = ""


def build_risk_score(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score por usuário + casa (brand).
    CORREÇÃO: Remove travas que escondiam a tabela e normaliza o ID do usuário.
    """
    df = df.copy()
    summary: Dict[str, Any] = {}

    # 1. Mapeamento de colunas expandido
    user_col = find_col(df, [
        "user_id", "id_usuario", "player_id", "userid", "user", 
        "jogador", "customer_id", "account_id", "login", "username"
    ])
    
    brand_col = find_col(df, ["brand", "brand_id", "brands", "casa", "crm_brand_id"])
    amount_col = find_col(df, ["amount", "valor", "value", "profit", "win", "ganho", "total", "net"])
    status_col = find_col(df, ["status", "state", "resultado", "kyc_status", "withdraw_status"])
    rollback_col = find_col(df, ["rollback", "rollback_count", "qtd_rollback"])
    freespin_col = find_col(df, ["freespin", "freespins", "free_spins", "free_spin", "fs_qtd", "bonus_qtd"])
    created_col = find_col(df, ["created_at", "data", "timestamp", "dt_created", "cadastro_at"])

    # Se não achar coluna de usuário, não tem como gerar score
    if not user_col:
        return df, {"risk_users": []}

    # Tratamento da Brand (Casa)
    if brand_col:
        df[brand_col] = df[brand_col].fillna("Sem casa")
        # Normaliza string e remove .0
        df[brand_col] = df[brand_col].astype(str).str.strip().str.replace(".0", "", regex=False)
        mapping = {"1": "7k", "3": "vera", "4": "cassino"}
        df[brand_col] = df[brand_col].replace(mapping)
    else:
        df["__brand_tmp__"] = "Geral"
        brand_col = "__brand_tmp__"

    # Agrupamento Principal
    grp = df.groupby(user_col, dropna=False)

    # --- CÁLCULO DE MÉTRICAS ---
    
    # Inicializa DataFrame de risco com o índice sendo o ID do usuário
    risk = pd.DataFrame(index=grp.groups.keys())
    risk.index.name = user_col 
    
    # 1) Volume de linhas
    risk["volume_linhas"] = grp.size()

    # 2) Brand principal (pega a mais frequente)
    try:
        # Método otimizado para pegar a moda
        risk["brand"] = df.groupby(user_col)[brand_col].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else "N/A")
    except Exception:
        risk["brand"] = "N/A"

    # 3) Soma de valor (Z-Score)
    if amount_col and is_numeric(df[amount_col]):
        risk["soma_valor"] = grp[amount_col].sum(min_count=1).fillna(0)
        mu = risk["soma_valor"].mean()
        sd = risk["soma_valor"].std(ddof=0)
        if sd == 0: sd = 1
        risk["z_valor"] = (risk["soma_valor"] - mu) / sd
    else:
        risk["soma_valor"] = 0.0
        risk["z_valor"] = 0.0

    # 4) Taxa de Rejeição
    if status_col:
        st = to_lower_str(df[status_col])
        df["_is_rejected_"] = st.isin(
            ["rejected", "negado", "denied", "failed", "refused", "2", "5", "❌ negado", "reprovado", "error"]
        ).astype(int)
        risk["rejected_rate"] = df.groupby(user_col)["_is_rejected_"].mean()
    else:
        risk["rejected_rate"] = 0.0

    # 5) Rollbacks
    if rollback_col:
        if is_numeric(df[rollback_col]):
            risk["rollback_qtd"] = grp[rollback_col].sum().fillna(0)
        else:
            df["_rb_flag_"] = df[rollback_col].notna().astype(int)
            risk["rollback_qtd"] = df.groupby(user_col)["_rb_flag_"].sum()
    else:
        risk["rollback_qtd"] = 0.0

    # 6) Freespins
    if freespin_col:
        if is_numeric(df[freespin_col]):
            risk["freespins_qtd"] = grp[freespin_col].sum().fillna(0)
        else:
            df["_fs_flag_"] = df[freespin_col].notna().astype(int)
            risk["freespins_qtd"] = df.groupby(user_col)["_fs_flag_"].sum()
    else:
        risk["freespins_qtd"] = 0.0

    # 7) Max Ganho em 1h
    risk["max_ganho_1h"] = 0.0
    # Mantendo a lógica original se houver colunas, mas garantindo default 0.0
    if (created_col and amount_col and is_numeric(df[amount_col]) 
        and len(df) <= cfg.get("max_rows_for_1h_calc", 200_000)):
        try:
            dt = coerce_datetime(df[created_col])
            df["_dt_"] = dt
            
            def max_1h_sum(g):
                g = g.dropna(subset=["_dt_"]).sort_values("_dt_")
                if g.empty: return 0.0
                vals = g[amount_col].values
                times = g["_dt_"].values.astype("datetime64[ns]")
                max_sum = 0.0
                j = 0
                for i in range(len(g)):
                    while (times[i] - times[j]) > np.timedelta64(1, "h"):
                        j += 1
                    s = vals[j: i + 1].sum()
                    if s > max_sum: max_sum = s
                return float(max_sum)

            max1h = df.groupby(user_col).apply(max_1h_sum, include_groups=False)
            risk["max_ganho_1h"] = max1h
        except Exception:
            pass # Falha silenciosa no calculo complexo para não parar o script

    # --- CÁLCULO DO SCORE FINAL ---
    # Normalização por Rank. Se só tiver 1 usuário, o rank funciona sem quebrar.
    def calc_rank(s):
        if s.nunique() <= 1: return 0.5 
        return s.rank(pct=True)

    risk["risk_score"] = (
        calc_rank(risk["volume_linhas"]) * 0.30 +
        calc_rank(risk["z_valor"].clip(lower=0)) * 0.35 +
        calc_rank(risk["rejected_rate"]) * 0.20 +
        calc_rank(risk["rollback_qtd"]) * 0.05 +
        calc_rank(risk["freespins_qtd"]) * 0.10
    )

    # Preenchimento de nulos no score final
    risk["risk_score"] = risk["risk_score"].fillna(0)

    # Ordena e pega Top 20 (SEMPRE RETORNA, SEM TRAVAS)
    top_risk = risk.sort_values("risk_score", ascending=False).head(20)
    
    # Reseta o index para transformar o ID do usuário em coluna
    top_risk = top_risk.reset_index()
    
    # Truque para garantir que o HTML mostre o ID:
    # O HTML procura por u.user_id, u.id_usuario, etc.
    # Vamos forçar uma coluna 'user_id' com o valor do ID encontrado.
    top_risk["user_id"] = top_risk[user_col]

    summary["risk_users"] = top_risk.to_dict(orient="records")
    return df, summary

def detect_alerts_and_insights(df: pd.DataFrame, cfg: dict, filename: str) -> Tuple[List[Alert], Dict[str, Any]]:
    alerts: List[Alert] = []
    insights: Dict[str, Any] = {}

    df = normalize_cols(df)
    proc_type = guess_process_type(filename, df)
    insights["process_type_guess"] = proc_type

    # --- CONFIGURAÇÕES ---
    rejected_threshold = cfg["alerts"].get("rejected_rate_threshold", 0.15)
    pending_threshold = cfg["alerts"].get("pending_rate_threshold", 0.30)
    spike_z = cfg["alerts"].get("spike_zscore", 3.0)

    # --- MAPEAMENTO DE COLUNAS ---
    status_col = find_col(df, ["status", "kyc_status_legivel", "status_kyc", "state", "resultado", "kyc_status_descricao", "financeiro_status"])
    brand_col = find_col(df, ["brand", "brand_id", "casa", "crm_brand_id", "brands"])
    proc_col = find_col(df, ["processor", "processor_name", "processadora", "operadora", "gateway", "financeiro_processor", "kyc_operator"])
    amount_col = find_col(df, ["amount", "valor", "value", "profit", "win", "ganho", "deposit_amount_value", "withdraw_amount_value", "financeiro_valor", "casino_valor"])
    game_col = find_col(df, [
    "game", "game_name", "provider_game", "nome_jogo", "primeiro_jogo", "casino_game_name",
    "jogo", "jogo_name", "jogo_nome", "title", "game_title", "slot_name", "slot", 
    "game_id_name", "top_game_name", "game_category", "categoria_jogo",
    "jogo_atual", "current_game", "last_game", "jogo_principal", "main_game"
])

    created_col = find_col(df, ["created_at", "data_cadastro", "kyc_data", "timestamp", "dt_created"])
    reason_col = find_col(df, ["reason", "motivo", "reject_reason", "kyc_reason", "liveness_reason", "motivo_provavel"])
    kyc_op_col = find_col(df, ["operation_type", "kyc_type", "kyc_event", "flow", "tipo_processo", "operation", "tipo", "etapa", "kyc_operation_type", "casino_type", "saque_type"])
    
    # Colunas específicas para Rollback e Freespin
    rollback_val_col = find_col(df, ["rollback_valor", "rollback_amount", "valor_rollback"])
    freespin_val_col = find_col(df, ["freespin_valor", "freespin_amount", "valor_freespin"])

    # Normalização
    if brand_col: df[brand_col] = df[brand_col].fillna("Indefinido").astype(str)
    if proc_col: df[proc_col] = df[proc_col].fillna("Indefinido").astype(str)

    # --- 1. STATUS GERAL ---
    rejected_rate = 0.0
    pending_rate = 0.0
    if status_col:
        st = to_lower_str(df[status_col])
        list_rej = ["rejected", "negado", "denied", "failed", "refused", "2", "5", "❌ negado", "reprovado", "error", "cancelado"]
        list_pen = ["pending", "pendente", "processing", "0", "4", "em análise", "analise", "🟡 em análise", "processando", "aguardando"]
        
        df["_is_rejected_"] = st.isin(list_rej).astype(int)
        df["_is_pending_"] = st.isin(list_pen).astype(int)
        
        rejected_rate = df["_is_rejected_"].mean()
        pending_rate = df["_is_pending_"].mean()
        
        insights["rejected_rate"] = rejected_rate
        insights["pending_rate"] = pending_rate

        if rejected_rate >= rejected_threshold:
            alerts.append(Alert(level="red" if rejected_rate >= rejected_threshold * 2 else "yellow",
                title=f"Rejeição Geral Alta ({proc_type})",
                detail=f"Taxa global: {rejected_rate:.1%} (limite {rejected_threshold:.1%})",
                file=filename, column=status_col))

    # --- 2. NOVOS INSIGHTS POR BRAND (ROLLBACK E FREESPIN) --- # NOVO
    if brand_col:
        # Rollback por Brand
        if rollback_val_col and is_numeric(df[rollback_val_col]):
            rb_stats = df.groupby(brand_col)[rollback_val_col].agg(['sum', 'count']).reset_index()
            insights["rollback_por_brand"] = rb_stats.set_index(brand_col).to_dict(orient='index')
        
        # Freespin por Brand
        if freespin_val_col and is_numeric(df[freespin_val_col]):
            fs_stats = df.groupby(brand_col)[freespin_val_col].agg(['sum', 'count']).reset_index()
            insights["freespin_por_brand"] = fs_stats.set_index(brand_col).to_dict(orient='index')

    # --- 3. INSIGHTS PADRÃO (Mantido lógica anterior) ---
    if brand_col:
        insights["brand_volume"] = df[brand_col].value_counts().head(10).to_dict()
    
    if proc_col:
        insights["top_processors_by_volume"] = df[proc_col].value_counts().head(10).to_dict()

    if kyc_op_col and brand_col:
        try:
            # Insight cruzado para IA entender onde está o problema
            kyc_pivot = df.groupby([brand_col, kyc_op_col]).size().unstack(fill_value=0)
            insights["kyc_ops_by_brand_type"] = kyc_pivot.to_dict(orient="index")
        except: pass

    # --- 4. PICOS DE VALOR ---
    if amount_col and is_numeric(df[amount_col]) and len(df) > 10:
        val_series = df[amount_col].fillna(0)
        z = (val_series - val_series.mean()) / (val_series.std(ddof=0) + 1e-9)
        max_z = float(z.max())
        insights["max_z_amount"] = max_z
        insights["financeiro_total"] = float(val_series.sum())

    return alerts, insights




# -----------------------------
# Resumo crítico
# -----------------------------


def build_critical_info(process_type: str, process_label: str, alerts: List[Alert]) -> Dict[str, str]:
    if not alerts:
        return {
            "level": "green",
            "title": "Sem pontos críticos",
            "headline": f"{process_label} — Sem pontos críticos relevantes",
        }

    priority = {"red": 3, "yellow": 2, "green": 1}
    best = max(alerts, key=lambda a: priority.get(a.level, 0))

    emoji = "🟥" if best.level == "red" else "🟨" if best.level == "yellow" else "🟩"
    headline = f"{emoji} {process_label} — {best.title}"

    return {
        "level": best.level,
        "title": best.title,
        "headline": headline,
    }


# --- INÍCIO DA FUNÇÃO DE ANÁLISES POR BRAND (SAFE MODE) ---
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg') # Garante que não abra janelas
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64

def gerar_analises_graficas(df):
    """
    Gera gráficos Cruzados (Brand x Status, Proc x Erro), Top Jogos por Financeiro
    e Distribuição de Free Spins e Rollbacks (Comparativo).
    """
    plt.switch_backend('Agg') 
    sns.set_theme(style="dark", rc={"axes.facecolor": "#121829", "figure.facecolor": "#0B0F19", "text.color": "#E6E9EF"})
    
    # --- ATUALIZAÇÃO DA PALETA DE CORES ---
    colors_map = {
        '7k': '#A0C62E',       # Verde
        'a7k': '#A0C62E',  
        '7kbet': '#A0C62E',
        'cassino': '#2F78D7',  # Azul
        'cassinobet': '#2F78D7',
        'vera': '#40DF4A',     # Verde
        'verabet': '#40DF4A',
        'outros': "#4E0739"
    }

    def get_color_dict(series):
        unique_brands = series.unique()
        palette = {}
        for b in unique_brands:
            key = str(b).lower().strip().replace(".0", "")
            if '7k' in key: val = colors_map['7k']
            elif 'cassino' in key: val = colors_map['cassino']
            elif 'vera' in key: val = colors_map['vera']
            else: val = colors_map['outros']
            palette[b] = val
        return palette

    # Helper para encontrar colunas (caso não seja global)
    def find_col_local(df, candidates):
        for c in df.columns:
            if str(c).lower() in candidates: return c
            if any(cand in str(c).lower() for cand in candidates): return c
        return None

    def is_numeric_local(series):
        return pd.api.types.is_numeric_dtype(series)

    imgs_b64 = {}

    def save_img():
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    try:
        # 1. PREPARAÇÃO
        if 'brand' not in df.columns: df['brand'] = 'Geral'
        df['brand'] = df['brand'].fillna("Indefinido").astype(str)
        
        # --- 1.1 VOLUME TOTAL POR CASA ---
        try:
            plt.figure(figsize=(8, 4))
            vol = df['brand'].value_counts().reset_index()
            vol.columns = ['brand', 'count']
            custom_pal = get_color_dict(vol['brand'])
            sns.barplot(data=vol, x='brand', y='count', hue='brand', palette=custom_pal, legend=False)
            plt.title('Volume Total de Operações por Casa')
            imgs_b64['volume_brand'] = save_img()
        except Exception as e:
            print(f"Erro grafico volume brand: {e}")

        # 2. FINANCEIRO CRUZADO (Brand x Status)
        cols_val = [c for c in df.columns if 'valor' in c or 'amount' in c]
        if cols_val:
            v_col = cols_val[0]
            if 'status' in df.columns:
                agg = df.groupby(['brand', 'status'])[v_col].sum().reset_index()
                top_s = agg.groupby('status')[v_col].sum().nlargest(3).index
                agg = agg[agg['status'].isin(top_s)]
                
                plt.figure(figsize=(9, 4))
                sns.barplot(data=agg, x='brand', y=v_col, hue='status', palette='viridis')
                plt.title(f'Volume Financeiro por Status ({v_col})')
                imgs_b64['financeiro_cruzado'] = save_img()

        # 3. HEATMAP DE ERROS
        proc_col = find_col_local(df, ['processor', 'processadora'])
        status_col = find_col_local(df, ['status', 'state'])
        
        if proc_col and status_col:
            ct = pd.crosstab(df[proc_col].fillna("N/A"), df[status_col].fillna("N/A"))
            if not ct.empty and ct.shape[0] > 1:
                plt.figure(figsize=(10, 5))
                sns.heatmap(ct, annot=True, fmt='d', cmap="Reds", cbar=False)
                plt.title(f'Mapa de Calor: {proc_col} vs {status_col}')
                plt.ylabel('')
                imgs_b64['heatmap_erros'] = save_img()

        # 4. PARETO DE MOTIVOS
        reason_col = find_col_local(df, ['reason', 'motivo', 'error'])
        if reason_col:
            top_r = df[reason_col].value_counts().head(15)
            if not top_r.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=top_r.values, y=top_r.index.astype(str), palette="magma")
                plt.title('Top 15 Motivos (Sem Filtros)')
                imgs_b64['pareto_motivos'] = save_img()

        # 5. TOP JOGOS POR BRAND (FINANCEIRO)
        rb_val_col = find_col_local(df, ["rollback_valor", "valor_rollback", "rollback_amount"])
        fs_val_col = find_col_local(df, ["freespin_valor", "valor_freespin", "freespin_amount"])
        game_col = find_col_local(df, ["game", "nome_jogo", "jogo_atual", "provider_game"])

        def plot_top_games_financeiro(val_column, label_metric):
            if val_column and game_col:
                # Tenta limpar moeda se a função existir globalmente, senão usa lambda simples
                try:
                    df[f"_clean_{label_metric}"] = df[val_column].apply(clean_ptbr_currency)
                except:
                    df[f"_clean_{label_metric}"] = pd.to_numeric(df[val_column], errors='coerce').fillna(0)
                
                df_filt = df[df[f"_clean_{label_metric}"] > 0].copy()
                
                if not df_filt.empty:
                    top_per_brand = (df_filt.groupby(['brand', game_col])[f"_clean_{label_metric}"]
                                     .sum().reset_index()
                                     .sort_values(['brand', f"_clean_{label_metric}"], ascending=[True, False])
                                     .groupby('brand').head(5))
                    
                    if not top_per_brand.empty:
                        plt.figure(figsize=(10, 6))
                        chart = sns.barplot(
                            data=top_per_brand, y=game_col, x=f"_clean_{label_metric}",
                            hue='brand', palette=get_color_dict(top_per_brand['brand'])
                        )
                        plt.title(f'Top 5 Jogos com Maior {label_metric.title()} (Financeiro)')
                        plt.xlabel('Valor (R$)')
                        plt.ylabel('')
                        for container in chart.containers:
                            labels = [f'{v.get_width():.0f}' if v.get_width() > 0 else '' for v in container]
                            chart.bar_label(container, labels=labels, padding=3, color='white', fontsize=8)
                        
                        imgs_b64[f'top_games_{label_metric}'] = save_img()

        if rb_val_col: plot_top_games_financeiro(rb_val_col, "rollback")
        if fs_val_col: plot_top_games_financeiro(fs_val_col, "freespin")

        # -------------------------------------------------------
        # === 6. NOVO: DISTRIBUIÇÃO DE FREE SPINS E ROLLBACKS (QUANTIDADE) ===
        # -------------------------------------------------------
        if game_col:
            df_game = df.copy()
            
            # --- Tenta encontrar contagem de Rollback ---
            rb_qty_col = find_col_local(df, ["rollback_count", "qtd_rollback", "rollback_qtd"])
            if rb_qty_col and is_numeric_local(df_game[rb_qty_col]):
                df_game['_rb_qtd'] = df_game[rb_qty_col].fillna(0)
            else:
                # Se for um arquivo específico de rollback, conta linhas
                rb_flag = find_col_local(df, ["rollback", "is_rollback"])
                if rb_flag:
                     df_game['_rb_qtd'] = df_game[rb_flag].apply(lambda x: 1 if str(x).lower() in ['1', 'true', 'sim'] else 0)
                elif "rollback" in str(df.columns).lower(): # fallback heurístico
                     df_game['_rb_qtd'] = 0 # não arriscar contar tudo se não tiver certeza
                else:
                     df_game['_rb_qtd'] = 0

            # --- Tenta encontrar contagem de Freespin ---
            fs_qty_col = find_col_local(df, ["freespin_count", "qtd_freespin", "freespin_qtd", "fs_qtd"])
            if fs_qty_col and is_numeric_local(df_game[fs_qty_col]):
                df_game['_fs_qtd'] = df_game[fs_qty_col].fillna(0)
            else:
                fs_flag = find_col_local(df, ["freespin", "free_spin", "is_freespin"])
                if fs_flag:
                    df_game['_fs_qtd'] = df_game[fs_flag].apply(lambda x: 1 if str(x).lower() in ['1', 'true', 'sim'] else 0)
                else:
                    df_game['_fs_qtd'] = 0

            # Agrupa e Soma
            game_stats = df_game.groupby(game_col)[['_rb_qtd', '_fs_qtd']].sum().reset_index()
            game_stats['Total'] = game_stats['_rb_qtd'] + game_stats['_fs_qtd']
            
            # Pega Top 15 Jogos com mais eventos
            top_games_qtd = game_stats.sort_values('Total', ascending=False).head(15)
            
            if not top_games_qtd.empty and top_games_qtd['Total'].sum() > 0:
                # Melt para formato longo (Seaborn friendly)
                df_melted = top_games_qtd.melt(id_vars=[game_col], value_vars=['_fs_qtd', '_rb_qtd'], 
                                               var_name='Tipo', value_name='Qtd')
                
                # Renomeia para legenda
                df_melted['Tipo'] = df_melted['Tipo'].replace({'_fs_qtd': 'Free Spin', '_rb_qtd': 'Rollback'})
                df_melted = df_melted[df_melted['Qtd'] > 0] # Remove barras vazias

                if not df_melted.empty:
                    plt.figure(figsize=(10, max(6, len(top_games_qtd)*0.4)))
                    
                    # Cores: Azul Céu (FS) e Vermelho Coral (RB)
                    colors_comp = {'Free Spin': '#87CEEB', 'Rollback': '#FF6F61'}
                    
                    chart = sns.barplot(
                        data=df_melted, y=game_col, x='Qtd', hue='Tipo', 
                        palette=colors_comp, edgecolor="#0B0F19"
                    )
                    
                    plt.title('Distribuição de Free Spins e Rollbacks por Jogo (Qtd)', color='white', weight='bold')
                    plt.xlabel('Quantidade de Eventos', color='#aab2c5')
                    plt.ylabel('')
                    plt.legend(title=None, frameon=False, labelcolor='white')
                    
                    for container in chart.containers:
                        labels = [f'{int(v.get_width())}' if v.get_width() > 0 else '' for v in container]
                        chart.bar_label(container, labels=labels, padding=3, color='white', fontsize=9, weight='bold')

                    sns.despine(left=True, bottom=False)
                    imgs_b64['distribuicao_rb_fs'] = save_img()

    except Exception as e:
        print(f"Erro grafico cruzado: {e}")

    return imgs_b64
# --- FIM DA FUNÇÃO ---



import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
import io
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
import io
from typing import List, Dict, Any

def make_charts(
    df: pd.DataFrame,
    cfg: dict,
    filename: str,
    process_type: str,
    insights: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Gera gráficos automáticos com QUEBRA POR BRAND e RÓTULOS DE DADOS.
    Inclui análise específica de Jogos por Rollback e Freespin (Qtd e Valor).
    """
    charts: List[Dict[str, Any]] = []
    
    # Verifica se gráficos estão habilitados na config
    if not cfg.get("charts", {}).get("enabled", True):
        return charts

    # Configuração Visual Dark (Backend Agg para não travar servidores)
    plt.switch_backend('Agg')
    sns.set_theme(style="dark", rc={
        "axes.facecolor": "#121829",
        "figure.facecolor": "#0B0F19",
        "text.color": "#E6E9EF",
        "axes.labelcolor": "#E6E9EF",
        "xtick.color": "#E6E9EF",
        "ytick.color": "#E6E9EF",
        "grid.color": "#222A3A",
        "axes.edgecolor": "#222A3A"
    })
    
    # --- PALETA DE CORES DAS BRANDS ---
    brand_colors = {
        '7k': '#A0C62E',       # Verde limão
        '7kbet': '#A0C62E',
        'cassino': '#2F78D7',  # Azul
        'cassino_pix': '#2F78D7',
        'vera': '#40DF4A',     # Verde folha
        'vera&john': '#40DF4A',
        'outros': "#BAD10A",   # Cinza
        'indefinido': "#B10B0B"
    }

    def get_palette(unique_brands):
        pal = []
        for b in unique_brands:
            s = str(b).lower().strip().replace(".0", "")
            if '7k' in s: c = brand_colors['7k']
            elif 'cassino' in s: c = brand_colors['cassino']
            elif 'vera' in s: c = brand_colors['vera']
            else: c = brand_colors['outros']
            pal.append(c)
        return pal

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Função auxiliar para encontrar coluna
    def find_col(df, candidates):
        for c in df.columns:
            if str(c).lower() in candidates: return c
            if any(cand in str(c).lower() for cand in candidates): return c
        return None

    # Identifica a coluna de BRAND
    brand_col = find_col(df, ["brand", "brand_id", "casa", "crm_brand_id", "brands"])
    
    if brand_col:
        df[brand_col] = df[brand_col].fillna("Indefinido").astype(str).str.replace(".0", "")
    else:
        df["_brand_tmp_"] = "Geral"
        brand_col = "_brand_tmp_"

    # =================================================================
    # 1. STACKED BARS (CATEGÓRICAS) COM LABELS
    # =================================================================
    priority_cols = ['status', 'processor', 'reason', 'kyc_status', 'operation_type', 'motivo', 'saque_motivo', 'kyc_tipo']
    
    cols_to_plot = []
    for c in df.columns:
        c_low = str(c).lower()
        if c == brand_col or "id" in c_low or "date" in c_low or "created" in c_low:
            continue
        # Removemos 'game' daqui para tratar especificamente na seção 4
        if "game" in c_low or "jogo" in c_low:
            continue
            
        if any(p in c_low for p in priority_cols) or (df[c].dtype == 'object' and 1 < df[c].nunique() < 40):
            cols_to_plot.append(c)
    
    cols_to_plot = list(set(cols_to_plot))[:15]

    for col in cols_to_plot:
        try:
            top_cats = df[col].value_counts().head(20).index
            df_plot = df[df[col].isin(top_cats)].copy()
            
            if not df_plot.empty:
                h = max(5, len(top_cats) * 0.5)
                fig, ax = plt.subplots(figsize=(11, h))
                
                sns.histplot(
                    data=df_plot, 
                    y=col, 
                    hue=brand_col, 
                    multiple="stack", 
                    palette=get_palette(df_plot[brand_col].unique()),
                    edgecolor="#0B0F19",
                    linewidth=0.5,
                    shrink=0.8,
                    ax=ax
                )
                
                # Labels
                for container in ax.containers:
                    labels = [f'{int(val)}' if (val := v.get_width()) > 0 else '' for v in container]
                    ax.bar_label(container, labels=labels, label_type='center', color='white', fontsize=8, weight='bold', padding=0)

                ax.set_title(f"Distribuição: {col}", color='white', weight='bold', loc='left')
                ax.set_xlabel("Volume", color='#aab2c5')
                ax.set_ylabel("")
                sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), frameon=False, title=None, ncol=3)
                sns.despine(left=True, bottom=False)
                
                charts.append({"title": f"Distribuição: {col}", "img_b64": fig_to_base64(fig)})
        except Exception as e:
            pass

    # =================================================================
    # 2. BOXPLOTS (NUMÉRICAS GERAIS)
    # =================================================================
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Removemos rollback e freespin daqui para tratar na seção 4
    target_num = [c for c in num_cols if any(x in str(c).lower() for x in ['amount', 'valor', 'ganho', 'win', 'bet'])]
    target_num = [c for c in target_num if not any(x in str(c).lower() for x in ['rollback', 'freespin', 'free_spin'])]
    
    for col in target_num:
        try:
            s = df[col].dropna()
            if not s.empty and s.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(data=df, x=col, y=brand_col, palette=get_palette(df[brand_col].unique()), ax=ax)
                ax.set_title(f"Dispersão: {col.upper()}", color='white', loc='left')
                sns.despine(left=True)
                charts.append({"title": f"Dispersão: {col}", "img_b64": fig_to_base64(fig)})
        except: pass

    # =================================================================
    # 3. LINHA TEMPORAL
    # =================================================================
    date_col = find_col(df, ["created_at", "data", "timestamp", "dt_created", "time"])
    if date_col:
        try:
            temp_df = df.copy()
            temp_df['_dt_plot'] = pd.to_datetime(temp_df[date_col], errors='coerce')
            temp_df = temp_df.dropna(subset=['_dt_plot'])
            
            if not temp_df.empty:
                vol_por_hora = temp_df.groupby([temp_df['_dt_plot'].dt.hour, brand_col]).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(10, 5))
                vol_por_hora.plot(kind='line', ax=ax, marker='o', linewidth=2)
                
                lines = ax.get_lines()
                for line, lbl in zip(lines, vol_por_hora.columns):
                    s = str(lbl).lower()
                    color = brand_colors['outros']
                    if '7k' in s: color = brand_colors['7k']
                    elif 'cassino' in s: color = brand_colors['cassino']
                    elif 'vera' in s: color = brand_colors['vera']
                    line.set_color(color)
                    
                    x_data, y_data = line.get_data()
                    for x, y in zip(x_data, y_data):
                        if y > 0:
                            ax.annotate(f'{int(y)}', xy=(x, y), xytext=(0, 5), textcoords='offset points', color=color, fontsize=8, weight='bold', ha='center')
                
                ax.set_xticks(range(0, 24))
                ax.set_title(f"Fluxo Horário por Brand", color='white', weight='bold')
                ax.legend(frameon=False, labelcolor='linecolor')
                sns.despine()
                charts.append({"title": "Fluxo Temporal", "img_b64": fig_to_base64(fig)})
        except: pass

    # =================================================================
    # 4. TOP JOGOS: ROLLBACK E FREESPIN (QTD E VALOR POR BRAND)
    # =================================================================
    
    # 1. Identificar coluna de Jogo
    game_col = find_col(df, ['game', 'game_name', 'nome_jogo', 'provider_game', 'casino_game_name'])
    
    if game_col:
        # Definição dos eixos de análise
        analises = [
            {
                "tipo": "Rollback",
                "col_val": find_col(df, ['rollback_valor', 'rollback_amount', 'valor_rollback']),
                "col_qtd": find_col(df, ['rollback_count', 'qtd_rollback', 'rollback']),
                "file_flag": "rollback" in filename.lower()
            },
            {
                "tipo": "Freespin",
                "col_val": find_col(df, ['freespin_valor', 'freespin_amount', 'valor_freespin', 'bonus_amount']),
                "col_qtd": find_col(df, ['freespin_count', 'qtd_freespin', 'freespin', 'free_spin']),
                "file_flag": "freespin" in filename.lower() or "bonus" in filename.lower()
            }
        ]

        for item in analises:
            # --- Lógica de Quantidade ---
            # Se tiver coluna de qtd, usa ela. Se não tiver, mas o arquivo for do tipo (ex: arquivo de rollback), conta as linhas.
            df_plot = pd.DataFrame()
            metric_col = None
            title_suffix = ""

            # Preparação dos dados para QTD
            if item["col_qtd"] and pd.api.types.is_numeric_dtype(df[item["col_qtd"]]):
                # Se existe coluna numérica de quantidade (ex: qtd_rollback)
                df_temp = df[df[item["col_qtd"]] > 0].copy()
                metric_col = item["col_qtd"]
                title_suffix = "Quantidade"
            elif item["file_flag"]:
                # Se não tem coluna de quantidade, mas o arquivo é desse tipo, assumimos count=1 por linha
                df_temp = df.copy()
                df_temp["_count_tmp_"] = 1
                metric_col = "_count_tmp_"
                title_suffix = "Ocorrências"
            else:
                df_temp = pd.DataFrame() # Pula se não tiver dados

            # Gerar Gráfico de QUANTIDADE
            if not df_temp.empty and metric_col:
                try:
                    # Top 15 Jogos por Volume Total
                    top_games = df_temp.groupby(game_col)[metric_col].sum().nlargest(15).index
                    df_filtered = df_temp[df_temp[game_col].isin(top_games)]

                    if not df_filtered.empty:
                        h = max(5, len(top_games) * 0.5)
                        fig, ax = plt.subplots(figsize=(11, h))

                        sns.histplot(
                            data=df_filtered,
                            y=game_col,
                            weights=metric_col, # IMPORTANTE: Usa weights para somar a qtd
                            hue=brand_col,
                            multiple="stack",
                            palette=get_palette(df_filtered[brand_col].unique()),
                            edgecolor="#0B0F19",
                            linewidth=0.5,
                            shrink=0.8,
                            ax=ax
                        )

                        # Labels QTD
                        for container in ax.containers:
                            labels = [f'{int(val)}' if (val := v.get_width()) > 0 else '' for v in container]
                            ax.bar_label(container, labels=labels, label_type='center', color='white', fontsize=8, weight='bold')

                        ax.set_title(f"Top Jogos {item['tipo']} ({title_suffix})", color='white', weight='bold', loc='left')
                        ax.set_xlabel("Volume", color='#aab2c5')
                        ax.set_ylabel("")
                        sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), frameon=False, title=None, ncol=3)
                        sns.despine(left=True)
                        
                        charts.append({"title": f"Jogos: {item['tipo']} (Qtd)", "img_b64": fig_to_base64(fig)})
                except Exception as e:
                    print(f"Erro plot qtd {item['tipo']}: {e}")

            # --- Lógica de Valor (R$) ---
            if item["col_val"] and pd.api.types.is_numeric_dtype(df[item["col_val"]]):
                try:
                    df_val = df[df[item["col_val"]] > 0].copy()
                    if not df_val.empty:
                        # Top 15 Jogos por Valor Total
                        top_games_val = df_val.groupby(game_col)[item["col_val"]].sum().nlargest(15).index
                        df_filtered_val = df_val[df_val[game_col].isin(top_games_val)]

                        if not df_filtered_val.empty:
                            h = max(5, len(top_games_val) * 0.5)
                            fig, ax = plt.subplots(figsize=(11, h))

                            sns.histplot(
                                data=df_filtered_val,
                                y=game_col,
                                weights=item["col_val"], # IMPORTANTE: Soma o valor financeiro
                                hue=brand_col,
                                multiple="stack",
                                palette=get_palette(df_filtered_val[brand_col].unique()),
                                edgecolor="#0B0F19",
                                linewidth=0.5,
                                shrink=0.8,
                                ax=ax
                            )

                            # Labels VALOR (Formatado simples)
                            for container in ax.containers:
                                # Formata para inteiro se for grande, para não poluir
                                labels = [f'{int(val)}' if (val := v.get_width()) > 1 else '' for v in container]
                                ax.bar_label(container, labels=labels, label_type='center', color='white', fontsize=7.5, weight='bold')

                            ax.set_title(f"Top Jogos {item['tipo']} (Valor Financeiro)", color='white', weight='bold', loc='left')
                            ax.set_xlabel("Valor Total", color='#aab2c5')
                            ax.set_ylabel("")
                            sns.move_legend(ax, "lower right", bbox_to_anchor=(1, 1), frameon=False, title=None, ncol=3)
                            sns.despine(left=True)

                            charts.append({"title": f"Jogos: {item['tipo']} (Valor)", "img_b64": fig_to_base64(fig)})
                except Exception as e:
                    print(f"Erro plot valor {item['tipo']}: {e}")

    # =================================================================
    # 5. EVOLUÇÃO DIÁRIA - TOP JOGOS (Corrigido: agora dentro da função)
    # =================================================================
    if game_col and date_col:
        try:
            temp_df = df.copy()
            temp_df['_dt_plot'] = pd.to_datetime(temp_df[date_col], errors='coerce')
            temp_df = temp_df.dropna(subset=['_dt_plot'])

            top_games = temp_df[game_col].value_counts().head(5).index
            temp_df = temp_df[temp_df[game_col].isin(top_games)]

            if not temp_df.empty:
                ts = temp_df.groupby([
                    temp_df['_dt_plot'].dt.date,
                    game_col
                ]).size().unstack(fill_value=0)

                fig, ax = plt.subplots(figsize=(12, 5))
                ts.plot(ax=ax, linewidth=2)

                ax.set_title("Evolução Diária - Top Jogos", color='white', weight='bold')
                ax.set_ylabel("Volume")
                ax.set_xlabel("Data")
                ax.legend(frameon=False)
                sns.despine()

                charts.append({
                    "title": "Jogos - Evolução Temporal",
                    "img_b64": fig_to_base64(fig)
                })
        except Exception as e:
            print(f"Erro temporal jogos: {e}")

    return charts


# -----------------------------
# Helpers de erro Openai
# -----------------------------


def clean_openai_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()

    if "401" in low or "authorization required" in low:
        return (
            "Openai retornou 401 (não autorizado). Verifique se a PPLX_API_KEY é válida "
            "e se sua conta tem acesso à API."
        )
    if "<html" in low or "cloudflare" in low or "openresty" in low:
        return (
            "Openai retornou uma página HTML em vez de JSON. "
            "Provavelmente problema de autenticação ou bloqueio na conta."
        )
    if len(msg) > 400:
        msg = msg[:400] + "..."
    return msg

# -----------------------------
# IA (Openai – resumo EXECUTIVO)
# -----------------------------

def openai_summary(
    chatgpt_client,
    cfg,
    file_summary,
    alerts,
    insights,
    filename,
    process_type,
    sample_csv_text,
    baseline,
    risk_users,
):


    # Prepara um JSON limpo para a IA não se perder com dados inúteis
    resumo_dados = {
        "tipo_arquivo": process_type,
        "total_linhas": file_summary.get("rows"),
        "kpis_gerais": {
            "taxa_rejeicao": insights.get("rejected_rate"),
            "taxa_pendencia": insights.get("pending_rate"),
            "total_financeiro": insights.get("financeiro_total"),
            "total_rejeitado_financeiro": insights.get("financeiro_rejeitado")
        }
    }

    # Adiciona blocos específicos apenas se existirem (para economizar tokens e focar a IA)
    if "kyc_detailed_analysis" in insights:
        resumo_dados["KYC_Detalhado_Por_Casa"] = insights["kyc_detailed_analysis"]
    
    if "processadora_performance" in insights:
        resumo_dados["Performance_Processadoras"] = insights["processadora_performance"]
        
    if "top_games_ocorrencias" in insights:
        resumo_dados["Top_Games"] = insights["top_games_ocorrencias"]
    
    if "top_reject_reasons" in insights:
        resumo_dados["Motivos_Erro"] = insights["top_reject_reasons"]

    # Prompt Otimizado para Análise Cruzada
    system_prompt = """
Você é um Analista Sênior de Operações iGaming (Cassino/Esportes).
Sua missão é cruzar os dados fornecidos e encontrar a CAUSA RAIZ dos problemas.

REGRAS DE ANÁLISE OBRIGATÓRIAS:

1. **SE O ARQUIVO FOR KYC:**
   - Não dê apenas a taxa geral. Analise a tabela `KYC_Detalhado_Por_Casa`.
   - Compare: Qual casa tem maior rejeição? O problema é no Onboarding ou no Liveness?
   - Exemplo: "Na casa 7k, o Liveness tem 40% de rejeição, enquanto na Vera é apenas 5%."

2. **SE O ARQUIVO FOR FINANCEIRO (SAQUES/DEPÓSITOS):**
   - Use a tabela `Performance_Processadoras`.
   - Identifique qual processadora está gerando as falhas (Rejeitados).
   - Identifique qual processadora está lenta (Pendentes/Processando).
   - Se houver dados financeiros, cite quanto dinheiro foi rejeitado.

3. **SE O ARQUIVO FOR CASINO (FREESPIN/ROLLBACK):**
   - Cite os jogos que mais aparecem em `Top_Games`.
   - Diga se é um problema concentrado em um jogo/provedor ou generalizado.

ESTRUTURA DO RELATÓRIO:
### 🚨 Resumo Executivo
(Veredito em 2 linhas: A operação está saudável ou crítica? Onde está o fogo?)

### 🔍 Análise Cruzada (Drill Down)
(Use os dados detalhados acima. Cite nomes de casas e processadoras. Compare desempenhos.)

### 📉 Principais Ofensores
- Motivos de erro mais comuns.
- Jogos ou Processadoras com pior desempenho.

### ✅ Ações Recomendadas
(3 bullet points práticos para o operador de plantão).

Responda em Português do Brasil. Seja técnico e direto.
"""

    user_content = f"""
CONTEXTO DO ARQUIVO ({filename}):
{json.dumps(resumo_dados, ensure_ascii=False, indent=2)}

AMOSTRA DE DADOS (CSV):
{sample_csv_text[:5000]}
"""

    resp = chatgpt_client.chat.completions.create(
        model=cfg.get("openai", {}).get("model", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2, # Temperatura baixa para ser mais analítico e menos criativo
        max_tokens=1000,
    )
    return resp.choices[0].message.content.strip()


# -----------------------------
# IA (Openai – OCORRÊNCIAS)   até aqui
# -----------------------------


def openai_occurrences(
    chatgpt_client: OpenAI,
    cfg: dict,
    nome_arquivo: str,
    process_type: str,
    sample_csv_text: str,
    insights: Dict[str, Any],
) -> Dict[str, Any]:
    system_prompt = """
Você é um analista de dados focado em monitoramento iGaming.
Seu objetivo é identificar os principais tipos de ocorrências/problemas presentes nos dados
(ex: "índice de rejeição KYC elevado", "saques negados por falta de saldo",
"pico de depósitos em uma processadora específica",
"uso intensivo de freespins em poucos usuários",
"concentração de volume em 1 jogo ou 1 marca", etc.).

Use linguagem de monitoramento: "pontos críticos", "comportamentos a acompanhar",
"usuários sob observação", evitando levantar afirmações de fraude certa.

Você receberá:
- tipo_processo inferido
- um JSON compacto de insights agregados (rejeição, pendência, brand_volume,
  processadoras, jogos, usuários de risco, rollbacks, freespins, etc.)
- uma amostra em texto das linhas CSV.

IMPORTANTE: devolva a resposta APENAS em JSON válido, no formato EXATO:

{
  "arquivo": "nome_arquivo.csv",
  "tipo_processo": "rollback | freespins | saques | depositos | kyc | analises",
  "resumo_geral": "texto com visão geral do que o arquivo mostra",
  "ocorrencias": [
    {
      "nome": "nome curto da ocorrência ou tipo de problema",
      "ocorrencias": 123,
      "eixo": "kyc | transacional | bonus | jogo | operacional",
      "prioridade": "baixa | media | alta",
      "resumo": "explicação curta do que é essa ocorrência/problema"
    }
  ]
}

Regras adicionais:
- "ocorrencias" deve ser uma lista de objetos, cada um com nome, ocorrencias (número inteiro),
  eixo, prioridade e resumo.
- Use as colunas (status, motivo, brand, processor, horário, jogo, valores, rollbacks, freespins, etc.)
  e os insights agregados para inferir os tipos.
- Foque em coisas relevantes para monitoramento de pontos críticos e gargalos
  (KYC, saques, depósitos, bônus, jogos, operação).
- Se o arquivo estiver estável, ainda assim crie poucas ocorrências do tipo
  "Comportamento normal" com prioridade "baixa".
- Não escreva nada fora do JSON (sem texto antes ou depois).
"""

    payload = {
        "arquivo": nome_arquivo,
        "tipo_processo": process_type,
        "insights": insights,
    }

    user_content = (
        f"Contexto do arquivo:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Abaixo está uma amostra das linhas do CSV (formato texto):\n\n"
        f"{sample_csv_text[:8000]}"
    )

    resp = chatgpt_client.chat.completions.create(
        model=cfg.get("openai", {}).get("model", "gpt-4.1-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=cfg.get("openai", {}).get("max_output_tokens", 600),
    )
    content = resp.choices[0].message.content

    try:
        data = json.loads(content)
    except Exception:
        data = {
            "arquivo": nome_arquivo,
            "tipo_processo": process_type,
            "resumo_geral": str(content)[:2000],
            "ocorrencias": [],
        }

    if not data.get("arquivo"):
        data["arquivo"] = nome_arquivo
    if not data.get("tipo_processo"):
        data["tipo_processo"] = process_type
    if "resumo_geral" not in data or data["resumo_geral"] is None:
        data["resumo_geral"] = ""
    if "ocorrencias" not in data or not isinstance(data["ocorrencias"], list):
        data["ocorrencias"] = []

    ocorrencias_normalizadas: List[Dict[str, Any]] = []
    for oc in data["ocorrencias"]:
        if not isinstance(oc, dict):
            continue
        nome = str(oc.get("nome", "")).strip()
        resumo = str(oc.get("resumo", "")).strip()
        eixo = str(oc.get("eixo", "")).strip().lower() or "operacional"
        prioridade = str(oc.get("prioridade", "")).strip().lower() or "media"
        try:
            qtd = int(oc.get("ocorrencias", 0))
        except Exception:
            qtd = 0
        if not nome:
            continue
        ocorrencias_normalizadas.append(
            {
                "nome": nome,
                "ocorrencias": qtd,
                "eixo": eixo,
                "prioridade": prioridade,
                "resumo": resumo,
            }
        )

    ocorrencias_normalizadas.sort(key=lambda x: x.get("ocorrencias", 0), reverse=True)
    data["ocorrencias"] = ocorrencias_normalizadas
    return data




# -----------------------------
# IA (Gemini – RESERVA)
# -----------------------------


def gemini_summary(cfg: dict, file_summary: dict, alerts: List[Alert], insights: Dict[str, Any]) -> str:
    if not HAVE_GEMINI:
        return "Gemini indisponível."
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "GEMINI_API_KEY não encontrada."

    genai.configure(api_key=api_key)
    model_name = cfg.get("gemini", {}).get("model", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Você é um analista sênior de dados especializado em iGaming (casas de apostas, cassino e esportivas).
Responda SEMPRE em português (pt-BR), em formato de tópicos, com foco em turno de operação.

Gere um resumo executivo curto e objetivo contendo:
- Visão geral da situação atual.
- Principais pontos de atenção (picos, quedas, concentrações, desvios relevantes).
- Oportunidades ou comportamentos positivos relevantes.
- Uma seção final "Ações recomendadas" com bullets claros e acionáveis.

Regras:
- Seja direto; evite texto genérico ou prolixo.
- Não crie campos, métricas ou dimensões que não existam nos dados fornecidos.
- Se o cenário estiver estável, deixe isso explícito e recomende apenas monitoramento contínuo.



Resumo estatístico:
{json.dumps(file_summary, ensure_ascii=False)}

Insights:
{json.dumps(insights, ensure_ascii=False)}

Alertas:
{json.dumps([a.__dict__ for a in alerts], ensure_ascii=False)}
"""
    try:
        r = model.generate_content(prompt)
        return (r.text or "").strip() or "Sem resumo Gemini."
    except Exception as e:
        return f"Gemini falhou: {e}"


def gemini_dashboard_spec(cfg: dict, file_summary: dict, insights: Dict[str, Any], alerts: List[Alert]) -> Dict[str, Any]:
    if not HAVE_GEMINI:
        return {"enabled": False, "error": "Gemini indisponível."}

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"enabled": False, "error": "GEMINI_API_KEY não encontrada."}

    genai.configure(api_key=api_key)
    model_name = cfg.get("gemini", {}).get("model", "gemini-1.5-pro")
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Você é um analista sênior de dados especializado em iGaming (casas de apostas, cassino e esportivas).
Responda SEMPRE em português (pt-BR), em formato de tópicos, com foco em turno de operação.

Gere um resumo executivo curto e objetivo contendo:
- Visão geral da situação atual.
- Principais pontos de atenção (picos, quedas, concentrações, desvios relevantes).
- Oportunidades ou comportamentos positivos relevantes.
- Uma seção final "Ações recomendadas" com bullets claros e acionáveis.

Regras:
- Seja direto e operacional; evite texto prolixo ou genérico.
- Não crie campos, métricas ou dimensões que não existam nos dados fornecidos.
- Se o cenário estiver estável, deixe isso explícito e recomende apenas monitoramento contínuo.



Dados:
Resumo estatístico:
{json.dumps(file_summary, ensure_ascii=False)}

Insights:
{json.dumps(insights, ensure_ascii=False)}

Alertas:
{json.dumps([a.__dict__ for a in alerts], ensure_ascii=False)}

Contexto Command Center (CORRELAÇÕES – OLHO VIVO):
- Estruture o dashboard como um "Command Center" com seções de:
  - Experiência do Jogador
  - Crescimento de Depósitos
  - Crescimento da Base de Jogadores
  - Segmentos de Usuários (alto valor)
  - Usuários de maior risco

Instruções de estrutura do JSON de dashboard:
- O JSON gerado deve conter, no mínimo, as chaves:
  "dashboard_title",
  "time_range_default",
  "filters",
  "panels",
  "alerts".
- Cada item em "filters" deve ter: id, label, type, column/columns e, quando fizer sentido, default_value.
- Cada item em "panels" deve ter, no mínimo:
  - id
  - title
  - section (uma das seções do Command Center)
  - description (curta, focada em ação)
  - type (timeseries, bar, pie, scatter, heatmap, single_value, table)
  - criticality (high, medium, low)
  - kpi_goal (o que esse painel ajuda a monitorar)
  - query (descrição textual de como agregar a partir do CSV)
  - breakdowns (lista de dimensões para corte, quando aplicável).
- O array "alerts" do dashboard deve se inspirar nos itens de Alertas recebidos e conter, para cada alerta:
  - id
  - name
  - description
  - severity (critical, high, medium, low)
  - related_panel_ids
  - condition (condição de disparo, em texto)
  - suggested_actions (lista curta de ações operacionais).

Importante:
- Use o Resumo estatístico para identificar colunas de valor, datas, segmentos e outliers.
- Use Insights para priorizar quais painéis criar e como descrever kpi_goal.
- Use Alertas para popular o array "alerts" do JSON de dashboard.
- Não escreva nada além do JSON final do dashboard.
"""



# -----------------------------
# Relatório HTML
# -----------------------------




HTML_TEMPLATE = r"""
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <title>CMD Análises — Relatório</title>
  <style>
    :root{
      --bg:#0b0f19; --card:#121829; --card2:#0e1322; --border:#222a3a;
      --text:#e6e9ef; --muted:#aab2c5;
      --green:#7CFFA3; --yellow:#FFE27C; --red:#FF7C7C;
    }
    body{font-family:Inter,Arial; margin:18px; background:var(--bg); color:var(--text);}
    h1{font-size:22px; margin:6px 0 2px}
    h2{font-size:18px; margin:0 0 6px}
    h3{font-size:14px; margin:12px 0 6px; color:#f5f7fb;}
    .tiny{font-size:12px; color:var(--muted);}
    .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px;}
    .card{
      background:var(--card); border:1px solid var(--border); padding:14px;
      border-radius:14px; margin-bottom:14px; box-shadow:0 6px 24px rgba(0,0,0,.35);
    }
    .badge{display:inline-block; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:700;}
    .green{background:#0e2a1a; color:var(--green);}
    .yellow{background:#2a250e; color:var(--yellow);}
    .red{background:#2a0e0e; color:var(--red);}
    img{max-width:100%; border-radius:10px; border:1px solid #2a3347; margin:6px 0 12px}
    table{border-collapse:collapse; width:100%; font-size:12px;}
    th,td{border:1px solid var(--border); padding:6px; text-align:left;}
    details{background:var(--card2); border:1px solid #1f2740; border-radius:10px; padding:8px; margin-top:6px;}
    summary{cursor:pointer; font-weight:700; color:#dce2ef;}
    pre{white-space:pre-wrap; font-size:12px; color:#d5d9e2;}
    .kpi{display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;}
    .kpi .pill{background:#0f1628; border:1px dashed #25304a; padding:6px 10px; border-radius:999px; font-size:12px;}
  </style>
</head>
<body>

<h1>CMD Análises — Relatório automático</h1>
<div class="tiny">Gerado em {{ generated_at }}</div>

{% for f in files %}
  <div class="card">
    <h2>{{ f.filename }}</h2>
    <div class="tiny">
      Tipo detectado: <b>{{ f.process_label }}</b>
      <span style="opacity: 1 !important; color: var(--text) !important;">({{ f.process_type }})</span>
    </div>

    <div class="tiny" style="margin-top:4px; opacity: 1 !important; color: var(--text) !important;">
      <span class="badge {{ f.critical_level }}" style="font-weight: normal !important;">{{ f.critical_level.upper() }}</span>
      <b>{{ f.critical_headline }}</b>
    </div>

    <div class="kpi">
      <div class="pill">Linhas: <b>{{ f.summary.rows }}</b></div>
      <div class="pill">Rejeição: <b>{{ (f.insights.rejected_rate*100)|round(1) if f.insights.rejected_rate is defined else 0 }}%</b></div>
      <div class="pill">Em análise: <b>{{ (f.insights.pending_rate*100)|round(1) if f.insights.pending_rate is defined else 0 }}%</b></div>
      {% if f.insights.max_z_amount is defined %}
        <div class="pill">Pico de valor (z): <b>{{ f.insights.max_z_amount|round(2) }}</b></div>
      {% endif %}
      {% if f.insights.baseline is defined %}
        <div class="pill">
          Baseline rejeição: <b>{{ (f.insights.baseline.rejected_rate_mean*100)|round(1) if f.insights.baseline.rejected_rate_mean is defined else 0 }}%</b>
        </div>
      {% endif %}
    </div>

    <h3>Alertas / Pontos críticos</h3>
    <div class="grid">
      {% for a in f.alerts %}
        <div class="card" style="margin:0;">
          <span class="badge {{a.level}}" style="font-weight: normal !important;">{{ a.level.upper() }}</span>
          <b>{{a.title}}</b>
          <div class="tiny">{{a.detail}}</div>
        </div>
      {% endfor %}
    </div>

    <h3>Resumo IA (Principal – Openai)</h3>
    <details open>
      <summary>Ver resumo IA</summary>
      <pre style="opacity: 1 !important; color: var(--text) !important; background: var(--card2); font-size: 13px;">{{ f.ai_summary }}</pre>
    </details>

    <h3>Mapa de Ocorrências (pontos críticos)</h3>
    <details>
      <summary>Ver ocorrências/chaves de atenção</summary>
      <pre style="opacity: 1 !important; color: var(--text) !important; background: var(--card2); font-size: 13px;">{{ f.occurrences_json }}</pre>
    </details>

    <h3>Resumo IA (Gemini)</h3>
    <details>
      <summary>Ver resumo Gemini</summary>
      <pre style="opacity: 1 !important; color: var(--text) !important; background: var(--card2); font-size: 13px;">{{ f.gemini_summary }}</pre>
    </details>

    <h3>Usuários em observação (score alto)</h3>
    {% if f.risk_users|length > 0 %}
      <table>
        <tr>
          <th>Casa</th>
          <th>Usuário</th>
          <th>Score</th>
          <th>Volume linhas</th>
          <th>Soma valor</th>
          <th>Rejeição</th>
          <th>Qtd rollback</th>
          <th>Qtd freespins</th>
          <th>Max ganho 1h</th>
        </tr>
        {% for u in f.risk_users %}
          <tr>
            <td>{{ u.brand if u.brand is defined else "" }}</td>
            <td>
              {{ u.user_id if u.user_id is defined
                 else u.id_usuario if u.id_usuario is defined
                 else u.player_id if u.player_id is defined
                 else "" }}
            </td>
            <td>{{ (u.risk_score*100)|round(1) if u.risk_score is defined else "" }}</td>
            <td>{{ u.volume_linhas if u.volume_linhas is defined else "" }}</td>
            <td>{{ u.soma_valor|round(2) if u.soma_valor is defined else "" }}</td>
            <td>{{ (u.rejected_rate*100)|round(1) if u.rejected_rate is defined else "" }}%</td>
            <td>{{ u.rollback_qtd if u.rollback_qtd is defined else "" }}</td>
            <td>{{ u.freespins_qtd if u.freespins_qtd is defined else "" }}</td>
            <td>{{ u.max_ganho_1h|round(2) if u.max_ganho_1h is defined else "" }}</td>
          </tr>
        {% endfor %}
      </table>
    {% else %}
      <div class="tiny">Nenhum usuário se destacou com pontos críticos relevantes no score.</div>
    {% endif %}

    <h3>Gráficos executivos</h3>
    {% if f.charts|length > 0 %}
      <div class="grid">
        {% for c in f.charts %}
          <div class="card" style="margin:0;">
            <div class="tiny">{{ c.title }}</div>
            <img src="data:image/png;base64,{{ c.img_b64 }}" alt="{{ c.title }}"/>
          </div>
        {% endfor %}
      </div>
    {% else %}
      <div class="tiny">Gráficos desativados ou sem dados relevantes.</div>
    {% endif %}

    <h3>Resumo JSON bruto (debug)</h3>
    <details>
      <summary>Ver JSON bruto deste arquivo</summary>
      <pre style="opacity: 1 !important; color: var(--text) !important; background: var(--card2);">{{ f.summary_json }}</pre>
    </details>

    <h3>Insights JSON bruto (debug)</h3>
    <details>
      <summary>Ver insights JSON</summary>
      <pre style="opacity: 1 !important; color: var(--text) !important; background: var(--card2);">{{ f.insights_json }}</pre>
    </details>

  </div>
{% endfor %}

<hr/>

<h2>Resumo agregado do dia</h2>
<div class="tiny">Quando gerado como relatório diário.</div>

<details>
  <summary>Ver resumo IA agregado</summary>
  <pre style="opacity: 1 !important; color: var(--text) !important; background: var(--card2);">{{ ctx.ia_summary }}</pre>
</details>

<h3>Tabela de arquivos processados</h3>
<table>
  <tr>
    <th>Arquivo</th>
    <th>Tipo</th>
    <th>Linhas</th>
    <th>Rejeição</th>
    <th>Pendente</th>
    <th>Alertas</th>
  </tr>
  {% for f in files %}
    <tr>
      <td>{{ f.filename }}</td>
      <td>{{ f.process_type }}</td>
      <td>{{ f.summary.rows }}</td>
      <td>{{ (f.insights.rejected_rate*100)|round(1) if f.insights.rejected_rate is defined else 0 }}%</td>
      <td>{{ (f.insights.pending_rate*100)|round(1) if f.insights.pending_rate is defined else 0 }}%</td>
      <td>{{ f.alerts|length }}</td>
    </tr>
  {% endfor %}
</table>

<details>
  <summary>JSON agregado (debug)</summary>
  <pre style="opacity: 1 !important; color: var(--text) !important; background: var(--card2);">{{ ctx.agg_json }}</pre>
</details>

</body>
</html>
"""


def send_email(
    subject: str,
    html_body: str,
    logger,
    attachments: Optional[List[Tuple[str, bytes]]] = None,
    force_recipients: Optional[List[str]] = None,
):
    if not CONFIG.get("email", {}).get("enabled", True):
        logger.info("Envio de e-mail está desabilitado em CONFIG.email.enabled.")
        return

    to_env = [x.strip() for x in EMAIL_TO.split(",") if x.strip()]
    base_conf_recip = CONFIG["email"].get("recipients", [])
    recipients = force_recipients if force_recipients else (to_env or base_conf_recip)
    recipients = list(dict.fromkeys([r.strip() for r in recipients if r.strip()]))

    if not recipients:
        logger.warning("Nenhum destinatário de e-mail configurado. Abortando envio.")
        return

    msg = EmailMessage()
    msg["Subject"] = EMAIL_SUBJECT_PREFIX + subject
    msg["From"] = CONFIG["email"].get("from", SMTP_USER)
    msg["To"] = ", ".join(recipients)
    msg.set_content("Versão HTML do relatório em anexo.")
    msg.add_alternative(html_body, subtype="html")

    if attachments:
        for fname, content in attachments:
            msg.add_attachment(
                content,
                maintype="application",
                subtype="octet-stream",
                filename=fname,
            )

    logger.info("Enviando e-mail para: %s", msg["To"])

    context = ssl.create_default_context()
    try:
        if SMTP_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                if CONFIG["email"].get("use_tls", True):
                    server.starttls(context=context)
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        logger.info("E-mail enviado com sucesso.")
    except Exception as e:
        logger.error("Falha ao enviar e-mail: %s", e)


# -----------------------------
# Notificação Windows
# -----------------------------


def notify(title: str, message: str):
    if not HAVE_NOTIFY:
        return
    try:
        safe_title = (title or "")[:60]
        safe_message = (message or "")[:200]
        notification.notify(title=safe_title, message=safe_message, timeout=10)
    except Exception:
        pass


# -----------------------------
# OPENAI client
# -----------------------------

def build_openai_client(cfg, logger):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PPLX_API_KEY")
    if not api_key:
        raise RuntimeError("Nenhuma API KEY encontrada. Defina OPENAI_API_KEY no ambiente.")

    base_url = cfg.get("openai", {}).get("base_url", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


# -----------------------------
# Helper: print dos gráficos executivos
# -----------------------------
import re
import base64
from io import BytesIO
from PIL import Image

def gerar_print_graficos(html: str) -> bytes:
    m = re.search(r"h3>Gráficos executivos</h3>(.*?)(<h3|</body>)", html, re.S)
    if not m:
        return b""
    bloco = m.group(1)

    imgs_b64 = re.findall(r'src="data:image/png;base64,([^"]+)"', bloco)
    if not imgs_b64:
        return b""

    pil_imgs = []
    for b64 in imgs_b64:
        try:
            data = base64.b64decode(b64)
            pil_imgs.append(Image.open(BytesIO(data)).convert("RGBA"))
        except Exception:
            continue

    if not pil_imgs:
        return b""

    w_total = sum(im.width for im in pil_imgs)
    h_max = max(im.height for im in pil_imgs)
    canvas = Image.new("RGBA", (w_total, h_max), (11, 15, 25, 255))

    x = 0
    for im in pil_imgs:
        canvas.paste(im, (x, 0))
        x += im.width

    buf = BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# -----------------------------
# Geração de relatório por arquivo
# -----------------------------


# ==== BLOCO CSV (MANTER COMO ESTÁ) ====


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    summary = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    return summary


def dataframe_sample_csv(df: pd.DataFrame, max_rows: int = 100) -> str:
    buf: List[str] = []
    subset = df.head(max_rows)
    buf.append(",".join(map(str, subset.columns)))
    for _, row in subset.iterrows():
        vals = [str(x) for x in row.values]
        buf.append(",".join(vals))
    return "\n".join(buf)


def render_html_report(files_ctx: List[Dict[str, Any]], agg_ctx: Dict[str, Any]) -> str:
    tmpl = Template(HTML_TEMPLATE)
    html = tmpl.render(
        files=files_ctx,
        ctx=agg_ctx,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return html

import os
from typing import Optional, Dict, Any

from openai import OpenAI


def analyze_file(
    path: str,
    cfg: dict,
    logger,
    chatgpt_client: Optional[OpenAI],
    baselines: Dict[str, Any],
) -> Dict[str, Any]:

    logger.info("Processando arquivo: %s", path)
    try:
        df = read_csv_smart(path)
    except Exception as e:
        logger.error("Falha ao ler %s: %s", path, e)
        return {
            "filename": os.path.basename(path),
            "error": str(e),
            "alerts": [Alert(level="red", title="Erro de leitura CSV", detail=str(e), file=path)],
            "insights": {},
            "summary": {"rows": 0, "cols": 0, "columns": [], "dtypes": {}},
        }

    df = normalize_cols(df)
    summary = summarize_dataframe(df)

    filename = os.path.basename(path)
    proc_type = guess_process_type(filename, df)
    alerts, insights = detect_alerts_and_insights(df, cfg, filename)

    df_risk, risk_summary = build_risk_score(df, cfg)
    risk_users = risk_summary.get("risk_users", [])
    insights["risk_users"] = risk_users

    baselines_for_type = baselines.get(proc_type, {})
    baseline_this = baselines_for_type.get("stats", {})

    rej_rate = insights.get("rejected_rate")
    if rej_rate is not None:
        hist = baselines_for_type.get("rejected_rate_hist", [])
        hist.append(float(rej_rate))
        if len(hist) > 50:
            hist = hist[-50:]
        baselines_for_type["rejected_rate_hist"] = hist
        baselines_for_type["stats"] = {
            "rejected_rate_mean": float(np.mean(hist)),
            "rejected_rate_std": float(np.std(hist)),
        }
        baselines[proc_type] = baselines_for_type
        insights["baseline"] = baselines_for_type["stats"]

    process_label = process_type_label(proc_type)

    sample_csv = dataframe_sample_csv(df, max_rows=cfg.get("ai_sample_rows", 800))

    ai_summary = ""
    occurrences_raw: Dict[str, Any] = {}
    gemini_sum = ""
    gemini_dash_spec: Dict[str, Any] = {}

    if chatgpt_client:
        try:
            ai_summary = openai_summary(
                chatgpt_client=chatgpt_client,
                cfg=cfg,
                file_summary=summary,
                alerts=alerts,
                insights=insights,
                filename=filename,
                process_type=proc_type,
                sample_csv_text=sample_csv,
                baseline=baseline_this,
                risk_users=risk_users,
            )
        except Exception as e:
            logger.warning("IA summary falhou em %s: %s", filename, e, exc_info=True)
            ai_summary = f"Falha ao gerar resumo IA (openai): {clean_openai_error(e)}"


        try:
            occurrences_raw = openai_occurrences(
                chatgpt_client=chatgpt_client,
                cfg=cfg,
                nome_arquivo=filename,
                process_type=proc_type,
                sample_csv_text=sample_csv,
                insights=insights,  # envia insights completos
            )
        except Exception as e:
            logger.warning("IA occurrences falhou em %s: %s", filename, e, exc_info=True)
            occurrences_raw = {
                "arquivo": filename,
                "tipo_processo": proc_type,
                "resumo_geral": f"Falha ao gerar mapa de ocorrências: {clean_openai_error(e)}",
                "ocorrencias": [],
            }
    else:
        ai_summary = "Openai desativado ou indisponível. Sem resumo IA principal."
        occurrences_raw = {
            "arquivo": filename,
            "tipo_processo": proc_type,
            "resumo_geral": "Openai desativado/indisponível. Sem mapa de ocorrências detalhado.",
            "ocorrencias": [],
        }

    if CONFIG.get("gemini", {}).get("enabled", False):
        try:
            gemini_sum = gemini_summary(CONFIG, summary, alerts, insights)
        except Exception as e:
            gemini_sum = f"Gemini falhou: {e}"

        try:
            gemini_dash_spec = gemini_dashboard_spec(CONFIG, summary, insights, alerts)
        except Exception as e:
            gemini_dash_spec = {"enabled": False, "error": f"Gemini spec falhou: {e}"}
    else:
        gemini_sum = "Gemini desabilitado na configuração."
        gemini_dash_spec = {"enabled": False, "error": "Gemini desabilitado na configuração."}

    # Geração dos gráficos executivos (Padrão)
    # Para KYC não colocar " - kyc" no título dos gráficos
    charts = make_charts(
        df,
        cfg,
        filename,
        "" if proc_type == "kyc" else proc_type,
        insights,
    )

    # Crítico / headline
    critical = build_critical_info(proc_type, process_label, alerts)

    # DEBUG: conferir se os insights de KYC por casa estão vindo
    print("DEBUG insights keys:", insights.keys())
    print("DEBUG kyc_ops_by_brand:", insights.get("kyc_ops_by_brand"))

    # -------------------------------------------------------------
    # 1. GERA OS GRÁFICOS NOVOS (incluindo o comparativo RB vs FS)
    # -------------------------------------------------------------
    graficos_extras = gerar_analises_graficas(df) 
    
    # 2. INJETA O GRÁFICO DE DESTAQUE NO RELATÓRIO
    if 'distribuicao_rb_fs' in graficos_extras:
        charts.insert(0, {
            "title": "Destaque: Free Spins vs Rollbacks por Jogo",
            "img_b64": graficos_extras['distribuicao_rb_fs']
        })

    file_ctx = {
        "filename": filename,
        "process_type": proc_type,
        "process_label": process_label,
        "summary": summary,
        "insights": insights,
        "alerts": alerts,
        "ai_summary": ai_summary,
        "occurrences": occurrences_raw,
        "occurrences_json": json.dumps(occurrences_raw, ensure_ascii=False, indent=2),
        "gemini_summary": gemini_sum,
        "gemini_dash_spec": gemini_dash_spec,
        "gemini_dash_spec_json": json.dumps(gemini_dash_spec, ensure_ascii=False, indent=2),
        "summary_json": json.dumps(summary, ensure_ascii=False, indent=2),
        "insights_json": json.dumps(insights, ensure_ascii=False, indent=2),
        "charts": charts,
        "risk_users": risk_users,
        
        "critical_level": critical["level"],
        "critical_title": critical["title"],
        "critical_headline": critical["headline"],
        
        "graficos_extra": graficos_extras,
    }
    return file_ctx

   



def montar_prompt_protocolo_imagem(ocr_text: str) -> str:
    """
    Prompt para a IA gerar um protocolo executivo de iGaming a partir do texto OCR.
    Serve para qualquer área: KYC, saques, depósitos, rollback, cassino, esportivas,
    freespins, campanhas/promoções, etc.
    """
    return f"""
Você é um analista sênior especializado em iGaming (casas de apostas online), com experiência em:
- Cadastros / KYC / Onboarding
- Depósitos, saques, chargebacks, rollback
- Cassino, esportivas, freespins, bônus, promoções e campanhas
- Comportamento transacional, risco, fraude e estabilidade operacional

Você recebeu o TEXTO OCR de um print de dashboard operacional (24h, 1h, etc.).
Seu objetivo é gerar um PROTOCOLO executivo de monitoramento sobre a situação atual,
sem inventar números que não estejam no texto.

Regras gerais:
- Responda sempre em português (pt-BR).
- Seja direto e operacional, como em um reporte de turno para Operações/Comercial/Risco.
- O painel pode ser de qualquer área: KYC, Saques, Depósitos, Rollback,
  Apostas e Ganhos, Freespins, Cassino, Esportivas, Campanhas/Tráfego, etc.
- Use SOMENTE o texto OCR para identificar casas, área, janela de tempo e comportamento.
- Se o gráfico e os valores indicarem comportamento normal/estável, deixe isso explícito.
- NÃO invente picos, janelas críticas ou instabilidade se o texto não mencionar isso
  ou se a linha real estiver próxima da linha esperada.

Atenção importante:
- Muitos painéis trazem textos do tipo "DOWN = alerta de nenhuma conclusão registrada nos últimos 20 minutos".
- Isso é apenas a definição da REGRA de alerta, NÃO significa que o alerta está ativo agora.
- Só trate como alerta atual se o painel indicar explicitamente o estado DOWN, vermelho,
  ou texto claro de ocorrência recente (por exemplo: mensagem destacada, contagem de eventos, etc.).
- Se o painel estiver com status UP e sem destaques em vermelho, deixe claro que a regra existe,
  mas que NÃO há alerta disparado nesse ponto no momento atual.

Critério de Severidade:
- Use 🟩 Normal quando:
  * não há menção clara a falha, erro ou incidente;
  * os valores são baixos ou dentro do padrão histórico;
  * o gráfico oscila em torno de uma faixa relativamente estável.
- Use 🟨 Atenção quando houver desvio relevante, concentração anormal,
  aumento súbito ou queda brusca com impacto potencial em operação ou receita.
- Use 🟥 Crítico apenas se houver indício de falha sistêmica, indisponibilidade,
  risco operacional/fraude elevado ou impacto financeiro significativo.

FORMATO DE SAÍDA (obrigatório):

Severidade: (🟥 Crítico / 🟨 Atenção / 🟩 Normal)
Horário: (data e janela aproximada; use Last 1 hour, Last 24 hours, horários do eixo, etc.)
Casa: (7K, Cassino, Vera, ou "Múltiplas"; se não der para saber, use "Geral")
Área: (KYC / Cadastros, Saques, Depósitos, Rollback, Apostas e Ganhos, Cassino,
       Esportivas, Bônus/Freespins, Campanhas/Tráfego, etc.)

Descrição
[2–3 parágrafos explicando o que está acontecendo agora:
picos ou quedas de volume, normalização, descolamento entre casas,
concentração em alguma etapa, impacto em receita e/ou risco OU estabilidade operacional.]

Observações
- [bullet 1 com ponto crítico ou comportamento relevante]
- [bullet 2]
- [bullet 3]
(Adicione mais bullets se necessário.)

Possíveis Causas
- [hipóteses de causa de negócio: campanha ativa, tráfego pago/afiliado,
   ajuste de limite, problema em processadora, promoção específica, comportamento de jogo, etc.]

Ação Recomendada
- [ações práticas: validar campanhas, checar processadoras, monitorar fraude,
   revisar KYC, acompanhar recorrência do padrão, etc.]

Agora, gere o protocolo com base SOMENTE nas informações e padrões que você consegue inferir
do texto OCR abaixo. Se algo não estiver claro (ex.: horário exato), use termos aproximados.
Se não for possível identificar anomalia clara, deixe explícito que o cenário é estável e
recomende apenas monitoramento contínuo.

Texto OCR do dashboard:
-----------------------
{ocr_text}
-----------------------
"""




# -----------------------------
# Helper: processar UMA IMAGEM (dashboard)
# -----------------------------

def process_single_image(
    path: str,
    cfg: dict,
    logger: logging.Logger,
    chat_client: Optional[OpenAI],
) -> None:
    """
    Processa um print de dashboard (JPG/PNG) como um incidente de iGaming:
    - roda OCR
    - gera um protocolo executivo (Severidade, Casa, Área, descrição, causas, ações)
    - gera um HTML usando o MESMO template dos CSVs e abre no navegador.
    """
    filename = os.path.basename(path)
    logger.info("Processando IMAGEM: %s", filename)

    # 1) OCR do print
    try:
        ocr_text = ocr_dashboard(path)
    except Exception as e:
        logger.error("Falha no OCR da imagem %s: %s", filename, e, exc_info=True)
        return

    # 2) Monta prompt específico de iGaming para IMAGEM
    prompt = montar_prompt_protocolo_imagem(ocr_text)

    # 3) Summary mínimo só para alimentar o HTML_TEMPLATE
    file_summary = {
        "rows": 1,
        "cols": 1,
        "columns": ["texto_ocr"],
        "dtypes": {"texto_ocr": "string"},
    }
    alerts: List[Alert] = []
    insights: Dict[str, Any] = {"ocr_text": ocr_text}
    baselines: Dict[str, Any] = {}
    risk_users: List[Dict[str, Any]] = []
    proc_type = "analises"
    process_label = "Dashboard (imagem)"

    ai_summary = ""
    occurrences_raw: Dict[str, Any] = {}

    # 4) Chama a IA com o prompt de protocolo (sem sample_csv_text)
    if chat_client:
        try:
            resp = chat_client.chat.completions.create(
                model=cfg["openai"]["model"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um analista sênior de iGaming especializado em monitorar operações "
                            "de casas de apostas (cassino e esportivas) e em gerar protocolos executivos "
                            "para turnos de operação, com foco em risco, estabilidade e impacto de negócio."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=cfg["openai"].get("max_output_tokens", 700),
                temperature=0.2,
            )
            ai_summary = resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(
                "IA summary (imagem) falhou em %s: %s", filename, e, exc_info=True
            )
            ai_summary = f"Falha ao gerar protocolo IA (imagem): {clean_openai_error(e)}"

        occurrences_raw = {
            "arquivo": filename,
            "tipo_processo": proc_type,
            "resumo_geral": ai_summary[:600],
            "ocorrencias": [],
        }
    else:
        ai_summary = "Openai desativado ou indisponível. Sem protocolo IA para imagem."
        occurrences_raw = {
            "arquivo": filename,
            "tipo_processo": proc_type,
            "resumo_geral": ai_summary,
            "ocorrencias": [],
        }

    # 5) Monta file_ctx compatível com HTML_TEMPLATE
    critical = {
        "level": "green",
        "title": "Resumo de dashboard (imagem)",
        "headline": f"Dashboard (imagem) — {filename}",
    }

    file_ctx = {
        "filename": filename,
        "process_type": proc_type,
        "process_label": process_label,
        "summary": file_summary,
        "insights": insights,
        "alerts": alerts,
        "ai_summary": ai_summary,
        "occurrences": occurrences_raw,
        "occurrences_json": json.dumps(occurrences_raw, ensure_ascii=False, indent=2),
        "gemini_summary": "",
        "gemini_dash_spec": {"enabled": False},
        "gemini_dash_spec_json": json.dumps({"enabled": False}, ensure_ascii=False),
        "summary_json": json.dumps(file_summary, ensure_ascii=False, indent=2),
        "insights_json": json.dumps(insights, ensure_ascii=False, indent=2),
        "charts": [],
        "risk_users": risk_users,
        "critical_level": critical["level"],
        "critical_title": critical["title"],
        "critical_headline": critical["headline"],
    }

    # ========= FEEDBACK MANUAL DE CRITICIDADE (IMAGEM) =========
    try:
        print("\n============================================")
        print(f"Painel analisado (imagem): {file_ctx.get('filename')}")
        crit_level = file_ctx.get("critical_level", "green").upper()
        crit_headline = file_ctx.get(
            "critical_headline",
            file_ctx.get("process_label", "Dashboard (imagem)"),
        )
        print(f"Criticidade automática: {crit_level} - {crit_headline}")
        fb = input(
            "De 0 a 10, qual o grau de criticidade que você enxerga nesse painel? "
        )
        fb_val = int(fb)
    except Exception:
        fb_val = None

    if fb_val is not None:
        logger.info(
            f"[Feedback][IMG] Criticidade manual = {fb_val} para {file_ctx.get('filename')}"
        )
        # aqui depois você pode salvar em JSON para calibrar
    # ============================================================

    # 6) Gera HTML e abre no navegador (igual CSV)
    out_folder = cfg["output_folder"]
    ensure_dir(out_folder)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_html = os.path.join(
        out_folder,
        f"{os.path.splitext(filename)[0]}_{ts_str}_imagem.html",
    )

    agg_ctx = {"ia_summary": "", "agg_json": "{}"}
    html = render_html_report([file_ctx], agg_ctx)

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Resumo de imagem salvo → %s", out_html)

    try:
        webbrowser.open("file://" + os.path.abspath(out_html))
    except Exception:
        pass



# -----------------------------
# Relatório diário agregado
# -----------------------------




def generate_daily_report(cfg: dict, logger, chatgpt_client: Optional[OpenAI]):
    input_folder = cfg["input_folder"]
    pattern = os.path.join(input_folder, cfg["file_glob"])
    files = glob.glob(pattern)
    today_files = files_modified_today(files)
    logger.info("Gerando relatório diário com %d arquivos de hoje.", len(today_files))

    baselines = load_baselines()
    per_file_ctx: List[Dict[str, Any]] = []

    for fpath in today_files:
        ctx = analyze_file(fpath, cfg, logger, chatgpt_client, baselines)
        per_file_ctx.append(ctx)

    save_baselines(baselines)

    agg: Dict[str, Any] = {
        "total_files": len(per_file_ctx),
        "files": [c["filename"] for c in per_file_ctx],
    }

    try:
        prompt = {
            "tipo": "resumo_diario",
            "arquivos": [
                {
                    "filename": c["filename"],
                    "process_type": c.get("process_type"),
                    "summary": c.get("summary"),
                    "insights": c.get("insights"),
                    "alerts": [a.__dict__ for a in c.get("alerts", [])],
                }
                for c in per_file_ctx
            ],
        }

        if chatgpt_client:
            system_prompt = """
Você é um coordenador de monitoramento iGaming.
Gere um resumo diário curto e objetivo para troca de turno, em português.
Use linguagem de pontos críticos / observação, sem cravar fraude.
"""
            user_content = json.dumps(prompt, ensure_ascii=False)
            resp = chatgpt_client.chat.completions.create(
                model=cfg.get("openai", {}).get("model", "gpt-4.1-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.25,
                max_tokens=cfg.get("openai", {}).get("max_output_tokens", 700),
            )
            text = resp.choices[0].message.content
            agg["ia_summary"] = text.strip() if isinstance(text, str) else str(text)
        else:
            agg["ia_summary"] = "Openai desativado/indisponível. Sem resumo diário IA."
    except Exception as e:
        logger.warning("Falha ao gerar resumo diário IA: %s", e)
        agg["ia_summary"] = f"Falha ao gerar resumo diário IA: {clean_openai_error(e)}"

    agg["agg_json"] = json.dumps(agg, ensure_ascii=False, indent=2)

    html = render_html_report(per_file_ctx, agg)

    out_folder = cfg["output_folder"]
    ensure_dir(out_folder)
    today_str = date.today().strftime("%Y-%m-%d")
    out_path = os.path.join(out_folder, f"CMD_Analises_Diario_{today_str}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Relatório diário salvo em %s", out_path)

    try:
        webbrowser.open("file://" + os.path.abspath(out_path))
    except Exception:
        pass

    recipients = ALWAYS_RECIPIENTS + CRITICAL_ONLY_RECIPIENTS
    recipients = list(dict.fromkeys(recipients))

    send_email(
        subject=f"Relatório diário CMD Análises ({today_str})",
        html_body=html,
        logger=logger,
        attachments=[(os.path.basename(out_path), open(out_path, "rb").read())],
        force_recipients=recipients,
    )

    notify("CMD Análises – Diário", f"Relatório diário gerado: {os.path.basename(out_path)}")


# =====================================================================
# CSVHandler_FIXED — vigia pasta e processa TODO arquivo novo (CSV ou imagem)
# =====================================================================


class CSVHandler_FIXED(FileSystemEventHandler):
    """
    Handler robusto:
    - Detecta arquivo novo (on_created)
    - Detecta arquivo sobrescrito (on_modified)
    - Gera relatório automático
    - Envia e-mail
    - Notifica no Windows
    """
    def __init__(self, cfg, logger, chat_client):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self.chat_client = chat_client
        self.baselines = load_baselines()

    def on_created(self, event):
        if event.is_directory:
            return
        time.sleep(1)
        self._process(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        time.sleep(1)
        self._process(event.src_path)

    def _process(self, path: str):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            self.logger.info(f"[Watcher] Detectado CSV: {path}")
            # Usa o mesmo fluxo completo do analista sênior (análise, HTML, e-mail, notify, feedback)
            process_single_file(path, self.cfg, self.logger, self.chat_client)

        elif ext in IMG_EXTS:
            # fluxo específico para imagens de dashboard
            self.logger.info(f"[Watcher] Detectada IMAGEM: {path}")
            process_single_image(path, self.cfg, self.logger, self.chat_client)
        else:
            # ignora outros tipos de arquivo
            return



def process_single_file(path: str, cfg: dict, logger, chat_client: Optional[OpenAI]):
    """
    Processa um único CSV:
    - analisa
    - atualiza baselines
    - gera HTML
    - abre no navegador
    - envia e-mail
    - manda notificação Windows
    - no final pede feedback de criticidade (0–10)
    """
    input_folder = cfg["input_folder"]
    ensure_dir(input_folder)

    if not os.path.isfile(path):
        logger.error("Arquivo não encontrado: %s", path)
        return

    baselines = load_baselines()
    ctx = analyze_file(path, cfg, logger, chat_client, baselines)
    save_baselines(baselines)

    out_folder = cfg["output_folder"]
    ensure_dir(out_folder)
    filename = os.path.basename(path)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_html = os.path.join(out_folder, f"{filename}_{ts_str}.html")

    agg_ctx = {"ia_summary": "", "agg_json": "{}"}
    html = render_html_report([ctx], agg_ctx)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"[Single] Relatório salvo → {out_html}")

    # --- NOVO: gerar print consolidado dos gráficos executivos ---
    png_graficos = gerar_print_graficos(html)
    attachments: List[Tuple[str, bytes]] = [
        (os.path.basename(out_html), open(out_html, "rb").read())
    ]
    if png_graficos:
        attachments.append(("graficos_executivos.png", png_graficos))
    # -------------------------------------------------------------

    critical_levels = {"green", "yellow", "red"}
    alerts_criticos = sum(
        1 for a in ctx.get("alerts", []) if a.level in critical_levels
    )

    menu_tipo = map_proctype_to_menu_tipo(ctx.get("process_type", "analises"))
    ref_date = datetime.now().date()

    logger.info(
        f"[INDEX] tipo={menu_tipo} html={out_html} alerts={alerts_criticos}"
    )

    appendreportindex(
        tipo=menu_tipo,
        htmlpath=out_html,
        alertscriticos=alerts_criticos,
        refdate=ref_date,
    )

    try:
        webbrowser.open("file:///" + os.path.abspath(out_html))
    except Exception:
        pass

    has_critical = any(a.level in critical_levels for a in ctx.get("alerts", []))
    if has_critical:
        recips = ALWAYS_RECIPIENTS + CRITICAL_ONLY_RECIPIENTS
    else:
        recips = ALWAYS_RECIPIENTS
    recips = list(dict.fromkeys(recips))

    crit_level = ctx.get("critical_level", "green").upper()
    crit_headline = ctx.get("critical_headline", ctx.get("process_label", "Relatório"))

    # ---- corpo de e-mail mais limpo ----
    # corta tudo que vem depois de "Resumo JSON bruto (debug)"
    if "Resumo JSON bruto" in html:
        html_email = html.split("Resumo JSON bruto", 1)[0]
    else:
        html_email = html
    # ---- remover também os gráficos executivos do corpo ----
    if "Gráficos executivos" in html_email:
        html_email = html_email.split("Gráficos executivos", 1)[0]
    # --------------------------------------------------------

    send_email(
        subject=f"[{crit_level}] {crit_headline} – {filename}",
        html_body=html_email,  # usa a versão enxuta no e-mail
        logger=logger,
        attachments=attachments,  # HTML + PNG dos gráficos
        force_recipients=recips,
    )

    title = f"CMD Análises – {crit_headline}"
    msg = (
        f"{len(ctx.get('alerts', []))} alertas • nível {crit_level} "
        f"• tipo: {ctx.get('process_type')}"
    )
    notify(title, msg)

    # ========= FEEDBACK MANUAL DE CRITICIDADE (CSV) =========
    try:
        print("\n============================================")
        print(f"Arquivo analisado: {ctx.get('filename')}")
        print(f"Criticidade automática: {crit_level} - {crit_headline}")
        fb = input(
            "Depois de ver o relatório, de 0 a 10, qual o grau de criticidade que você enxerga nesse arquivo? "
        )
        fb_val = int(fb)
    except Exception:
        fb_val = None

    if fb_val is not None:
        logger.info(
            f"[Feedback] Criticidade manual = {fb_val} para {ctx.get('filename')}"
        )
    # ========================================================


    title = f"CMD Análises – {crit_headline}"
    msg = (
        f"{len(ctx.get('alerts', []))} alertas • nível {crit_level} "
        f"• tipo: {ctx.get('process_type')}"
    )
    notify(title, msg)

    # ========= FEEDBACK MANUAL DE CRITICIDADE (CSV) =========
    try:
        print("\n============================================")
        print(f"Arquivo analisado: {ctx.get('filename')}")
        print(f"Criticidade automática: {crit_level} - {crit_headline}")
        fb = input(
            "Depois de ver o relatório, de 0 a 10, qual o grau de criticidade que você enxerga nesse arquivo? "
        )
        fb_val = int(fb)
    except Exception:
        fb_val = None

    if fb_val is not None:
        logger.info(
            f"[Feedback] Criticidade manual = {fb_val} para {ctx.get('filename')}"
        )
        # aqui depois você pode salvar em JSON para calibrar
    # ========================================================



def main():
    parser = argparse.ArgumentParser(description="CMD Analises Bot - análise automática de CSVs.")
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Gera apenas relatório diário agregado dos arquivos de hoje e encerra.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Processa apenas um arquivo CSV específico (caminho completo) e encerra.",
    )
    args = parser.parse_args()

    cfg = CONFIG
    logger = setup_logger(cfg["output_folder"])
    logger.info("Iniciando CMD_Analises_Bot...")

    chat_client = None
    try:
        chat_client = build_openai_client(cfg, logger)
    except Exception as e:
        logger.error("Falha ao criar client Perplexity/OpenAI: %s", e)

    if args.file:
        process_single_file(args.file, cfg, logger, chat_client)
        return

    if args.daily:
        generate_daily_report(cfg, logger, chat_client)
        return

    input_folder = cfg["input_folder"]
    ensure_dir(input_folder)

    event_handler = CSVHandler_FIXED(cfg, logger, chat_client)
    observer = Observer()
    observer.schedule(event_handler, input_folder, recursive=False)
    observer.start()

    logger.info(f"Watcher iniciado em {input_folder} (modo padrão). Aguardando novos CSVs ou imagens...")
    notify("CMD Análises – Watcher (padrão)", f"Monitorando {input_folder} por novos CSVs.")

    daily_cfg = cfg.get("daily_report", {})
    daily_enabled = daily_cfg.get("enabled", False)
    daily_hour = int(daily_cfg.get("hour", 18))
    last_daily_date = None

    try:
        while True:
            if daily_enabled:
                now = datetime.now()
                if now.hour >= daily_hour:
                    if last_daily_date is None or last_daily_date != now.date():
                        logger.info("Gerando relatório diário automático...")
                        try:
                            generate_daily_report(cfg, logger, chat_client)
                            last_daily_date = now.date()
                        except Exception as e:
                            logger.error(f"Erro ao gerar relatório diário automático: {e}")
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Encerrando watcher...")
        observer.stop()

    observer.join()
    logger.info("CMD_Analises_Bot finalizado.")

if __name__ == "__main__":
    main()
