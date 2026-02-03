# -*- coding: utf-8 -*-
"""
CÉREBRO ANALÍTICO + IA INTERNA (VERSION FIXED)
----------------------------------------------
- Core: orquestração, memória, aprendizado, decisão
- Analytics: métricas, sportsbook, fraude, comparações entre períodos
- Vision: comparação de imagens
- Text Rules: regras inteligentes em blocos de texto
- IA: conector interno (via OpenAI API compatível com Perplexity)

Este módulo foi higienizado para não conflitar com configurações de rede do Windows.
"""

from __future__ import annotations

import os
import json
import time
import math
import hashlib
import sqlite3
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

# ==============================================================================
# 🧼 1. HIGIENIZAÇÃO DE AMBIENTE (NETWORK CLEANER)
# ==============================================================================
# Remove configurações de Proxy/SSL do Windows que bloqueiam a conexão da IA.
# Isso garante que este script rode limpo, sem afetar ou ser afetado pelo shell.
def _sanitize_environment():
    keys_to_remove = ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY"]
    cleaned = False
    for k in keys_to_remove:
        if k in os.environ:
            del os.environ[k]
            cleaned = True
    if cleaned:
        print(">>> [Cérebro] Ambiente de rede higienizado para evitar erros de SSL.")

_sanitize_environment()

# ==============================================================================
# ⚙️ CONFIGURAÇÕES GERAIS
# ==============================================================================

try:
    from openai import OpenAI  # SDK compatível com Perplexity
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False
    print(">>> [Aviso] Biblioteca 'openai' não instalada. Funcionalidades de IA desativadas.")

# Tratamento de API Key: remove espaços em branco que causam erro 401
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("BRAIN_IA_MODEL", "gpt-4o") # Ajustado para modelo padrão vision/texto

# Pasta de regras em texto (memória do cérebro) - Uso de raw string (r"") para Windows
CEREBRO_DIR = r"C:\Olho Vivo\cerebro"

# =========================
# Utils
# =========================

def now_ts() -> float:
    return time.time()

def iso_time(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = now_ts()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# =========================
# Estruturas principais
# =========================

@dataclass
class BrainEvent:
    ts: float
    type: str
    brand: str = ""
    source: str = ""
    payload: Dict[str, Any] = None
    meta: Dict[str, Any] = None

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}
        if self.meta is None:
            self.meta = {}

@dataclass
class BrainResult:
    ok: bool
    route: str
    severity: str
    headline: str
    details: Dict[str, Any]
    recommendations: List[str]
    learned: Dict[str, Any]

# =========================
# Memória (SQLite + JSON)
# =========================

class Memory:
    def __init__(self, db_path: str = "brain_memory.db", json_shadow_path: str = "brain_memory_shadow.json"):
        self.db_path = db_path
        self.json_shadow_path = json_shadow_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._create_tables()
        self._shadow_cache: Dict[str, Any] = self._load_shadow()

    def _create_tables(self):
        cur = self._conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            iso TEXT,
            type TEXT,
            brand TEXT,
            source TEXT,
            payload_json TEXT,
            meta_json TEXT,
            hash TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts REAL,
            iso TEXT,
            name TEXT,
            route TEXT,
            enabled INTEGER,
            rule_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts REAL,
            iso TEXT,
            kind TEXT,
            key TEXT,
            value_json TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            updated_ts REAL,
            iso TEXT,
            brand TEXT,
            metric TEXT,
            window TEXT,
            mean REAL,
            std REAL,
            p95 REAL,
            n INTEGER
        );
        """)
        self._conn.commit()

    def _load_shadow(self) -> Dict[str, Any]:
        if os.path.exists(self.json_shadow_path):
            try:
                with open(self.json_shadow_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"learning": [], "patterns": {}, "stats": {}}
        return {"learning": [], "patterns": {}, "stats": {}}

    def persist_shadow(self):
        try:
            with open(self.json_shadow_path, "w", encoding="utf-8") as f:
                json.dump(self._shadow_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Eventos
    def add_event(self, ev: BrainEvent) -> int:
        payload_json = json.dumps(ev.payload or {}, ensure_ascii=False)
        meta_json = json.dumps(ev.meta or {}, ensure_ascii=False)
        h = sha1_text(f"{ev.ts}|{ev.type}|{ev.brand}|{payload_json}|{meta_json}")
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO events (ts, iso, type, brand, source, payload_json, meta_json, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ev.ts, iso_time(ev.ts), ev.type, ev.brand, ev.source, payload_json, meta_json, h))
        self._conn.commit()
        return cur.lastrowid

    def get_recent_events(
        self,
        limit: int = 200,
        type_filter: Optional[str] = None,
        brand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        q = "SELECT ts, iso, type, brand, source, payload_json, meta_json FROM events"
        cond: List[str] = []
        params: List[Any] = []
        if type_filter:
            cond.append("type = ?")
            params.append(type_filter)
        if brand:
            cond.append("brand = ?")
            params.append(brand)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = cur.execute(q, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "ts": r[0],
                "iso": r[1],
                "type": r[2],
                "brand": r[3],
                "source": r[4],
                "payload": json.loads(r[5] or "{}"),
                "meta": json.loads(r[6] or "{}"),
            })
        return out

    def get_events_by_time_window(
        self,
        brand: Optional[str],
        type_filter: Optional[str],
        since_ts: float,
        until_ts: float,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        q = "SELECT ts, iso, type, brand, source, payload_json, meta_json FROM events WHERE ts BETWEEN ? AND ?"
        params: List[Any] = [since_ts, until_ts]
        if type_filter:
            q += " AND type = ?"
            params.append(type_filter)
        if brand:
            q += " AND brand = ?"
            params.append(brand)
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = cur.execute(q, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "ts": r[0],
                "iso": r[1],
                "type": r[2],
                "brand": r[3],
                "source": r[4],
                "payload": json.loads(r[5] or "{}"),
                "meta": json.loads(r[6] or "{}"),
            })
        return out

    # Rules
    def upsert_rule(self, name: str, route: str, rule: Dict[str, Any], enabled: bool = True) -> int:
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO rules (created_ts, iso, name, route, enabled, rule_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now_ts(), iso_time(), name, route, 1 if enabled else 0, json.dumps(rule, ensure_ascii=False)))
        self._conn.commit()
        return cur.lastrowid

    def list_rules(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        if enabled_only:
            rows = cur.execute("""
                SELECT id, iso, name, route, enabled, rule_json
                FROM rules WHERE enabled = 1
                ORDER BY id DESC
            """).fetchall()
        else:
            rows = cur.execute("""
                SELECT id, iso, name, route, enabled, rule_json
                FROM rules ORDER BY id DESC
            """).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": r[0],
                "iso": r[1],
                "name": r[2],
                "route": r[3],
                "enabled": bool(r[4]),
                "rule": json.loads(r[5] or "{}"),
            })
        return out

    # Patterns
    def save_pattern(self, kind: str, key: str, value: Dict[str, Any]):
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO patterns (created_ts, iso, kind, key, value_json)
            VALUES (?, ?, ?, ?, ?)
        """, (now_ts(), iso_time(), kind, key, json.dumps(value, ensure_ascii=False)))
        self._conn.commit()
        self._shadow_cache.setdefault("patterns", {}).setdefault(kind, {})
        self._shadow_cache["patterns"][kind][key] = value
        self.persist_shadow()

    def get_pattern(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        v = self._shadow_cache.get("patterns", {}).get(kind, {}).get(key)
        if v is not None:
            return v
        cur = self._conn.cursor()
        row = cur.execute("""
            SELECT value_json FROM patterns
            WHERE kind = ? AND key = ?
            ORDER BY id DESC LIMIT 1
        """, (kind, key)).fetchone()
        if not row:
            return None
        return json.loads(row[0] or "{}")

    # Baseline
    def upsert_baseline(self, brand: str, metric: str, window: str, mean: float, std: float, p95: float, n: int):
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO baselines (updated_ts, iso, brand, metric, window, mean, std, p95, n)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_ts(), iso_time(), brand, metric, window, mean, std, p95, n))
        self._conn.commit()

    def get_latest_baseline(self, brand: str, metric: str, window: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        row = cur.execute("""
            SELECT updated_ts, iso, mean, std, p95, n
            FROM baselines
            WHERE brand = ? AND metric = ? AND window = ?
            ORDER BY id DESC LIMIT 1
        """, (brand, metric, window)).fetchone()
        if not row:
            return None
        return {"updated_ts": row[0], "iso": row[1], "mean": row[2], "std": row[3], "p95": row[4], "n": row[5]}

# =========================
# Learning Engine
# =========================

class LearningEngine:
    def __init__(self, memory: Memory):
        self.memory = memory

    def learn(self, ev: BrainEvent, result: BrainResult) -> Dict[str, Any]:
        entry = {
            "ts": ev.ts,
            "iso": iso_time(ev.ts),
            "type": ev.type,
            "brand": ev.brand,
            "route": result.route,
            "severity": result.severity,
            "headline": result.headline,
        }
        self.memory._shadow_cache.setdefault("learning", []).append(entry)
        if len(self.memory._shadow_cache["learning"]) > 2000:
            self.memory._shadow_cache["learning"] = self.memory._shadow_cache["learning"][-2000:]
        self.memory.persist_shadow()

        if result.severity in ("YELLOW", "RED"):
            key = f"{ev.brand}:{result.route}"
            self.memory.save_pattern("alerts", key, {
                "last_ts": ev.ts,
                "last_iso": iso_time(ev.ts),
                "severity": result.severity,
                "headline": result.headline,
                "details": result.details,
            })

        return {"saved_learning": True}

    def fit_baseline_from_samples(
        self, brand: str, metric: str, window: str, samples: List[float]
    ) -> Optional[Dict[str, Any]]:
        if not samples:
            return None
        s = sorted(samples)
        n = len(s)
        mean = sum(s) / n
        var = sum((x - mean) ** 2 for x in s) / max(1, (n - 1))
        std = math.sqrt(var)
        p95 = s[int(clamp(math.floor(0.95 * (n - 1)), 0, n - 1))]
        self.memory.upsert_baseline(brand=brand, metric=metric, window=window, mean=mean, std=std, p95=p95, n=n)
        return {"brand": brand, "metric": metric, "window": window, "mean": mean, "std": std, "p95": p95, "n": n}

# =========================
# Decision Engine
# =========================

class DecisionEngine:
    def __init__(self, memory: Memory):
        self.memory = memory

    def route(self, ev: BrainEvent) -> str:
        for r in self.memory.list_rules(enabled_only=True):
            rule = r["rule"]
            if self._match_rule(ev, rule):
                return r["route"]

        t = (ev.type or "").lower()
        if t in ("metrics", "kpis", "anomaly"):
            return "analytics.metrics"
        if t in ("fraud", "risk"):
            return "analytics.fraud"
        if t in ("vision", "image", "compare"):
            return "vision.compare"
        if t in ("train", "baseline"):
            return "learning.baseline"
        if t in ("sportsbook", "bets", "apostas"):
            return "analytics.sportsbook"
        if t in ("compare_periods", "investigacao", "investigacao_periodos"):
            return "analytics.compare_periods"
        return "core.unknown"

    def _match_rule(self, ev: BrainEvent, rule: Dict[str, Any]) -> bool:
        when = rule.get("when", {})
        if "type" in when and (ev.type != when["type"]):
            return False
        if "brand" in when and (ev.brand != when["brand"]):
            return False
        if "payload.has" in when:
            need = when["payload.has"]
            if need not in (ev.payload or {}):
                return False
        return True

# =========================
# Metrics Engine
# =========================

class MetricsEngine:
    def __init__(self, memory: Memory):
        self.memory = memory

    def analyze(self, ev: BrainEvent) -> Tuple[str, Dict[str, Any], List[str]]:
        payload = ev.payload or {}
        window = payload.get("window", "5m")
        metrics = payload.get("metrics", {})
        samples = payload.get("samples")
        metric_name_for_baseline = payload.get("baseline_metric")

        details: Dict[str, Any] = {"window": window, "metrics": metrics}
        recs: List[str] = []

        if isinstance(samples, list) and metric_name_for_baseline:
            fitted = LearningEngine(self.memory).fit_baseline_from_samples(
                brand=ev.brand or "UNKNOWN",
                metric=metric_name_for_baseline,
                window=window,
                samples=[safe_float(x) for x in samples],
            )
            details["baseline_fitted"] = fitted
            recs.append(f"Baseline atualizado para {metric_name_for_baseline} ({window}).")

        anomaly_hits = []
        for m, v in (metrics or {}).items():
            v = safe_float(v, default=None)
            if v is None:
                continue
            b = self.memory.get_latest_baseline(ev.brand or "UNKNOWN", m, window)
            if not b:
                continue
            mean = safe_float(b["mean"])
            std = max(1e-9, safe_float(b["std"], 0.0))
            z = (v - mean) / std if std > 0 else 0.0
            if v > safe_float(b["p95"]) or z >= 3.0:
                anomaly_hits.append({
                    "metric": m,
                    "value": v,
                    "mean": mean,
                    "std": std,
                    "p95": b["p95"],
                    "z": z,
                })

        details["anomaly_hits"] = anomaly_hits
        if len(anomaly_hits) >= 3:
            sev = "RED"
            recs.append("Múltiplas métricas anômalas, revisar processo completo.")
        elif len(anomaly_hits) >= 1:
            sev = "YELLOW"
            recs.append("Anomalia isolada detectada, investigar contexto.")
        else:
            sev = "GREEN"
            recs.append("Sem anomalias relevantes.")

        return sev, details, recs

# =========================
# Fraud Detector
# =========================

class FraudDetector:
    def __init__(self, memory: Memory):
        self.memory = memory

    def score(self, ev: BrainEvent) -> Tuple[str, Dict[str, Any], List[str]]:
        p = ev.payload or {}
        profile = p.get("profile", {}) or {}
        signals = p.get("signals", {}) or {}

        deposit = safe_float(profile.get("deposit", 0))
        withdraw = safe_float(profile.get("withdraw", 0))
        avg_ticket = safe_float(profile.get("avg_ticket", profile.get("avg", 0)))
        ftd = safe_float(profile.get("ftd", 0))

        score = 0.0
        reasons: List[str] = []

        if deposit > 0 and withdraw >= deposit * 0.9:
            score += 2.0
            reasons.append("Saque alto vs depósito (>= 90%).")
        if avg_ticket >= 1000:
            score += 1.0
            reasons.append("Ticket médio elevado (>= 1000).")
        if ftd >= 500 and withdraw >= 0.8 * ftd:
            score += 1.0
            reasons.append("FTD alto com saque próximo do FTD.")
        if signals.get("many_accounts_same_device"):
            score += 2.0
            reasons.append("Múltiplas contas no mesmo device.")
        if signals.get("kyc_failed_many_times"):
            score += 1.5
            reasons.append("Muitas reprovações no KYC.")
        if signals.get("fast_deposit_withdraw"):
            score += 1.5
            reasons.append("Depósito e saque em curto período.")
        if signals.get("freespin_abuse"):
            score += 2.0
            reasons.append("Uso intenso de freespin em curto prazo e alto valor.")
        if signals.get("sportsbook_matchfix_risk"):
            score += 2.0
            reasons.append("Padrão suspeito de apostas em partida específica (match-fixing).")

        score = clamp(score, 0.0, 10.0)

        if score >= 5.0:
            sev = "RED"
        elif score >= 2.5:
            sev = "YELLOW"
        else:
            sev = "GREEN"

        details = {
            "user": p.get("user", {}),
            "profile": profile,
            "signals": signals,
            "risk_score": score,
            "reasons": reasons,
        }

        recs: List[str] = []
        if sev == "RED":
            recs += [
                "Segurar pagamento se política permitir.",
                "Revisar device/IP/cadastro/KYC.",
                "Cruzar com campanhas, afiliados e padrão de jogo.",
            ]
        elif sev == "YELLOW":
            recs += [
                "Monitorar nas próximas janelas.",
                "Coletar mais evidências (device/IP/velocity).",
            ]
        else:
            recs += ["Perfil sem sinais fortes de fraude no modelo atual."]

        return sev, details, recs

# =========================
# Sportsbook Engine
# =========================

class SportsbookEngine:
    def __init__(self, memory: Memory):
        self.memory = memory

    def analyze(self, ev: BrainEvent) -> Tuple[str, Dict[str, Any], List[str]]:
        payload = ev.payload or {}
        window = payload.get("window", "5m")
        metrics = payload.get("metrics", {}) or {}
        signals = payload.get("signals", {}) or {}

        details: Dict[str, Any] = {"window": window, "metrics": metrics, "signals": signals}
        recs: List[str] = []
        sev = "GREEN"
        critical_flags: List[str] = []

        hlwr = safe_float(metrics.get("live_win_rate", 0))
        if hlwr >= 0.75:
            sev = "YELLOW"
            critical_flags.append("KYC/Liveness está demorando mais ou live bets com win-rate alto.")
            recs.append("Comparar tempo médio de hoje vs ontem e checar se há fila ou bug de settlement.")

        scash = safe_float(metrics.get("cashout_suspicious_rate", 0))
        if scash >= 0.20:
            sev = "RED"
            critical_flags.append("Muitos cashouts suspeitos em janela curta.")
            recs.append("Investigar cashouts por usuário, jogo, liga e horário.")

        if signals.get("same_bet_many_users"):
            sev = "RED"
            critical_flags.append("Muitos usuários com mesma aposta (possível informação privilegiada).")
            recs.append("Cruzar contas, dispositivos e afiliação para detecção de rede.")

        if signals.get("odd_inconsistency"):
            sev = "YELLOW"
            critical_flags.append("Odds divergentes do feed/provedores.")
            recs.append("Comparar odds internas com provedores e corrigir latência/gargalo de atualização.")

        if not recs:
            recs.append("Sem criticidades fortes detectadas em sportsbook na janela atual.")

        details["critical_flags"] = critical_flags
        return sev, details, recs

# =========================
# Comparação entre períodos
# =========================

class PeriodComparator:
    def __init__(self, memory: Memory):
        self.memory = memory

    def compare(self, ev: BrainEvent) -> Tuple[str, Dict[str, Any], List[str]]:
        p = ev.payload or {}
        brand = ev.brand or p.get("brand")
        type_filter = p.get("type_filter")
        seconds = int(p.get("window_seconds", 3600))
        now = now_ts()
        today_events = self.memory.get_events_by_time_window(
            brand=brand,
            type_filter=type_filter,
            since_ts=now - seconds,
            until_ts=now,
            limit=500,
        )
        yesterday_events = self.memory.get_events_by_time_window(
            brand=brand,
            type_filter=type_filter,
            since_ts=now - seconds * 2 - 24 * 3600,
            until_ts=now - seconds - 24 * 3600,
            limit=500,
        )

        details: Dict[str, Any] = {
            "current_window_events": len(today_events),
            "previous_window_events": len(yesterday_events),
            "brand": brand,
            "type_filter": type_filter,
        }
        recs: List[str] = []
        if not today_events and not yesterday_events:
            return "GREEN", details, ["Sem dados suficientes para comparação de períodos."]

        diff = len(today_events) - len(yesterday_events)
        details["count_diff"] = diff
        if diff > 0:
            recs.append(f"Volume de eventos {type_filter or 'gerais'} aumentou em {diff} na janela atual vs ontem.")
        elif diff < 0:
            recs.append(f"Volume de eventos {type_filter or 'gerais'} reduziu em {-diff} na janela atual vs ontem.")
        else:
            recs.append("Volume de eventos está estável vs ontem na mesma janela.")

        sev = "GREEN"
        if abs(diff) > max(10, 0.3 * max(1, len(yesterday_events))):
            sev = "YELLOW"
            recs.append("Variação relevante de volume, investigar processos críticos.")

        return sev, details, recs

# =========================
# Vision (imagens)
# =========================

class ImageComparator:
    def __init__(self, memory: Memory):
        self.memory = memory
        self._cv2 = None
        self._cosine = None
        # Importação segura de dependências pesadas
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            pass
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            self._cosine = cosine_similarity
        except ImportError:
            pass

    def compare(self, ev: BrainEvent) -> Tuple[str, Dict[str, Any], List[str]]:
        p = ev.payload or {}
        img1 = p.get("img1")
        img2 = p.get("img2")
        if not img1 or not img2:
            return "YELLOW", {"error": "img1/img2 não informados"}, ["Informe caminhos de img1 e img2."]
        if (not os.path.exists(img1)) or (not os.path.exists(img2)):
            return "YELLOW", {"error": "arquivo não encontrado", "img1": img1, "img2": img2}, ["Verifique os caminhos das imagens."]

        if self._cv2 is not None and self._cosine is not None:
            cv2 = self._cv2
            cosine_similarity = self._cosine

            def emb(path: str):
                img = cv2.imread(path)
                # Se falhar ao ler (arquivo corrompido), evitar crash
                if img is None:
                    return None
                img = cv2.resize(img, (128, 128))
                e = img.flatten().astype("float32") / 255.0
                return e

            e1 = emb(img1)
            e2 = emb(img2)
            
            if e1 is None or e2 is None:
                 return "YELLOW", {"error": "falha leitura imagem"}, ["Imagem corrompida ou formato inválido."]

            sim = float(cosine_similarity([e1], [e2])[0][0])
            details = {"similarity": sim, "method": "cv2+cosine"}
        else:
            # Fallback para hash simples se libs não existirem
            def file_hash(path: str) -> str:
                h = hashlib.sha1()
                try:
                    with open(path, "rb") as f:
                        while True:
                            b = f.read(1024 * 1024)
                            if not b:
                                break
                            h.update(b)
                except Exception:
                    return ""
                return h.hexdigest()
            h1 = file_hash(img1)
            h2 = file_hash(img2)
            sim = 1.0 if h1 and h2 and h1 == h2 else 0.0
            details = {"similarity": sim, "method": "file_hash"}

        if sim >= 0.90:
            sev = "GREEN"
            recs = ["Imagens muito semelhantes."]
        elif sim >= 0.75:
            sev = "YELLOW"
            recs = ["Imagens parcialmente semelhantes, revisar manualmente."]
        else:
            sev = "RED"
            recs = ["Imagens pouco semelhantes, possível fraude documental dependendo do contexto."]
        return sev, details, recs

# =========================
# IA interna (OpenAI/Perplexity)
# =========================

class IAConnector:
    def __init__(self, enabled: bool = True):
        # Verifica se temos biblioteca e chave válida (não vazia)
        self.enabled = enabled and HAVE_OPENAI and bool(OPENAI_API_KEY)
        self._client: Optional[OpenAI] = None
        if self.enabled:
            try:
                self._client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            except Exception:
                self.enabled = False
                print(">>> [Cérebro] Falha ao iniciar cliente OpenAI. IA desativada.")

    def summarize(self, context: Dict[str, Any]) -> Optional[str]:
        if not self.enabled or self._client is None:
            return None
        try:
            system_prompt = (
                "Você é um cérebro analítico de monitoramento de iGaming/sportsbook. "
                "Responda SEMPRE em português (pt-BR), em tom investigativo, direto e operacional. "
                "Devolva tópicos curtos."
            )
            payload_compacto = json.dumps(context, ensure_ascii=False)[:9000]

            resp = self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload_compacto},
                ],
                temperature=0.25,
                max_tokens=700,
            )
            text = resp.choices[0].message.content
            return text.strip() if isinstance(text, str) else str(text)
        except Exception:
            # Em caso de erro de rede ou chave, silencia para não travar o processo principal
            return None

# =========================
# RuleTextEngine
# =========================

class RuleTextEngine:
    def __init__(self, base_dir: str = CEREBRO_DIR):
        self.base_dir = base_dir
        self.regras: List[Dict[str, Any]] = []
        self._load_regras()

    def _load_regras(self):
        # Cria pasta se não existir, evitando erro
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            self.regras.clear()
            for fname in os.listdir(self.base_dir):
                if not fname.lower().endswith((".txt", ".md")):
                    continue
                path = os.path.join(self.base_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.regras.append({
                        "arquivo": path,
                        "nome": fname,
                        "conteudo": content,
                    })
                except Exception:
                    continue
        except Exception:
            pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_regras": len(self.regras),
            "arquivos": [r["arquivo"] for r in self.regras],
        }

    def apply_with_ia(
        self,
        ia: IAConnector,
        ev: BrainEvent,
        engine_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not ia.enabled or ia._client is None:
            return {}

        regras_texto = "\n\n".join(
            [f"### {r['nome']}\n{r['conteudo']}" for r in self.regras]
        )[:12000]

        context = {
            "event": asdict(ev),
            "engine_details": engine_details,
        }

        system_prompt = (
            "Você é o CÉREBRO de fraude/risco.\n"
            "Responda APENAS em JSON válido com o formato:\n"
            "{\n"
            '  "alertas_extras": [ {"titulo": str, "descricao": str, "criticidade": "MEDIA"|"ALTA"} ],\n'
            '  "explicacao": "texto curto"\n'
            "}\n"
        )
        user_content = (
            f"REGRAS:\n{regras_texto}\n\n"
            f"DADOS:\n{json.dumps(context, ensure_ascii=False)[:8000]}"
        )

        try:
            resp = ia._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"} # Força JSON se modelo suportar
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)
        except Exception:
            return {}

# =========================
# Brain (orquestrador)
# =========================

class Brain:
    def __init__(
        self,
        db_path: str = "brain_memory.db",
        json_shadow_path: str = "brain_memory_shadow.json",
        ia_enabled: bool = True,
    ):
        self.memory = Memory(db_path=db_path, json_shadow_path=json_shadow_path)
        self.learning = LearningEngine(self.memory)
        self.decision = DecisionEngine(self.memory)

        self.metrics = MetricsEngine(self.memory)
        self.fraud = FraudDetector(self.memory)
        self.sportsbook = SportsbookEngine(self.memory)
        self.comparator = PeriodComparator(self.memory)
        self.vision = ImageComparator(self.memory)

        self.ia = IAConnector(enabled=ia_enabled)
        self.rules_text = RuleTextEngine(base_dir=CEREBRO_DIR)

        self._ensure_default_rules()

    def _ensure_default_rules(self):
        if self.memory.list_rules(enabled_only=False):
            return
        # Regras padrão
        defaults = [
            ("Route metrics", "analytics.metrics", {"when": {"type": "metrics"}, "then": {"route": "analytics.metrics"}}),
            ("Route fraud", "analytics.fraud", {"when": {"type": "fraud"}, "then": {"route": "analytics.fraud"}}),
            ("Route vision", "vision.compare", {"when": {"type": "vision"}, "then": {"route": "vision.compare"}}),
            ("Route baseline", "learning.baseline", {"when": {"type": "train"}, "then": {"route": "learning.baseline"}}),
            ("Route sportsbook", "analytics.sportsbook", {"when": {"type": "sportsbook"}, "then": {"route": "analytics.sportsbook"}}),
            ("Route compare", "analytics.compare_periods", {"when": {"type": "compare_periods"}, "then": {"route": "analytics.compare_periods"}}),
        ]
        for name, route, rule in defaults:
            self.memory.upsert_rule(name, route, rule, enabled=True)

    def think(self, command: Union[Dict[str, Any], BrainEvent]) -> BrainResult:
        if isinstance(command, BrainEvent):
            ev = command
        else:
            ev = BrainEvent(
                ts=now_ts(),
                type=command.get("type", "command"),
                brand=command.get("brand", ""),
                source=command.get("source", "external"),
                payload=command.get("payload", command.get("data", {})) or {},
                meta=command.get("meta", {}) or {},
            )

        self.memory.add_event(ev)
        route = self.decision.route(ev)
        sev, details, recs = self._execute_route(route, ev)

        # Resumo IA (Opcional)
        if self.ia.enabled:
             context_for_ia = {
                "event": asdict(ev),
                "severity": sev,
                "details": details,
            }
             ia_summary = self.ia.summarize(context_for_ia)
             if ia_summary:
                 details["ia_summary"] = ia_summary

        headline = self._headline(route, sev, details)
        result = BrainResult(
            ok=True,
            route=route,
            severity=sev,
            headline=headline,
            details=details,
            recommendations=recs,
            learned={},
        )
        learned = self.learning.learn(ev, result)
        result.learned = learned
        return result

    def _execute_route(self, route: str, ev: BrainEvent) -> Tuple[str, Dict[str, Any], List[str]]:
        if route == "analytics.metrics":
            sev, details, recs = self.metrics.analyze(ev)
        elif route == "analytics.fraud":
            sev, details, recs = self.fraud.score(ev)
        elif route == "analytics.sportsbook":
            sev, details, recs = self.sportsbook.analyze(ev)
        elif route == "analytics.compare_periods":
            sev, details, recs = self.comparator.compare(ev)
        elif route == "vision.compare":
            sev, details, recs = self.vision.compare(ev)
        elif route == "learning.baseline":
            return self._train_baseline(ev)
        else:
            return "YELLOW", {"warning": "rota desconhecida"}, ["Rota indefinida."]

        # Regras de texto (IA extra)
        if self.ia.enabled:
             extra = self.rules_text.apply_with_ia(self.ia, ev, details)
             if extra:
                 details["brain_text_rules"] = extra
                 for a in extra.get("alertas_extras", []):
                     crit = (a.get("criticidade") or "").upper()
                     if crit == "ALTA":
                         sev = "RED"
                     elif crit == "MEDIA" and sev == "GREEN":
                         sev = "YELLOW"
        
        return sev, details, recs

    def _train_baseline(self, ev: BrainEvent) -> Tuple[str, Dict[str, Any], List[str]]:
        p = ev.payload or {}
        brand = ev.brand or p.get("brand", "UNKNOWN")
        metric = p.get("metric")
        window = p.get("window", "5m")
        samples = p.get("samples", [])
        if not metric or not isinstance(samples, list) or len(samples) < 10:
            return "YELLOW", {"error": "amostras insuficientes"}, ["Mínimo 10 amostras necessárias."]
        
        fitted = self.learning.fit_baseline_from_samples(
            brand=brand, metric=metric, window=window, samples=[safe_float(x) for x in samples]
        )
        return "GREEN", {"trained": fitted}, ["Baseline treinado."]

    def _headline(self, route: str, severity: str, details: Dict[str, Any]) -> str:
        # Simplificado para evitar quebras
        if route == "analytics.metrics":
             return f"Métricas: {severity}"
        if route == "analytics.fraud":
             return f"Risco: {details.get('risk_score', 0)}"
        return f"{route} processado ({severity})"

# =========================
# Demonstração rápida
# =========================

if __name__ == "__main__":
    # Esta parte só roda se você executar o arquivo diretamente.
    # Se importar, nada acontece.
    print(">>> Iniciando Cérebro em modo de teste...")
    brain = Brain(ia_enabled=True)
    
    # Teste básico
    res = brain.think({
        "type": "metrics", 
        "brand": "Teste", 
        "payload": {"metrics": {"deposits": 100}}
    })
    print(f"Resultado: {res.headline}")
