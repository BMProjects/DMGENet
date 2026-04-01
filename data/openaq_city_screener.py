"""
OpenAQ 城市筛选器
实现 doc/openaq_city_screening_protocol.md 定义的 5 层筛选协议

用法:
  python data/openaq_city_screener.py              # Phase 1: 元数据筛选
  python data/openaq_city_screener.py --phase 2    # Phase 2: 下载通过城市的完整数据
  python data/openaq_city_screener.py --phase 2 --city Bangkok

API Key 从项目根目录 .env 文件读取 (OPENAQ_API_KEY=...)
"""

import os, sys, json, time, math, argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from openaq import OpenAQ
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.environ.get("OPENAQ_API_KEY", "")
if not API_KEY:
    sys.exit("错误: 未找到 OPENAQ_API_KEY，请在项目根目录 .env 文件中设置")

ROOT      = Path(__file__).resolve().parent.parent
DOC_DIR   = ROOT / "doc"
DATA_DIR  = ROOT / "dataset" / "OpenAQ_Cities"
CACHE_DIR = ROOT / ".openaq_cache"
for d in (DOC_DIR, DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── 候选城市池 ────────────────────────────────────────────────────────────────
# 聚焦快速工业化/高污染城市；Tier A/B/C 由数据质量决定，用户在结果中人工复选
CANDIDATE_CITIES = [
    # 南亚
    {"city": "Mumbai",      "country": "IN"},
    {"city": "Kolkata",     "country": "IN"},
    {"city": "Chennai",     "country": "IN"},
    {"city": "Hyderabad",   "country": "IN"},
    {"city": "Pune",        "country": "IN"},
    {"city": "Ahmedabad",   "country": "IN"},
    {"city": "Lahore",      "country": "PK"},
    {"city": "Karachi",     "country": "PK"},
    {"city": "Dhaka",       "country": "BD"},
    {"city": "Kathmandu",   "country": "NP"},
    # 东南亚
    {"city": "Bangkok",     "country": "TH"},
    {"city": "Jakarta",     "country": "ID"},
    {"city": "Hanoi",       "country": "VN"},
    {"city": "Manila",      "country": "PH"},
    # 中亚 / 东亚
    {"city": "Ulaanbaatar", "country": "MN"},
    {"city": "Chengdu",     "country": "CN"},
    {"city": "Wuhan",       "country": "CN"},
    # 中东
    {"city": "Tehran",      "country": "IR"},
    {"city": "Riyadh",      "country": "SA"},
    # 非洲
    {"city": "Nairobi",     "country": "KE"},
    {"city": "Accra",       "country": "GH"},
    {"city": "Cairo",       "country": "EG"},
    # 南美
    {"city": "Bogota",      "country": "CO"},
    {"city": "Lima",        "country": "PE"},
    {"city": "Santiago",    "country": "CL"},
    {"city": "Sao Paulo",   "country": "BR"},
]

# ─── 协议参数 ─────────────────────────────────────────────────────────────────
REQUIRED_PARAMS   = {"pm25", "pm10", "no2", "so2", "o3", "co"}
MIN_STATIONS      = 8       # 协议 6.1-A
MIN_COMMON_HOURS  = 8760    # 1 年, 协议 6.3
PM25_COV_THRESH   = 0.85    # 协议 6.5
JOINT_COV_THRESH  = 0.65    # 协议 6.5
SYNC_THRESH       = 0.70    # 协议 6.4: ≥70% 时刻有≥70% 站点在线
SYNC_STATION_FRAC = 0.70

# 评分权重 (协议 7.4 补充量化)
W_STATIONS = 0.25
W_TIME     = 0.30
W_FEATURE  = 0.25
W_SYNC     = 0.20

# SDK 返回的单位直接使用，无需手动过滤（官方 SDK 保证 µg/m³ / ppm 等标准单位）


# ─── 缓存 (断点续传) ──────────────────────────────────────────────────────────

def _load(key):
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None

def _save(key, obj):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(obj, default=str))


# ─── Module 1: 城市候选发现 ───────────────────────────────────────────────────

def module1_discover(client: OpenAQ) -> list[dict]:
    """拉取各候选城市的 location 列表，初步过滤站点数和坐标完整率"""
    print("\n" + "="*60)
    print("Module 1: 城市候选发现")
    print("="*60)

    rows = []
    for entry in CANDIDATE_CITIES:
        city, country = entry["city"], entry["country"]
        key = f"m1_{country}_{city}"
        locs_raw = _load(key)

        if locs_raw is not None:
            print(f"  [缓存] {city}, {country}: {len(locs_raw)} 站")
        else:
            print(f"  查询 {city}, {country} ...", end=" ", flush=True)
            locs_raw = []
            page = 1
            while True:
                resp = client.locations.list(
                    city=city, country=country,
                    limit=200, page=page
                )
                batch = resp.results
                locs_raw.extend([
                    {
                        "id":       loc.id,
                        "name":     loc.name,
                        "lat":      loc.coordinates.latitude  if loc.coordinates else None,
                        "lon":      loc.coordinates.longitude if loc.coordinates else None,
                        "provider": loc.provider.name if loc.provider else "unknown",
                        "is_monitor": getattr(loc, "is_monitor", True),
                        "sensors":  [
                            {
                                "id":    s.id,
                                "param": s.parameter.name,
                                "units": s.parameter.units,
                                "display": s.parameter.display_name,
                            }
                            for s in (loc.sensors or [])
                        ],
                        "datetime_first": str(loc.datetime_first.utc) if getattr(loc, "datetime_first", None) else "",
                        "datetime_last":  str(loc.datetime_last.utc)  if getattr(loc, "datetime_last",  None) else "",
                    }
                    for loc in batch
                ])
                if len(batch) < 200:
                    break
                page += 1
            _save(key, locs_raw)
            print(f"{len(locs_raw)} 站")
            time.sleep(0.5)

        if not locs_raw:
            continue

        n_total = len(locs_raw)
        n_coords = sum(1 for l in locs_raw if l["lat"] and l["lon"])
        providers = {l["provider"] for l in locs_raw}

        # 最近数据时间（时间临近性，供用户人工参考）
        last_dates = [l["datetime_last"] for l in locs_raw if l["datetime_last"]]
        most_recent = max(last_dates) if last_dates else ""

        rows.append({
            "city": city, "country": country,
            "station_count": n_total,
            "geo_coverage":  round(n_coords / n_total, 3) if n_total else 0,
            "provider_count": len(providers),
            "providers":     "; ".join(sorted(providers)),
            "most_recent_data": most_recent[:10],  # 用户判断时间临近性
            "passes_m1": n_total >= MIN_STATIONS and (n_coords / n_total >= 0.9 if n_total else False),
            "_locs": locs_raw,
        })

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_locs"} for r in rows])
    df = df.sort_values("station_count", ascending=False)
    df.to_csv(DOC_DIR / "openaq_m1_discovery.csv", index=False)
    print(f"\n  保存: {DOC_DIR / 'openaq_m1_discovery.csv'}")

    passing = df[df["passes_m1"]]
    print(f"  通过 M1 ({MIN_STATIONS}+ 站点 + 坐标≥90%): {len(passing)} / {len(df)} 城市")
    for _, r in passing.iterrows():
        recency = f"  最新数据: {r['most_recent_data']}" if r['most_recent_data'] else ""
        print(f"    ✓ {r['city']:15s} {r['country']}  {r['station_count']} 站{recency}")

    return rows


# ─── Module 2: 站点变量画像 ───────────────────────────────────────────────────

def module2_profile(m1_rows: list[dict]) -> list[dict]:
    """从 M1 结果中解析各站点的变量覆盖情况（无需额外 API 调用，利用 locations 的 sensors 字段）"""
    print("\n" + "="*60)
    print("Module 2: 站点与变量画像")
    print("="*60)

    profile_rows = []
    for entry in m1_rows:
        if not entry["passes_m1"]:
            continue
        city, country = entry["city"], entry["country"]
        print(f"  {city}, {country}")

        for loc in entry["_locs"]:
            if not loc["lat"] or not loc["lon"]:
                continue
            # 从 sensors 列表中归纳该站点具备的参数
            params_present = {s["param"].lower().replace(".", "")
                              for s in loc["sensors"]}
            # 规范化: pm2.5 → pm25
            params_norm = set()
            for p in params_present:
                p2 = p.replace(".", "").replace(" ", "")
                if p2 in ("pm25", "pm2.5"):    params_norm.add("pm25")
                elif p2 == "pm10":             params_norm.add("pm10")
                elif p2 == "no2":              params_norm.add("no2")
                elif p2 == "so2":              params_norm.add("so2")
                elif p2 in ("o3", "ozone"):    params_norm.add("o3")
                elif p2 == "co":               params_norm.add("co")

            n_req = len(params_norm & REQUIRED_PARAMS)
            # sensor_id map: param → sensor_id (SDK 已提供)
            sensor_map = {}
            for s in loc["sensors"]:
                p = s["param"].lower().replace(".", "").replace(" ", "")
                if p in ("pm25", "pm2.5"):  key = "pm25"
                elif p in ("pm10",):        key = "pm10"
                elif p in ("no2",):         key = "no2"
                elif p in ("so2",):         key = "so2"
                elif p in ("o3","ozone"):   key = "o3"
                elif p in ("co",):          key = "co"
                else:                       continue
                if key not in sensor_map:
                    sensor_map[key] = s["id"]

            profile_rows.append({
                "city": city, "country": country,
                "location_id":   loc["id"],
                "location_name": loc["name"],
                "lat": loc["lat"], "lon": loc["lon"],
                "provider":  loc["provider"],
                "has_pm25":  "pm25" in params_norm,
                "n_req_params": n_req,
                "available_params": "; ".join(sorted(params_norm)),
                "datetime_first": loc["datetime_first"][:10],
                "datetime_last":  loc["datetime_last"][:10],
                "_sensor_map": sensor_map,
            })

        city_pm25 = sum(1 for r in profile_rows
                        if r["city"] == city and r["has_pm25"])
        print(f"    有 PM2.5 的站点: {city_pm25}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_sensor_map"} for r in profile_rows])
    df.to_csv(DOC_DIR / "openaq_m2_station_profiles.csv", index=False)
    print(f"\n  保存: {DOC_DIR / 'openaq_m2_station_profiles.csv'}")
    return profile_rows


# ─── Module 3: 完备性统计与城市评分 ──────────────────────────────────────────

def _common_hours(stations: list[dict]) -> tuple[str, str, int]:
    """利用 datetime_first/last 字段估算公共时间窗（无需下载数据）"""
    starts, ends = [], []
    for s in stations:
        f = s.get("datetime_first", "")
        l = s.get("datetime_last",  "")
        if f and l and len(f) >= 10 and len(l) >= 10:
            try:
                starts.append(pd.Timestamp(f, tz="UTC"))
                ends.append(  pd.Timestamp(l, tz="UTC"))
            except Exception:
                pass
    if not starts:
        return "", "", 0
    c_start = max(starts)
    c_end   = min(ends)
    if c_end <= c_start:
        return str(c_start)[:10], str(c_end)[:10], 0
    hours = int((c_end - c_start).total_seconds() / 3600)
    return str(c_start)[:10], str(c_end)[:10], hours


def _city_score(n_stations, common_hours, joint_cov, sync_cov) -> float:
    s_stat = min(n_stations / 12.0, 1.0)
    s_time = min(common_hours / 17520.0, 1.0)  # 2 年满分
    return W_STATIONS*s_stat + W_TIME*s_time + W_FEATURE*joint_cov + W_SYNC*sync_cov


def module3_screen(client: OpenAQ, m2_rows: list[dict]) -> list[dict]:
    """
    5 层筛选 + 城市评分。
    覆盖率 / 同步率用 7 天样本数据估算（避免全量下载）。
    """
    print("\n" + "="*60)
    print("Module 3: 完备性统计与城市评分")
    print("="*60)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in m2_rows:
        if r["has_pm25"] and r["lat"] and r["lon"]:
            groups[(r["city"], r["country"])].append(r)

    city_results = []

    for (city, country), stations in groups.items():
        print(f"\n  {city}, {country}  ({len(stations)} 有 PM2.5 + 坐标的站点)")

        # 层 1: 站点数
        if len(stations) < MIN_STATIONS:
            print(f"    ✗ 站点不足 {len(stations)} < {MIN_STATIONS}")
            continue

        # 层 2: 变量筛选 — 选 n_req_params ≥ 3 的站点
        cands = [s for s in stations if s["n_req_params"] >= 3]
        if len(cands) < MIN_STATIONS:
            print(f"    ✗ 多参数站点不足 ({len(cands)})")
            continue
        working = sorted(cands, key=lambda x: x["n_req_params"], reverse=True)[:16]

        # 层 3: 公共时间窗 (从 metadata 估算)
        c_start, c_end, c_hours = _common_hours(working)
        print(f"    公共时间窗: {c_hours}h  ({c_start} ~ {c_end})")
        if c_hours < MIN_COMMON_HOURS:
            print(f"    ✗ 公共时间窗不足")
            continue

        # 层 4 & 5: 下载最近 7 天样本 × 前 8 站 PM2.5 估算覆盖率 / 同步率
        sample_end   = pd.Timestamp.now(tz="UTC").normalize()
        sample_start = sample_end - pd.Timedelta(days=7)
        # 限制在公共时间窗内
        t0 = max(pd.Timestamp(c_start, tz="UTC"), sample_start)
        t1 = min(pd.Timestamp(c_end,   tz="UTC"), sample_end)
        if t1 <= t0:
            t0 = pd.Timestamp(c_start, tz="UTC")
            t1 = t0 + pd.Timedelta(days=7)

        idx = pd.date_range(t0, t1, freq="h", tz="UTC")
        mat = {}  # location_id → pd.Series

        for st in working[:8]:
            sid = st["_sensor_map"].get("pm25")
            if not sid:
                continue
            cache_key = f"m3_{sid}_{t0.date()}_{t1.date()}"
            cached = _load(cache_key)
            if cached is not None:
                rows_data = cached
            else:
                rows_data, page = [], 1
                while True:
                    resp = client.measurements.list(
                        sensors_id=sid, data="hourly",
                        datetime_from=t0.isoformat(),
                        datetime_to=  t1.isoformat(),
                        limit=1000, page=page,
                    )
                    batch = resp.results
                    rows_data.extend([
                        {"dt": str(m.period.datetime_from.utc), "v": m.value}
                        for m in batch
                        if m.value is not None
                    ])
                    if len(batch) < 1000:
                        break
                    page += 1
                _save(cache_key, rows_data)
                time.sleep(0.3)

            if rows_data:
                df = pd.DataFrame(rows_data)
                df["dt"] = pd.to_datetime(df["dt"], utc=True)
                s = df.set_index("dt")["v"].drop_duplicates()
                mat[st["location_id"]] = s.reindex(idx)

        if mat:
            mat_df = pd.DataFrame(mat, index=idx)
            pm25_cov  = mat_df.notna().mean().mean()
            pm25_cmin = mat_df.notna().mean().min()
            n_thresh  = max(1, math.ceil(len(mat) * SYNC_STATION_FRAC))
            # 同步覆盖率: ≥70% 站点同时有观测的时刻占比 (协议 6.4 精确定义)
            sync_cov  = (mat_df.notna().sum(axis=1) >= n_thresh).mean()
            # 联合特征覆盖率近似: PM2.5 覆盖率 × 平均变量完整度
            param_frac = np.mean([s["n_req_params"] / len(REQUIRED_PARAMS)
                                  for s in working[:8]])
            joint_cov = float(pm25_cov * param_frac)
            max_gap   = int(mat_df.isna().all(axis=1).astype(int)
                           .groupby((mat_df.notna().any(axis=1)).cumsum())
                           .sum().max()) if len(mat_df) else 0
        else:
            pm25_cov = pm25_cmin = sync_cov = joint_cov = 0.0
            max_gap = 9999

        print(f"    PM2.5 覆盖率: {pm25_cov:.0%} (min {pm25_cmin:.0%})")
        print(f"    同步覆盖率:   {sync_cov:.0%}")
        print(f"    联合特征覆盖: {joint_cov:.0%}")

        pass_all = (pm25_cov  >= PM25_COV_THRESH  and
                    joint_cov >= JOINT_COV_THRESH  and
                    sync_cov  >= SYNC_THRESH)
        score = _city_score(len(working), c_hours, joint_cov, sync_cov)
        tier  = "A" if pass_all and score >= 0.65 else ("B" if pass_all else "C")

        print(f"    {'✓' if pass_all else '✗'} Tier={tier}  score={score:.3f}")

        # 最近数据日期（供用户判断时间临近性）
        recent_dates = [s["datetime_last"] for s in working if s["datetime_last"]]
        most_recent  = max(recent_dates) if recent_dates else ""

        city_results.append({
            "city": city, "country": country,
            "working_stations": len(working),
            "common_start": c_start, "common_end": c_end, "common_hours": c_hours,
            "most_recent_data": most_recent,          # ← 时间临近性，供用户人工参考
            "pm25_coverage_mean": round(pm25_cov,  3),
            "pm25_coverage_min":  round(pm25_cmin, 3),
            "joint_feature_coverage": round(joint_cov, 3),
            "sync_coverage":    round(sync_cov, 3),
            "max_gap_hours":    max_gap,
            "passes_all": pass_all,
            "city_score": round(score, 3),
            "tier": tier,
            "_working": working,
        })

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_working"}
                        for r in city_results])
    if not df.empty:
        df = df.sort_values("city_score", ascending=False)
    df.to_csv(DOC_DIR / "openaq_candidate_cities.csv", index=False)
    print(f"\n  保存: {DOC_DIR / 'openaq_candidate_cities.csv'}")

    print("\n  ══ 筛选结果（请人工确认最终候选城市）══")
    print(f"  {'城市':<18} {'国家'} {'Tier'} {'score':>6} {'时间h':>7} "
          f"{'pm25':>5} {'sync':>5} {'最新数据':>12}")
    print("  " + "-"*75)
    for _, r in df.iterrows():
        mark = "★" if r["tier"] == "A" else ("·" if r["tier"] == "B" else " ")
        print(f"  {mark} {r['city']:<17} {r['country']}   {r['tier']}  "
              f"{r['city_score']:>5.3f}  {r['common_hours']:>6d}  "
              f"{r['pm25_coverage_mean']:>4.0%}  {r['sync_coverage']:>4.0%}  "
              f"{r['most_recent_data']:>12}")
    print("\n  ★ = Tier A (推荐主 benchmark)  · = Tier B (补充验证)  空 = Tier C (内部)")
    print("  注: 时间临近性请结合'最新数据'列人工判断，代码不自动加权。")

    _save("m3_results", [{k: v for k, v in r.items() if k != "_working"}
                          for r in city_results])
    return city_results


# ─── Module 4: 完整数据下载 + 气象补源 + NPZ 生成 ──────────────────────────────

def _fetch_open_meteo(lat, lon, start, end) -> pd.DataFrame:
    import requests as req
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ("temperature_2m,relative_humidity_2m,"
                   "wind_speed_10m,wind_direction_10m,surface_pressure"),
        "wind_speed_unit": "ms", "timezone": "UTC",
    }
    for attempt in range(4):
        try:
            r = req.get(url, params=params, timeout=60)
            r.raise_for_status()
            h = r.json().get("hourly", {})
            df = pd.DataFrame(h)
            df["datetime"] = pd.to_datetime(df["time"], utc=True)
            return df.drop(columns=["time"]).set_index("datetime")
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt)
    return pd.DataFrame()

METEO_COLS     = ["temperature_2m", "relative_humidity_2m",
                  "wind_speed_10m", "wind_direction_10m", "surface_pressure"]
POLLUTANT_COLS = ["pm25", "pm10", "no2", "so2", "o3", "co"]
ALL_COLS       = POLLUTANT_COLS + METEO_COLS   # 11 维，语义统一


def module4_build(client: OpenAQ, m3_rows: list[dict], target_city: str = None):
    print("\n" + "="*60)
    print("Module 4: 完整数据下载 + 气象补源")
    print("="*60)

    for entry in m3_rows:
        city, country = entry["city"], entry["country"]
        if target_city and city.lower() != target_city.lower():
            continue
        if entry["tier"] not in ("A", "B"):
            continue

        print(f"\n  构建 {city}, {country}  Tier={entry['tier']}")
        city_dir = DATA_DIR / f"{country}_{city.replace(' ', '_')}"
        city_dir.mkdir(parents=True, exist_ok=True)

        working  = entry["_working"]
        c_start, c_end = entry["common_start"], entry["common_end"]
        idx = pd.date_range(c_start, c_end, freq="h", tz="UTC")
        T, N, F = len(idx), len(working), len(ALL_COLS)
        print(f"  {T}h × {N} 站 × {F} 维")

        cube = np.full((T, N, F), np.nan, dtype=np.float32)

        for s_idx, st in enumerate(working):
            loc_id, lat, lon = st["location_id"], st["lat"], st["lon"]
            sensor_map = st["_sensor_map"]
            print(f"  [{s_idx+1}/{N}] {st['location_name'][:35]}")

            for f_idx, param in enumerate(POLLUTANT_COLS):
                sid = sensor_map.get(param)
                if not sid:
                    continue
                cache_key = f"m4_{sid}_{c_start}_{c_end}"
                cached = _load(cache_key)
                if cached is not None:
                    rows_data = cached
                else:
                    rows_data, page = [], 1
                    while True:
                        resp = client.measurements.list(
                            sensors_id=sid, data="hourly",
                            datetime_from=f"{c_start}T00:00:00Z",
                            datetime_to=  f"{c_end}T23:59:59Z",
                            limit=1000, page=page,
                        )
                        batch = resp.results
                        rows_data.extend([
                            {"dt": str(m.period.datetime_from.utc), "v": m.value}
                            for m in batch if m.value is not None
                        ])
                        if len(batch) < 1000:
                            break
                        page += 1
                        time.sleep(0.1)
                    _save(cache_key, rows_data)

                if rows_data:
                    df = pd.DataFrame(rows_data)
                    df["dt"] = pd.to_datetime(df["dt"], utc=True)
                    s = df.set_index("dt")["v"].drop_duplicates().reindex(idx)
                    cube[:, s_idx, f_idx] = s.values

            # 气象补源 (Open-Meteo, 协议 9.1 方案 A)
            mkey = f"m4_meteo_{lat:.4f}_{lon:.4f}_{c_start}_{c_end}"
            mcached = _load(mkey)
            if mcached is not None:
                mdf = pd.DataFrame(mcached)
                mdf["datetime"] = pd.to_datetime(mdf["datetime"], utc=True)
                mdf = mdf.set_index("datetime")
            else:
                mdf = _fetch_open_meteo(lat, lon, c_start, c_end)
                _save(mkey, mdf.reset_index().to_dict("records"))

            if not mdf.empty:
                for m_idx, mc in enumerate(METEO_COLS):
                    if mc in mdf.columns:
                        cube[:, s_idx, len(POLLUTANT_COLS)+m_idx] = (
                            mdf[mc].reindex(idx).values)

        # 线性插值填补 (与北京一致)
        from scipy.interpolate import interp1d
        for si in range(N):
            for fi in range(F):
                col = cube[:, si, fi]
                nan = np.isnan(col)
                if nan.all():
                    cube[:, si, fi] = 0
                elif nan.any():
                    xs = np.where(~nan)[0]; ys = col[~nan]
                    fn = interp1d(xs, ys, kind="linear",
                                  bounds_error=False, fill_value=(ys[0], ys[-1]))
                    col[nan] = fn(np.where(nan)[0])
                    cube[:, si, fi] = col

        np.save(str(city_dir / "raw_aligned.npy"), cube)

        # Min-Max 归一化 + 滑窗数据集
        pm25_idx = 0
        _min = np.nanmin(cube[:, :, pm25_idx])
        _max = np.nanmax(cube[:, :, pm25_idx])
        normed = np.zeros_like(cube)
        for fi in range(F):
            fmin, fmax = np.nanmin(cube[:,:,fi]), np.nanmax(cube[:,:,fi])
            if fmax - fmin > 1e-8:
                normed[:,:,fi] = (cube[:,:,fi] - fmin) / (fmax - fmin)

        SEQ = 72
        for pred_len in [1, 6, 12, 24]:
            ns = T - SEQ - pred_len + 1
            if ns < 200:
                continue
            X = np.zeros((ns, N, SEQ, F), dtype=np.float32)
            y = np.zeros((ns, N, pred_len), dtype=np.float32)
            for i in range(ns):
                X[i] = normed[i:i+SEQ].transpose(1,0,2)
                y[i] = normed[i+SEQ:i+SEQ+pred_len, :, pm25_idx].T
            n_tr = int(ns*0.6); n_vl = int(ns*0.2)
            out = city_dir / "train_val_test_data" / f"{SEQ}_{pred_len}"
            out.mkdir(parents=True, exist_ok=True)
            np.savez(str(out/"train_PM25.npz"), X=X[:n_tr],           y=y[:n_tr])
            np.savez(str(out/"val_PM25.npz"),   X=X[n_tr:n_tr+n_vl], y=y[n_tr:n_tr+n_vl])
            np.savez(str(out/"test_PM25.npz"),  X=X[n_tr+n_vl:],     y=y[n_tr+n_vl:])
            np.save(str(out/"scaler_PM25.npy"), np.array([_min, _max]))
            print(f"    pred={pred_len}: {X.shape} → {out}")

        # 站点元数据
        meta = {
            "city": city, "country": country,
            "common_start": c_start, "common_end": c_end,
            "stations": [s["location_id"] for s in working],
            "station_details": [
                {"id": s["location_id"], "name": s["location_name"],
                 "lat": s["lat"], "lon": s["lon"],
                 "provider": s["provider"], "params": s["available_params"]}
                for s in working
            ],
            "features": ALL_COLS,
        }
        (city_dir / "station_metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))

        # 筛选报告 (协议 12.3)
        report = "\n".join([
            f"# {city}, {country} — DMGENet 筛选报告",
            f"生成: {datetime.now():%Y-%m-%d %H:%M}",
            f"## 结果: Tier {entry['tier']}  score={entry['city_score']}",
            f"- 公共时间窗: {c_start} ~ {c_end}  ({entry['common_hours']}h)",
            f"- PM2.5 覆盖: {entry['pm25_coverage_mean']:.0%}",
            f"- 联合特征覆盖: {entry['joint_feature_coverage']:.0%}",
            f"- 同步覆盖: {entry['sync_coverage']:.0%}",
            f"- 最新数据: {entry['most_recent_data']}",
            f"## {len(working)} 个使用站点",
        ] + [f"{i+1}. {s['location_name']} (id={s['location_id']}, "
             f"{s['lat']:.4f},{s['lon']:.4f})" for i, s in enumerate(working)]
        + [f"## 特征集合 (11 维)",
           "| # | 特征 | 来源 |", "|---|------|------|"]
        + [f"| {i} | {c} | {'Open-Meteo' if i>=6 else 'OpenAQ'} |"
           for i, c in enumerate(ALL_COLS)])
        fn = DOC_DIR / f"{country}_{city.replace(' ','_')}_screening_report.md"
        fn.write_text(report)
        print(f"    报告: {fn}")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="1", choices=["1", "2"])
    parser.add_argument("--city", default=None, help="Phase 2 只处理指定城市")
    args = parser.parse_args()

    with OpenAQ(api_key=API_KEY) as client:
        if args.phase == "1":
            m1 = module1_discover(client)
            m2 = module2_profile(m1)
            module3_screen(client, m2)
            print("\n下一步: 查看 doc/openaq_candidate_cities.csv，")
            print("人工选择目标城市后运行:")
            print("  python data/openaq_city_screener.py --phase 2 --city 城市名")

        elif args.phase == "2":
            # 恢复 M3 结果并补回 _working
            m3_saved = _load("m3_results")
            if not m3_saved:
                sys.exit("请先运行 --phase 1")
            # 需要重建 _working (含 _sensor_map)
            m1 = module1_discover(client)
            m2 = module2_profile(m1)
            # 将 _sensor_map 注入到 m3
            m2_idx = {r["location_id"]: r for r in m2}
            for entry in m3_saved:
                city_locs = [
                    m2_idx[lid]
                    for r in m2 if r["city"] == entry["city"]
                    for lid in [r["location_id"]]
                    if lid in m2_idx and m2_idx[lid]["has_pm25"]
                ]
                entry["_working"] = sorted(
                    city_locs, key=lambda x: x["n_req_params"], reverse=True
                )[:16]
            module4_build(client, m3_saved, target_city=args.city)


if __name__ == "__main__":
    main()
