"""
RouteRider — Backend (main.py)
Flask API powering multi-stop delivery route optimization.

Pipeline per /optimize request:
  1. Validate input
  2. Build N×N travel time graph (OSRM Table API) — cached
  3. Nearest Neighbor TSP — O(N²) greedy baseline
  4. 2-opt improvement — iteratively uncross the route
  5. Fetch road polylines + turn-by-turn per segment (OSRM Route API)
  6. Compute ETAs per stop
  7. Compute analytics — fuel cost, time saved, 2-opt improvement
  8. Return full response
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OSRM_BASE            = "http://router.project-osrm.org"
MAX_STOPS            = 9
FUEL_PRICE_PER_LITER = 67.0   # ₱/L — Petron RON 95 approximate
MOTO_KM_PER_LITER    = 40.0   # km/L — average Philippine delivery motorcycle

# Philippines bounding box — reject coordinates clearly outside the country
LAT_MIN, LAT_MAX =  4.0,  21.5
LNG_MIN, LNG_MAX = 116.0, 127.0

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY CACHE
#
# Why: OSRM Table API takes ~1-2 seconds per call.
# If the same stop set is submitted again (e.g. rider recomputes without
# changing stops), we return the cached matrix instantly.
#
# Key: frozenset of rounded (lat, lng) tuples — order-independent, hashable.
# Limitation: resets when main.py restarts (in-memory only).
# ─────────────────────────────────────────────────────────────────────────────

_matrix_cache: dict = {}

def _cache_key(all_points: list) -> frozenset:
    """
    Converts list of { lat, lng } dicts into a hashable frozenset of tuples.
    Rounded to 5 decimal places (~1.1m) so minor GPS drift still hits the cache.
    """
    return frozenset(
        (round(float(p['lat']), 5), round(float(p['lng']), 5))
        for p in all_points
    )

# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION
#
# Validates each point and the full request body before any OSRM calls.
# Returns (is_valid: bool, error_message: str | None).
# ─────────────────────────────────────────────────────────────────────────────

def validate_point(p: dict, label: str):
    """Validates a single { name, lat, lng } point."""
    for key in ('name', 'lat', 'lng'):
        if key not in p:
            return False, f"{label} missing required field: '{key}'"
    if not isinstance(p['name'], str) or not p['name'].strip():
        return False, f"{label} has an invalid or empty 'name'"
    try:
        lat = float(p['lat'])
        lng = float(p['lng'])
    except (TypeError, ValueError):
        return False, f"{label} has non-numeric lat/lng values"
    if not (LAT_MIN <= lat <= LAT_MAX):
        return False, f"{label} latitude {lat:.4f} is outside the Philippines"
    if not (LNG_MIN <= lng <= LNG_MAX):
        return False, f"{label} longitude {lng:.4f} is outside the Philippines"
    return True, None


def validate_request(body: dict):
    """Validates the full /optimize request body."""
    if not isinstance(body, dict):
        return False, "Request body must be a JSON object"
    gz = body.get('ground_zero')
    if not gz:
        return False, "Missing 'ground_zero' — rider start point is required"
    ok, err = validate_point(gz, "ground_zero")
    if not ok:
        return False, err
    stops = body.get('stops')
    if not isinstance(stops, list):
        return False, "'stops' must be an array"
    if len(stops) < 1:
        return False, "Need at least 1 delivery stop"
    if len(stops) > MAX_STOPS:
        return False, f"Too many stops — maximum is {MAX_STOPS}"
    for i, stop in enumerate(stops):
        ok, err = validate_point(stop, f"stops[{i}]")
        if not ok:
            return False, err
    # Duplicate coordinate check (~11m precision)
    seen = set()
    for p in [gz] + stops:
        key = (round(float(p['lat']), 4), round(float(p['lng']), 4))
        if key in seen:
            return False, f"Duplicate location near ({p['lat']}, {p['lng']})"
        seen.add(key)
    return True, None

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH — Build (N+1)×(N+1) Travel Time Matrix via OSRM Table API
#
# This IS the weighted directed graph from the DSA proposal:
#   Nodes  = ground_zero + all delivery stops
#   Edges  = every possible pair (i, j)
#   Weight = road travel time in seconds
#
# OSRM computes all N² pairs in one API call — much faster than N² Route calls.
# Results are cached so repeated submissions skip OSRM entirely.
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(all_points: list) -> list:
    """
    Returns (N+1)×(N+1) adjacency matrix where matrix[i][j] = seconds.
    Index 0 = ground_zero, indices 1..N = delivery stops in original input order.
    """
    key = _cache_key(all_points)
    if key in _matrix_cache:
        print(f"[CACHE HIT] Matrix for {len(all_points)} points")
        return _matrix_cache[key]

    coords = ";".join(f"{p['lng']},{p['lat']}" for p in all_points)
    url    = f"{OSRM_BASE}/table/v1/driving/{coords}?annotations=duration"
    print(f"[OSRM TABLE] Fetching {len(all_points)}×{len(all_points)} matrix...")
    res  = requests.get(url, timeout=15)
    data = res.json()
    if data.get("code") != "Ok":
        raise Exception(f"OSRM Table API error: {data.get('message', 'unknown')}")

    matrix = data["durations"]
    _matrix_cache[key] = matrix
    print(f"[CACHE STORE] Matrix cached for {len(all_points)} points")
    return matrix

# ─────────────────────────────────────────────────────────────────────────────
# TSP STEP 1 — Nearest Neighbor Heuristic  O(N²)
#
# Always starts at ground_zero (index 0).
# At each step, moves to the closest unvisited stop by road travel time.
# Produces a greedy baseline — fast but may contain crossed paths.
# 2-opt (below) fixes those crossings.
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbor_tsp(matrix: list, start: int = 0) -> list:
    """
    Returns list of point indices starting with ground_zero.
    e.g. [0, 3, 1, 2] = GZ → stop3 → stop1 → stop2
    Time complexity: O(N²)
    """
    n         = len(matrix)
    unvisited = set(range(1, n))  # delivery stops only
    route     = [start]           # ground zero fixed at front

    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda j: matrix[current][j])
        route.append(nearest)
        unvisited.remove(nearest)

    return route

# ─────────────────────────────────────────────────────────────────────────────
# TSP STEP 2 — 2-opt Improvement
#
# Problem: NN can produce crossed routes (A→D→B→C when A→B→D→C is shorter).
#
# Fix: try reversing every sub-segment [i..j] of the route.
# If the reversal reduces total travel time, accept it.
# Repeat until no reversal helps (convergence).
#
# route[0] = ground_zero — always kept fixed.
# Only delivery stop positions (route[1:]) are reordered.
#
# Time complexity: O(N²) per pass, typically 2-5 passes.
# For N≤9: completes in microseconds.
# ─────────────────────────────────────────────────────────────────────────────

def two_opt(route: list, matrix: list) -> list:
    """
    Improves TSP route by reversing sub-segments that reduce total cost.
    Returns the same index-format list as nearest_neighbor_tsp.
    """
    def route_cost(r):
        return sum(matrix[r[i]][r[i+1]] for i in range(len(r) - 1))

    best      = route[:]
    best_cost = route_cost(best)
    improved  = True

    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                # Reverse the sub-segment from position i to j
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                cost      = route_cost(candidate)
                if cost < best_cost:
                    best      = candidate
                    best_cost = cost
                    improved  = True

    return best

# ─────────────────────────────────────────────────────────────────────────────
# ROAD SEGMENT — Polyline + Turn-by-Turn via OSRM Route API
#
# Called once per consecutive stop pair in the optimized route.
# Returns the actual road-following path and navigation instructions.
# ─────────────────────────────────────────────────────────────────────────────

def get_segment(point_a: dict, point_b: dict) -> tuple:
    """
    Returns: (polyline [[lat,lng],...], steps [...], duration_s, distance_m)
    """
    coords = f"{point_a['lng']},{point_a['lat']};{point_b['lng']},{point_b['lat']}"
    url    = (f"{OSRM_BASE}/route/v1/driving/{coords}"
              f"?overview=full&geometries=geojson&steps=true")
    res  = requests.get(url, timeout=15)
    data = res.json()
    if data.get("code") != "Ok":
        raise Exception(f"OSRM Route error: {point_a['name']} → {point_b['name']}")

    route_obj = data["routes"][0]
    leg       = route_obj["legs"][0]
    geometry  = route_obj["geometry"]["coordinates"]

    steps = []
    for s in leg["steps"]:
        maneuver = s.get("maneuver", {})
        steps.append({
            "instruction": maneuver.get("type", "proceed"),
            "modifier":    maneuver.get("modifier", ""),
            "road":        s.get("name", "").strip(),
            "distance":    round(s.get("distance", 0)),
        })

    # OSRM returns [lng,lat] — flip to [lat,lng] for Leaflet
    polyline = [[c[1], c[0]] for c in geometry]
    return polyline, steps, leg["duration"], leg["distance"]

# ─────────────────────────────────────────────────────────────────────────────
# ETA CALCULATION
#
# Cumulative sum of segment durations starting from now.
# Returns ISO 8601 strings so JavaScript can parse with new Date(str).
#
# Example (start = 2:00 PM):
#   segment[0] = 600s  → Stop 1 ETA = "...T14:10:00"
#   segment[1] = 900s  → Stop 2 ETA = "...T14:25:00"
# ─────────────────────────────────────────────────────────────────────────────

def compute_etas(segment_durations: list, start_time: datetime) -> list:
    """
    Returns list of ISO 8601 strings, one per delivery stop.
    """
    etas   = []
    cursor = start_time
    for seconds in segment_durations:
        cursor = cursor + timedelta(seconds=seconds)
        etas.append(cursor.isoformat())
    return etas

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS
#
# fuel_liters   = total_km / MOTO_KM_PER_LITER
# fuel_cost_php = fuel_liters × FUEL_PRICE_PER_LITER
# time_saved    = unoptimized duration − optimized duration (minutes)
# two_opt_pct   = how much 2-opt improved over nearest neighbor (%)
#
# "time_saved" answers: how much faster is the TSP-optimized route vs
# visiting stops in the order the rider originally typed them?
# ─────────────────────────────────────────────────────────────────────────────

def compute_analytics(total_distance_m: float,
                      optimized_s: float,
                      unoptimized_s: float,
                      improvement_pct: float) -> dict:
    total_km  = total_distance_m / 1000
    liters    = round(total_km / MOTO_KM_PER_LITER, 3)
    cost      = round(liters * FUEL_PRICE_PER_LITER, 2)
    saved_min = round(max(0, unoptimized_s - optimized_s) / 60)

    return {
        "fuel_liters":             liters,
        "fuel_cost_php":           cost,       # Philippine Peso
        "time_saved_min":          saved_min,  # minutes saved vs unoptimized
        "two_opt_improvement_pct": improvement_pct,
        "km_per_liter":            MOTO_KM_PER_LITER,
        "price_per_liter":         FUEL_PRICE_PER_LITER,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — POST /optimize
#
# Full pipeline:
#   1. Validate request
#   2. Build adjacency matrix (cached OSRM table)
#   3. Nearest Neighbor TSP
#   4. 2-opt improvement
#   5. Compute unoptimized duration (for time_saved)
#   6. Fetch road segments (polyline + steps per leg)
#   7. Compute ETAs
#   8. Compute analytics
#   9. Return response
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/optimize", methods=["POST"])
def optimize():

    # 1. Validate
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    ok, err = validate_request(body)
    if not ok:
        return jsonify({"error": err}), 400

    ground_zero    = body["ground_zero"]
    original_stops = body["stops"]

    try:
        # 2. Build graph
        # all_points[0] = GZ, [1..N] = stops in original input order
        all_points = [ground_zero] + original_stops
        matrix     = build_graph(all_points)

        # 3. Nearest Neighbor TSP
        nn_route = nearest_neighbor_tsp(matrix, start=0)
        nn_cost  = sum(matrix[nn_route[i]][nn_route[i+1]]
                       for i in range(len(nn_route) - 1))

        # 4. 2-opt improvement
        opt_route = two_opt(nn_route, matrix)
        opt_cost  = sum(matrix[opt_route[i]][opt_route[i+1]]
                        for i in range(len(opt_route) - 1))
        improvement_pct = round((1 - opt_cost / nn_cost) * 100, 1) if nn_cost > 0 else 0.0
        print(f"[TSP] NN: {nn_cost:.0f}s | 2-opt: {opt_cost:.0f}s | Saved: {improvement_pct}%")

        # Map optimized index list back to point dicts (skip index 0 = GZ)
        ordered_stops = [all_points[i] for i in opt_route[1:]]

        # 5. Unoptimized duration — original input order [0,1,2,...,N]
        original_route = list(range(len(all_points)))
        unopt_s = sum(matrix[original_route[i]][original_route[i+1]]
                      for i in range(len(original_route) - 1))

        # 6. Fetch road segments
        route_points     = [ground_zero] + ordered_stops
        segments         = []
        total_duration_s = 0.0
        total_distance_m = 0.0

        for i in range(len(route_points) - 1):
            polyline, steps, duration, distance = get_segment(
                route_points[i], route_points[i + 1]
            )
            segments.append({
                "from":     route_points[i]["name"],
                "to":       route_points[i + 1]["name"],
                "polyline": polyline,
                "steps":    steps,
                "duration": round(duration),
                "distance": round(distance),
            })
            total_duration_s += duration
            total_distance_m += distance

        # 7. Compute ETAs — attach eta_iso to each ordered stop
        etas = compute_etas(
            [seg["duration"] for seg in segments],
            datetime.now()
        )
        for i, stop in enumerate(ordered_stops):
            stop["eta_iso"] = etas[i]

        # 8. Compute analytics
        analytics = compute_analytics(
            total_distance_m,
            total_duration_s,
            unopt_s,
            improvement_pct
        )

        # 9. Return
        return jsonify({
            "ground_zero":        ground_zero,
            "ordered_stops":      ordered_stops,
            "segments":           segments,
            "total_duration_min": round(total_duration_s / 60),
            "total_distance_km":  round(total_distance_m / 1000, 2),
            "analytics":          analytics,
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "OSRM timed out. Try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach OSRM. Check internet."}), 503
    except Exception as e:
        print(f"[ERROR] /optimize: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT — DELETE /cache  (dev utility — clears the matrix cache)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/cache", methods=["DELETE"])
def clear_cache():
    count = len(_matrix_cache)
    _matrix_cache.clear()
    return jsonify({"cleared": count})


if __name__ == "__main__":
    print("RouteRider backend — http://127.0.0.1:5000")
    print(f"  Fuel: ₱{FUEL_PRICE_PER_LITER}/L  |  Economy: {MOTO_KM_PER_LITER} km/L")
    app.run(debug=True, port=5000)