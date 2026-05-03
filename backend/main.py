"""
RouteRider — Backend (main.py)
Flask API that powers multi-stop delivery route optimization.

Responsibilities:
  1. Input validation        — reject bad data before touching OSRM
  2. OSRM matrix caching     — skip redundant API calls for repeated stop sets
  3. Nearest Neighbor TSP    — fast greedy route order (O(N²))
  4. 2-opt improvement       — refine NN result by uncrossing path segments
  5. Road segment fetching   — polyline + turn-by-turn steps per leg via OSRM
  6. ETA calculation         — cumulative arrival times per stop
  7. Fuel & analytics        — estimated fuel cost, litres, and time savings
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

# Philippines bounding box — reject coordinates outside this range
LAT_MIN, LAT_MAX =  4.0,  21.5
LNG_MIN, LNG_MAX = 116.0, 127.0

# ─────────────────────────────────────────────────────────────────────────────
# CACHE
#
# Why: OSRM Table API takes ~1-2 seconds. If the same stop set is submitted
# again, we return the cached matrix instantly without hitting OSRM.
#
# Key: frozenset of (lat, lng) tuples — order-independent and hashable.
# Lifetime: in-memory only (resets when main.py restarts).
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
#
# Checks performed:
#   - Required keys present (name, lat, lng)
#   - lat/lng are numeric
#   - Coordinates inside Philippines bounding box
#   - No duplicate locations
# ─────────────────────────────────────────────────────────────────────────────

def validate_point(p: dict, label: str):
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

    # Duplicate coordinate check (rounded to ~11m precision)
    seen = set()
    for p in [gz] + stops:
        key = (round(float(p['lat']), 4), round(float(p['lng']), 4))
        if key in seen:
            return False, f"Duplicate location detected near ({p['lat']}, {p['lng']})"
        seen.add(key)

    return True, None

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH: Build (N+1)×(N+1) Travel Time Matrix via OSRM Table API
#
# This IS the weighted directed graph from your DSA proposal:
#   Nodes  = ground_zero + all delivery stops
#   Edges  = every possible pair (i, j)
#   Weight = road travel time in seconds
#
# OSRM computes all N² pairs in one request (much faster than N² Route calls).
# Result is cached so repeated submissions with same stops skip OSRM entirely.
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(all_points: list) -> list:
    """
    Returns (N+1)×(N+1) adjacency matrix where matrix[i][j] = seconds.
    Index 0 = ground_zero, indices 1..N = delivery stops.
    """
    key = _cache_key(all_points)
    if key in _matrix_cache:
        print(f"[CACHE HIT] {len(all_points)}×{len(all_points)} matrix")
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
    print(f"[CACHE STORE] Matrix cached ({len(all_points)} points)")
    return matrix

# ─────────────────────────────────────────────────────────────────────────────
# TSP STEP 1: Nearest Neighbor Heuristic  O(N²)
#
# Start at ground_zero (index 0).
# At each step, move to the closest unvisited stop by road travel time.
# Continue until all stops are visited.
#
# This is the greedy baseline — fast but may produce "crossed" routes.
# 2-opt (below) refines the result.
# ─────────────────────────────────────────────────────────────────────────────

def nearest_neighbor_tsp(matrix: list, start: int = 0) -> list:
    """
    Greedy TSP with fixed start at ground_zero (index 0).
    Returns list of indices e.g. [0, 3, 1, 2]
    Meaning: ground_zero → stop3 → stop1 → stop2
    Time complexity: O(N²)
    """
    n         = len(matrix)
    unvisited = set(range(1, n))  # delivery stops only — GZ never re-visited
    route     = [start]

    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda j: matrix[current][j])
        route.append(nearest)
        unvisited.remove(nearest)

    return route

# ─────────────────────────────────────────────────────────────────────────────
# TSP STEP 2: 2-opt Improvement
#
# Problem: NN can produce crossed routes (A→D→B→C when A→B→D→C is shorter).
#
# 2-opt fix: try reversing every sub-segment [i+1..j] of the route.
# If the reversal reduces total cost, accept it and keep searching.
# Repeat until no reversal improves the route (convergence).
#
# Ground zero (route[0]) is always kept fixed.
# Only delivery stops (route[1:]) are reordered.
#
# Time complexity: O(N²) per pass, typically 2-5 passes to converge.
# For N≤9: runs in microseconds.
#
# DSA relevance: this is a local search / iterative improvement algorithm.
# NN gives a feasible solution; 2-opt improves it toward the optimum.
# ─────────────────────────────────────────────────────────────────────────────

def two_opt(route: list, matrix: list) -> list:
    """
    Improves TSP route by iteratively reversing sub-segments.
    route[0] (ground_zero) is fixed — only route[1:] is reordered.

    Returns improved route in the same index format as nearest_neighbor_tsp.
    """

    def route_cost(r: list) -> float:
        """Sum of edge weights along the route."""
        return sum(matrix[r[i]][r[i+1]] for i in range(len(r) - 1))

    best       = route[:]
    best_cost  = route_cost(best)
    improved   = True

    while improved:
        improved = False
        # i and j iterate over delivery stop positions only (skip index 0 = GZ)
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                # Reverse the sub-segment from i to j (inclusive)
                # Example: [GZ, A, B, C, D] with i=1, j=3 → [GZ, C, B, A, D]
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                cost      = route_cost(candidate)

                if cost < best_cost:
                    best      = candidate
                    best_cost = cost
                    improved  = True  # found improvement — do another full pass

    return best

# ─────────────────────────────────────────────────────────────────────────────
# ROAD SEGMENT: Polyline + Turn-by-Turn via OSRM Route API
#
# Called once per consecutive stop pair in the optimized route.
# Returns the actual road-following path (not straight line) and
# turn-by-turn navigation steps.
# ─────────────────────────────────────────────────────────────────────────────

def get_segment(point_a: dict, point_b: dict) -> tuple:
    """
    Fetches road route between two points.
    Returns: (polyline [[lat,lng],...], steps [...], duration_s, distance_m)
    """
    coords = f"{point_a['lng']},{point_a['lat']};{point_b['lng']},{point_b['lat']}"
    url    = (f"{OSRM_BASE}/route/v1/driving/{coords}"
              f"?overview=full&geometries=geojson&steps=true")

    res  = requests.get(url, timeout=15)
    data = res.json()

    if data.get("code") != "Ok":
        raise Exception(
            f"OSRM Route error: {point_a['name']} → {point_b['name']}"
        )

    route_obj = data["routes"][0]
    leg       = route_obj["legs"][0]
    geometry  = route_obj["geometry"]["coordinates"]  # [[lng,lat], ...]

    # Parse turn-by-turn steps
    steps = []
    for s in leg["steps"]:
        maneuver = s.get("maneuver", {})
        steps.append({
            "instruction": maneuver.get("type", "proceed"),
            "modifier":    maneuver.get("modifier", ""),  # "left", "right", "straight"
            "road":        s.get("name", "").strip(),
            "distance":    round(s.get("distance", 0)),   # metres for this step
        })

    # Flip [lng,lat] → [lat,lng] for Leaflet
    polyline = [[c[1], c[0]] for c in geometry]

    return polyline, steps, leg["duration"], leg["distance"]

# ─────────────────────────────────────────────────────────────────────────────
# ETA CALCULATION
#
# Walks through segment durations and accumulates arrival times.
# Returns ISO 8601 strings so JavaScript can parse with new Date(str).
#
# Example (start = 2:00 PM):
#   segment[0] = 600s  → Stop 1 ETA = 2:10 PM  ("...T14:10:00")
#   segment[1] = 900s  → Stop 2 ETA = 2:25 PM  ("...T14:25:00")
#   segment[2] = 300s  → Stop 3 ETA = 2:30 PM  ("...T14:30:00")
# ─────────────────────────────────────────────────────────────────────────────

def compute_etas(segment_durations: list, start_time: datetime) -> list:
    """
    Returns list of ISO timestamp strings, one per delivery stop.
    segment_durations: [float] seconds, one per segment
    start_time: datetime when rider departs ground zero
    """
    etas   = []
    cursor = start_time
    for seconds in segment_durations:
        cursor = cursor + timedelta(seconds=seconds)
        etas.append(cursor.isoformat())  # "2025-04-28T14:10:22.000000"
    return etas

# ─────────────────────────────────────────────────────────────────────────────
# FUEL & ANALYTICS
#
# Computes:
#   fuel_liters    = total_km / MOTO_KM_PER_LITER
#   fuel_cost_php  = fuel_liters * FUEL_PRICE_PER_LITER
#   time_saved_min = (unoptimized duration - optimized duration) in minutes
#
# "time_saved" answers the question: how much faster is the TSP-optimized
# route vs just visiting stops in the order the rider typed them?
# ─────────────────────────────────────────────────────────────────────────────

def compute_analytics(total_distance_m: float,
                      optimized_s: float,
                      unoptimized_s: float,
                      improvement_pct: float) -> dict:
    """
    Returns fuel estimate and time savings analytics dict.
    """
    total_km   = total_distance_m / 1000
    liters     = round(total_km / MOTO_KM_PER_LITER, 3)
    cost       = round(liters * FUEL_PRICE_PER_LITER, 2)
    saved_min  = round(max(0, unoptimized_s - optimized_s) / 60)

    return {
        "fuel_liters":             liters,
        "fuel_cost_php":           cost,
        "time_saved_min":          saved_min,
        "two_opt_improvement_pct": improvement_pct,
        "km_per_liter":            MOTO_KM_PER_LITER,
        "price_per_liter":         FUEL_PRICE_PER_LITER,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: POST /optimize
#
# Full pipeline:
#   1. Parse + validate request body
#   2. Build adjacency matrix (cached OSRM table call)
#   3. Nearest Neighbor TSP → initial route order
#   4. 2-opt → refined route order
#   5. Compute unoptimized duration (for time_saved analytics)
#   6. Fetch road segments (polyline + steps per leg)
#   7. Compute ETAs from now
#   8. Compute fuel + analytics
#   9. Return full response
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/optimize", methods=["POST"])
def optimize():

    # ── 1. Parse + validate ────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    ok, err = validate_request(body)
    if not ok:
        return jsonify({"error": err}), 400

    ground_zero    = body["ground_zero"]
    original_stops = body["stops"]   # original input order — used for analytics

    try:
        # ── 2. Build graph ─────────────────────────────────────────────────
        # all_points[0] = GZ, [1..N] = stops in original input order
        all_points = [ground_zero] + original_stops
        matrix     = build_graph(all_points)

        # ── 3. Nearest Neighbor TSP ────────────────────────────────────────
        nn_route = nearest_neighbor_tsp(matrix, start=0)
        nn_cost  = sum(matrix[nn_route[i]][nn_route[i+1]]
                       for i in range(len(nn_route) - 1))

        # ── 4. 2-opt improvement ───────────────────────────────────────────
        opt_route = two_opt(nn_route, matrix)
        opt_cost  = sum(matrix[opt_route[i]][opt_route[i+1]]
                        for i in range(len(opt_route) - 1))

        improvement_pct = round((1 - opt_cost / nn_cost) * 100, 1) if nn_cost > 0 else 0.0
        print(f"[TSP] NN: {nn_cost:.0f}s | 2-opt: {opt_cost:.0f}s | "
              f"Saved: {improvement_pct}%")

        # Map optimized index list back to point dicts (skip index 0 = GZ)
        ordered_stops = [all_points[i] for i in opt_route[1:]]

        # ── 5. Unoptimized duration (for analytics) ────────────────────────
        # Original route = [0, 1, 2, ..., N] — stops in the order rider typed them
        original_route = list(range(len(all_points)))
        unopt_s = sum(matrix[original_route[i]][original_route[i+1]]
                      for i in range(len(original_route) - 1))

        # ── 6. Fetch road segments ─────────────────────────────────────────
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
                "duration": round(duration),   # seconds
                "distance": round(distance),   # metres
            })
            total_duration_s += duration
            total_distance_m += distance

        # ── 7. Compute ETAs ────────────────────────────────────────────────
        etas = compute_etas(
            [seg["duration"] for seg in segments],
            datetime.now()
        )
        # Attach eta_iso to each stop so frontend can display it directly
        for i, stop in enumerate(ordered_stops):
            stop["eta_iso"] = etas[i]  # "2025-04-28T14:10:22"

        # ── 8. Compute analytics ───────────────────────────────────────────
        analytics = compute_analytics(
            total_distance_m,
            total_duration_s,
            unopt_s,
            improvement_pct
        )

        # ── 9. Return response ─────────────────────────────────────────────
        return jsonify({
            "ground_zero":        ground_zero,
            "ordered_stops":      ordered_stops,      # optimized, each has eta_iso
            "segments":           segments,            # road polylines + steps
            "total_duration_min": round(total_duration_s / 60),
            "total_distance_km":  round(total_distance_m / 1000, 2),
            "analytics":          analytics,           # fuel, cost, time saved
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "OSRM timed out. Try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach OSRM. Check internet."}), 503
    except Exception as e:
        print(f"[ERROR] /optimize: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: DELETE /cache  — clears the matrix cache (dev utility)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/cache", methods=["DELETE"])
def clear_cache():
    count = len(_matrix_cache)
    _matrix_cache.clear()
    return jsonify({"cleared": count})


if __name__ == "__main__":
    print("RouteRider backend — http://127.0.0.1:5000")
    print(f"  Fuel: ₱{FUEL_PRICE_PER_LITER}/L at {MOTO_KM_PER_LITER}km/L")
    app.run(debug=True, port=5000)