from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

OSRM_BASE = "http://router.project-osrm.org"

# ─── GRAPH: Build (N+1)×(N+1) travel time matrix via OSRM ───────────────────
# Index 0 = ground_zero, indices 1..N = delivery stops
def build_graph(all_points):
    """
    all_points: [ground_zero, stop1, stop2, ... stopN]
    Returns full adjacency matrix where matrix[i][j] = travel time in seconds.
    """
    coords = ";".join(f"{p['lng']},{p['lat']}" for p in all_points)
    url = f"{OSRM_BASE}/table/v1/driving/{coords}?annotations=duration"
    res = requests.get(url, timeout=10)
    data = res.json()
    if data.get("code") != "Ok":
        raise Exception("OSRM table API failed")
    return data["durations"]


# ─── TSP: Nearest Neighbor — fixed start at index 0 (ground zero) ────────────
def nearest_neighbor_tsp(matrix, start=0):
    """
    Greedy TSP always starting from ground zero (index 0).
    Only delivery stop indices (1..N) are reordered.
    Time complexity: O(N²)
    Returns: ordered list of indices starting with 0.
    """
    n = len(matrix)
    unvisited = set(range(1, n))   # ground zero is never in unvisited pool
    route = [start]                # always start from ground zero

    while unvisited:
        current = route[-1]
        nearest = min(unvisited, key=lambda j: matrix[current][j])
        route.append(nearest)
        unvisited.remove(nearest)

    return route  # e.g. [0, 3, 1, 2] means: ground_zero → stop3 → stop1 → stop2


# ─── SEGMENT: Road polyline + turn-by-turn between two points ────────────────
def get_segment(point_a, point_b):
    coords = f"{point_a['lng']},{point_a['lat']};{point_b['lng']},{point_b['lat']}"
    url = f"{OSRM_BASE}/route/v1/driving/{coords}?overview=full&geometries=geojson&steps=true"
    res = requests.get(url, timeout=10)
    data = res.json()
    if data.get("code") != "Ok":
        raise Exception("OSRM route API failed")

    leg = data["routes"][0]["legs"][0]
    geometry = data["routes"][0]["geometry"]["coordinates"]
    steps = [
        {
            "instruction": s["maneuver"]["type"] + " onto " + s.get("name", "road"),
            "distance": round(s["distance"]),
            "road": s.get("name", "")
        }
        for s in leg["steps"]
    ]
    polyline = [[c[1], c[0]] for c in geometry]  # [lng,lat] → [lat,lng] for Leaflet
    return polyline, steps, leg["duration"], leg["distance"]


# ─── ENDPOINT ─────────────────────────────────────────────────────────────────
@app.route("/optimize", methods=["POST"])
def optimize():
    body = request.json
    ground_zero = body.get("ground_zero")   # { name, lat, lng }
    stops = body.get("stops", [])           # [{ name, lat, lng }, ...]

    if not ground_zero:
        return jsonify({"error": "Ground zero (start point) is required"}), 400
    if len(stops) < 1:
        return jsonify({"error": "Need at least 1 delivery stop"}), 400
    if len(stops) > 9:
        return jsonify({"error": "Max 9 stops"}), 400

    try:
        # all_points[0] = ground_zero, all_points[1..N] = delivery stops
        all_points = [ground_zero] + stops

        # Step 1: Build full (N+1)×(N+1) adjacency matrix
        matrix = build_graph(all_points)

        # Step 2: TSP — always starts at index 0 (ground zero), reorders stops 1..N
        order = nearest_neighbor_tsp(matrix, start=0)
        # order[0] is always 0 (ground_zero), skip it for ordered_stops list
        ordered_stops = [all_points[i] for i in order[1:]]

        # Step 3: Build segments — ground_zero → stop1 → stop2 → ... → stopN
        route_points = [ground_zero] + ordered_stops
        segments = []
        total_duration = 0
        total_distance = 0

        for i in range(len(route_points) - 1):
            polyline, steps, duration, distance = get_segment(route_points[i], route_points[i + 1])
            segments.append({
                "from": route_points[i]["name"],
                "to": route_points[i + 1]["name"],
                "polyline": polyline,
                "steps": steps,
                "duration": round(duration),
                "distance": round(distance)
            })
            total_duration += duration
            total_distance += distance

        return jsonify({
            "ground_zero": ground_zero,
            "ordered_stops": ordered_stops,       # delivery stops in optimized order
            "segments": segments,                  # includes ground_zero → stop1 leg
            "total_duration_min": round(total_duration / 60),
            "total_distance_km": round(total_distance / 1000, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)