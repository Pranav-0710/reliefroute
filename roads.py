"""Road routing on OpenStreetMap through a Valhalla server (no API key).

Gives live operations real road distances and drive times for planning, and the road geometry each
driver follows. Every active hazard is sent as an area to avoid, so routes go around flooded or
blocked roads. Callers fall back to straight-line estimates when the router is unreachable.

ROUTER_URL picks the server (default: the public FOSSGIS instance); ROUTER_URL=off disables it.
"""
import json
import math
import os
import threading
from collections import OrderedDict

import httpx

from env import SPEED_KMH, _xy, distance_km

DEFAULT_URL = "https://valhalla1.openstreetmap.de"
COSTING = "truck"
MAX_TABLE_POINTS = 40  # locations per distance-table request; larger waves plan on estimates
MAX_SHAPE_POINTS = 300  # per route, keeps the shared operation document small
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


class NoRoute(RuntimeError):
    """The router answered, but no road connects the points without crossing an avoided area."""


def enabled():
    return os.getenv("OFFLINE") != "1" and os.getenv("ROUTER_URL", "").lower() != "off"


def _url():
    return (os.getenv("ROUTER_URL") or DEFAULT_URL).rstrip("/")


def _km_straight(a, b):
    (x1, y1), (x2, y2) = _xy(*a), _xy(*b)
    return math.hypot(x2 - x1, y2 - y1)


def _avoid(hazards, points):
    """Hazards as polygons to avoid. A hazard covering a stop itself is left out: the truck still has
    to get there, and the router rejects locations inside excluded areas."""
    polys = []
    for h in hazards:
        r = max(0.2, float(h.get("radius_km", 0.5)))
        if any(_km_straight(p, (h["lat"], h["lng"])) <= r for p in points):
            continue
        dlat, dlng = r / 110.57, r / (111.32 * math.cos(math.radians(h["lat"])))
        ring = [[round(h["lng"] + dlng * math.cos(a), 6), round(h["lat"] + dlat * math.sin(a), 6)]
                for a in (2 * math.pi * k / 10 for k in range(10))]
        polys.append(ring + [ring[0]])
    return polys


def _post(path, body):
    key = path + json.dumps(body, sort_keys=True)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    r = httpx.post(_url() + path, json=body, timeout=12)
    if r.status_code >= 400:
        msg, code = r.text[:120], None
        try:
            body = r.json()
            msg, code = body.get("error", msg), body.get("error_code")
        except ValueError:
            pass
        if r.status_code == 400 and (code == 442 or "no path" in str(msg).lower()):
            raise NoRoute(f"router: {msg}")
        raise RuntimeError(f"router {r.status_code}: {msg}")
    data = r.json()
    with _cache_lock:
        _cache[key] = data
        while len(_cache) > 256:
            _cache.popitem(last=False)
    return data


def _loc(p):
    return {"lat": round(p[0], 6), "lon": round(p[1], 6)}


def table(points, hazards):
    """{(a, b): road km} for every ordered pair of distinct points, around all hazards.
    None when routing is disabled or there are too many points; raises when the router fails."""
    pts = list(dict.fromkeys((round(a, 5), round(b, 5)) for a, b in points))
    if not enabled() or len(pts) < 2 or len(pts) > MAX_TABLE_POINTS:
        return None
    locs = [_loc(p) for p in pts]
    body = {"sources": locs, "targets": locs, "costing": COSTING, "units": "kilometers"}
    avoid = _avoid(hazards, pts)
    if avoid:
        body["exclude_polygons"] = avoid
    out = {}
    for row in _post("/sources_to_targets", body)["sources_to_targets"]:
        for c in row:
            if c["from_index"] != c["to_index"] and c.get("distance") is not None:
                out[(pts[c["from_index"]], pts[c["to_index"]])] = c["distance"]
    return out


def _decode(shape, precision=6):
    """Decode an encoded polyline (Valhalla uses 6 decimal places) into [[lat, lng], ...]."""
    coords, index, lat, lng, f = [], 0, 0, 0, 10 ** precision
    while index < len(shape):
        for which in (0, 1):
            result, shift = 0, 0
            while True:
                b = ord(shape[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if which == 0:
                lat += delta
            else:
                lng += delta
        coords.append([round(lat / f, 5), round(lng / f, 5)])
    return coords


def _thin(coords):
    step = max(1, math.ceil(len(coords) / MAX_SHAPE_POINTS))
    out = coords[::step]
    return out if out[-1] == coords[-1] else out + [coords[-1]]


def route(points, hazards):
    """Drive the points in order on real roads. Returns {"km", "minutes", "legs": [{"km", "minutes"}],
    "shape": [[lat, lng], ...], "source": "road"}; None when disabled; raises when the router fails."""
    if not enabled():
        return None
    body = {"locations": [_loc(p) for p in points], "costing": COSTING, "units": "kilometers"}
    avoid = _avoid(hazards, points)
    if avoid:
        body["exclude_polygons"] = avoid
    trip = _post("/route", body)["trip"]
    legs = [{"km": round(l["summary"]["length"], 2), "minutes": round(l["summary"]["time"] / 60, 1)} for l in trip["legs"]]
    shape = [pt for l in trip["legs"] for pt in _decode(l["shape"])]
    return {"km": round(sum(l["km"] for l in legs), 1), "minutes": round(sum(l["minutes"] for l in legs), 1),
            "legs": legs, "shape": _thin(shape), "source": "road"}


def estimate(points):
    """Same shape as route(), from straight-line distances: the fallback when no router answers."""
    legs = [{"km": round(distance_km(a, b), 2), "minutes": round(distance_km(a, b) / SPEED_KMH * 60, 1)}
            for a, b in zip(points, points[1:])]
    return {"km": round(sum(l["km"] for l in legs), 1), "minutes": round(sum(l["minutes"] for l in legs), 1),
            "legs": legs, "shape": [[round(a, 5), round(b, 5)] for a, b in points], "source": "estimate"}
