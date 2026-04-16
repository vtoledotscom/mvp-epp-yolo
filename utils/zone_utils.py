from __future__ import annotations


def box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box_a: list[int], box_b: list[int]) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    return (ix2 - ix1) * (iy2 - iy1)


def box_center(box: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def point_in_rectangle(point: tuple[int, int], rect: dict) -> bool:
    return (
        rect["x1"] <= point[0] <= rect["x2"]
        and rect["y1"] <= point[1] <= rect["y2"]
    )


def point_in_polygon(point: tuple[int, int], polygon_points: list[list[int]] | list[tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon_points)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def point_in_zone(point: tuple[int, int], zone: dict | None) -> bool:
    if not zone:
        return True

    zone_type = zone.get("type", "rectangle")

    if zone_type == "rectangle":
        return point_in_rectangle(point, zone)

    if zone_type == "polygon":
        points = zone.get("points", [])
        return point_in_polygon(point, points)

    raise ValueError(f"Tipo de zona no soportado: {zone_type}")


def rectangle_overlap_ratio(box: list[int], rect: dict) -> float:
    inter = intersection_area(box, [rect["x1"], rect["y1"], rect["x2"], rect["y2"]])
    area = box_area(box)
    if area <= 0:
        return 0.0
    return inter / area


def polygon_overlap_proxy(box: list[int], polygon: dict) -> float:
    """
    Aproximación liviana para polígonos:
    - 1.0 si el centro está dentro
    - 0.0 si no
    """
    center = box_center(box)
    return 1.0 if point_in_polygon(center, polygon.get("points", [])) else 0.0


def overlap_ratio_with_zone(box: list[int], zone: dict | None) -> float:
    if not zone:
        return 1.0

    zone_type = zone.get("type", "rectangle")
    if zone_type == "rectangle":
        return rectangle_overlap_ratio(box, zone)
    if zone_type == "polygon":
        return polygon_overlap_proxy(box, zone)

    raise ValueError(f"Tipo de zona no soportado: {zone_type}")


def is_person_in_ignore_zones(person_box: list[int], ignore_zones: list[dict] | None) -> bool:
    if not ignore_zones:
        return False

    center = box_center(person_box)
    for zone in ignore_zones:
        if point_in_zone(center, zone):
            return True
    return False


def is_person_in_inspection_zone(
    person_box: list[int],
    inspection_zone: dict | None,
    min_overlap: float = 0.25,
) -> bool:
    if not inspection_zone:
        return True

    overlap = overlap_ratio_with_zone(person_box, inspection_zone)
    center = box_center(person_box)

    if overlap >= float(min_overlap):
        return True

    return point_in_zone(center, inspection_zone)