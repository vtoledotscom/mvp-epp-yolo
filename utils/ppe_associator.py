"""
utils/ppe_associator.py
=======================

Asociación exclusiva de EPP por persona.

Corrección importante:
- ya no mezcla casco de una persona con chaleco de otra
- casco y chaleco se buscan primero dentro de la misma persona
- una detección de EPP se asigna a una sola persona
"""

from collections import defaultdict


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    return (ix2 - ix1) * (iy2 - iy1)


def _overlap_ratio(box, region):
    area = _box_area(box)
    if area <= 0:
        return 0.0
    return _intersection_area(box, region) / area


def _in_region(pt, region):
    x1, y1, x2, y2 = region
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


def _head_region(pbox):
    x1, y1, x2, y2 = pbox
    w = x2 - x1
    h = y2 - y1
    return [
        int(x1 + 0.10 * w),
        y1,
        int(x2 - 0.10 * w),
        int(y1 + 0.32 * h),
    ]


def _torso_region(pbox):
    x1, y1, x2, y2 = pbox
    w = x2 - x1
    h = y2 - y1
    return [
        int(x1 + 0.08 * w),
        int(y1 + 0.22 * h),
        int(x2 - 0.08 * w),
        int(y1 + 0.85 * h),
    ]


def _inside_person(box, person_box, margin=20):
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = person_box
    return (
        x1 >= px1 - margin
        and y1 >= py1 - margin
        and x2 <= px2 + margin
        and y2 <= py2 + margin
    )


def _score_candidate(epp_box, region_box, person_box):
    score = 0.0

    overlap = _overlap_ratio(epp_box, region_box)
    score += overlap

    if _in_region(_center(epp_box), region_box):
        score += 0.50

    if _inside_person(epp_box, person_box, margin=20):
        score += 0.25

    return score


def _build_person_assignments(persons, detections, class_name, region_key, min_score):
    assignments = {}
    candidate_pairs = []

    items = [d for d in detections if d.get("class_name") == class_name]

    for p in persons:
        person_id = p["person_id"]
        person_box = p["box"]
        region_box = p[region_key]

        for idx, det in enumerate(items):
            det_box = det["box"]
            score = _score_candidate(det_box, region_box, person_box)

            if score >= min_score:
                candidate_pairs.append((score, person_id, idx, det))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    used_persons = set()
    used_items = set()

    for score, person_id, idx, det in candidate_pairs:
        if person_id in used_persons or idx in used_items:
            continue

        assignments[person_id] = {
            "det": det,
            "score": score,
        }
        used_persons.add(person_id)
        used_items.add(idx)

    return assignments


def evaluate_all_persons(persons: list, detections: list) -> list:
    prepared = []

    for person in persons:
        pbox = person["box"]
        prepared.append({
            "person_id": person["person_id"],
            "box": pbox,
            "head_box": _head_region(pbox),
            "torso_box": _torso_region(pbox),
        })

    helmet_assignments = _build_person_assignments(
        persons=prepared,
        detections=detections,
        class_name="helmet",
        region_key="head_box",
        min_score=0.25,
    )

    vest_assignments = _build_person_assignments(
        persons=prepared,
        detections=detections,
        class_name="vest",
        region_key="torso_box",
        min_score=0.30,
    )

    result = []

    for item in prepared:
        person_id = item["person_id"]

        helmet_data = helmet_assignments.get(person_id)
        vest_data = vest_assignments.get(person_id)

        best_helmet = helmet_data["det"] if helmet_data else None
        best_vest = vest_data["det"] if vest_data else None

        helmet_score = helmet_data["score"] if helmet_data else 0.0
        vest_score = vest_data["score"] if vest_data else 0.0

        helmet_ok = best_helmet is not None
        vest_ok = best_vest is not None

        result.append({
            "person_id": person_id,
            "person_box": item["box"],
            "head_box": item["head_box"],
            "torso_box": item["torso_box"],
            "helmet_ok": helmet_ok,
            "vest_ok": vest_ok,
            "has_helmet": helmet_ok,
            "has_vest": vest_ok,
            "helmet": best_helmet,
            "vest": best_vest,
            "helmet_score": helmet_score,
            "vest_score": vest_score,
        })

    return result


def evaluate_person(person: dict, detections: list):
    """
    Compatibilidad hacia atrás.
    Evalúa una sola persona, pero usando la misma lógica.
    """
    results = evaluate_all_persons([person], detections)
    if not results:
        return {
            "person_id": person.get("person_id"),
            "person_box": person.get("box"),
            "head_box": None,
            "torso_box": None,
            "helmet_ok": False,
            "vest_ok": False,
            "has_helmet": False,
            "has_vest": False,
            "helmet": None,
            "vest": None,
            "helmet_score": 0.0,
            "vest_score": 0.0,
        }
    return results[0]