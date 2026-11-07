"""The unified class list every dataset is mapped onto, and label-name aliases.

Class order is canonical: the builder writes the classes that actually occur in the built
dataset, in this order, so a model trained on the drone datasets only has
{drone, bird, airplane, helicopter}, and one trained together with the team's own
frames also has ball / card / hand. AimBot maps model class *names* to logical
classes (aimbot/perception/detectors.py), so indices never need to match across models.

bird / airplane / helicopter are explicit *non-target* classes (the Halmstad dataset
labels them). Learning them as classes, rather than as background, teaches the
detector what is not a drone, and the operator HUD shows them as non-targets.
"""

from __future__ import annotations

import re

CLASSES = ["drone", "bird", "airplane", "helicopter", "ball", "card", "hand"]

ALIASES = {
    # drones (dataset label names seen in MAV-VID, DUT, Det-Fly, Drone-vs-Bird, Anti-UAV, Roboflow ...)
    "drone": "drone", "drones": "drone", "dron": "drone", "uav": "drone", "uavs": "drone", "mav": "drone", "suas": "drone",
    "multirotor": "drone", "multicopter": "drone", "quadcopter": "drone", "quadrotor": "drone", "quad": "drone",
    "hexacopter": "drone", "octocopter": "drone", "dji": "drone", "mavic": "drone", "phantom": "drone",
    "inspire": "drone", "matrice": "drone", "tello": "drone", "parrot": "drone", "solo": "drone",
    "target": "drone", "target_uav": "drone", "uav_target": "drone", "drone_target": "drone",
    # confusers (Halmstad non-drone videos)
    "bird": "bird", "birds": "bird",
    "airplane": "airplane", "aeroplane": "airplane", "plane": "airplane", "aircraft": "airplane", "airliner": "airplane",
    "helicopter": "helicopter", "heli": "helicopter", "helicopters": "helicopter",
    # the project's other targets (team-recorded data)
    "ball": "ball", "tennis ball": "ball", "tennis_ball": "ball", "tennisball": "ball", "sports ball": "ball",
    "card": "card", "index card": "card", "index_card": "card", "indexcard": "card",
    "hand": "hand", "hands": "hand",
}


def normalize_label(raw, class_map: dict | None = None, default: str | None = None) -> str | None:
    """Dataset label -> unified class name, or None to drop the box.

    ``class_map`` (per source, from datasets/sources.yaml) wins over the built-in aliases;
    mapping a label to ``null``/``""`` drops it. ``default`` is used for labels that are
    not recognised at all (single-class drone datasets whose only label is e.g. "0").
    """
    if raw is None:
        return default
    key = str(raw).strip()
    if class_map:
        for k in (key, key.lower()):
            if k in class_map:
                v = class_map[k]
                return str(v) if v else None
    low = key.lower().replace("-", " ").strip()
    if low in ALIASES:
        return ALIASES[low]
    low2 = re.sub(r"[\s_]+", " ", low)
    if low2 in ALIASES:
        return ALIASES[low2]
    if low2.replace(" ", "_") in ALIASES:
        return ALIASES[low2.replace(" ", "_")]
    for word in re.split(r"[\s_/]+", low2):            # e.g. "dji mavic 2" / "uav_small"
        if word in ALIASES and ALIASES[word] == "drone":
            return "drone"
    return default
