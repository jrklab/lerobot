"""brain/markers.py on synthetic front-camera images: a marker drawn where the camera would see it
is measured back to where it was."""
# ruff: noqa: N806

import cv2
import numpy as np
import pytest
from brain.cameras import Lens
from brain.markers import DICTIONARY, MARKER_SIZE_M, find_markers

K = np.array([[420.0, 0, 320], [0, 420.0, 240], [0, 0, 1]])
LENS = Lens(K, np.zeros(5))
# The camera 0.1 m behind the arm's origin and 0.05 m above the floor plane z=-0.1, looking along
# +x (camera z forward = arm x, camera x right = arm -y, camera y down = arm -z).
T_BASE_CAM = np.eye(4)
T_BASE_CAM[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
T_BASE_CAM[:3, 3] = [-0.1, 0.0, -0.05]


def _render(mid, centre, heading_deg, blank=False):
    """A 640x480 grey image of marker `mid` (white margin around it) upright with its middle at
    `centre` (arm frame), its face turned heading_deg from facing the camera. blank: the paper
    only (black where it is), to paste it onto a face."""
    side = 210  # 7 cells (5x5 and the border) of 30 px
    margin = 35  # 10 mm on the 60 mm marker, as printed (make_markers.py)
    tile = np.pad(cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), mid, side),
                  margin, constant_values=255)
    if blank:
        tile[:] = 0
    a = np.radians(heading_deg)
    normal = np.array([-np.cos(a), -np.sin(a), 0.0])  # out of the face, towards the robot
    across = np.array([-normal[1], normal[0], 0.0])  # the face's left-right, seen from in front
    up = np.array([0.0, 0.0, 1.0])
    full = MARKER_SIZE_M * (side + 2 * margin) / side
    h = full / 2
    world = [centre - across * h + up * h, centre + across * h + up * h,
             centre + across * h - up * h, centre - across * h - up * h]  # TL TR BR BL seen from the front
    T = np.linalg.inv(T_BASE_CAM)
    px = np.array([K @ (T[:3, :3] @ p + T[:3, 3]) for p in world])
    px = (px[:, :2] / px[:, 2:]).astype(np.float32)
    n = tile.shape[0]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]), px)
    return cv2.warpPerspective(tile, H, (640, 480), borderValue=255)


@pytest.mark.parametrize("x", [0.30, 0.45])
@pytest.mark.parametrize("heading", [0.0, 25.0, -30.0])
def test_a_marker_is_measured_where_it_is(x, heading):
    # From the place's distances (the reach point is ~35 cm from the camera). The facing comes
    # from the 6 cm between the side edges: a few mm there is ~5 deg; at 60 cm (the synthetic
    # marker ~30 px across) up to 12 deg.
    centre = np.array([x, 0.06, -0.05])
    found = find_markers(_render(90, centre, heading), LENS, T_BASE_CAM)
    assert [m.id for m in found] == [90]
    m = found[0]
    assert np.linalg.norm(m.centre - centre) < 0.015 * (x + 0.1)  # within 1.5% of its distance
    assert m.heading_deg() == pytest.approx(heading, abs=6.0)


def test_markers_not_on_a_container_are_ignored():
    # The calibration boards' ids (0-68) can be in view too.
    assert find_markers(_render(12, np.array([0.45, 0.0, -0.05]), 0.0), LENS, T_BASE_CAM) == []


def _render_face(mid, face_centre, face_top, marker_mid_z, background=200, dark_behind=False):
    """A grey image: a dark box face (its top at face_top, arm-frame z) square to the camera, with
    marker `mid` and a white margin on it, its middle at marker_mid_z; a light room behind.
    dark_behind: something as dark as the face behind its middle (a black bag, Nov 11)."""
    T = np.linalg.inv(T_BASE_CAM)

    def px(p):
        q = K @ (T[:3, :3] @ np.asarray(p, float) + T[:3, 3])
        return q[:2] / q[2]

    img = np.full((480, 640), background, np.uint8)
    if dark_behind:
        img[:, 270:370] = 25
    x, y = face_centre[0], face_centre[1]
    floor = -0.1
    face = np.array([px([x, y + 0.12, face_top]), px([x, y - 0.12, face_top]),
                     px([x, y - 0.12, floor]), px([x, y + 0.12, floor])], np.int32)
    cv2.fillConvexPoly(img, face, 25)
    marker = _render(mid, np.array([x, y, marker_mid_z]), 0.0)
    paper = _render(mid, np.array([x, y, marker_mid_z]), 0.0, blank=True)
    img[paper < 255] = marker[paper < 255]  # the marker's paper (margin) over the face
    return img


@pytest.mark.parametrize("top", [0.0, 0.015])
def test_the_top_of_the_marked_face_is_measured(top):
    from brain.markers import face_top_z

    img = _render_face(91, np.array([0.40, 0.0]), top, -0.055)
    (m,) = find_markers(img, LENS, T_BASE_CAM)
    assert face_top_z(img, m, LENS, T_BASE_CAM) == pytest.approx(top, abs=0.004)


def test_containers_are_marked_by_name():
    from brain.markers import CONTAINER_MARKER_IDS, marker_ids

    assert marker_ids("black shoe box") == CONTAINER_MARKER_IDS
    assert marker_ids("paper cup") == ()


def test_a_marker_seen_nearly_edge_on_is_not_measured():
    # Nov 6: id 90 on the box's far side, at the frame's edge, read 13 cm from the camera.
    assert find_markers(_render(90, np.array([0.30, 0.0, -0.05]), 70.0), LENS, T_BASE_CAM) == []


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_a_marker_stuck_on_turned_is_measured_the_same(turns, monkeypatch):
    # Nov 6: id 90 went on the box a quarter turn round.
    import brain.markers as mk

    centre = np.array([0.35, 0.03, -0.05])
    img = _render(90, centre, 15.0)
    detect = cv2.aruco.ArucoDetector.detectMarkers

    def turned(self, grey):
        corners, ids, rej = detect(self, grey)
        return tuple(np.roll(c, turns, axis=1) for c in corners), ids, rej

    monkeypatch.setattr(mk.cv2.aruco.ArucoDetector, "detectMarkers", turned)
    (m,) = find_markers(img, LENS, T_BASE_CAM)
    assert np.linalg.norm(m.centre - centre) < 0.01 and m.heading_deg() == pytest.approx(15.0, abs=6.0)


def test_the_top_is_found_where_the_background_behind_it_is_as_dark_as_the_face():
    # Nov 11: a black bag behind the box over the marker carried the one-column walk up past the
    # top: the 11.5 cm box read 13.4-17.8 cm. The columns beside it, over the light room, end at it.
    from brain.markers import face_top_z

    img = _render_face(91, np.array([0.40, 0.0]), 0.015, -0.055, dark_behind=True)
    (m,) = find_markers(img, LENS, T_BASE_CAM)
    assert face_top_z(img, m, LENS, T_BASE_CAM) == pytest.approx(0.015, abs=0.004)
