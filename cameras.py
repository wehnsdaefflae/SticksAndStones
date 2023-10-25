import cv2


def is_available(source: int) -> bool:
    cap = cv2.VideoCapture(source)
    available = cap is not None and cap.isOpened()
    cap.release()
    return available


def return_camera_indexes() -> list[int]:
    # checks the first 10 indexes.
    index = 0
    arr = list()
    i = 10
    while 0 < i:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def get_last_camera_index() -> int:
    for i in range(10, -1, -1):
        if is_available(i):
            return i
    raise IndexError("No camera index found.")


print(get_last_camera_index())