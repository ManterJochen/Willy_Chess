import cv2 as cv
import numpy as np
from src.board.saddle import SaddlePointDetector
from src.board.symetric_solver import ChessboardSolver
from src.board.visualize import visualize_reconstruction

def open_camera(camera_id: int = 0) -> None:
    """
    Open the camera and display a live video stream.

    Args:
        camera_id: Camera index (0 = default camera).
    """
    cap = cv.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"Kamera mit ID {camera_id} konnte nicht geöffnet werden."
        )

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Kein Kamerabild empfangen.")
                break
            image = cv.resize(frame, (640, 480), interpolation=cv.INTER_AREA)
            points = SaddlePointDetector(image, 50, True).detect()
            solver = ChessboardSolver(lattice_points=points)
            chess_grid_points, basis_vectors, _ = solver.estimate_chess_grid(lattice_points=points)
            H = solver.estimate_homography(points, chess_grid_points)
            
            visualize_reconstruction(
                image,
                points,
                chess_grid_points,
                H,
                basis_vectors)

            # Press 'q' to exit.
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    open_camera()
