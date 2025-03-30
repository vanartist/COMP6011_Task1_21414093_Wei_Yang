from ultralytics import YOLO
import time

def main():
    start_time = time.time()
    model = YOLO("yolov8s-seg.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="yolo_f1",
        name="yolov_f1",
        verbose=True,
    )

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()