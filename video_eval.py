import cv2
import os

image_folder = 'results/Lidar_Viz_Preds'

images = [img for img in os.listdir(image_folder) if img.startswith("lcheck") and img.endswith(".jpg")]
images.sort()

if not images:
    print("No images found in the directory that match the pattern.")
    exit()

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

while True:
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Display each frame for 100 ms (10 FPS)
            cv2.destroyAllWindows()
            exit()
