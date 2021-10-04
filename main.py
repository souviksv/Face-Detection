import cv2
import time
from fps import FPS

from face_detector import FaceDetector

def visualize(image, boxes, scores):

    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box
       
        h = ymax - ymin
        w = xmax - xmin

        im_w, im_h = image.shape[:2]

        ymin = int(max(ymin - (w * 0.35), 0))
        ymax = int(min(ymax + (w * 0.35), im_w))

        xmin = int(max(xmin - (h * 0.35), 0))
        xmax = int(min(xmax + (h * 0.35), im_h))

        cropped_face = image[ymin:ymax, xmin:xmax]
        
        cv2.imshow('CroppedImage', cropped_face)
        cv2.waitKey(1)

        outline = (0,255,0)

        score = round(score, 3)

        cv2.rectangle(image, (xmin, ymin), (min(xmax, image.shape[1]), min(ymax, image.shape[0])), outline, thickness=2)
        cv2.putText(image, str(score), (int(xmin)+5, int(ymin) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, outline, thickness=2)
    
    return image


 
def main():
    fps = FPS()
    MODEL_PATH = './model.pb'
    face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()

        if frame is None:
            print("Frame not found !!!")
            break

        resize_frame = cv2.resize(frame, (640, 480))
        rgb_image = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)

        start_time = time.perf_counter()

        boxes, scores = face_detector(rgb_image, score_threshold=0.3)

        detect_cost = time.perf_counter() - start_time
        disp_image = visualize(resize_frame, boxes, scores)

        print(f"Detection Cost: {detect_cost * 1000:.2f}ms")

        cv2.imshow("Detected Faces", disp_image)
        key = cv2.waitKey(1) & 0xff

        print("FPS:{}".format(fps()))

        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()




    
