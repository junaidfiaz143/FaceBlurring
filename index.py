import os
import cv2
import numpy as np
import tensorflow as tf

PATH_TO_CKPT = os.path.join("inference_graph", "frozen_inference_graph.pb")

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

threshold = 0.5  # if predicted image has more than 50% of confidence

cap = cv2.VideoCapture("videoali.mp4")
# cap = cv2.VideoCapture("rtsp://192.168.137.61:8554/live.sdp")
# cap = cv2.VideoCapture(0) #this is for WEBCAM

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    width  = cap.get(3) # float
    height = cap.get(4) # float

    if ret == True:
        frame_expanded = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        for index, score in enumerate(scores[0]):
            if round(score, 2) > threshold:
                if classes[0][index] == 1.0:
                    ymin = int(boxes[0][index][0] * height)
                    xmin = int(boxes[0][index][1] * width)
                    ymax = int(boxes[0][index][2] * height)
                    xmax = int(boxes[0][index][3] * width)

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 4)
                    frame[ymin:ymax, xmin:xmax] = cv2.blur(frame[ymin:ymax, xmin:xmax], (50, 50))
                    cv2.putText(frame, "FACE " + str(round(score*100, 2)) + "%", (xmin, ymin-13), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1, cv2.LINE_AA) 

        # Display the resulting frame
        cv2.imshow("Face Blurring", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("+-----------------+")
        print("NO FRAMES DETECTED!")
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()