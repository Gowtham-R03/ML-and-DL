import cv2
import matplotlib.pyplot as plt
config_file ='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels = []
file_name = 'coco.names'

thresh = 0.55

def warn():
    img5 = cv2.imread("warning.png")
    cv2.imshow("WARNING",img5)
    
    
    
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2 = 127.5 scaling
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("webcam")
font_scale = 2
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret,frame = cap.read()
    ClassIndex, confidece,bbox = model.detect(frame,confThreshold = thresh)
    print(ClassIndex,bbox)
   
    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale = font_scale, color = (0,255,0), thickness = 3)
                cv2.putText(frame,str(round(conf*100,2)),(boxes[0]+200,boxes[1]+30),font, fontScale = font_scale, color = (0,255,0), thickness = 3)
    cv2.imshow('object detection in video',frame)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break

if (ClassIndex == [1]):
        warn()
       

cap.release()
cv2.destroyAllWindows()