from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import numpy as np
import cv2

#prepare GroundingDINO model
base_model = GroundingDINO(ontology=CaptionOntology({"hand": "hand"}))


def crop_img_bb(img,hand_bb,pad,show=False):
    if len(img.shape)==2:
        h,w=img.shape
    elif len(img.shape)==3:
        h,w,_=img.shape
    else:
        raise Exception("Invalid image shape")
    hand_bb=[int(bb) for bb in hand_bb]
    img_crop=img[max(0,hand_bb[1]-pad):min(h,hand_bb[3]+pad),max(0,hand_bb[0]-pad):min(w,hand_bb[2]+pad)]
    if show:
        import cv2
        # Display the image
        cv2.imshow("Image with Bounding Box", img_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_crop

def get_bb(results):
    bbx_list=results.xyxy
    conf_list=results.confidence
    if len(conf_list)==0:
        return []
    max_conf_arg=np.argmax(conf_list)
    bb=bbx_list[max_conf_arg]
    return bb

class WristDet_mediapipe:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp_hands = mp.solutions.hands
    
    def get_kypts(self,image):
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
                # image = cv2.imread(path)
                height, width, _ = image.shape
                # Detect hands in the image
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                xy_vals=[]
                z_vals=[]
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(21):
                            x = int(hand_landmarks.landmark[i].x * width)
                            y = int(hand_landmarks.landmark[i].y * height)
                            z=hand_landmarks.landmark[i].z
                            xy_vals.append((x,y))
                            z_vals.append(z)
                if len(xy_vals)==42:
                    closest_hand=np.argmin([z_vals[0],z_vals[21]])
                    start_coord=0 if closest_hand==0 else 21
                    xy_vals=xy_vals[start_coord:start_coord+21]
        return image,xy_vals




def get_kpts(img_path):
    results = base_model.predict(img_path)
    bb=get_bb(results)

    wrst=WristDet_mediapipe()
    img=cv2.imread(img_path)
    pad=80 
    img_crop= crop_img_bb(img,bb,pad,show=False)
    image,xy_vals=wrst.get_kypts(img_crop)
    x_vals=[int(val[0]+bb[0]-pad) for val in xy_vals]
    y_vals=[int(val[1]+bb[1]-pad) for val in xy_vals]
    kpt_dict={"x":x_vals,"y":y_vals}
    return kpt_dict

#how to use
#first keypoint is wrist. i.e kpt_dict['x'][0],kpt_dict['y'][0] is wrist
img_path=r'D:\CPR_extracted\P4\s_3\kinect\color\00550.jpg'
kpt_dict=get_kpts(img_path)







