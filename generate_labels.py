import cv2
import numpy as np
import dlib
import math
from tqdm import tqdm
from utilities import *
from model import mfnet, iresnet, losses
import torch
import torchvision.transforms as T
from PIL import Image

def cal_R(ldmk):
    model_points = np.array([
            [6.825897, 6.760612, 4.402142],
            [1.330353, 7.122144, 6.903745],
            [-1.330353, 7.122144, 6.903745 ],
            [-6.825897, 6.760612, 4.402142],
            [5.311432, 5.485328, 3.987654],
            [1.789930, 5.393625, 4.413414],
            [-1.789930, 5.393625, 4.413414],
            [-5.311432, 5.485328, 3.987654],
            [2.005628, 1.409845, 6.165652],
            [-2.005628, 1.409845, 6.165652],
            [2.774015, -2.080775, 5.048531],
            [-2.774015, -2.080775, 5.048531],
            [0.000000, -3.116408, 6.097667],
            [0.000000, -7.415691, 4.070434]])
    points = np.array([(p.x, p.y) for p in ldmk.parts()],dtype="double")
    points_68 = np.array([points[17],points[21],points[22],points[26],points[36],points[39],points[42],points[45],points[31],points[35],points[48],points[54],points[57],points[8]],dtype="double")
    size=img.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype = "double")
    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, points_68, camera_matrix, dist_coeffs)
    theta = np.linalg.norm(rotation_vector)
    r = rotation_vector / theta
    R_ = np.array([[0, -r[2][0], r[1][0]],
            [r[2][0], 0, -r[0][0]],
            [-r[1][0], r[0][0], 0]])
    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r * r.T + np.sin(theta) * R_
    return R

def isRotationMatrix(R):
    Rt = np.transpose(R)  
    shouldBeIdentity = np.dot(Rt, R)   
    I = np.identity(3, dtype=R.dtype)         
    n = np.linalg.norm(I - shouldBeIdentity)   
    return n < 1e-6                           
 
 
def rotationMatrixToAngles(R):
    assert (isRotationMatrix(R))   
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])  
    singular = sy < 1e-6  
    if not singular:  
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:            
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)  
        z = 0
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
    return x, y, z

def estimate_blur_CPBD(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    blur_score = np.mean(sobel_mag) * 0.01
    return blur_score

def img_tensor(imgPath):
    img = Image.open(imgPath).convert("RGB")
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = transform(img)
    data = img_tensor.unsqueeze(dim=0)
    return data

if __name__ == '__main__':
    data_path = './'
    outfile_path = "./"
    dlib_weight = "./weights/shape_predictor_68_face_landmarks.dat"
    mfn_weight = "./weights/MFN_Factors.pth"
    fr_weight = "./"                                                # the FR backbone checkpoint training via CR-FIQA
    header_weight = "./"                                            # the header checkpoint training via CR-FIQA

    datatxt, id_dict, peop_num = get_image_paths(data_path)
    predictor = dlib.shape_predictor(dlib_weight)
    detector = dlib.get_frontal_face_detector() 
    
    mfn_net = mfnet.CLS_Model(mfnet.MobileFaceNet([112, 112], 512, output_name = 'GDC')).cuda()
    mfn_net = load_net_param(mfn_net, mfn_weight)
    mfn_net = mfn_net.eval()
    fr_net = iresnet.iresnet50(dropout=0.4, num_features=512, use_se=False, qs=1).cuda()
    fr_net = fr_net.eval()
    fr_net = load_net_param(fr_net, fr_weight)
    fr_header = losses.CR_FIQA_LOSS(in_features=512, out_features=peop_num, s=64.0, m=0.50).cuda()
    fr_header = fr_header.eval()
    fr_header = load_net_param(fr_header, header_weight)

    outfile = open(outfile_path, 'w')

    for image_path in tqdm(datatxt):
        image = cv2.imread(image_path)
        blur_score = estimate_blur_CPBD(image)
        BLUR = '2' if blur_score >= 0.7 else '1' if blur_score >= 0.35 else '0'

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_data = img_tensor(image_path).cuda()

        try: 
            face = detector(img, 2)[0]
            x0,y0,x1,y1 = face.left(),face.top(),face.right(),face.bottom()
            ldmk = predictor(img, face) 
            R = cal_R(ldmk)
            pitch, yaw, roll = rotationMatrixToAngles(R)
            print(f"Pitch: {pitch}; Yaw: {yaw}; Roll: {roll}")
            POSE = '2' if abs(yaw) <= 10.0 else '1' if abs(yaw) <= 25.0 else '0'
        except:
            POSE = '0'

        with torch.no_grad():
            _, expr_out, illum_out, occ_out, _ = mfn_net(tensor_data)
            expr_out = torch.argmax(expr_out, dim=1).cpu().detach().numpy()     #[0: typical expression, 1: exaggerated expression]
            illum_out = torch.argmax(illum_out, dim=1).cpu().detach().numpy()   #[0: normal lighting, 1: extreme lighting]
            occ_out = torch.argmax(occ_out, dim=1).cpu().detach().numpy()       #[0: “unobstructed”, 1: “obstructed”]
            EXPRESSION = 1 if expr_out==0 else 0                                #Reverse the lables
            ILLUMINATION = 1 if illum_out==0 else 0
            OCCLUSION = 1 if occ_out==0 else 0

            id = torch.tensor(id_dict[image_path.split('/')[-2]]).cuda().to(torch.long).unsqueeze(dim=0)
            _, _, ccs, nnccs = fr_header(fr_net(tensor_data)[0], id)
            qs = ccs / nnccs
            QUALITY = qs.cpu().detach().numpy()
            
            out_res = f"{image_path}\t{QUALITY}\t{BLUR}\t{POSE}\t{EXPRESSION}\t{ILLUMINATION}\t{OCCLUSION}"
            outfile.write(out_res+'\n')

    outfile.close()