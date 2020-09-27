from __future__ import division
from yoloObjectDetection.util import *
from yoloObjectDetection.darknet import Darknet

class ObjectDetector:
    def __init__(self, confidence, nms_threshold, numClasses, resolution):
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.numClasses = numClasses
        self.resolution = resolution

        self.cfgFile = "yoloObjectDetection/cfg/yolov3.cfg"
        self.weightsFile = "yoloObjectDetection/yolov3.weights"
        self.classes = load_classes('yoloObjectDetection/data/coco.names')
        self.CUDA = torch.cuda.is_available()

        self.model = Darknet(self.cfgFile)

        # If CUDA is detected, need to ensure data and model are both located in the GPU
        # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
        if self.CUDA:
            self.model = self.model.cuda()

        self.model.eval()
        self.model.load_weights(self.weightsFile)
        self.model.net_info["height"] = self.resolution
        self.inp_dim = int(self.model.net_info["height"])

    def PrepImage(self, frame):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """
        orignalImageFrame = frame
        dimensions = orignalImageFrame.shape[1], orignalImageFrame.shape[0]
        img = cv2.resize(orignalImageFrame, (self.inp_dim, self.inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orignalImageFrame, dimensions

    #ret and frame are obtained through the videoCapture
    def detectObject(self, ret, frame):
        if ret:
            image, originalImage, dim = self.PrepImage(frame)
            print("Detecting Object...")
            if self.CUDA:
                image = image.cuda()

            output = self.model(Variable(image), self.CUDA)
            output = write_results(output, self.confidence, self.numClasses, nms= True, nms_conf=self.nms_threshold)
            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim

            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            for tensor in output:
                c1 = tuple(tensor[1:3].int())
                c2 = tuple(tensor[3:5].int())
                cls = int(tensor[-1])
                label = "{0}".format(self.classes[cls])
                if (label == "cell phone"):
                    boundingBox = (c1[0].item(), c1[1].item(), c2[0].item() - c1[0].item(), c2[1].item() - c1[1].item())
                    return boundingBox

        return None
