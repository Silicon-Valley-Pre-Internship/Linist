import os
import tqdm
import logging
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
from skimage import io, transform


logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def __work_mode__(path: str):
    if os.path.isfile(path):  # Input is file
        return "file"
    if os.path.isdir(path):  # Input is dir
        return "dir"
    else:
        return "no"


def process(input_path, model_name="u2net"):

    if input_path is None:
        raise Exception("Bad parameters! Please specify input path and output path.")

    model = U2NET(model_name)  # Load model
    preprocessing_method = BoundingBoxDetectionFastRcnn()
    postprocessing_method = RemovingTooTransparentBordersHardAndBlurringHardBorders()
    wmode = __work_mode__(input_path)  # Get work mode
    if wmode == "file":  # File work mode
        image = model.process_image(input_path, preprocessing_method, postprocessing_method)
        return image
    elif wmode == "dir":  # Dir work mode
        # Start process
        files = os.listdir(input_path)
        for file in tqdm.tqdm(files, ascii=True, desc='Remove Background', unit='image'):
            file_path = os.path.join(input_path, file)
            image = model.process_image(file_path, preprocessing_method, postprocessing_method)
    else:
        raise Exception("Bad input parameter! Please indicate the correct path to the file or folder.")

logger = logging.getLogger(__name__)

class U2NET:
    """U^2-Net model interface"""
    def __init__(self, name="u2net"):
        self.Variable = Variable
        self.torch = torch
        self.U2NET_DEEP = U2NET_DEEP
        self.U2NETP_DEEP = U2NETP_DEEP

        if name == 'u2net':  # Load model
            logger.debug("Loading a U2NET model (176.6 mb) with better quality but slower processing.")
            net = self.U2NET_DEEP()
        elif name == 'u2netp':
            logger.debug("Loading a U2NETp model (4 mb) with lower quality but fast processing.")
            net = self.U2NETP_DEEP()
        else:
            raise Exception("Unknown u2net model!")
        try:
            path = '/Users/gimgiho/Desktop/Linist/backend/app/u2net.pth' # model path
            if self.torch.cuda.is_available():
                net.load_state_dict(self.torch.load(path))
                net.cuda()
            else:
                net.load_state_dict(self.torch.load(path, map_location="cpu"))
        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained model found! Run setup.sh or setup.bat to download it!")
        net.eval()
        self.__net__ = net  # Define model

    def process_image(self, data, preprocessing=None, postprocessing=None):
        if isinstance(data, str):
            logger.debug("Load image: {}".format(data))
        image, org_image = self.__load_image__(data)  # Load image
        if image is False or org_image is False:
            return False
        if preprocessing:  # If an algorithm that preprocesses is specified,
            # then this algorithm should immediately remove the background
            image = preprocessing.run(self, image, org_image)
        else:
            image = self.__get_output__(image, org_image)  # If this is not, then just remove the background
        if postprocessing:  # If a postprocessing algorithm is specified, we send it an image without a background
            image = postprocessing.run(self, image, org_image)
        return image

    def __get_output__(self, image, org_image):
        start_time = time.time()  # Time counter
        image = image.type(self.torch.FloatTensor)
        if self.torch.cuda.is_available():
            image = self.Variable(image.cuda())
        else:
            image = self.Variable(image)
        mask, d2, d3, d4, d5, d6, d7 = self.__net__(image)  # Predict mask
        logger.debug("Mask prediction completed")
        # Normalization
        logger.debug("Mask normalization")
        mask = mask[:, 0, :, :]
        mask = self.__normalize__(mask)
        # Prepare mask
        logger.debug("Prepare mask")
        mask = self.__prepare_mask__(mask, org_image.size)
        # Apply mask to image
        logger.debug("Apply mask to image")
        empty = Image.new("RGBA", org_image.size)
        image = Image.composite(org_image, empty, mask)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image

    def __load_image__(self, data):
        image_size = 320  # Size of the input and output image for the model
        if isinstance(data, str):
            try:
                image = io.imread(data)  # Load image if there is a path
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data)
                return False, False
            pil_image = Image.fromarray(image)
        else:
            image = np.array(data)  # Convert PIL image to numpy arr
            pil_image = data
        image = transform.resize(image, (image_size, image_size), mode='constant')  # Resize image
        image = self.__ndrarray2tensor__(image)  # Convert image from numpy arr to tensor
        return image, pil_image

    def __ndrarray2tensor__(self, image: np.ndarray):
        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image /= np.max(image)
        if image.shape[2] == 1:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        return self.torch.from_numpy(tmp_img)

    def __normalize__(self, predicted):
        """Normalize the predicted map"""
        ma = self.torch.max(predicted)
        mi = self.torch.min(predicted)
        out = (predicted - mi) / (ma - mi)
        return out

    @staticmethod
    def __prepare_mask__(predict, image_size):
        """Prepares mask"""
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        mask = Image.fromarray(predict_np * 255).convert("L")
        mask = mask.resize(image_size, resample=Image.BILINEAR)
        return mask

def method_detect(method: str):
    return RemovingTooTransparentBordersHardAndBlurringHardBorders()

class RemovingTooTransparentBordersHardAndBlurringHardBorders:

    def __init__(self):
        import cv2
        import skimage
        import numpy as np
        self.cv2 = cv2
        self.skimage = skimage
        self.np = np

        self.model = None
        self.prep_image = None
        self.orig_image = None

    @staticmethod
    def __extact_alpha_channel__(image):
        # Extract just the alpha channel
        alpha = image.split()[-1]
        # Create a new image with an opaque black background
        bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
        # Copy the alpha channel to the new image using itself as the mask
        bg.paste(alpha, mask=alpha)
        return bg.convert("RGB")

    def __blur_edges__(self, imaged):
        image = self.np.array(imaged)
        image = self.cv2.cvtColor(image, self.cv2.COLOR_RGBA2BGRA)
        # extract alpha channel
        a = image[:, :, 3]
        # blur alpha channel
        ab = self.cv2.GaussianBlur(a, (0, 0), sigmaX=2, sigmaY=2, borderType=self.cv2.BORDER_DEFAULT)
        # stretch so that 255 -> 255 and 127.5 -> 0
        # noinspection PyUnresolvedReferences
        aa = self.skimage.exposure.rescale_intensity(ab, in_range=(140, 255), out_range=(0, 255))
        # replace alpha channel in input with new alpha channel
        out = image.copy()
        out[:, :, 3] = aa
        image = self.cv2.cvtColor(out, self.cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(image)

    def __remove_too_transparent_borders__(self, mask, tranp_val=31):
        mask = self.np.array(mask.convert("L"))
        height, weight = mask.shape
        for h in range(height):
            for w in range(weight):
                val = mask[h, w]
                if val > tranp_val:
                    mask[h, w] = 255
                else:
                    mask[h, w] = 0
        return Image.fromarray(mask)

    def run(self, _, image, orig_image):
        mask = self.__remove_too_transparent_borders__(self.__extact_alpha_channel__(image))
        empty = Image.new("RGBA", orig_image.size)
        image = Image.composite(orig_image, empty, mask)
        image = self.__blur_edges__(image)
        return image


class BoundingBoxDetectionFastRcnn:
    def __init__(self):
        self.__fast_rcnn__ = FastRcnn()
        self.model = None
        self.prep_image = None
        self.orig_image = None

    @staticmethod
    def trans_paste(bg_img, fg_img, box=(0, 0)):
        fg_img_trans = Image.new("RGBA", bg_img.size)
        fg_img_trans.paste(fg_img, box, mask=fg_img)
        new_img = Image.alpha_composite(bg_img, fg_img_trans)
        return new_img

    @staticmethod
    def __orig_object_border__(border, orig_image, resized_image, indent=16):
        x_factor = resized_image.shape[1] / orig_image.size[0]
        y_factor = resized_image.shape[0] / orig_image.size[1]
        xmin, ymin, xmax, ymax = [int(x) for x in border]
        if ymin < 0:
            ymin = 0
        if ymax > resized_image.shape[0]:
            ymax = resized_image.shape[0]
        if xmax > resized_image.shape[1]:
            xmax = resized_image.shape[1]
        if xmin < 0:
            xmin = 0
        if x_factor == 0:
            x_factor = 1
        if y_factor == 0:
            y_factor = 1
        border = (int(xmin / x_factor) - indent,
                  int(ymin / y_factor) - indent, int(xmax / x_factor) + indent, int(ymax / y_factor) + indent)
        return border

    def run(self, model, prep_image, orig_image):
        _, resized_image, results = self.__fast_rcnn__.process_image(orig_image)

        classes = self.__fast_rcnn__.class_names
        bboxes = results['bboxes']
        ids = results['ids']
        scores = results['scores']

        object_num = len(bboxes)  # We get the number of all objects in the photo

        if object_num < 1:  # If there are no objects, or they are not found,
            # we try to remove the background using standard tools
            return model.__get_output__(prep_image, orig_image)
        else:
            # Check that all arrays match each other in size
            if ids is not None and not len(bboxes) == len(ids):
                return model.__get_output__(prep_image,
                                            orig_image)  # we try to remove the background using standard tools
            if scores is not None and not len(bboxes) == len(scores):
                return model.__get_output__(prep_image, orig_image)
                # we try to remove the background using standard tools
        objects = []
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < 0.5:
                continue
            if ids is not None and ids.flat[i] < 0:
                continue
            object_cls_id = int(ids.flat[i]) if ids is not None else -1
            if classes is not None and object_cls_id < len(classes):
                object_label = classes[object_cls_id]
            else:
                object_label = str(object_cls_id) if object_cls_id >= 0 else ''
            object_border = self.__orig_object_border__(bbox, orig_image, resized_image)
            objects.append([object_label, object_border])
        if objects:
            if len(objects) == 1:
                return model.__get_output__(prep_image, orig_image)
                # we try to remove the background using standard tools
            else:
                obj_images = []
                for obj in objects:
                    border = obj[1]
                    obj_crop = orig_image.crop(border)
                    # TODO: make a special algorithm to improve the removal of background from images with people.
                    if obj[0] == "person":
                        obj_img = model.process_image(obj_crop)
                    else:
                        obj_img = model.process_image(obj_crop)
                    obj_images.append([obj_img, obj])
                image = Image.new("RGBA", orig_image.size)
                for obj in obj_images:
                    image = self.trans_paste(image, obj[0], obj[1][1])
                return image
        else:
            return model.__get_output__(prep_image, orig_image)


class FastRcnn:
    def __init__(self):
        from gluoncv import model_zoo, data
        from mxnet import nd
        self.model_zoo = model_zoo
        self.data = data
        self.nd = nd
        logger.debug("Loading Fast RCNN neural network")
        self.__net__ = self.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc',
                                                pretrained=True)  # Download the pre-trained model, if one is missing.
        # noinspection PyUnresolvedReferences
        self.class_names = self.__net__.classes

    def __load_image__(self, data_input):
        if isinstance(data_input, str):
            try:
                data_input = Image.open(data_input)
                # Fix https://github.com/OPHoperHPO/image-background-remove-tool/issues/19
                data_input = data_input.convert("RGB")
                image = np.array(data_input)  # Convert PIL image to numpy arr
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data_input)
                return False, False
        else:
            # Fix https://github.com/OPHoperHPO/image-background-remove-tool/issues/19
            data_input = data_input.convert("RGB")
            image = np.array(data_input)  # Convert PIL image to numpy arr
        x, resized_image = self.data.transforms.presets.rcnn.transform_test(self.nd.array(image))
        return x, image, resized_image

    def process_image(self, image):
        start_time = time.time()  # Time counter
        x, image, resized_image = self.__load_image__(image)
        ids, scores, bboxes = [xx[0].asnumpy() for xx in self.__net__(x)]
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image, resized_image, {"ids": ids, "scores": scores, "bboxes": bboxes}


class MaskRcnn:
    def __init__(self):
        from gluoncv import model_zoo, utils, data
        from mxnet import nd
        self.model_zoo = model_zoo
        self.utils = utils
        self.data = data
        self.nd = nd
        logger.debug("Loading Mask RCNN neural network")
        self.__net__ = self.model_zoo.get_model('mask_rcnn_resnet50_v1b_coco',
                                                pretrained=True)  # Download the pre-trained model, if one is missing.
        # noinspection PyUnresolvedReferences
        self.class_names = self.__net__.classes

    def __load_image__(self, data_input):
        if isinstance(data_input, str):
            try:
                data_input = Image.open(data_input)
                # Fix https://github.com/OPHoperHPO/image-background-remove-tool/issues/19
                data_input = data_input.convert("RGB")
                image = np.array(data_input)  # Convert PIL image to numpy arr
            except IOError:
                logger.error('Cannot retrieve image. Please check file: ' + data_input)
                return False, False
        else:
            # Fix https://github.com/OPHoperHPO/image-background-remove-tool/issues/19
            data_input = data_input.convert("RGB")
            image = np.array(data_input)  # Convert PIL image to numpy arr
        x, resized_image = self.data.transforms.presets.rcnn.transform_test(self.nd.array(image))
        return x, image, resized_image

    def process_image(self, image):
        start_time = time.time()  # Time counter
        x, image, resized_image = self.__load_image__(image)
        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in self.__net__(x)]
        masks, _ = self.utils.viz.expand_mask(masks, bboxes, (image.shape[1], image.shape[0]), scores)
        logger.debug("Finished! Time spent: {}".format(time.time() - start_time))
        return image, resized_image, {"ids": ids, "scores": scores, "bboxes": bboxes,
                                      "masks": masks}

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

    return src


# RSU-7
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET_DEEP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET_DEEP, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)


### U^2-Net small ###
class U2NETP_DEEP(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP_DEEP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)
