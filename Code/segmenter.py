import scipy.io
import cv2
import os

os.chdir('/content')
mat = scipy.io.loadmat('data/color150.mat')

print(mat['colors'][1])
for i in range(len(mat['colors'])):
    mat['colors'][i] = [0, 0, 0]
mat['colors'][1] = [255, 255, 255]  # originally it is 180 120 120
print(mat['colors'][1])
scipy.io.savemat('data/color150_white.mat', mat)

for i in range(len(mat['colors'])):
    mat['colors'][i] = [255, 255, 255]
mat['colors'][1] = [0, 0, 0]  # originally it is 180 120 120
print(mat['colors'][1])
scipy.io.savemat('data/color150_black.mat', mat)

# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

colors = scipy.io.loadmat('data/color150_white.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index + 1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    # display(PIL.Image.fromarray(im_vis))
    return PIL.Image.fromarray(im_vis)


# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()


def visualize_result(img, pred, index=None):
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index + 1]}:')
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    return im_vis
    return PIL.Image.fromarray(im_vis)
    # display(PIL.Image.fromarray(im_vis))


def return_building_images_from_localfile(img):
    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    pil_image = PIL.Image.open(img).convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return visualize_result(img_original, pred)


from torchvision import transforms


def return_building_images_from_tensor(img):
    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])
    pil_image = transforms.ToPILImage()(img).convert("RGB")
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]
    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
        # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    return visualize_result(img_original, pred)
