import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2


def starSimulator(resolution=360, sigma=5, lightPollution=4):
    """
    input resolution(int), sigma(float), lightPollution(float)
    returns grids of r/g/b light intensity of size(resolution, resolution, 3)
    """
    # generate a grid
    x, y = np.meshgrid(np.linspace(0, resolution, resolution),
                       np.linspace(0, resolution, resolution))
    # calculate the Gaussian distribution
    dst = np.sqrt((x-resolution/2)*((x-resolution/2)) +
                  (y-resolution/2)*(y-resolution/2))
    source = np.exp(-((dst)**2 / (2.0 * sigma**2)))
    # add a background of light pollution, and normalize
    source += (lightPollution*np.ones((resolution, resolution)))
    source /= (1 + lightPollution)
    rgb_source = source[:, :, np.newaxis].repeat(3, axis=2)
    # rgb_source of size(resolution, resolution, 3)

    return rgb_source


def exposureSimulator(rgb_source, integrationTime):
    """
    input rgb_source of size(resolution, resolution, 3), integrationTime(int)
    returns a RGBchannel image of size (resolution, resolution, 3)
    """
    if not integrationTime:
        rgbChannels = np.array(rgb_source)

    else:
        # total signal, equivalent to summing up many exposures
        rgbChannels = rgb_source * integrationTime
        # sum after poisson process acts same (on probability) as sum before poisson process
        rgbChannels = np.random.poisson(rgbChannels)
        rgbChannels = rgbChannels / float(integrationTime)

    return rgbChannels


def plotColoredImage(exposureTime, source):
    # this will make imageGenerator return expexted value
    exposureTime.insert(0, None)

    # dimension of subplots
    sub = 2
    fig, axs = plt.subplots(sub, sub)

    # plot!
    for i in range(2):
        for j in range(2):
            coloredImage = exposureSimulator(source, exposureTime[i + 2*j])
            # take the average of (a range of row pixels) along the center column
            vResolution, hResolution, channels = np.shape(coloredImage)
            profile = coloredImage[int(
                vResolution/2-1):int(vResolution/2+1), :, 0].mean(axis=0)

            # some matplotlib code from StackOverFlow :)
            divider = make_axes_locatable(axs[i, j])
            right_ax = divider.append_axes(
                "right", 0.7, pad=0.1, sharey=axs[i, j])
            right_ax.yaxis.set_tick_params(labelleft=False)
            right_ax.set_xlabel('Intensity')
            right_ax.autoscale(enable=False)
            right_ax.set_xlim(right=1.25)
            v_prof, = right_ax.plot(profile, np.arange(
                profile.shape[0]), 'r-', linewidth=0.3)

            title = f"exp time {exposureTime[i + 2*j]} units"
            if i == 0 and j == 0:
                title = "expected value"
            axs[i, j].set_title(title)
            axs[i, j].imshow(coloredImage)
    plt.show()


def readImage(path):
    """
    input image path
    returns a RGBchannel image of size (resolution, resolution, 3)
    """
    # read directories containing utf-8 characters
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                         cv2.IMREAD_UNCHANGED).astype(np.float32)
    # cv2 reads color as BGR while matplotlib display as RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # decrease resolution
    height = 480
    aspectRatio = np.shape(image)[1] / np.shape(image)[0]
    width = int(height * aspectRatio)
    image = cv2.resize(image, (width, height))

    # scale to 0~1 float
    image /= 255.0

    return image


exp_time = [10, 33, 100]

# use simulated star as source
# sauce = starSimulator(resolution=360, sigma=5, lightPollution=4)

# use custom file as source
sauce = readImage("Rosette.jpg")

plotColoredImage(exp_time, sauce)
