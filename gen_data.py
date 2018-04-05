from constant import *
import numpy as np
import cv2

def convertTranslated(images, initImgSize, transSize, finalImgSize):
    size_diff = finalImgSize - transSize
    newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
    imgCoord = np.zeros([batch_size,2])
    for k in range(batch_size):
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))
        image = cv2.resize(image, dsize=(transSize, transSize), interpolation=cv2.INTER_NEAREST)
        # generate and save random coordinates
        randX = np.random.randint(0, size_diff)
        randY = np.random.randint(0, size_diff)
        imgCoord[k,:] = np.array([randX, randY])
        # padding
        image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
        # plt.imshow(image, cmap='gray')
        # plt.show()
        newimages[k, :] = np.reshape(image, (finalImgSize*finalImgSize))

    return newimages, imgCoord


def convertTranslated_mix(images, initImgSize, transSizes, finalImgSize):

    newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
    imgCoord = np.zeros([batch_size,2])
    for k in range(batch_size):
        rand_int = np.random.randint(0,len(transSizes))
        transSize = transSizes[rand_int]
        size_diff = finalImgSize - transSize
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))
        image = cv2.resize(image, dsize=(transSize, transSize), interpolation=cv2.INTER_NEAREST)
        # generate and save random coordinates
        randX = np.random.randint(0, size_diff)
        randY = np.random.randint(0, size_diff)
        imgCoord[k,:] = np.array([randX, randY])
        # padding
        image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)), 'constant', constant_values = (0))
        # plt.imshow(image, cmap='gray')
        # plt.show()
        newimages[k, :] = np.reshape(image, (finalImgSize*finalImgSize))

    return newimages, imgCoord


def convertCluttered_mix(images, initImgSize, transSizes, finalImgSize):
    imgCoord = np.zeros([batch_size,2])
    newimages = np.zeros([batch_size, finalImgSize * finalImgSize])
    clutter_size = int(MNIST_SIZE / 2)

    for k in range(batch_size):
        rand_int = np.random.randint(0, len(transSizes))
        transSize = transSizes[rand_int]
        size_diff = finalImgSize - transSize
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))
        image = cv2.resize(image, dsize=(transSize, transSize), interpolation=cv2.INTER_NEAREST)
        # generate and save random coordinates
        # Random1 = Random(217)
        randX_img = np.random.randint(0, size_diff)
        randY_img = np.random.randint(0, size_diff)
        imgCoord[k, :] = np.array([randX_img, randY_img])
        # clutter
        clutter = np.reshape(images[np.random.randint(0, batch_size), :], (initImgSize, initImgSize))
        num1 = np.random.randint(0, 2)
        num2 = np.random.randint(0, 2)
        clutter = clutter[num1 * clutter_size:num1 * clutter_size + clutter_size,
                  num2 * clutter_size:num2 * clutter_size + clutter_size]

        # if the clutter cannot fit
        if size_diff / 2 < clutter_size:
            rand_num = np.random.randint(0, 2)
            if rand_num:
                if randX_img > size_diff / 2:
                    clutter_x = 0
                else:
                    clutter_x = finalImgSize - clutter_size
                clutter_y = np.random.randint(0, finalImgSize - clutter_size)
            else:
                if randY_img > size_diff / 2:
                    clutter_y = 0
                else:
                    clutter_y = finalImgSize - clutter_size
                clutter_x = np.random.randint(0, finalImgSize - clutter_size)
        else:
            # padding
            clutter_x = np.random.randint(0, finalImgSize - clutter_size)
            clutter_y = np.random.randint(0, finalImgSize - clutter_size)
            while True:
                if randX_img - clutter_size < clutter_x < randX_img + transSize and randY_img - clutter_size < clutter_y < randY_img + transSize:
                    clutter_x = np.random.randint(0, finalImgSize - int(MNIST_SIZE / 2))
                    clutter_y = np.random.randint(0, finalImgSize - int(MNIST_SIZE / 2))
                else:
                    break
        image_pad = np.zeros((finalImgSize, finalImgSize))
        image_pad[clutter_x:clutter_x + clutter_size, clutter_y:clutter_y + clutter_size] = clutter
        image_pad[randX_img:randX_img + transSize, randY_img:randY_img + transSize] = image
        # plt.imshow(image_pad, cmap='gray')
        # plt.show()
        newimages[k, :] = np.reshape(image_pad, (finalImgSize * finalImgSize))

    return newimages, imgCoord


def convertCluttered(images, initImgSize, transSize, finalImgSize):
    clutter_size = int(MNIST_SIZE/2)
    size_diff = finalImgSize - transSize
    newimages = np.zeros([batch_size, finalImgSize*finalImgSize])
    imgCoord = np.zeros([batch_size,2])
    for k in range(batch_size):
        image = images[k, :]
        image = np.reshape(image, (initImgSize, initImgSize))
        image = cv2.resize(image, dsize=(transSize, transSize), interpolation=cv2.INTER_NEAREST)
        # generate and save random coordinates
        # Random1 = Random(217)
        randX_img = np.random.randint(0, size_diff)
        randY_img = np.random.randint(0, size_diff)
        imgCoord[k,:] = np.array([randX_img, randY_img])
        #clutter
        clutter = np.reshape(images[np.random.randint(0,batch_size), :], (initImgSize, initImgSize))
        num1 = np.random.randint(0,2)
        num2 = np.random.randint(0,2)
        clutter = clutter[num1*clutter_size:num1*clutter_size+clutter_size, num2*clutter_size:num2*clutter_size+clutter_size]

        # if the clutter cannot fit
        if size_diff/2 < clutter_size:
            rand_num = np.random.randint(0,2)
            if rand_num:
                if randX_img > size_diff/2:
                    clutter_x = 0
                else:
                    clutter_x = finalImgSize - clutter_size
                clutter_y = np.random.randint(0, finalImgSize - clutter_size)
            else:
                if randY_img > size_diff/2:
                    clutter_y = 0
                else:
                    clutter_y = finalImgSize - clutter_size
                clutter_x = np.random.randint(0, finalImgSize - clutter_size)
        else:
            # padding
            clutter_x = np.random.randint(0, finalImgSize - clutter_size)
            clutter_y = np.random.randint(0, finalImgSize - clutter_size)
            while True:
                if randX_img - clutter_size < clutter_x < randX_img + transSize and randY_img - clutter_size < clutter_y < randY_img + transSize:
                    clutter_x = np.random.randint(0, finalImgSize - int(MNIST_SIZE / 2))
                    clutter_y = np.random.randint(0, finalImgSize - int(MNIST_SIZE / 2))
                else:
                    break
        image_pad = np.zeros((finalImgSize, finalImgSize))
        image_pad[clutter_x:clutter_x+clutter_size, clutter_y:clutter_y+clutter_size] = clutter
        image_pad[randX_img:randX_img+transSize, randY_img:randY_img+transSize] = image
        # plt.imshow(image_pad, cmap='gray')
        # plt.show()
        newimages[k, :] = np.reshape(image_pad, (finalImgSize*finalImgSize))

    return newimages, imgCoord