# -*- encoding: utf-8 -*-

import argparse
# import datetime
import multiprocessing
import random
# import threading
from PIL import Image
from scipy import fft
import pandas as pd
# from scipy.stats import norm
from scipy.signal import find_peaks
import scipy.stats
# from statsmodels.tsa import tsatools, stattools
from torch.utils.data.distributed import DistributedSampler

from scipy import stats
# from statsmodels.tsa.stattools import adfuller, coint
import numpy as np
# import keras
import requests
# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from PIL import Image
# from keras.preprocessing import image
import cv2
from imgaug import augmenters as iaa
import torch, torchvision
import os
from random import sample

import torch.nn.functional as F
from statsmodels.tsa.stattools import coint
from torch import nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as utils
# from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import shutil
import time
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import math
from pathlib import Path
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
# import statsmodels.api as sm
# import transformers
import cupy as cp

from tqdm import tqdm

from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable


def get_class_name():
    class_index_dict = datasets.ImageFolder("/data/imagenet_2012/val").classes
    return class_index_dict


class_name = get_class_name()


# def keras_dataGenerator(source_path, target_path, img_num):
#     '''
#     # 函数作用是将source_path文件夹中的图片进行随机干扰变换，然后将其输出到目标文件夹。
#     :param source_path: 源文件夹，存放用于干扰的图片
#     :param target_path: 目标文件夹，存放源文件夹中图片干扰之后的图片
#     :param img_num: 需要生成的干扰图片的数目
#     :return: 无
#     '''
#
#     fill_mode = ["reflect", "wrap", "nearest"]
#     datagen = image.ImageDataGenerator(
#         zca_whitening=True,
#         rotation_range=30,
#         width_shift_range=0.03,
#         height_shift_range=0.03,
#         shear_range=0.5,
#         zoom_range=0.1,
#         channel_shift_range=100,
#         horizontal_flip=True,
#         fill_mode=fill_mode[np.random.randint(3)]
#     )
#     gen_data = datagen.flow_from_directory(source_path,
#                                            batch_size=1,
#                                            shuffle=False,
#                                            save_to_dir=target_path,
#                                            save_prefix="gen",
#                                            target_size=(224, 224))
#     while (img_num > 0):
#         gen_data.next()
#         img_num -= 1


## 生成固定干扰的图片，为了便于两个图片协同变换。方法弃用
def generate_coin(sigma):  # crop, fliplr, sigma, connor, flipud, x_s, y_s, x_t, y_t, rotate, shear, order, mode):

    seq = iaa.Sequential({  # 建立一个名为seq的实例，定义增强方法，用于增强
        # iaa.Crop(px=(crop, crop+1)),  # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
        # # iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
        # iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
        #
        # iaa.Fliplr(fliplr),
        iaa.GaussianBlur(sigma=sigma)
        # iaa.contrast.LinearContrast(connor, per_channel=True),
        # iaa.Flipud(flipud),
        #
        # iaa.Affine(
        #     scale={"x": x_s, "y": y_s},
        #     translate_percent={"x": x_t, "y": y_t},
        #     rotate=rotate,
        #     shear=shear,
        #     order=order,
        #     mode=mode
        # )
    })
    return seq


## 生成固定干扰的图片，为了便于两个图片协同变换。
def generate_coinT(aug_num, thread_num, img_list_dir="/home/python/Image/exper_v1/marker_C/",
                   save_dir="/home/python/Image/exper_v1/marker_C/"):
    # aug_num = args[0]
    # thread_num = args[1]

    print("线程序号：", thread_num)
    try:
        imglist = np.load(img_list_dir + "/result/data_imglist_" + str(thread_num) + ".npy",
                          allow_pickle=True)
        print(np.shape(imglist))  # (1, ..,.,.)
        img_num = 1

        # sleepawhile(3)

        imglist = [imglist]
        # crop_ind = 0
        # fliplr_ind = 0
        # sigma_ind = 1
        # connor_ind = 0
        # flipud_ind = 0
        # x_s_ind = 0
        # rotate_ind = 0
        # shear_ind = 0
        # order_ind = 0
        # mode_ind = 0

        for i in range(aug_num):
            #     crop =  random.randint(0, 20)
            # crop = (0.01 * np.abs(np.sin(i / 3.14))) if crop_ind else 0
            # crop = int(20 * (i/aug_num)) if crop_ind else 0
            # fliplr = (i % 2) if fliplr_ind else 0
            # # sigma = 5 * np.abs(np.cos(i / 3.14)) if sigma_ind else 0
            # sigma = 5 * (i/aug_num) # if sigma_ind else 0
            #
            # # (0.75-1.5)
            # # connor = np.sin(i / 3.14) * 0.75 + 0.75 if connor_ind else 1
            # connor = (i/aug_num) * 0.75 + 0.75 if connor_ind else 1
            #
            # flipud = i % 2 if flipud_ind else 0
            #
            # if x_s_ind:
            #     # x_s = np.sin(i / 3.14) * 0.4 + 0.8
            #     # y_s = np.sin(i / 3.14) * 0.4 + 0.8
            #     # x_t = np.sin(i / 3.14) * 0.4 - 0.2
            #     # y_t = np.sin(i / 3.14) * 0.4 - 0.2
            #     x_s = (i/aug_num) * 0.4 + 0.8
            #     y_s = (i/aug_num) * 0.4 + 0.8
            #     x_t = (i/aug_num) * 0.4 - 0.2
            #     y_t = (i/aug_num) * 0.4 - 0.2
            # else:
            #     x_s = 1
            #     y_s = 1
            #     x_t = 0
            #     y_t = 0
            #
            # # rotate = int(90 * np.sin(i / 3.14) - 45) if rotate_ind else 0
            # rotate = int(90 * (2*i/aug_num-1)) if rotate_ind else 0
            # # shear = int(32 * np.sin(i / 3.14) - 16) if shear_ind else 0
            # shear = int(32 * (2*i/aug_num-1) - 16) if shear_ind else 0
            # order = (i % 2) if order_ind else 0
            #
            # c = ["edge", "symmetric", "reflect", "wrap"]
            # mode = c[i % 4] if mode_ind else "constant"
            # mode = "constant"

            # if i % 50 == 0:
            #     print(i)

            # seq = generate_coin(sigma) #crop, fliplr, sigma, connor, flipud, x_s, y_s, x_t, y_t, rotate, shear, order, mode)

            # 变换的种类，可以在此选择，然后找出有意义的干扰。
            # seq = iaa.Sequential({  # 建立一个名为seq的实例，定义增强方法，用于增强
            #     # iaa.Crop(px=(crop, crop+1)),  # 对图像进行crop操作，随机在距离边缘的0到16像素中选择crop范围
            #     # # iaa.Fliplr(0.5),  # 对百分之五十的图像进行做左右翻转
            #     # iaa.GaussianBlur((0, 1.0)),  # 在模型上使用0均值1方差进行高斯模糊
            #     #
            #     # iaa.Fliplr(fliplr),
            #     iaa.GaussianBlur(sigma=sigma)
            #     # iaa.contrast.LinearContrast(connor, per_channel=True),
            #     # iaa.Flipud(flipud),
            #     #
            #     # iaa.Affine(
            #     #     scale={"x": x_s, "y": y_s},
            #     #     translate_percent={"x": x_t, "y": y_t},
            #     #     rotate=rotate,
            #     #     shear=shear,
            #     #     order=order,
            #     #     mode=mode
            #     # )
            # })

            # 作为marker_C的时候是需要进行调制的,但是marker_B不需要
            seq = iaa.GaussianBlur(sigma=(6 + 6 * np.sin(7 * i * np.pi / 180)))

            # marker_B
            # seq = iaa.GaussianBlur((0, 1.0))
            images_aug = seq.augment_images(imglist)
            #  这里得出的结果应该是50*图片尺寸，我们只需要直接进行保存就好了，不用全部进行存储
            for img in range(img_num):
                # cv2.imwrite("/home/python/Image/mid_result/aug/" + str(thread_num) + "/" + str(img) + "/val/class/" + str(i) +".jpg", images_aug[img])
                # print(np.shape(images_aug[img]))
                cv2.imwrite(
                    save_dir + str(thread_num) + "/val/class/" + str(i) + ".jpg",
                    images_aug[img])

            # for j in range(img_num):
            #     aug_list[j].append(images_aug[j])

            if i % 333 == 0:
                print("线程", thread_num, "进度：", (i / aug_num))


    except Exception as e:
        print(e)
    # print("线程", thread_num, "开始存储")
    # for j in range(img_num):  # （0-49）
    #     for k in range(aug_num):
    #         cv2.imwrite("/home/python/Image/mid_result/aug/" + str(thread_num) + '/' + str(j) + '/val/class/' + str(k) + '.jpg', aug_list[j][k])
    #


# 计算各种评估方式的方法
## 首先是预处理方法，为了删除左右异常值
def pre_handle(point):
    """
    方法是为了删除异常点，选择删除两边5%的点
    :param point: 要删除点的数组
    :return: 删除节点之后的数组
    """

    # v1
    # min_ = np.percentile(point, 2.5)  # 2.5%分位数
    # max_ = np.percentile(point, 97.5)
    # n = np.shape(point)[0]
    # count = int(n * 0.95)
    # z = count
    # point_new = np.zeros(n, )
    # k = 0
    # for i in range(n):
    #     if min_ <= point[i] < max_ and count > 0:
    #         point_new[k] = point[i]
    #         k += 1
    #         count -= 1
    #     elif count <= 0:
    #         point_new[k] = point[i]
    #         k += 1
    # point_new = point_new[:z]
    # point = point_new

    # v2
    # n = 0
    # miu = point.mean()
    # sigma = point.std()
    # if sigma == 0:
    #     return point
    # for i in range(np.shape(point)[0]):
    #     if np.abs(point[i] - miu) / sigma < 3:
    #         point[n] = point[i]
    #         n += 1
    # if n < 100:
    #     print(n)
    # return np.resize(point, (n,))

    # v3
    n = 0
    miu = point.mean()
    sigma = point.std()
    if sigma == 0:
        return point

    m_max = miu + 2 * sigma
    m_min = miu - 2 * sigma

    point = (point - m_min) / (m_max - m_min)

    return point


# 进行分箱
def fenxiang(sequence, Bin_number=10, max_=1, min_=0, del_zero=False):
    """
    将数据进行分箱，是为了将离散的数据进行概率化。
    有两种选择： 直接数字分箱或者将其拟合数据再分箱
    :param min_:
    :param max_:
    :param del_zero:
    :param sequence: 数据结果
    :param Bin_number: 箱的个数。默认值100
    :return: 返回分享结果，即为概率的结果数组。
    """
    sequence = sequence.reshape((-1,))

    sequence = (sequence - min_) / (max_ - min_)

    p = np.ones(Bin_number, )

    bins = []
    for low in range(0, 100, int(100 / Bin_number)):
        bins.append((low / 100, (low + 100 / Bin_number) / 100))
    #     print(bins)

    for j in range(np.shape(sequence)[0]):
        for i in range(0, len(bins)):
            if sequence[j] == 0 and del_zero:
                break
            if bins[i][0] <= sequence[j] < bins[i][1]:
                p[i] += 1
    for i in range(Bin_number):
        p[i] = p[i] / (np.shape(sequence)[0] + Bin_number)
    return p


# 计算js散度
def JS_divergence(point_1, point_2, del_zero=False, num_bins=100):
    """
    计算js散度的函数
    :param num_bins:
    :param del_zero:
    :param point_1: point表示要计算js散度的两个值，一般都是相同长度的概率数组。
    :param point_2:
    :return: 返回js散度
    """
    global js

    try:
        point_1 = point_1.reshape(-1, )
        point_2 = point_2.reshape(-1, )
        x_1 = pre_handle(point_1)
        x_2 = pre_handle(point_2)

        min_ = min(min(x_1), min(x_2))
        max_ = max(max(x_1), max(x_2))

        # p = fenxiang(x_1, 10, max_, min_, del_zero)
        # q = fenxiang(x_2, 10, max_, min_, del_zero)

        # # max0 = max(np.max(x_1), np.max(x_2))
        # # min0 = min(np.min(x_1), np.min(x_2))
        # # bins = np.linspace(min0 - 1e-4, max0 - 1e-4, num=num_bins)
        # # PDF1 = pd.cut(x_1, bins).value_counts() / len(x_1)
        # # PDF2 = pd.cut(x_2, bins).value_counts() / len(x_2)
        # # p = PDF1.values
        # # q = PDF2.values
        # M = (p + q) / 2
        # js = 0.5 * entropy(p, M) + 0.5 * entropy(q, M)

        bins = [n / 10 for n in range(0, 11, 1)]  # 箱子边界，0-1之内
        hist_1, bin_edges_1 = np.histogram(x_1, bins)
        hist_2, bin_edges_2 = np.histogram(x_2, bins)
        js = scipy.spatial.distance.jensenshannon(hist_1 + 1, hist_2 + 1)



    except Exception as e:
        print(e)
    return js


def JS_divergence_1(point_1, point_2):
    point_1 = (point_1 + 1e-10)  # /  (point_1.sum() + 1e-7)
    point_2 = (point_2 + 1e-10)  # / (point_2.sum() + 1e-7)
    # print(point_1.sum(), point_2.sum())
    # print(point_1)

    # M = (point_1 + point_2) / 2
    # distance = 0.5 * scipy.stats.entropy(point_1, M) + 0.5 * scipy.stats.entropy(point_2, M)

    n = np.shape(point_2)[0]
    a = np.arange(n)
    distance = wasserstein_distance(a, a, point_1, point_2)
    return distance


# 计算W距离
def W_divergence(point_1, point_2, del_zero=False):
    """
    计算W推土机距离的函数
    :param del_zero:
    :param point_1: 分别是两个相同长度概率数组
    :param point_2:
    :return: 返回W距离
    """

    x_1 = pre_handle(point_1)
    x_2 = pre_handle(point_2)

    p = fenxiang(x_1, 100, del_zero)
    q = fenxiang(x_2, 100, del_zero)
    w = wasserstein_distance(p, q)

    return w


# 计算序列的js散度，就是求每个序列与平均值js然后再平均
def kl_Bin_cal(sequence, Normalize=True, del_zero=False):
    '''
    首先计算n个序列的平均值，然后计算每个序列与平均值的js散度，然后将结果进行平均，作为返回值
    '''
    n = np.shape(sequence)[0]
    sequence = sequence.reshape(n, -1)

    if Normalize:
        sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))

    # 首先计算平均值，
    shape = np.shape(sequence)[1:]
    sum_ = np.zeros(shape=shape)

    for i in range(n):
        sum_ += sequence[i]
    average = sum_ / n

    kl_seq = np.zeros(n, )

    for i in range(n):
        kl_seq[i] = JS_divergence_1(average, sequence[i], del_zero=del_zero)

    # Bin_number = 100
    # q = fenxiang(average, Bin_number)
    #
    # #     print(q)
    #
    # kl_seq = np.zeros((n,))
    # for i in range(n):
    #     p = fenxiang(sequence[i], Bin_number)  # 返回分箱之后的概率
    #
    #     # for j in range(Bin_number - 1):
    #     #     kl_seq[i] += p[j] * np.log(p[j] / q[j])
    #     kl_seq[i] = JS_divergence_1(p, q, del_zero)
    #     if i % 20 == 0:
    #         print(kl_seq[i])
    # #         print(kl_seq[i])

    return kl_seq


# 计算数据的psnr
def psnr_cal(x, y):
    """
    计算两个向量或者矩阵的psnr
    :param x: 向量
    :param y:
    :return: psnr
    """
    PIXEL_MAX = 1
    #     x = x / PIXEL_MAX
    #     y = y /PIXEL_MAX
    #     PIXEL_MAX = 1
    ## mse = np.mean( (x/1 - y/1) ** 2 )
    mse = np.mean((x / 255. - y / 255.) ** 2)
    print(mse)
    if mse < 1.0e-10:
        return 100

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def cal_psnr(data, Standard=False, Normalize=False):
    """
    计算多维数据的psnr
    :param data: 数据
    :param Standard: 是否对数据进行标准化
    :param Normalize: 是否对数据进行归一化
    :return:
    """

    # 定义两开关：standard 为标准化，Normalize为归一化
    if Standard:
        data = (data - np.mean(data)) / (np.std(data))

    if Normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    data = data.reshape((1000, -1))

    shape = np.shape(data)[1]

    sum_ = np.zeros((shape,))
    for i in range(1000):
        sum_ = sum_ + data[i]
    layer_average = sum_ / 1000
    psnr_layer = np.zeros((1000,))
    for i in range(1000):
        # 可选，采用官方实现的psnr或者自己实现的psnr计算法
        # psnr_layer[i] = psnr_cal(layer_average, data[i])
        psnr_layer[i] = psnr(layer_average, data[i])
    return psnr_layer.mean()


# 计算数据的SSIM
def cal_ssim(sequence1, sequence2, shape):
    # sequence 是指输入向量，比如，第一层输出为(64, 27, 27),，shape=(64, 27, 27)

    # 然后计算L， C, S
    # 先计算出两个分布的均值和方差
    miu_1 = np.mean(sequence1)
    miu_2 = np.mean(sequence2)
    sigma_1 = np.sqrt(np.var(sequence1))
    sigma_2 = np.sqrt(np.var(sequence2))
    cov_12 = np.cov(sequence1, sequence2)

    K1 = 0.01
    K2 = 0.03
    L = 255

    C1 = np.square(K1 * L)
    C2 = np.square(K2 * L)
    C3 = C2 / 2

    L_XY = (2 * miu_1 * miu_2 + C1) / (np.square(miu_1) + np.square(miu_2) + C1)
    C_XY = (2 * sigma_1 * sigma_2 + C2) / (np.square(sigma_1) + np.square(sigma_2) + C2)
    S_XY = (cov_12[0][1] + C3) / (sigma_1 * sigma_2 + C3)
    return L_XY * C_XY * S_XY


# 计算序列的SSIM
def cal_seq_ssim(sequence, layer, Standard=False, Normalize=False):
    '''
    计算一个序列的ssim
    计算步骤，将每个维度的数据与所有维度的平均值做ssim，然后取平均值，得出结果。
    :param sequence: 序列，
    :param layer: 层数
    :param Standard: 标准化
    :param Normalize: 归一化
    :return: 返回序列的ssim
    '''
    if (layer <= 4):
        win_size = 5
    else:
        win_size = 3

    n = np.shape(sequence)[0]
    #     sequence = sequence.reshape((n, sequence.shape[2],-1))
    sequence = sequence.reshape((n, -1))

    shape = np.shape(sequence)[1:]

    #     print(shape)

    # 定义两开关：standard 为标准化，Normalize为归一化

    if (Standard):
        sequence = (sequence - np.mean(sequence)) / (np.var(sequence))

    if (Normalize):
        sequence = (sequence - np.min(sequence)) / (np.max(sequence) - np.min(sequence))

    # 首先计算平均值，
    sum_ = np.zeros(shape=shape, dtype=np.float32)
    for i in range(n):
        sum_ += sequence[i]
    average = sum_ / n

    # 将序列中每个值与均值做ssim比较，得出ssim向量，最后求一个均值，
    ssim_seq = np.zeros((n))
    for i in range(n):
        #         ssim_seq[i] = cal_ssim(average, sequence[i], shape=shape)
        #         print(type(average[0])), print(type(sequence[i][0]))

        ssim_seq[i] = ssim(X=average, Y=sequence[i], win_size=win_size)
    #         if (i<10):
    #             print(ssim_seq[i])
    return ssim_seq.mean()


# 关于模型的一些方法，用来获取模型中间输出或者准确度
class AverageMeter(object):
    '''
        获取平均网络精度
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), tratime=1):
    '''
    获取模型准确度，一般获取前1个和前5个的准确度。
    :param tratime:
    :param output: 输出结果
    :param target: 目标
    :param topk: 前几个准确度
    :return: 返回准确度
    '''
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data_loader(root, batch_size=16, workers=1, mode="val", pin_memory=False):
    '''
    使用数据集生成dataloader
    :param mode:
    :param root: 数据集的文件夹
    :param batch_size:
    :param workers:
    :param pin_memory:
    :return: 返回dataloader
    '''
    valdir = os.path.join(root, mode)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return val_loader


def save_checkpoint(state, is_best, filename='alex_checkpoint.pth'):
    '''
    保存模型的权重
    :param state: 状态，
    :param is_best:  是否是最优权重，也就是当前结果是不是最好的
    :param filename: 输出的文件名
    :return: 无返回值
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'alex_model_best_1.pth')


def adjust_learning_rate(optimizer, epoch, init_lr):
    """
    适应性学习率
    :param optimizer: 选择的优化器
    :param epoch:
    :param init_lr: 初始学习率
    :return: 无返回值
    """
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def imagenet_class_index_dic():
    with open('imagenet_class_index.txt', 'r') as f:
        index_list = f.readlines()

    index_dic = {}
    for i in range(len(index_list)):
        index_list[i] = index_list[i].strip('\n')
        index_split = index_list[i].split(' ')
        index_dic[index_split[0]] = index_split[1]
    return index_dic


def save_output(model, val_loader):
    model.eval()
    result = np.zeros((1, 1000))
    with torch.no_grad():
        for i, (input, _) in enumerate(val_loader):
            if i % 100 == 0:
                print(i)
            input = torch.autograd.Variable(input)
            output = model(input)
            result = np.concatenate((result, output.detach().cpu().numpy()))
    np.save("/data/mid_result/result.npy", result[1:])


# 验证js，通过node来对节点进行删除，将node转化为0、1向量，直接采用与中间结果相乘即可，
def validate_js(model, criterion, print_freq, node, tresh, thread_num, filename, class_name, is_cuda=False):
    # batch_time = AverageMeter()
    node = torch.from_numpy(node)

    model.eval()

    im_dict = imagenet_class_index_dic()
    this_ta = int(im_dict[class_name])

    this_num = 0
    all_num = 0

    for i in range(5):
        mid_result = np.load("/data/mid_result/mid_result_" + str(i) + ".npy")
        mid_result = torch.tensor(mid_result, dtype=torch.float32)

        for k in range(mid_result.size()[0]):
            mid_result[k] = mid_result[k].mul(node)
        # print(mid_result.size())
        # print(model.avgpool(mid_result).view(10000, 2048).size())
        output = model.fc(model.avgpool(mid_result).view(10000, 2048))
        this_tar = np.array([this_ta for i in range(mid_result.size()[0])])

        # print(output.size())

        target = np.zeros((200, 50))
        for j in range(200):
            target[j] = np.array([j + 200 * i] * 50)
        target = np.reshape(target, (-1,))

        # print(target)

        targ = np.argmax(output.detach().cpu().numpy(), axis=1)
        # print(this_tar.shape)
        # print(targ.shape)
        # print(target.shape)
        all_num += sum(targ == this_tar)
        this_num += sum((targ == target) & (targ == this_tar))

    print(all_num, this_num)

    # end = time.time()
    # for i, (input_, target) in enumerate(val_loader):
    #     if i % 10 == 0:
    #         print(thread_num, i)
    #     with torch.no_grad():
    #         # output = model(input_)
    #         if is_cuda:
    #             model = model.cuda()
    #             node = node.cuda()
    #             input_ = input_.cuda()
    #
    #         try:
    #             # compute output
    #             #             output = model(input)
    #             mid=mid_result[i]
    #             # print(mid.dtype, node.dtype)
    #             # for k in range(mid.size()[0]):
    #             #     mid[k] = mid[k] * node
    #             mid = mid.mul(node)
    #         except Exception as e:
    #             print(e)
    #         # print("mid_shape: ", (mid.size()))
    #         if model_name == "res50":
    #             mid = model.avgpool(mid)
    #             mid = mid.view(mid.size()[0], -1)
    #             output = model.fc(mid)
    #         else:
    #             mid = mid.view(mid.size()[0], -1)
    #             output = model.classifier(mid)
    #
    #         # loss = criterion(output, target)
    #         # measure accuracy and record loss
    #
    #         # print(output)
    #         # print(target)
    #         this_tar = np.array([this_ta for i in range(np.shape(input_)[0])])
    #         targ = np.argmax(output.detach().cpu().numpy(), axis=1)

    # if is_cuda:
    #     target = target.cuda()

    # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    # # losses.update(loss.item(), input_.size(0))
    # top1.update(prec1[0], input_.size(0))
    # top5.update(prec5[0], input_.size(0))
    # # measure elapsed time
    # batch_time.update(time.time() - end)
    # end = time.time()
    # if i % print_freq == 0:
    #     print(str(thread_num), 'Test: [{0}/{1}]\t'
    #                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    #                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    #         i, len(val_loader), batch_time=batch_time,
    #         top1=top1, top5=top5))

    acc = 0.999 + (2 * this_num - all_num) / 50000
    recall = this_num / 50
    prec = this_num / all_num
    fpr = (all_num - this_num) / 49950

    with open("/home/python/Image/" + experience_id + "/" + filename + ".txt",
              "a+") as f:
        f.write(
            'save ' + str(
                thread_num) + ", percent: " + str(
                node.mean()) + ", all_num: " + str(
                all_num) + ", this_num: " + str(
                this_num) + ", prec: " + str(
                prec) + ' :   * acc {prec:.3f} recall {recall:.3f} \n'.format(
                prec=acc,
                recall=recall))
    print(thread_num, "over!!")
    return recall, prec, fpr
    # return top1.avg, top5.avg


# 直接验证
def validate(val_loader, model, criterion, print_freq, filename, is_cuda=False):
    '''
    使用验证集对当前模型进行评估
    :param val_loader: 使用验证集生成的dataloader
    :param model: 模型
    :param criterion: 评价标准
    :param print_freq: 打印频率
    :return: 最终的准确率
    '''
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    # a = [
    #     [0, 120, 143, 166, 189, 210, 233, 256, 279, 300, 323, 346, 369, 391, 413, 436, 459, 481, 503, 526, 549, 571,
    #      594, 616, 639, 661, 684, 706, 729, 751, 774, 797, 819, 841, 864, 887, 909, 931, 954, 977],
    #     [1, 121, 144, 167, 19, 211, 234, 257, 28, 301, 324, 347, 37, 392, 414, 437, 46, 482, 504, 527, 55, 572, 595,
    #      617, 64, 662, 685, 707, 73, 752, 775, 798, 82, 842, 865, 888, 91, 932, 955, 978],
    #     [10, 122, 145, 168, 190, 212, 235, 258, 280, 302, 325, 348, 370, 393, 415, 438, 460, 483, 505, 528, 550, 573,
    #      596, 618, 640, 663, 686, 708, 730, 753, 776, 799, 820, 843, 866, 889, 910, 933, 956, 979],
    #     [100, 123, 146, 169, 191, 213, 236, 259, 281, 303, 326, 349, 371, 394, 416, 439, 461, 484, 506, 529, 551, 574,
    #      597, 619, 641, 664, 687, 709, 731, 754, 777, 8, 821, 844, 867, 89, 911, 934, 957, 98],
    #     [101, 124, 147, 17, 192, 214, 237, 26, 282, 304, 327, 35, 372, 395, 417, 44, 462, 485, 507, 53, 552, 575, 598,
    #      62, 642, 665, 688, 71, 732, 755, 778, 80, 822, 845, 868, 890, 912, 935, 958, 980],
    #     [102, 125, 148, 170, 193, 215, 238, 260, 283, 305, 328, 350, 373, 396, 418, 440, 463, 486, 508, 530, 553, 576,
    #      599, 620, 643, 666, 689, 710, 733, 756, 779, 800, 823, 846, 869, 891, 913, 936, 959, 981],
    #     [103, 126, 149, 171, 194, 216, 239, 261, 284, 306, 329, 351, 374, 397, 419, 441, 464, 487, 509, 531, 554, 577,
    #      6, 621, 644, 667, 69, 711, 734, 757, 78, 801, 824, 847, 87, 892, 914, 937, 96, 982],
    #     [104, 127, 15, 172, 195, 217, 24, 262, 285, 307, 33, 352, 375, 398, 42, 442, 465, 488, 51, 532, 555, 578, 60,
    #      622, 645, 668, 690, 712, 735, 758, 780, 802, 825, 848, 870, 893, 915, 938, 960, 983],
    #     [105, 128, 150, 173, 196, 218, 240, 263, 286, 308, 330, 353, 376, 399, 420, 443, 466, 489, 510, 533, 556, 579,
    #      600, 623, 646, 669, 691, 713, 736, 759, 781, 803, 826, 849, 871, 894, 916, 939, 961, 984],
    #     [106, 129, 151, 174, 197, 219, 241, 264, 287, 309, 331, 354, 377, 4, 421, 444, 467, 49, 511, 534, 557, 58, 601,
    #      624, 647, 67, 692, 714, 737, 76, 782, 804, 827, 85, 872, 895, 917, 94, 962, 985],
    #     [107, 13, 152, 175, 198, 22, 242, 265, 288, 31, 332, 355, 378, 40, 422, 445, 468, 490, 512, 535, 558, 580, 602,
    #      625, 648, 670, 693, 715, 738, 760, 783, 805, 828, 850, 873, 896, 918, 940, 963, 986],
    #     [108, 130, 153, 176, 199, 220, 243, 266, 289, 310, 333, 356, 379, 400, 423, 446, 469, 491, 513, 536, 559, 581,
    #      603, 626, 649, 671, 694, 716, 739, 761, 784, 806, 829, 851, 874, 897, 919, 941, 964, 987],
    #     [109, 131, 154, 177, 2, 221, 244, 267, 29, 311, 334, 357, 38, 401, 424, 447, 47, 492, 514, 537, 56, 582, 604,
    #      627, 65, 672, 695, 717, 74, 762, 785, 807, 83, 852, 875, 898, 92, 942, 965, 988],
    #     [11, 132, 155, 178, 20, 222, 245, 268, 290, 312, 335, 358, 380, 402, 425, 448, 470, 493, 515, 538, 560, 583,
    #      605, 628, 650, 673, 696, 718, 740, 763, 786, 808, 830, 853, 876, 899, 920, 943, 966, 989],
    #     [110, 133, 156, 179, 200, 223, 246, 269, 291, 313, 336, 359, 381, 403, 426, 449, 471, 494, 516, 539, 561, 584,
    #      606, 629, 651, 674, 697, 719, 741, 764, 787, 809, 831, 854, 877, 9, 921, 944, 967, 99],
    #     [111, 134, 157, 18, 201, 224, 247, 27, 292, 314, 337, 36, 382, 404, 427, 45, 472, 495, 517, 54, 562, 585, 607,
    #      63, 652, 675, 698, 72, 742, 765, 788, 81, 832, 855, 878, 90, 922, 945, 968, 990],
    #     [112, 135, 158, 180, 202, 225, 248, 270, 293, 315, 338, 360, 383, 405, 428, 450, 473, 496, 518, 540, 563, 586,
    #      608, 630, 653, 676, 699, 720, 743, 766, 789, 810, 833, 856, 879, 900, 923, 946, 969, 991],
    #     [113, 136, 159, 181, 203, 226, 249, 271, 294, 316, 339, 361, 384, 406, 429, 451, 474, 497, 519, 541, 564, 587,
    #      609, 631, 654, 677, 7, 721, 744, 767, 79, 811, 834, 857, 88, 901, 924, 947, 97, 992],
    #     [114, 137, 16, 182, 204, 227, 25, 272, 295, 317, 34, 362, 385, 407, 43, 452, 475, 498, 52, 542, 565, 588, 61,
    #      632, 655, 678, 70, 722, 745, 768, 790, 812, 835, 858, 880, 902, 925, 948, 970, 993],
    #     [115, 138, 160, 183, 205, 228, 250, 273, 296, 318, 340, 363, 386, 408, 430, 453, 476, 499, 520, 543, 566, 589,
    #      610, 633, 656, 679, 700, 723, 746, 769, 791, 813, 836, 859, 881, 903, 926, 949, 971, 994],
    #     [116, 139, 161, 184, 206, 229, 251, 274, 297, 319, 341, 364, 387, 409, 431, 454, 477, 5, 521, 544, 567, 59,
    #      611, 634, 657, 68, 701, 724, 747, 77, 792, 814, 837, 86, 882, 904, 927, 95, 972, 995],
    #     [117, 14, 162, 185, 207, 23, 252, 275, 298, 32, 342, 365, 388, 41, 432, 455, 478, 50, 522, 545, 568, 590, 612,
    #      635, 658, 680, 702, 725, 748, 770, 793, 815, 838, 860, 883, 905, 928, 950, 973, 996],
    #     [118, 140, 163, 186, 208, 230, 253, 276, 299, 320, 343, 366, 389, 410, 433, 456, 479, 500, 523, 546, 569, 591,
    #      613, 636, 659, 681, 703, 726, 749, 771, 794, 816, 839, 861, 884, 906, 929, 951, 974, 997],
    #     [119, 141, 164, 187, 209, 231, 254, 277, 3, 321, 344, 367, 39, 411, 434, 457, 48, 501, 524, 547, 57, 592, 614,
    #      637, 66, 682, 704, 727, 75, 772, 795, 817, 84, 862, 885, 907, 93, 952, 975, 998],
    #     [12, 142, 165, 188, 21, 232, 255, 278, 30, 322, 345, 368, 390, 412, 435, 458, 480, 502, 525, 548, 570, 593,
    #      615, 638, 660, 683, 705, 728, 750, 773, 796, 818, 840, 863, 886, 908, 930, 953, 976, 999]]

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # bianhuan = np.array(a).T.reshape(-1, )

        # for s in range(np.shape(target)[0]):
        # target[s] = (torch.tensor(bianhuan[target[s]]))

        if (is_cuda):
            target = target.cuda()
            input = input.cuda()
            model = model.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    with open("/home/python/Image/" + experience_id + "/" + filename + ".txt",
              "a+") as f:
        f.write('save 1, percent:   * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} \n'.format(
            top1=top1,
            top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate_class(model, criterion, print_freq, class_num, scr_dir="/home/python/Image/val/",
                   val_dir="/home/Image/valdate_class/val/", is_cuda=False):
    '''
    验证单个类别的准确率
    :param val_loader:
    :param model:
    :param criterion:
    :param print_freq:
    :param class_num: 表示类别的序号
    :param is_cuda:
    :return: 最终的准确率
    '''

    shutil.rmtree(val_dir)
    shutil.copytree(scr_dir + str(class_num),
                    val_dir + str(class_num))

    val_loader = data_loader(val_dir + "/../")

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, _) in enumerate(val_loader):
        target = torch.ones(np.shape(input)[0], dtype=torch.int64) * class_num
        if (is_cuda):
            target = target.cuda()
            input = input.cuda()
            model = model.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


# 直接获取第n层的中间输出
def get_n_mid(data_loader, model, layer, is_cuda=False):
    '''
    获取模型第n层的输出，此时模型直接表示为alexnet，结果直接用list
    :param data_loader:
    :param model:
    :param layer:
    :param cuda:
    :return:
    '''
    model.eval()
    geshi_1 = np.array(
        ((1, 3, 224, 224), (1, 64, 27, 27), (1, 192, 13, 13), (1, 384, 13, 13), (1, 256, 13, 13), (1, 256, 6, 6)))
    geshi_2 = np.array(((1, 4096), (1, 4096), (1, 1000)))

    if layer <= 5:
        layer_result = torch.zeros(geshi_1[layer].tolist())
    else:
        layer_result = torch.zeros(geshi_2[layer - 6].tolist())

    if (is_cuda):
        model = model.cuda()
        layer_result = layer_result.cuda()

    with torch.no_grad():
        for i, (input, _) in enumerate(data_loader):

            model_f = model.features
            # model_c = model.classifier

            input_var = torch.autograd.Variable(input)

            if is_cuda:
                input_var = input_var.cuda()

            if layer == 0:
                layer_result = torch.cat((layer_result, input_var), 0)
                continue

            cov_re_pool_1 = model_f[2](model_f[1](model_f[0](input_var)))
            if layer == 1:
                layer_result = torch.cat((layer_result, cov_re_pool_1), 0)
                continue

            cov_re_pool_2 = model_f[5](model_f[4](model_f[3](cov_re_pool_1)))
            if layer == 2:
                layer_result = torch.cat((layer_result, cov_re_pool_2), 0)
                continue

            cov_re_pool_3 = model_f[7](model_f[6](cov_re_pool_2))
            if layer == 3:
                layer_result = torch.cat((layer_result, cov_re_pool_3), 0)
                continue

            cov_re_pool_4 = model_f[9](model_f[8](cov_re_pool_3))
            if layer == 4:
                layer_result = torch.cat((layer_result, cov_re_pool_4), 0)
                continue

            cov_re_pool_5 = model_f[12](model_f[11](model_f[10](cov_re_pool_4)))
            # 添加avgpool层
            cov_re_pool_5 = model.avgpool(cov_re_pool_5)
            if layer == 5:
                layer_result = torch.cat((layer_result, cov_re_pool_5), 0)
                continue

            cov_re_pool_5 = cov_re_pool_5.view((-1, 6 * 6 * 256))

            linear_relu_1 = model.classifier[2](model.classifier[1](model.classifier[0](cov_re_pool_5)))
            if layer == 6:
                layer_result = torch.cat((layer_result, linear_relu_1), 0)
                continue

            linear_relu_2 = model.classifier[5](model.classifier[4](model.classifier[3](linear_relu_1)))
            if (layer == 7):
                layer_result = torch.cat((layer_result, linear_relu_2), 0)
                continue

            linear_relu_3 = model.classifier[6](linear_relu_2)
            if (layer == 8):
                layer_result = torch.cat((layer_result, linear_relu_3), 0)

    if layer <= 5:
        layer_result = layer_result[1:, :, :, :]
    else:
        layer_result = layer_result[1:, :]

    return layer_result.cpu().detach().numpy()


def get_midresult(data_loader, model, augnum, shape, batch_size, is_cuda=False):
    result = np.zeros(np.concatenate(([augnum], shape)))
    model = model.eval()
    if is_cuda:
        model = model.cuda()
    for i, (input_, _) in enumerate(data_loader):
        if is_cuda:
            input_ = input_.cuda()
        if model_name != "res50":
            data = model.features(input_).cpu().detach().numpy()
        else:
            data = (model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(
                model.bn1(model.conv1(input_))
            ))))))).cpu().detach().numpy()
        for j in range(np.shape(input_)[0]):
            result[i * batch_size + j] = data[j]
    return result


# 从x到y层的过程
def from_layer_x_to_y(layer_x, x, model):
    # 将x层的输出输入到x+1层得到x+1层的输出
    model = model.cpu()

    model_f = model.features
    model_c = model.classifier

    if x == 0:
        layer_y = model_f[2](model_f[1](model_f[0](torch.Tensor(layer_x))))
    elif x == 1:
        layer_y = model_f[5](model_f[4](model_f[3](torch.Tensor(layer_x))))
    elif x == 2:
        layer_y = (model_f[7](model_f[6](torch.Tensor(layer_x))))
    elif x == 3:
        layer_y = (model_f[9](model_f[8](torch.Tensor(layer_x))))
    elif x == 4:
        layer_y = model_f[12](model_f[11](model_f[10](torch.Tensor(layer_x))))
    elif x == 5:
        n = np.shape(layer_x)[0]
        layer_x = layer_x.reshape(n, -1)
        layer_y = (model_c[2](model_c[1](torch.Tensor(layer_x))))
    elif x == 6:
        layer_y = model_c[5](model_c[4](model_c[3](torch.Tensor(layer_x))))
    else:
        layer_y = (model_c[6](torch.Tensor(layer_x)))

    return layer_y


# 从上一层直接经过标准化之后输入到下一层
def before_to_after(layer_result, x, model):
    '''
    将单层输出输入进行bn之后输出到下一层，并返回下一层的结果
    :param layer_result: 单层输出
    :param x: 层数
    :param model: 模型
    :return: 模型最终输出和经过bn后输出到下一层之后的结果
    '''

    # 将第0层结果进行标准化
    #     layer_result = nn.BatchNorm2d(torch.from_array(layer_0_result))
    # 按照通道进行标准化，第二个维度为通道
    if (np.ndim(layer_result) == 4):
        for i in range(np.shape(layer_result)[1]):
            layer_result[:, i, :, :] = (layer_result[:, i, :, :] - np.mean(layer_result[:, i, :, :])) / np.std(
                layer_result[:, i, :, :])
    else:
        layer_result = (layer_result - np.mean(layer_result)) / np.std(layer_result)

    layer_next = from_layer_x_to_y(layer_result, x, model)

    result_trans_x = from_layer_x_to_y(layer_x=layer_result, x=x, model=model)

    for i in range(x + 1, 8):
        result_trans_x = from_layer_x_to_y(result_trans_x, i, model)

    return [result_trans_x, layer_next]

    # np.save(trans_dir + "result_trans_" + str(x) + ".npy", result_trans_x.detach().cpu().numpy())
    #
    # np.save(trans_dir + "layer_" + str(x + 1) + ".npy", layer_next.detach().cpu().numpy())


# 检查数据的偏序关系
# - 考虑一下步骤
#     1. 首先得到三个类别的数据 (1, 100, 200)
#     2. 将其输入网络后得出中间层输出，例如输出4096层为ndarray为[3, 4096]
#     3. 然后考虑单个节点的偏序关系，查看基层之间时候改变，并与最终层比较，因为最终分类层的偏序关系才是正确的。


# 构建数据集文件夹
# train_dir = "/home/python/Image/tiny_image/tiny/train/"


# 构建训练数据集
def make_train_dir(train_dir, class_num, image_num):
    sample_dir = sample(os.listdir(train_dir), class_num)
    others = []
    one = []
    for i in range(class_num):
        if (i == 0):
            img = sample(os.listdir(train_dir + sample_dir[i] + '/images'), image_num)
            for j in range(image_num):
                img[j] = train_dir + sample_dir[i] + '/images/' + img[j]
            one.extend(img)
        else:
            img = sample(os.listdir(train_dir + sample_dir[i] + '/images'), image_num)
            for j in range(image_num):
                img[j] = train_dir + sample_dir[i] + '/images/' + img[j]
            others.append(img)  # others.extend(img) 直接将很多类合并在一起，而这里我们不合并，之后再合并
    one_imglist = []
    others_imglist = []

    for img in one:
        one_imglist.append(cv2.imread(img))

    for link in others:
        img_list = []
        for img in link:
            img_list.append(cv2.imread(img))
        others_imglist.append(img_list)
    return [one_imglist, others_imglist]


# 构建训练数据集
def make_dir(train_dir, filename, class_num, image_num, save_dir):
    """
    直接构建class_num类的数据，平均每个类选取image_num个图片
    :param save_dir:
    :param train_dir: 训练文件夹
    :param class_num: 文件数量
    :param image_num: 图片数量
    :return:
    """
    # sample_dir = sample(os.listdir(train_dir), class_num)
    # global file
    data = []

    # print(sample_dir)

    # for _, _, files in os.walk(train_dir + "/n02088364/"):
    #     file = files

    for i in range(class_num):
        img = sample(os.listdir(train_dir + filename + "/" + filename), image_num)
        # img = os.listdir(train_dir + filename)
        print(img)
        for j in range(image_num):
            img[j] = train_dir + filename + "/" + filename + "/" + img[j]
            # img[j] = train_dir + sample_dir[i] + '/' + img[j]
        # print(img)

        data.append(img)

    data_imglist = []

    for cls in data:
        data_img = []
        for img in cls:
            # print(cv2.imread(img))
            data_img.append(cv2.imread(img))
        data_imglist.append(data_img)

    # print(np.shape(data_imglist))
    # print(data_imglist)
    print("构建数据完成！！")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(np.shape(data_imglist)[0]):
        for j in range(np.shape(data_imglist)[1]):
            # np.save("/home/python/Image/result/data_imglist_"+str(i)+".npy", data_imglist[i])
            np.save(save_dir + "/data_imglist_" + str(j) + ".npy", data_imglist[i][j])

    print("写入data_imglist完毕！")

    # return data_imglist


# 测试两个js散度的区别，按照js的大小关系进行区别。
def test_dif(js_old, js_new):
    # np.reshape(js_old, [-1, ])
    # np.reshape(js_new, [-1, ])
    # count = 0.0
    # for i in range(np.shape(js_old)[0]):
    #     # if float(js_old[i])==0:
    #     #     continue
    #     for j in range(np.shape(js_new)[0]):
    #         # if (float(js_old[j])==0):
    #         #     continue
    #         if js_new[i]-js_new[j]>=0.01 and (float(js_old[i]) - float(js_old[j]))>0.01 or (js_new[i] - js_new[j] < -0.01 and float(js_old[i]) - float(js_old[j]) < -0.01):
    #             print(js_new[i], js_new[j])
    #             print(js_old[i], js_old[j])
    #             count+=1
    # return count/(np.shape(js_new)[0]*(np.shape(js_old)[0])/2)

    old_sort = np.argsort(js_old)
    new_sort = np.argsort(js_new)
    return np.sqrt(mean_squared_error(new_sort, old_sort))  # 当随机选择一个验证集计算第五层结果是，两个差别的RMSE=407


# 
def js_criterion():
    parse = argparse.ArgumentParser()
    parse.add_argument("--learning_rate", type=float, default=0.01, help="initial learning rate")
    parse.add_argument("--test_dir", default="/home/python/Image/data_test/", help="initial test dir")
    parse.add_argument("--train_dir", default="/home/python/Image/tiny_image/tiny/train/",
                       help="initial train dir")
    parse.add_argument("--class_num", default=10, help="test class num")
    parse.add_argument("--image_num", default=50, help="test images num")
    parse.add_argument("--aug_num", default=1000, help="test augment num")
    parse.add_argument("--val_dir", default="/home/python/Image", help="validate dir")
    parse.add_argument("--print_freq", default=100, help="print frequency")

    args, unparsed = parse.parse_known_args()
    data_dir = args.test_dir
    train_dir = args.train_dir
    class_num = args.class_num
    image_num = args.image_num
    aug_num = args.aug_num
    print_freq = args.print_freq
    val_dir = args.val_dir

    # 首先删除之前建立的数据目录
    # if os.listdir(data_dir):
    #     shutil.rmtree(data_dir)
    #     os.mkdir(data_dir)

    print("开始添加数据干扰！")
    one_list, others_list = make_train_dir(train_dir, class_num=class_num, image_num=image_num)
    print("干扰添加完成！")
    one = one_list
    others = others_list
    # 得出结果为onelist shape: (50, 1000, 64, 64, 3)
    # otherslist shape : (9, 50, 64, 64, 3)
    # 首先将其数据输出为图片，然后制作dataloader
    # 首先需要设置目录
    if not Path(data_dir):
        os.makedirs(path=data_dir)
    # 输出one和others数据
    # 目录结构为 -datadir  -one/others  -i  -val -class -images
    # 首先输出one

    # 进行10词交换对比，保留的是需要删除点的坐标。如果是第五层则维度为[256, 6, 6]
    del_point = []

    print("开始计算需要删除的点！")
    for t in range(10):
        print("第", t + 1, "次计算！")
        # 每次都要对数据做一次恢复
        one_list = one
        others_list = others

        # 并且删除之前建立的数据目录
        if os.listdir(data_dir):
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)

        if t != 0:
            # 进行交换
            one_list, others_list[t - 1] = others_list[t - 1], one_list

        one_list = generate_coinT(one_list, aug_num=aug_num)
        for i in range(np.shape(one_list)[0]):
            os.makedirs(data_dir + '/one/' + str(i) + '/val/class/')
            for j in range(np.shape(one_list)[1]):
                cv2.imwrite(data_dir + '/one/' + str(i) + '/val/class/aug_' + str(j) + '.jpg', one_list[i][j])
        # 然后输出others
        os.makedirs(data_dir + '/others/val/class/')
        k = 0
        for i in range(np.shape(others_list)[0]):
            for j in range(np.shape(others_list)[1]):
                cv2.imwrite(data_dir + '/others/val/class/' + str(k) + '.jpg', others_list[i][j])
                k += 1

        print('文件结构梳理完成')

        # 构建data_loader
        # one_data_loader
        one_data_loader = []
        for i in range(image_num):
            one_data_loader.append(data_loader(data_dir + "/one/" + str(i)))
        others_data_loader = data_loader(data_dir + "/others/")

        print("构建dataloader完成 ----- ")

        model = torchvision.models.alexnet(pretrained=True)
        one_mid5_result = []
        for i in range(image_num):
            one_mid5_result.append(get_n_mid(one_data_loader[i], model=model, layer=5, is_cuda=False))
        others_mid5_result = get_n_mid(others_data_loader, model=model, layer=5, is_cuda=False)

        print("中间输出获取完成------ ")

        js = np.zeros([50, 256, 6, 6])
        for i in range(50):
            for j in range(256):
                for k in range(6):
                    for r in range(6):
                        js[i][j][k][r] = JS_divergence_1(one_mid5_result[i][:, j, k, r], others_mid5_result[:, j, k, r],
                                                         del_zero=True)

        print("JS 计算完毕")

        # 根据js方差来进行筛选

        jstd_tresh = np.percentile(js.std(axis=0), 75)
        del_p = np.zeros([256, 6, 6])
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    if (js[:, i, j, k].std() > jstd_tresh):
                        del_p[i][j][k] = -1

        del_point.append(del_p)

        # val_loader = data_loader(val_dir)
        # criterion = torch.nn.CrossEntropyLoss().cuda()
        # validate_js(val_loader, criterion=criterion, js=js, model=model, print_freq=100,
        #             percent=75, is_cuda=True)

        # print("模型评估完成，可与原始模型对比")
    return del_point


def test_js_zero():
    # 首先得到一些中间输出
    for i in range(4):
        loader = data_loader("/home/python/Image/test_stand/one/" + str(i + 2) + "/")
        data = get_n_mid(data_loader=loader, model=torchvision.models.alexnet(pretrained=True), layer=5, is_cuda=False)
        print(np.shape(data))

        js_old = kl_Bin_cal(data)
        js_new = kl_Bin_cal(data, del_zero=True)

        print(test_dif(js_old, js_new))


from random import randint


# 使用Mypool重写，让子线程也可以创建新的子线程
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def sleepawhile(t):
    print("Sleeping %i seconds..." % t)
    time.sleep(t)
    return t


# 方法重写过了，弃用
def cal_mid(data_dir="/home/python/Image/mid_result/", aug_num=1000):
    """
    这里是想得到每个类别进行相同转换之后的中间输出，并且不经过转化之后的中间输出，
    :param:  输入为数据文件名称，将其设计为文件夹，并且输入网络进行输出。shape: [class_num, image_num]
    :return: 结果定义为两个ndarray:
                    经过干扰之后的输出: aug_res: [class_num, image_num, aug_num, 256, 6, 6]
                    未经干扰之后的输出：ori_res: [class_num, image_num, 256, 6, 6]
    """
    # 构建数据存储变量
    class_num = 10  # np.shape(data_imglist)[0]
    image_num = 50  # np.shape(data_imglist)[1]
    # ori_res = np.zeros((class_num, image_num, 256, 6, 6))
    #
    # model = torchvision.models.alexnet(pretrained=True)
    #
    # # # 设计文件目录
    # shutil.rmtree(data_dir+"/aug/")
    #
    # # 将对应分类的图片输出到对应文件夹, 当文件存在时，下面代码可以不需要
    # # for i in range(class_num):
    # #     os.makedirs(data_dir + str(i) + "/val/class/")
    # # for i in range(class_num):
    # #     for j in range(image_num):
    # #         cv2.imwrite(data_dir + str(i) + '/val/class/' + str(j) + '.jpg', data_imglist[i][j])
    #
    # # print("原始数据输出至文件夹完毕！")
    #
    #
    # # 下面进行干扰数据生成，并得到中间结果
    # # 直接读取数据，然后干扰之后再输出到文件， 当文件存在时，下面代码可以注释掉
    # for i in range(class_num):
    #     for j in range(image_num):
    #         os.makedirs(data_dir + "/aug/" + str(i) + "/" + str(j) + "/val/class/")
    #
    # # for i in range(class_num): # （0-9）
    # #     print("干扰数据写入类序：" , i)
    # #     aug_img = generate_coinT(data_imglist[i], aug_num=aug_num)
    #
    #     # for j in range(image_num): # （0-49）
    #     #     for k in range(aug_num):
    #     #         cv2.imwrite(data_dir + '/aug/' + str(i) + '/' + str(j) + '/val/class/'+str(k)+'.jpg', aug_img[j][k])
    #
    # #
    # # for i in range(10):
    # #     args = [aug_num, i]
    # #     generate_coinT(args)
    #
    #
    #
    # print("多线程实现添加干扰！！")
    # #
    # # # 多线程实现
    # pool = Pool(processes=10)
    # # args = zip([aug_num for i in range(class_num)], [thread_num for thread_num in range(class_num)])
    # # pool.map(generate_coinT, args)
    #
    # for index in range(10):
    #     pool.apply_async(func=generate_coinT, args=((aug_num, index), ))
    #
    # pool.close()
    # pool.join()
    #
    #
    # print("干扰数据写入文件完毕！")

    # 制作原始数据的data_loader, 数目为num_class
    # loaders = []
    # for i in range(class_num):
    #     loaders.append(data_loader(data_dir + str(i)))
    #
    # # 然后根据loaders经过网络得到中间输出。
    # for i in range(class_num):
    #     ori_res[i] = get_n_mid(loaders[i], model=model, layer=5)
    #
    # np.save("/home/python/Image/result/origin_res.npy", ori_res)

    # print("原始网络结果获取完毕")
    # 多线程实现
    pool_1 = MyPool(10)
    for index in range(10):
        pool_1.apply_async(func=augMidres, args=(index,))
    # pool_1.map_async(augMidres, range(10))
    pool_1.close()
    pool_1.join()


def mult_process(function, aug_num):
    print("多线程实现添加干扰！！")

    # 多线程实现
    pool = Pool(processes=10)
    # args = zip([data_imglist[i] for i in range(class_num)], [aug_num for i in range(class_num)], [thread_num for thread_num in range(class_num)])
    # pool.map(generate_coinT, args)

    for i in range(10):
        pool.apply_async(func=function, args=((aug_num, i),))

    pool.close()
    pool.join()


# 获取添加干扰后的数据的中间结果
def augMidres(index, model, augnum, shape, data_dir="/home/python/Image/exper_v1/marker_C/", is_cuda=False):
    # 构建loaders

    global aug_res, layer_result
    try:

        print("augmid计算：", index)
        batch_size = 100
        data_load = data_loader(data_dir + str(index), batch_size=batch_size, pin_memory=False)

        # layer_result = torch.zeros((1, 256, 6, 6))
        # with torch.no_grad():
        #     for i, (input_, _) in enumerate(data_load):
        #         if (i % 3):
        #             print(index, i)
        #         input_var = torch.autograd.Variable(input_)
        #         output = model.features(input_var)
        #         # print(np.shape(output))
        #         layer_result = torch.cat((layer_result, output), 0)
        #     # print(np.shape(layer_result))
        # layer_result = (layer_result[1:, :, :, :].cpu().detach().numpy())

        layer_result = get_midresult(data_loader=data_load, model=model, batch_size=batch_size, augnum=augnum,
                                     shape=shape, is_cuda=is_cuda)

    except Exception as e:
        print(e)
    # assert np.shape(layer_result) == (1000, 256, 6, 6)
    if not os.path.exists(data_dir + "/result/"):
        os.makedirs(data_dir + "/result/")

    np.save(data_dir + "/result/mid_result_" + str(index) + ".npy", layer_result)

    print(index, "线程干扰数据结果计算并存储完毕")


# 获取干扰数据的中间输出与原始数据中间输出的js散度
def test_stand(z):
    """
    本函数作用在于将每个aug_data的数据与原始数据进行对比，并且每个类得出image_num个js散度
    在输入函数之前要将数据进行合并，取出单个类干扰数据和其他数据综合进行对比。
    :param z:
    :param thread_num:
    :param aug_data: 干扰之后数据的中间输出 shape: [image_num, aug_num, 256, 6, 6]
    :param origin_data: 原始数据中间输出 shape: [class_num*image_num, 256, 6, 6]
    :return: js散度结果， shape: [256, 6, 6, image_num]
    """
    thread_num = z[0]
    origin_data = z[1]

    print("这是第", thread_num, "条进程！！")
    aug_data = np.load("/home/python/Image/result/aug_res_" + str(thread_num) + ".npy")

    image_num = np.shape(aug_data)[0]

    js = np.zeros((256, 6, 6, image_num))

    print("开始计算kl散度: 线程" + str(thread_num))
    for i in range(256):

        if i % 32 == 0:
            print("进程", thread_num, "进度为： ", i / 256)

        for j in range(6):
            for k in range(6):
                for s in range(image_num):
                    js[i][j][k][s] = JS_divergence_1(aug_data[s, :, i, j, k], origin_data[i, j, k])
    np.save("/home/python/Image/result/js_" + str(thread_num) + ".npy", js)


def duoxiancheng():
    # print("开始读取文件")
    # aug_data = np.load("/home/python/Image/aug_data.npy")
    # ori_data = np.load("/home/python/Image/ori_data.npy")
    # print("读取文件完毕")

    ori_data = np.load("/home/python/Image/result/origin_res.npy")

    class_num = np.shape(ori_data)[0]
    image_num = np.shape(ori_data)[1]

    origin_data = np.zeros((class_num * image_num, 256, 6, 6))
    for i in range(class_num):
        for j in range(image_num):
            origin_data[i * image_num + j] = ori_data[i][j]

    print("处理数据完成！")

    pool = Pool(processes=10)
    args = zip([i for i in range(10)], [origin_data for i in range(10)])
    pool.map(test_stand, args)
    pool.close()
    pool.join()
    print("结束")


def getNode(thread_num):
    js_i = np.load("/home/python/Image/result/js_" + str(thread_num) + ".npy")
    # print(np.shape(js_i))
    js_tresh = np.percentile(np.std(js_i, axis=3), 50)
    # print(js_tresh)
    node = np.zeros((256, 6, 6))
    for i in range(256):
        for j in range(6):
            for k in range(6):
                if np.std(js_i[i, j, k, :]) > js_tresh:
                    node[i, j, k] = 1

    return node


def get_node():
    nodes = []
    for i in range(10):
        nodes.append(getNode(i))
    np.save("/home/python/Image/result/nodes.npy", nodes)


# # 测试多线程
# def thread(url):
#     r = requests.get(url, headers=None, stream=True, timeout=30)
#     # print(r.status_code, r.headers)
#     headers = {}
#     all_thread = 1
#     # 获取视频大小
#     file_size = int(r.headers['content-length'])
#     # 如果获取到文件大小，创建一个和需要下载文件一样大小的文件
#     if file_size:
#         fp = open('2012train.tar', 'wb')
#         fp.truncate(file_size)
#         print('文件大小：' + str(int(file_size / 1024 / 1024)) + "MB")
#         fp.close()
#     # 每个线程每次下载大小为5M
#     size = 5242880
#
#     # 当前文件大小需大于5M
#     if file_size > size:
#         # 获取总线程数
#         all_thread = int(file_size / size)
#         # 设最大线程数为10，如总线程数大于10
#         # 线程数为10
#         if all_thread > 10:
#             all_thread = 10
#     part = file_size // all_thread
#     threads = []
#     starttime = datetime.datetime.now().replace(microsecond=0)
#     for i in range(all_thread):
#         # 获取每个线程开始时的文件位置
#         start = part * i
#         # 获取每个文件结束位置
#         if i == all_thread - 1:
#             end = file_size
#         else:
#             end = start + part
#         if i > 0:
#             start += 1
#         headers = headers.copy()
#         headers['Range'] = "bytes=%s-%s" % (start, end)
#         t = threading.Thread(target=Handler, name='th-' + str(i),
#                              kwargs={'start': start, 'end': end, 'url': url, 'filename': '2012train.tar',
#                                      'headers': headers})
#         t.setDaemon(True)
#         threads.append(t)
#     # 线程开始
#     for t in threads:
#         time.sleep(0.2)
#         t.start()
#     # 等待所有线程结束
#     for t in threads:
#         t.join()
#     endtime = datetime.datetime.now().replace(microsecond=0)
#     print('用时：%s' % (endtime - starttime))
#
# def Handler(start, end, url, filename, headers={}):
#     tt_name = threading.current_thread().getName()
#     print(tt_name + ' is begin')
#     r = requests.get(url, headers=headers, stream=True)
#     total_size = end - start
#     downsize = 0
#     startTime = time.time()
#     with open(filename, 'r+b') as fp:
#         fp.seek(start)
#         var = fp.tell()
#         for chunk in r.iter_content(204800):
#             if chunk:
#                 fp.write(chunk)
#                 downsize += len(chunk)
#                 line = tt_name + '-downloading %d KB/s - %.2f MB， 共 %.2f MB'
#                 line = line % (
#                     downsize / 1024 / (time.time() - startTime), downsize / 1024 / 1024,
#                     total_size / 1024 / 1024)
#                 print(line, end='\r')


# 通过node修改网络并进行网络性能评估
def critBynodes(val_loader, model, node, tresh, is_cuda=False):
    '''
    使用验证集对当前模型进行评估
    :param val_loader: 使用验证集生成的dataloader
    :param model: 模型
    :param criterion: 评价标准
    :param print_freq: 打印频率
    :return: 最终的准确率
    '''

    model.eval()
    model_f = model.features
    model_c = model.classifier

    output_ = []
    for i, (input, _) in enumerate(val_loader):

        if (is_cuda):
            input = input.cuda()
            model = model.cuda()
            model_c = model_c.cuda()
            model_f = model_f.cuda()

        with torch.no_grad():
            # compute output
            mid = model_f(input)
            for k in range(mid.size()[0]):
                for r in range(256):
                    for s in range(6):
                        for t in range(6):
                            if node[:, r, s, t].std() >= tresh:
                                mid[k][r][s][t] = 0

            mid = mid.reshape(mid.size()[0], -1)
            output = model_c(mid)
            output_.extend(output)

    target = np.argmax(output, axis=1)
    print(target)
    from scipy import stats
    zhongshu = (stats.mode(target)[0][0])
    count = 0
    for i in range(50):
        if target[i] == zhongshu:
            count += 1

    # print(count/50)

    return count / 50


# 直接进行
def get_align():
    # for i in range(100):
    #     os.makedirs("/home/python/Image/class_100/"+str(i)+"/val/class/")

    # make_dir(train_dir="/home/python/Image/train/train/", class_num=100, image_num=1)
    # os.makedirs("/home/python/Image/class_100/val/class")

    pool = MyPool(100)
    for thread_num in range(100):
        pool.apply_async(generate_coinT, args=((1000, thread_num),))
    pool.close()
    pool.join()


# 批处理
def save_Snode(thread_num_1, thread_num, std_percent):
    try:
        print("进程序号：", thread_num_1, "---", thread_num)
        js = np.load("/home/python/Image/result/js_" + str(thread_num) + ".npy", allow_pickle=True)
        js_std = np.std(js, axis=3)
        # print(np.shape(js_std))
        js_tresh = np.percentile(js_std, std_percent)
        node = np.ones((256, 6, 6))
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    if js_std[i][j][k] > js_tresh:
                        node[i][j][k] = 0
        np.save("/home/python/Image/result/" + str(thread_num_1) + "_node_" + str(thread_num) + ".npy", node)
        print(np.shape(js))
    except Exception as e:
        print(e)


def save_node(thread_num_1, std_percent):
    #
    try:
        # print("外部线程：", thread_num_1)
        # pool = MyPool(10)
        # for thread_num in range(10):
        #     pool.apply_async(func=save_Snode, args=(thread_num_1, thread_num, std_percent,))
        # pool.close()
        # pool.join()

        node = np.zeros((256, 6, 6))
        for i in range(10):
            node += (np.load("/home/python/Image/result/" + str(thread_num_1) + "_node_" + str(i) + ".npy",
                             allow_pickle=True))

        np.save("/home/python/Image/result/criterion_" + str(std_percent) + ".npy", node)
    except Exception as e:
        print(e)


def vali_js(thread_num):
    try:
        print(thread_num, "线程启动！！！")

        model = torchvision.models.alexnet(pretrained=True)
        criterion = torch.nn.CrossEntropyLoss()
        print_freq = 100
        train_dataloader = data_loader(root="/home/python/Image/train/train", mode="val", pin_memory=True)
        node = np.load("/home/python/Image/result/criterion_" + str(30 + 5 * (thread_num % 10)) + ".npy")
        validate_js(val_loader=train_dataloader, model=model, criterion=criterion,
                    print_freq=print_freq, node=node, tresh=int(thread_num / 10 + 1), thread_num=thread_num,
                    is_cuda=True)

        print(thread_num, "线程写入结果完毕!")
    except Exception as e:
        print(e)


def class_100_mid(index):
    # 构建loaders

    global aug_res, layer_result
    try:

        print("计算类序：", index)
        data_dir = "/home/python/Image/class_100/"
        model = torchvision.models.alexnet(pretrained=True)
        loader = (data_loader(data_dir + str(index)))

        layer_result = torch.zeros((1, 256, 6, 6))
        with torch.no_grad():
            for i, (input_, _) in enumerate(loader):
                if (i == 0):
                    print(index, "----开始计算----")
                input_var = torch.autograd.Variable(input_)
                output = model.features(input_var)
                # print(np.shape(output))
                layer_result = torch.cat((layer_result, output), 0)
            # print(np.shape(layer_result))

        np.save("/home/python/Image/class_100/result/class_" + str(index) + "_res.npy",
                layer_result[1:, :, :, :])

        print(index, "数据结果计算并存储完毕")

    except Exception as e:
        print(e)


# 获取100类的中间结果
def get_100_mid():
    pool = MyPool(100)
    for thread_num in range(100):
        pool.apply_async(class_100_mid, args=(thread_num,))
    pool.close()
    pool.join()


def doub_son(thread_num, data, mean):
    print("线程启动：", thread_num)

    try:
        js = np.zeros((1000, 6, 6))
        # print(np.shape(data))
        # print(np.shape(mean))
        for i in range(6):
            print("线程", thread_num, "计算：", i)
            for j in range(6):
                for k in range(np.shape(data)[0]):
                    # print(np.shape(data[k, :, i, j]), np.shape(mean[:, i, j]))
                    js[k][i][j] = JS_divergence_1(data[k, :, i, j], mean[:, i, j])
        np.save("/home/python/Image/diff_class_2/result/gauss_ind/js_" + str(thread_num) + ".npy", js)
        print("线程完成：", thread_num)
    except Exception as e:
        print(e)


def cal_ttest(thread_num):
    print("ttest 线程：", thread_num)

    reason = [i + 1 for i in range(1000)]
    js = np.load("/home/python/Image/class_100/result/gauss_ind/js_" + str(thread_num) + ".npy")
    # test_ind_ = np.zeros((6, 6, 2))
    coint_result = np.zeros((6, 6))

    for j in range(6):
        print("ttest线程", thread_num, "计算：", j)
        for k in range(6):
            # test_ind_[j][k] = scipy.stats.ttest_ind(reason, js[:, j, k], equal_var=False)

            # a_price_diff = np.diff(reason)
            # b_price_diff = np.diff(js[:, j, k])
            temp = coint(reason, js[:, j, k])
            print(temp)
            if temp[1] < 0.05:
                coint_result[j][k] = 1

    np.save("/home/python/Image/class_100/result/gauss_ind/Gauss_blur_coint_" + str(thread_num) + ".npy",
            coint_result)

    print("ttest 计算完毕：", thread_num)


# 100类数据获取的结果应该是100个文件，每个维度为(1000, 256, 6, 6)
def doubleSamT():
    # 构建长度为1000的js序列， 然后对这个序列和1-1000的序列做双样本t检验

    # result = np.zeros((100, 1000, 256, 6, 6))
    # for i in range(100):
    #     result[i] = np.load("/home/python/Image/class_100/result/class_" + str(i) + "_res.npy")
    # result = np.swapaxes(result, 0, 1)
    # # print(np.shape(result))
    # mean = np.mean(result, axis=0)
    # print(np.shape(mean))
    #
    # pool = MyPool(256)
    #
    # for i in range(256):
    #     pool.apply_async(doub_son, args=(i, result[:, :, i, :, :], mean[:, i, :, :],))
    # pool.close()
    # pool.join()

    # 对每个点都查看是否相关，使用双样本t检验, 或者协整

    pool_1 = MyPool(1)
    for i in range(1):
        pool_1.apply_async(cal_ttest, args=(i,))
    pool_1.close()
    pool_1.join()


# 将结果进行保存
def save_result():
    with open("/home/python/Image/marker_C/result/one_class_result_1.txt", "a+") as savef:
        for i in range(9):
            with open("/home/python/Image/marker_C/result/one_class_result_" + str(i * 5 + 30) + ".txt",
                      "r") as sourcef:
                lines = sourcef.readlines()
                for line in lines:
                    savef.write(line)
                savef.write("\n")
                print(i)


def get_covari():
    ttest_ind = np.zeros((256, 6, 6, 2))
    for i in range(256):
        ttest_ind[i] = np.load(
            "/home/python/Image/class_100/result/gauss_ind/Gauss_blur_ttest_ind" + str(i) + ".npy")
    result = np.zeros((256, 6, 6))
    for i in range(256):
        for j in range(6):
            for k in range(6):
                if ttest_ind[i][j][k][1] > 0.05:
                    result[i][j][k] = 1
    np.save("/home/python/Image/class_100/result/p_value_cri.npy", result)


def get_final():
    final = np.zeros((256, 6, 6))
    for i in range(256):
        final[i] = np.load("/home/python/Image/class_100/result/gauss_ind/Gauss_blur_coint_" + str(i) + ".npy")
    for i in range(256):
        np.savetxt("/home/python/Image/class_100/result/final_coint" + str(i) + ".txt", final[i])
    save_result()


# 计算序列的js， 与之前的方法功能相同。不过改写为适用于多线程
def cal_2_js(result, thread_num, image_num, aug_num, filename, shape):
    print(np.shape(result))
    try:
        print(thread_num, "start: cal_2_js")

        standard = np.load("/data/gauss_test_res/n02123045/marker_C/result/mid_result_0.npy")

        js = np.zeros(np.concatenate(([shape[1]], [shape[2]], [image_num - 1])))
        for j in range(shape[1]):
            for k in range(shape[2]):
                # mean = np.mean(result[:, :, j, k], axis=0)
                # mean = result[0, :, j, k]
                mean = standard[:, thread_num, j, k]
                for s in range(image_num - 1):
                    js[j][k][s] = JS_divergence_1(result[s + 1, :, j, k], mean)

        np.save(WORK_DIR + experience_id + filename + "/marker_C/result/js_" + str(thread_num) + "withcat0.npy", js)
    except Exception as e:
        print(e)


# 验证不同类之间差距区分导致最终性能的改变
def vali_2_js(thread_num):
    try:
        print(thread_num, "线程启动！！！")

        model = torchvision.models.alexnet(pretrained=True)
        criterion = torch.nn.CrossEntropyLoss()
        print_freq = 100
        train_dataloader = data_loader(root=WORK_DIR + "/tiny_image/tiny/", mode="train", pin_memory=True,
                                       batch_size=16)
        node = np.load(WORK_DIR + "/diff_class_2/result/node_" + str(30 + 5 * thread_num) + ".npy")
        validate_js(val_loader=train_dataloader, model=model, criterion=criterion,
                    print_freq=print_freq, node=node, tresh=1, thread_num=thread_num,
                    is_cuda=True)

        print(thread_num, "线程写入结果完毕!")
    except Exception as e:
        print(e)


def test_01():
    # make_dir(train_dir=WORK_DIR+"/train/train/val/", class_num=2, image_num=1)
    # pool = MyPool(2)
    # for i in range(2):
    #
    #     pool.apply_async(generate_coinT, args=(([1000, i]), ))
    #
    # pool.close()
    # pool.join()

    # result = np.zeros((2, 1000, 256, 6, 6))
    # for i in range(2):
    #     loader = data_loader(WORK_DIR+"/diff_class_2/"+str(i))
    #     result[i] = get_n_mid(data_loader=loader, model=torchvision.models.alexnet(pretrained=True), layer=5, is_cuda=True)
    # np.save(WORK_DIR+"/diff_class_2/result/mid_result.npy", result)
    #
    # result = np.load(WORK_DIR+"/diff_class_2/result/mid_result.npy")
    #
    # print(np.shape(result))
    # pool = MyPool(256)
    #
    # for i in range(256):
    #     print(i)
    #     pool.apply_async(cal_2_js, args=(result[:, :, i, :, :], i, ))
    # pool.close()
    # pool.join()

    # for i in range(256):
    #     print(i)
    #     for j in range(6):
    #         for k in range(6):
    #             js[i][j][k] = JS_divergence_1(result[0, :, i, j, k], result[1, :, i, j, k])
    # np.save(WORK_DIR+"/diff_class_2/result/js.npy")

    # js = np.zeros((256, 6, 6))
    # for i in range(256):
    #     js[i] = np.load(WORK_DIR+"/diff_class_2/result/js" + str(i) + ".npy")
    #
    # for num in range(10):
    #     node = np.ones((256, 6, 6))
    #     js_tresh = np.percentile(js, 30 + 5 * num)
    #     for i in range(256):
    #         for j in range(6):
    #             for k in range(6):
    #                 if js[i, j, k] < js_tresh:
    #                     node[i, j, k] = 0
    #     np.save(WORK_DIR+"/diff_class_2/result/node_" + str(30 + 5 * num) + ".npy", node)

    pool = MyPool(10)
    for thread_num in range(10):
        pool.apply_async(vali_2_js, args=(thread_num,))
    pool.close()
    pool.join()

    # for i in range(10):
    #     vali_2_js(i)


# 计算原始数据的中间结果
def cal_ori():
    model = torchvision.models.alexnet(pretrained=True)
    for i in range(200):
        progress(i / 200)
        if not os.path.exists(WORK_DIR + experience_id + "/marker_C/result/origin/" + str(i) + "/val/class"):
            os.makedirs(WORK_DIR + experience_id + "/marker_C/result/origin/" + str(i) + "/val/class")
        img = np.load(WORK_DIR + "exper_v1/marker_C/result/data_imglist_" + str(i) + ".npy")
        cv2.imwrite(WORK_DIR + "exper_v1/marker_C/result/origin/" + str(i) + "/val/class/origin.jpg", img)
        data_load = data_loader(WORK_DIR + "exper_v1/marker_C/result/origin/" + str(i))
        np.save(WORK_DIR + experience_id + "/marker_C/result/origin/origin_" + str(i) + ".npy",
                get_n_mid(model=model, data_loader=data_load, layer=5))


# 计算给定标签的数据的js散度，这个主要是计算序列js
def cal_js(thread_num, filename, compare, degree):
    # try:

    print("start: degree: ", degree, "thread_num:", thread_num, "compare:", compare)
    js = np.zeros(shape)
    count = 0
    data = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(thread_num) + "_1.npy")
    comp = np.load(WORK_DIR + experience_id + filename + "/marker_C/result/xulie/xulie_" + str(compare) + "_0.npy")

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                js[i, j, k] = JS_divergence_1(data[:, i, j, k], comp[:, i, j, k])
                if js[i, j, k] < 1e-10:
                    # print((data[:, i, j, k] - comp[:, i, j, k]).mean())
                    count += 1
    print(thread_num, "js中为0的个数：", count)
    np.save(
        WORK_DIR + experience_id + filename + "/marker_C/result/js_" + str(degree) + "/js_" + str(thread_num) + ".npy",
        js)

    print("end: degree: ", degree, "thread_num:", thread_num)
    # except Exception as e:
    #     print(e)


# 通过方差计算删除节点，也就是以稳定性作为标准，结果保留为包含percent的文件。
def multi_get_acc(thread_num, filename, js_std):
    print("multi_get_acc, 计算：", thread_num)
    percent = thread_num * 5 + 30
    js_tresh = np.percentile(js_std, percent)
    node = np.ones(shape, dtype=np.float32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # 这里是删除方差大的节点，可以理解就是对于同类图片来说，我们需要保留能提取稳定特征的节点，但是也可以试试删除方差小的节点
                if js_std[i][j][k] > js_tresh:
                    # if js_std[i][j][k] < js_tresh:
                    node[i, j, k] = 0

    np.save(WORK_DIR + experience_id + filename + "/marker_C/result/node_" + str(percent) + ".npy", node)
    # node = np.load(WORK_DIR + "/marker_C/result/node_" + str(percent) + ".npy")

    # 对js做标准化，然后将其中不在3倍标准差的点删除
    # try:
    #     js = (js - np.mean(js)) / np.std(js)
    #     # 范围： [μ+σ+(2σ/10)*i, μ+σ+(2σ/10)*i];
    #     miu = js.mean()
    #     sigma = js.std()
    #     range_0 = miu - (sigma + (2 * sigma / 10 * thread_num))
    #     range_1 = miu + (sigma + (2 * sigma / 10 * thread_num))
    #     print(thread_num, range_0, range_1)
    #     print(js.shape)
    #
    #     node = np.reshape(list(map(lambda x: 1 if (range_1 > x > range_0) else 0, js.reshape(-1, ))), (256, 6, 6))
    #
    #     np.save(WORK_DIR + "/marker_C/result/sigma_node_" + str(thread_num) + ".npy", node)
    #     print("计算node完毕：", thread_num)
    #
    # except Exception as e:
    #     print(e)
    return

    model = torchvision.models.alexnet(pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    print_freq = 100
    # 整个数据集进行验证
    train_loader = data_loader(root=WORK_DIR + "/train/train/", mode="val")
    # 只采用选用的那个类进行验证
    # train_loader = data_loader(root="/home/tmp/pycharm_project_490/result/0/")
    validate_js(val_loader=train_loader, model=model, criterion=criterion,
                print_freq=print_freq, node=node, tresh=1, thread_num=thread_num)


# 计算给定数据之间的js散度
def cal_js_1(thread_num, target_dir, output_dir):
    print("JS计算开始:", thread_num)

    try:
        target = np.load(target_dir)
        output = np.load(output_dir + str(thread_num) + ".npy")

        js = np.zeros((256, 6, 6))
        for i in range(256):
            if i % 64 == 0:
                print(thread_num, "进程：", i / 256)
            for j in range(6):
                for k in range(6):
                    js[i, j, k] = JS_divergence_1(target[:, i, j, k], output[:, i, j, k])
                    print(js[i, j, k])
                    # sleepawhile(4)
                    # print(js[i, j, k])
                    # if np.abs(js[i, j, k] - 0.0) < 1e-6:
                    #     print((output[:, i, j, k] == target[:, i, j, k]).mean())
                    #     print(output[:, i, j, k].mean())
        np.save(WORK_DIR + experience_id + "/marker_A/result/js_" + str(thread_num) + ".npy", js)
    except Exception as e:
        print(e)


def getSomeNode(thread_num, thread_num_1):
    try:
        print(thread_num, thread_num_1, 'start')

        js = np.load(WORK_DIR + experience_id + "/marker_A/result/js_" + str(thread_num) + ".npy")
        print(js.shape)

        # 标准化有必要吗。。。
        # 首先进行标准化
        # js = (js - js.mean()) / js.std()

        js_kur = np.zeros((256, 6, 6))
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    js_kur[i, j, k] = get_kurto(js[:, i, j, k])

        node = np.ones((256, 6, 6))
        js_tresh = np.percentile(js_kur, 2.5 + 2.5 * thread_num_1)

        # 计算峰度作为节点筛选标准

        for i in range(256):
            for j in range(6):
                for k in range(6):
                    if js_kur[i, j, k] <= js_tresh:
                        node[i, j, k] = 0
        np.save(WORK_DIR + experience_id + "/marker_A/result/node_" + str(thread_num) + "_" + str(
            thread_num_1 * 2.5 + 2.5) + ".npy",
                node)
        print(thread_num, thread_num_1, "end")
    except Exception as e:
        print(e)


def handleJsGetNode(thread_num):
    try:
        print(thread_num, "start")

        # 标准化之后选取一定比例的点作为js几乎无变化的点
        # pool = MyPool()
        # for i in range(5):
        #     pool.apply_async(getSomeNode, args=(thread_num, i,))
        # pool.close()
        # pool.join()
        js = np.zeros((20, 256, 6, 6))
        for i in range(20):
            js[i] = np.load(WORK_DIR + experience_id + "/marker_A/result/js_" + str(i) + ".npy")

        js_kur = np.zeros((256, 6, 6))
        for i in range(256):
            for j in range(6):
                for k in range(6):
                    # print(js[:, i, j, k])
                    js_kur[i, j, k] = stats.kurtosis(js[:, i, j, k])
                    # print(js_kur[i, j, k])
                    # sleepawhile(2)
        # np.save(WORK_DIR+experience_id+"/marker_A/result/kurto_"+str(thread_num)+".npy", js_kur)
        print(js_kur)

        for percent_level in range(5):
            node = np.ones((256, 6, 6))
            js_tresh = np.percentile(js_kur, 2.5 + 2.5 * percent_level)
            # 计算峰度作为节点筛选标准
            for i in range(256):
                for j in range(6):
                    for k in range(6):
                        if js_kur[i, j, k] <= js_tresh:
                            node[i, j, k] = 0
            np.save(WORK_DIR + experience_id + "/marker_A/result/node_" + str(thread_num) + "_" + str(
                percent_level * 2.5 + 2.5) + ".npy",
                    node)

        print(thread_num, "end")
    except Exception as e:
        print(e)


# 频谱分析
def fft_test(js, thread_num):
    '''
    计算频谱
    :param thread_num:
    :param js: shape=[6, 6]
    :return:
    '''

    print(thread_num, "start")

    try:
        node = np.zeros((6, 6))
        node_4 = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                fft_y = fft(js[:, i, j])
                N = 1000
                x = np.arange(N)  # 频率个数
                abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
                angle_y = np.angle(fft_y)  # 取复数的角度
                normalization_y = abs_y / N  # 归一化处理（双边频谱）
                half_x = x[range(int(N / 2))]  # 取一半区间
                normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
                # return normalization_half_y
                peaks, _ = find_peaks(normalization_half_y, prominence=normalization_half_y[1:].max() / 2)

                if np.abs(peaks - 19).min() < 2:
                    node[i][j] = 1
                if np.abs(peaks - 19).min() < 4:
                    node_4[i][j] = 1
        np.save(WORK_DIR + experience_id + "/marker_D/result/fft_node_6_6_" + str(thread_num) + ".npy", node)
        np.save(WORK_DIR + experience_id + "/marker_D/result/fft_node_4_6_6_" + str(thread_num) + ".npy", node_4)
    except Exception as e:
        print(e)
    print(thread_num, "end")


# 通过js序列进行频谱分析得出满足条件的点
def get_fftNode():
    js = np.zeros((1000, 256, 6, 6))
    for i in range(1000):
        progress(i / 1000)
        js[i] = np.load(WORK_DIR + experience_id + "/marker_C/result/js_0/js_" + str(i + 1) + ".npy")
    pool = MyPool()
    for i in range(256):
        pool.apply_async(fft_test, args=(js[:, i, :, :], i,))
    pool.close()
    pool.join()


# 这是手写DW检验，不过后面用的是stats中的sapi
def DW_Stat(js_order):  # 德宾-瓦特逊检验， Durbin-Watson Statistics
    fenzi = 0
    fenmu = 0
    for i in range(1000):
        fenzi += np.square(js_order[i + 1] - js_order[i])
        fenmu += np.square(js_order[i])
    fenmu += np.square(js_order[1000])
    return fenzi / fenmu


# # 通过js序列计算dw
# def dw_stat_6_6(js_order, thread_num, mode="dw"):
#     print("start:", thread_num)
#
#     try:
#         if mode == "dw":
#             node = np.zeros((6, 6))
#             for i in range(6):
#                 for j in range(6):
#                     temp = sm.stats.durbin_watson(js_order[:, i, j])
#                     if temp > 1.779 or temp < 1.758:
#                         node[i][j] = 1
#                     np.save(WORK_DIR + "/marker_D/result/dw_" + str(thread_num) + "_6_6.npy", node)
#         else:
#             node01 = np.zeros((6, 6))
#             node05 = np.zeros((6, 6))
#             node10 = np.zeros((6, 6))
#             for i in range(6):
#                 for j in range(6):
#                     temp = stattools.coint([(1 + np.sin(7 * k * np.pi / 180)) for k in np.arange(1000)],
#                                            js_order[:, i, j])
#                     if temp[0] < temp[2][0] and temp[1] < 0.01:
#                         node01[i][j] = 1
#                     if temp[0] < temp[2][1] and temp[1] < 0.05:
#                         node05[i][j] = 1
#                     if temp[0] < temp[2][2] and temp[1] < 0.10:
#                         node10[i][j] = 1
#                     np.save(WORK_DIR + "/marker_D/result/node_10_" + str(thread_num) + ".npy", node10)
#                     np.save(WORK_DIR + "/marker_D/result/node_01_" + str(thread_num) + ".npy", node01)
#                     np.save(WORK_DIR + "/marker_D/result/node_05_" + str(thread_num) + ".npy", node05)
#     except Exception as e:
#         print(e)
#
#     print("end:", thread_num)
#     # 直接根据dw结果计算节点保存与否
#

# # 计算格兰杰因果关系检验的p值
# def granger_test(js_order, thread_num):
#     try:
#         print("start:", thread_num)
#         lag = np.zeros((6, 6))
#         gt_6_6 = np.zeros((6, 6))
#         for j in range(6):
#             for k in range(6):
#                 gt = stattools.grangercausalitytests(np.vstack((js_order[:, k, j], [(1 + np.sin(7 * i * np.pi / 180))
#                                                                                     for i in np.arange(1000)])).T,
#                                                      maxlag=20, verbose=False)
#
#                 lag[j][k] = np.argmax([(gt[i + 1][0]['params_ftest'][0]) for i in range(20)]) + 1
#                 gt_6_6[j][k] = gt[lag[j][k]][0]['params_ftest'][1]
#
#         np.save(WORK_DIR + experience_id + "/marker_D/result/gt_" + str(thread_num) + "_6_6.npy", gt_6_6)
#         print("end:", thread_num)
#     except Exception as e:
#         print(e)
#

# 通过p值计算节点
def get_D_Node(p):
    gt_p = np.zeros((256, 6, 6))
    for i in range(256):
        progress(i / 256)
        gt_p[i] = np.load(WORK_DIR + "/marker_C/result/gt_" + str(i) + "_6_6.npy")
    # node_1 = np.reshape(list(map(lambda x: 1 if x < 0.01 else 0, gt_p.reshape((-1,)))), (256, 6, 6))
    # node_5 = np.reshape(list(map(lambda x: 1 if x < 0.05 else 0, gt_p.reshape((-1,)))), (256, 6, 6))
    # np.save(WORK_DIR+"/marker_D/result/node_0.01.npy", node_1)
    # np.save(WORK_DIR+"/marker_D/result/node_0.05.npy", node_5)

    node_10 = np.reshape(list(map(lambda x: 1 if x < p else 0, gt_p.reshape((-1,)))), (256, 6, 6))
    np.save(WORK_DIR + "/marker_D/result/node_" + str(p) + ".npy", node_10)


# 进度条
def progress(title, percent, width=50):
    if percent > 1:
        percent = 1
    show_str = (('[%%-%ds]' % width) % (int(percent * width) * '#'))
    print('\r%s %s %.2f%%' % (title, show_str, float(percent * 100)))


# 测试原始数据，后面直接在summary中改变loader的root参数即可实现此功能
def test_ori():
    pool = MyPool()
    index = ["", "_4"]
    for i in range(10):
        for j in range(2):
            node_C = np.load(
                WORK_DIR + experience_id + "/marker_C/result/node/node_" + str(30 + i * 5) + ".npy").astype(np.int8)
            node_D = np.load(WORK_DIR + experience_id + "/marker_D/result/fft_node" + index[j] + ".npy").astype(np.int8)
            node = np.bitwise_or(node_C, node_D)

            # node = np.ones((256, 6, 6))

            percent = np.mean(node)
            model = torchvision.models.alexnet(pretrained=True)
            criterion = torch.nn.CrossEntropyLoss()
            print_freq = 1000
            node = node.astype(np.float32)
            loader = data_loader(root=WORK_DIR, mode="val")
            # loader = data_loader(root=WORK_DIR+"/team/", mode="train",
            #                      pin_memory=True)
            pool.apply_async(validate_js, args=(
                loader, model, criterion, print_freq, node, 1, 30 + i * 5 + j, percent, "CorD", False,))

    pool.close()
    pool.join()


# 测试hx的数据
# def test_hx():
#     js_1_test = np.load(WORK_DIR + "/marker_C/result/js_1/cat_js.npy")
#     js_1_test = np.diff(js_1_test, axis=0)
#     print(js_1_test.shape)
#     ngt = np.zeros((43264,))
#     lag = np.zeros((43264,))
#     for i in range(43264):
#         progress(i / 43264)
#         gt = stattools.grangercausalitytests(np.vstack((js_1_test[:, i], [
#             (1.1 + 0.9 * math.sin(math.pi * i * 7.0 / 180.0) + float(i) / 500.0 + np.random.randint(-10, 10) / 50.0) for
#             i in np.arange(999)])).T, maxlag=20, verbose=False)
#         lag[i] = np.argmax([(gt[i + 1][0]['params_ftest'][0]) for i in range(20)]) + 1
#         ngt[i] = gt[lag[i]][0]['params_ftest'][1]
#     np.save(WORK_DIR + "/marker_D/result/gt_hx.npy", ngt)
#

def get_kurto(data):
    mean_ = data.mean()
    std_ = data.std()
    kurto = np.mean((data - mean_) ** 4) / pow(std_, 4)
    return kurto


def generate_data(image, img_size, centre, col, row, index, main_dir):
    if not os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/"):
        os.makedirs("WORK_DIR+experience_id+main_dir" + "/marker_C/" + str(index) + "/val/class/")

    try:
        for j in range(col):
            if j % 2 == 0:
                for i in range(row):
                    x_0 = i * centre[0] / col
                    x_1 = j * centre[1] / row
                    y_0 = i * (img_size[0] - centre[2]) / col + centre[2]
                    y_1 = j * (img_size[1] - centre[3]) / row + centre[3]
                    # print(x_0, y_0, x_1, y_1)
                    imag2 = image.crop((x_0, x_1, y_0, y_1))
                    imag2.save(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/" + str(
                        j * row + i) + ".jpg")
            else:
                for i in range(row - 1, -1, -1):
                    x_0 = i * centre[0] / col
                    x_1 = j * centre[1] / row
                    y_0 = i * (img_size[0] - centre[2]) / col + centre[2]
                    y_1 = j * (img_size[1] - centre[3]) / row + centre[3]
                    # print(x_0, y_0, x_1, y_1)
                    imag2 = image.crop((x_0, x_1, y_0, y_1))
                    imag2.save(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/" + str(
                        j * row + row - 1 - i) + ".jpg")
    except Exception as e:
        print(e)


def generate_scale(img, img_size, center, augnum, index, main_dir):
    # print(index)
    # if not os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/"):
    #     print("文件夹不存在")
    #     os.makedirs("WORK_DIR+experience_id+main_dir" + "/marker_C/" + str(index) + "/val/class/")

    try:
        deltax = img_size[0] / 10 / augnum
        deltay = img_size[1] / 10 / augnum
        for i in range(augnum):
            x_0 = center[0] - deltax * i
            x_1 = center[1] - deltay * i
            y_0 = center[2] + deltax * i
            y_1 = center[3] + deltay * i
            img_temp = img.crop((x_0, x_1, y_0, y_1))
            img_temp.save(
                WORK_DIR + experience_id + main_dir + "/marker_C/" + str(index) + "/val/class/" + str(i) + ".jpg")
    except Exception as e:
        print(e)


def generate(filename):
    # main_dir=random.sample(os.listdir("./train/train/val/"), 1)
    # print(filename)

    # pool = MyPool()
    # for i in range(1):
    # print(random_class[i])

    # if filename != "None" and random_class[i] != filename:
    #     continue
    # main_dir = random_class[i]
    main_dir = filename
    # print(main_dir)
    if os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/val/"):
        # print("目录存在")
        shutil.rmtree(WORK_DIR + experience_id + main_dir + "/marker_C/val/")
    # if not os.path.exists(WORK_DIR + experience_id + main_dir + "/marker_C/val/"):
    #     os.makedirs(WORK_DIR + experience_id + main_dir + "/marker_C/val/")

    shutil.copytree(WORK_DIR + experience_id + "/../val/" + main_dir,
                    WORK_DIR + experience_id + main_dir
                    + "/marker_C/val/" + main_dir)
    print("????start: ", main_dir)
    files = (random.sample(os.listdir(WORK_DIR + "/train/train/val/" + main_dir + "/"), 50))
    for index in range(50):
        if not os.path.exists(
                WORK_DIR + experience_id + main_dir + "/marker_C/" + "/" + str(index) + "/val/class/"):
            os.makedirs(WORK_DIR + experience_id + "/" + main_dir + "/marker_C/" + str(index) + "/val/class")
        file = files[index]
        # print("start: ", file)
        img = Image.open(WORK_DIR + "/train/train/val/" + main_dir + "/" + file)
        img_size = img.size
        center = np.array([img_size[0] / 5, img_size[1] / 5, img_size[0] * 4 / 5, img_size[1] * 4 / 5])
        # pool.apply_async(generate_data, args=(img, img_size, center, 10, 10, index, main_dir,))
        generate_data(img, img_size, center, 10, 10, index, main_dir)
        # pool.apply_async(generate_scale, args=(img, img_size, center, 50, index, main_dir,))
        # generate_scale(img, img_size, center, 50, index, main_dir)
        # print("end: ", file)

    print("end: ", filename)

    # pool.close()
    # pool.join()


'''
实验步骤如下：
    1、首先对于每个类，获取其特征的300个节点，也就是保存权重矩阵。然后20个类6000个特征节点，其中第一层的去噪保留位置。
    2、对于6000个特征节点，通过训练集训练6000*20的分类矩阵，
    3、对验证集计算其准确率，与原模型对比
'''


def relu(point):
    data = np.array(point)
    for i in range(np.shape(data)[0]):
        if data[i] < 0:
            data[i] = 0.00001
    return data


def cal_w(point_1, point_2):
    point_1 = (point_1 + 1e-10)  # /  (point_1.sum() + 1e-7)
    point_2 = (point_2 + 1e-10)  # / (point_2.sum() + 1e-7)
    # print(point_1.sum(), point_2.sum())
    # print(point_1)

    # M = (point_1 + point_2) / 2
    # distance = 0.5 * scipy.stats.entropy(point_1, M) + 0.5 * scipy.stats.entropy(point_2, M)
    point_1 = relu(point_1)
    point_2 = relu(point_2)

    n = np.shape(point_2)[0]
    a = np.arange(n)
    distance = wasserstein_distance(a, a, point_1, point_2)
    return distance


# def pool_cal_w(label, thread_num, mode):
def pool_cal_w(args):
    try:
        label, thread_num, mode = args

        print(label, "  start   ", mode)
        normal = np.load("/data/result/" + label + "_normal.npy")
        if normal.ndim != 2:
            normal = normal.reshape((100, -1))
        node = np.load("/data/gauss_test_res/" + label + "/w_node.npy")
        delete = []
        node = node.reshape(-1, )
        for i in range(np.shape(node)[0]):
            if node[i] == 0:
                delete.append(i)

        if mode == "in":

            if os.path.exists("/data/result/new_test/" + label + "/in_w_" + str(thread_num) + ".npy"):
                print("end: ", thread_num)
                return
            data = np.load(
                "/data/gauss_test_res/" + label + "/marker_C/result/mid_result_" + str(thread_num) + ".npy")

            data = data.reshape(100, -1)
            data = np.delete(data, delete, 1)
            normal = np.delete(normal, delete, 1)

            # data : [100, 80000]
            # standard: [100, 80000]
            print("in start: ", thread_num)
            n = np.shape(data)[1]
            w = np.zeros((n,))
            for i in range(n):
                w[i] = cal_w(data[:, i], normal[:, i])
            np.save("/data//result/new_test/" + label + "/in_w_" + str(thread_num) + ".npy",
                    w)
            print("end: ", thread_num)
        elif mode == 'out':
            # if thread_num == class_name.index(label):
            #     return
            if os.path.exists("/data//result/new_test/" + label + "/out_w_" + str(thread_num) + ".npy"):
                print("end: ", thread_num)
                return
            # data : [20, 100, 80000]
            # standard: [100, 80000]

            # disk = list()
            # disk.append(class_name.index(label))
            x = np.random.randint(1000)
            while x == class_name.index(label):
                x = np.random.randint(1000)
            data = np.zeros((20, 100, 2048, 7, 7))
            for i in range(20):
                # x = np.random.randint(1000)
                # while x in disk:
                #     x = np.random.randint(1000)
                # disk.append(x)
                data[i] = np.load("/data//gauss_test_res/" + str(class_name[x])
                                  # data[i] = np.load("/data//gauss_test_res/" + str(class_name[thread_num])
                                  + "/marker_C/result/mid_result_" + str(i) + ".npy")
            data = data.reshape((20, 100, -1))
            data = np.delete(data, delete, 2)
            normal = np.delete(normal, delete, 1)

            print("out start: ", thread_num)
            n = np.shape(data)[2]
            k = np.shape(data)[0]
            w = np.zeros((k, n))
            for i in (range(n)):
                # if i % 20000 == 0:
                #     print(thread_num, i)
                for j in range(k):
                    w[j][i] = cal_w(data[j, :, i], normal[:, i])
            np.save("/data//result/new_test/" + label + "/out_w_" + str(thread_num) + ".npy",
                    w)
            print("end: ", thread_num)
    except Exception as e:
        print(e)


def concept_construct(label):
    try:
        print(label, "    start!")
        if not os.path.exists("/data//result/" + str(label) + "_normal.npy"):
            normal = np.zeros((20, 100, 2048, 7, 7))
            for epo in (range(20)):
                normal[epo] = np.load(
                    "/data//gauss_test_res/" + label + "/marker_C/result/mid_result_" + str(epo) + ".npy")

            normal = np.mean(normal, axis=0).reshape((100, -1))

            np.save("/data//result/" + str(label) + "_normal.npy", normal)

        print(label, "normal 结束")
        # 均值
    except Exception as e:
        print("出错了！！！", e)


def cc(label):
    try:
        # print(n)
        if not os.path.exists("/data//result/new_test/" + label + "/"):
            os.makedirs("/data/result/new_test/" + label + "/")

        pool1 = MyPool()
        # for thread_num in range(20):
        #
        # params = []
        pool1.map(pool_cal_w, [(label, i, "in") for i in range(20)])
        # for thread_num in range(20):
        #     pool1.apply_async(pool_cal_w, args=(label, thread_num, "out"))
        #     params.append((label, thread_num, "out"))

        pool1.map(pool_cal_w, [(label, thread_num, "out") for thread_num in range(20)])
        pool1.close()
        pool1.join()
        global weight_num
        weight_num += 1
        print(label, " Wasserstein 距离计算完毕！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！",
              weight_num / 1000)

    except Exception as e:
        print(e)


def cw():
    pool = MyPool()
    pool.map(cal_weight, class_name)
    pool.close()
    pool.join()


def cal_weight(label):
    try:
        if os.path.exists("/data/result/index_" + label + ".npy"):
            return
        newNodeNum = 300
        index = []
        weight = []
        node = np.load("/data/gauss_test_res/" + label + "/w_node.npy", allow_pickle=True)
        n = int(node.sum())
        in_w = np.zeros((20, n))
        for i in tqdm(range(20)):
            in_w[i] = np.load("/data/result/new_test/" + label + "/in_w_" + str(i) + ".npy")

        node = node.reshape((-1,))
        # new version
        # out_w = np.zeros((999, 20, n))
        # index_ = 0
        # for i in tqdm(range(1000)):
        #     if i == class_name.index[label]:
        #         continue
        #     out_w[index_] = np.load("/data/result/new_test/" + label + "/out_w_" + str(i) + ".npy")
        #     index_ += 1

        # old version
        out_w = []
        for dir in list(os.walk("/data/result/new_test/" + label))[0][2]:
            out_w.append(np.load("/data/result/new_test/" + label + "/" + dir))
        out_w = np.mean(out_w, axis=0)

        ###########################################################################################
        #
        # print("cupy")
        # out_w = cp.asarray(out_w)
        # in_w = cp.asarray(in_w)
        # rewardall = cp.median(out_w, axis=0) / (cp.median(in_w, 0) + 0.00000001)
        # print('x')
        # all_rewa = (out_w / (in_w + 0.00000001)).T
        # candidate = cp.argsort(rewardall)[-300:]
        #
        # candi_reward = all_rewa[candidate]
        # candi_cova = cp.corrcoef(candi_reward)
        #
        # out_sample_w = out_w.T
        # mid_w_del = in_w.T
        #
        # print(" 数据处理完毕， 开始计算组合！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！  ")
        #
        # for node_index in range(newNodeNum):
        #     cova = candi_cova[node_index]
        #
        #     union = []
        #
        #     union.extend(sample(list(candidate[[i for i, j in enumerate(cova) if j < -0.1]]), 5))
        #     union.extend(sample(list(candidate[[i for i, j in enumerate(cova) if 0.3 > j > -0.1]]), 5))
        #     union.append(candidate[node_index])
        #     union.extend(sample(list(cp.argsort(rewardall)[:-300]), 4))
        #     sampleNum = 19
        #
        #     x = len(union)
        #
        #     union = cp.array([int(x) for x in union])
        #
        #     reward = cp.log(out_sample_w[union] / mid_w_del[union])
        #
        #     V = cp.cov(reward)
        #     Er = cp.mean(reward, 1).reshape((x, 1))
        #     e = cp.ones((x, 1))
        #
        #     a = cp.dot(cp.dot(Er.T, cp.linalg.pinv(V)), Er)
        #     b = cp.dot(cp.dot(Er.T, cp.linalg.pinv(V)), e)
        #     A = cp.dot(cp.dot(cp.hstack((Er, e)).T, cp.linalg.pinv(V)), cp.hstack((Er, e)))
        #
        #     miuP = a / b
        #     k = cp.hstack((Er, e))
        #     c = cp.dot(cp.linalg.pinv(V), k)
        #     d = cp.dot(c, cp.linalg.pinv(A))
        #     w_star = cp.dot(d, cp.vstack((miuP, [1])))
        #
        #     w = w_star / cp.sum(w_star)  # 归一化
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #     #     print(reward.shape)
        #     # v = cp.cov(reward)
        #     # e = cp.ones((x, 1))
        #     # #     print(type(reward))
        #     # #     print([[int(cp.mean(x)) for x in reward]])
        #     # Er = cp.array([[int(cp.mean(x)) for x in reward]])
        #     # inver = cp.linalg.inv(v)
        #     # b1 = cp.dot(Er, inver)
        #     # b = cp.dot(b1, e)
        #     # c1 = cp.dot(e.T, inver)
        #     # c = cp.dot(c1, e)
        #     # a1 = cp.dot(Er, inver)
        #     # a = cp.dot(a1, Er.T)
        #     # d = a[0][0] * c[0][0] - b[0][0] ** 2
        #     # N = 0
        #     # #     miu = (b[0][0] ** 2 + d - c[0][0] * N * b[0][0]) / (b[0][0] * c[0][0] - N * c[0][0] * c[0][0])
        #     # miu = a[0][0] / b[0][0]
        #     # w_temp = cp.concatenate((Er, e.T)).T
        #     # Aa = cp.dot(cp.dot(w_temp.T, inver), w_temp)
        #     # w1 = cp.dot(cp.dot(inver, w_temp), cp.linalg.inv(Aa))
        #     # #     print(np.array(w1), miu)
        #     # print(miu)
        #     # mid = cp.array([[miu], [1.0]])
        #     # w = cp.dot(w1, mid)
        #     # w = cp.maximum(w, 0)
        #     # w = w / cp.sum(w)  # 归一化

        ###########################################################################################
        rewardall = np.median(out_w, axis=0) / (np.median(in_w, 0) + 0.00000001)
        all_rewa = (out_w / (in_w + 0.00000001)).T
        candidate = np.argsort(rewardall)[-300:]
        candi_reward = all_rewa[candidate]
        candi_cova = np.corrcoef(candi_reward)

        out_sample_w = out_w.T
        mid_w_del = in_w.T

        # print(label, " 数据处理完毕， 开始计算组合！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！  ")

        for node_index in range(newNodeNum):
            cova = candi_cova[node_index]

            union = []

            union.extend(sample(list(candidate[[i for i, j in enumerate(cova) if j < 0.1]]), 5))
            union.extend(sample(list(candidate[[i for i, j in enumerate(cova) if 0.7 > j > 0.1]]), 5))
            union.append(candidate[node_index])
            union.extend(sample(list(np.argsort(rewardall)[:-300]), 4))
            sampleNum = 19

            x = len(union)

            # one_reward = np.zeros((x, sampleNum))
            # for i in range(x):
            #     for j in range(sampleNum):
            #         if mid_w_del[union[i]][j] == 0:
            #             one_reward[i][j] = 1
            #         else:
            #             one_reward[i][j] = np.log(
            #                 (out_sample_w[union[i]][j] + 0.000001) / (mid_w_del[union[i]][j] + 0.0000000001)) \
            #                                / (np.std(mid_w_del[union[i]]) + 0.000001)
            #
            # V = np.cov(one_reward)
            # Er = np.mean(one_reward, 1).reshape((x, 1))
            # e = np.ones((x, 1))
            #
            # a = np.dot(np.dot(Er.T, np.linalg.pinv(V)), Er)
            # b = np.dot(np.dot(Er.T, np.linalg.pinv(V)), e)
            # A = np.dot(np.dot(np.hstack((Er, e)).T, np.linalg.pinv(V)), np.hstack((Er, e)))
            #
            # miuP = a / b
            # k = np.hstack((Er, e))
            # c = np.dot(np.linalg.pinv(V), k)
            # d = np.dot(c, np.linalg.pinv(A))
            # w_star = np.dot(d, np.vstack((miuP, [1])))

            reward = np.log(out_sample_w[union] / mid_w_del[union])
            # print(reward.shape)
            v = np.cov(reward)
            e = np.ones((x, 1))
            Er = np.array([[np.mean(x) for x in reward]])
            inver = np.linalg.inv(v)
            b1 = np.dot(Er, inver)
            b = np.dot(b1, e)
            c1 = np.dot(e.T, inver)
            c = np.dot(c1, e)
            a1 = np.dot(Er, inver)
            a = np.dot(a1, Er.T)
            d = a[0][0] * c[0][0] - b[0][0] ** 2
            N = 0
            miu = (b[0][0] ** 2 + d - c[0][0] * N * b[0][0]) / (b[0][0] * c[0][0] - N * c[0][0] * c[0][0])
            # miu = a[0][0] / b[0][0]
            w_temp = np.concatenate((Er, e.T)).T
            Aa = np.dot(np.dot(w_temp.T, inver), w_temp)
            w1 = np.dot(np.dot(inver, w_temp), np.linalg.inv(Aa))
            w = np.dot(w1, np.array([[miu], [1.0]]))
            w = np.maximum(w, 0)
            w = w / np.sum(w)  # 归一化

            #######################################################################
            next_uni = []
            for uni in union:
                tes = 0
                for x in range(np.shape(node)[0]):
                    tes += node[x]
                    if tes == uni + 1:
                        next_uni.append(x)
                        break
            index.append(next_uni)
            weight.append(w)
            if node_index % 100 == 0:
                print(label, node_index / newNodeNum)

        np.save("/data/result/index_" + label + ".npy", index)
        np.save("/data/result/weight_" + label + ".npy", weight)

        print(label, "    end!")
    except Exception as e:
        print(e)


# 对于每个类别，计算权值
def getNormal():
    # print(class_name)
    pool = MyPool()
    for cls in class_name:
        pool.apply_async(concept_construct, args=(cls,))
    pool.close()
    pool.join()
    print("计算normal结束！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")


def cal_wasserstein():
    # params = []
    # for cls in class_name:
    #     pool.apply_async(cc, args=(cls,))

    for label in class_name:
        if not os.path.exists("/data/result/new_test/" + label + "/"):
            os.makedirs("/data/result/new_test/" + label + "/")

    pool = MyPool()

    clss = np.load("/data/result/clss.npy")

    # params = [(label, i, "in") for label in class_name for i in range(20)]
    # params.extend([(label, thread_num, "out") for label in class_name for thread_num in range(20)])
    params = [(label, thread_num, "out") for label in clss for thread_num in range(10)]

    pool.map(pool_cal_w, params)
    # for thread_num in range(20):
    #     pool1.apply_async(pool_cal_w, args=(label, thread_num, "out"))
    #     params.append((label, thread_num, "out"))

    # pool.map(cc, class_name)
    pool.close()
    pool.join()
    print(" get_20 weight   计算结束!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def get(dir, indexes, weights, class_name, state, label_num):
    try:
        data = np.zeros((30000, 1))
        label = np.zeros((1,))

        model = torchvision.models.resnext50_32x4d(pretrained=True)
        print(dir)
        batch_size = 128
        val_loader = data_loader("/data/data/" + state + "/" + dir, batch_size=batch_size)
        with torch.no_grad():
            for i, (input_, _) in enumerate(val_loader):
                # progress(title="batch进度：", percent=i/100)
                input_ = torch.autograd.Variable(input_)
                output = (model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(
                    model.conv1(input_))))))))).cpu().detach().numpy()
                n = np.shape(output)[0]
                output = output.reshape((n, -1))
                # 输出维度为[batch_size, 2048, 7, 7]

                # print(n)

                feature = np.zeros((30000, n))
                for x in range(30000):
                    for y in range(15):
                        feature[x] += output[:, int(indexes[x][y])] * weights[x][y]
                data = np.concatenate((data, feature), axis=1)
                label = np.concatenate((label, np.ones(n, ) * label_num), axis=0)

        data = data[:, 1:]
        label = label[1:]
        print("data: ", np.shape(data))
        np.save("/data/result/" + state + "_data_" + dir + ".npy", data)
        np.save("/data/result/" + state + "_label_" + dir + ".npy", label)
    except Exception as e:
        print(e)


def consData():
    indexes = []
    weights = []
    for clsn in class_name:
        indexes.extend(np.load("/data/result/index_" + clsn + ".npy", allow_pickle=True))
        weights.extend(np.load("/data/result/weight_" + clsn + ".npy", allow_pickle=True))

    indexes = np.array(indexes)
    weights = np.array(weights)

    # indexes = np.load("/data/class21_index.npy")
    # weights = np.load("/data/class21_w.npy")

    # print("index:")
    # print(indexes.shape, weights.shape)

    val_cls_label = {}
    pool = MyPool()
    for cls in class_name:
        # pool.apply_async(get, args=(dir, indexes, weights, class_name, "train"))
        lab_num = class_name.index(cls)
        val_cls_label[cls] = lab_num
        pool.apply_async(get, args=(cls, indexes, weights, class_name, "val", lab_num,))
    pool.close()
    pool.join()

    np.save("/data/result/val_cls_label.npy", val_cls_label)


def getFeature(cls, indexes, weights):
    try:

        # if os.path.exists("/data/result/feature_" + cls + ".npy"):
        #     print(cls, " end train features 结束！！！！！！！！！！！！！！！！！！！！！！!")
        #     return
        print(cls, ' start 获取训练 features 数据！！！！！！！！！！！！！！！！！！！！')
        data_20 = np.zeros((20, 2048, 7, 7))
        for j in (range(20)):
            data = np.load(
                "/data/gauss_test_res/" + cls + "/marker_C/result/mid_result_" + str(j) + ".npy")
            data_20[j] = data[0]
        print(cls, " 读取数据完毕")

        data = np.reshape(data_20, (20, -1))
        feature = np.zeros((300000, 20))
        for x in (range(300000)):
            for y in range(15):
                feature[x] += data[:, int(indexes[x][y])] * weights[x][y]
        np.save("/data/result/feature_" + cls + ".npy", feature.T)
        print(cls, " end train features 结束！！！！！！！！！！！！！！！！！！！！！！!")
    except Exception as e:
        print(e)


def getTrain():
    indexes = []
    weights = []
    for clsn in class_name:
        indexes.extend(np.load("/data/result/index_" + clsn + ".npy", allow_pickle=True))
        weights.extend(np.load("/data/result/weight_" + clsn + ".npy", allow_pickle=True))
    indexes = np.array(indexes)
    weights = np.array(weights)

    print("index over!")
    # params = []
    # for i in range(1000):
    #     params.append([class_name[i], indexes, weights])
    pool = MyPool(20)
    for i in tqdm(range(1000)):
        pool.apply_async(getFeature, (class_name[i], indexes, weights,))
        # getFeature(class_name[i], indexes, weights)
    pool.close()
    pool.join()

    print(" 训练数据计算！！！！！！！！！！！！！over!!")


def getValMid(cls):
    try:
        if os.path.exists("/data/result/val_data_" + cls + ".npy"):
            print(cls, " 计算验证集中间结果！！          over  !")
            return

        print(cls, "start!")
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model.eval().cuda()
        val_loader = data_loader("/data/data/val/" + cls)
        data = np.zeros((1, 2048, 7, 7))
        for i, (input_, _) in enumerate(val_loader):
            input_ = torch.autograd.Variable(input_).cuda()
            output = (model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(
                model.conv1(input_))))))))).cpu().detach().numpy()

            data = np.concatenate([data, output], axis=0)
        data = data[1:]
        np.save("/data/result/val_data_" + cls + ".npy", data)
        print(cls, " 计算验证集中间结果！！          over  !")
    except Exception as e:
        print(cls, " 出错了哦：       ", e)


def getVal_feature(args):
    try:

        cls, indexes, weights = args
        # if os.path.exists("/data/result/val_feature_" + cls + ".npy"):
        #     print(cls, " 计算验证集features！！！！              end!")
        #     return

        print(cls, ' 计算验证集features！！！ start')
        data = np.load("/data/result/val_data_" + cls + ".npy")
        print(cls, " 验证集读取数据完毕")
        data = np.reshape(data, (50, -1))
        feature = np.zeros((300000, 50))
        for x in range(300000):
            for y in range(15):
                feature[x] += data[:, int(indexes[x][y])] * weights[x][y]
        np.save("/data/result/val_feature_" + cls + ".npy", feature.T)
        print(cls, " 计算验证集features！！！！              end!")
    except Exception as e:
        print(cls, "出错了哦            ", e)


def getVal():
    indexes = []
    weights = []
    for clsn in class_name:
        indexes.extend(np.load("/data/result/index_" + clsn + ".npy", allow_pickle=True))
        weights.extend(np.load("/data/result/weight_" + clsn + ".npy", allow_pickle=True))
    indexes = np.array(indexes)
    weights = np.array(weights)

    # pool = MyPool()
    # pool.map(getValMid, class_name)
    # pool.close()
    # pool.join()

    # for cls in class_name:
    #     getValMid(cls)

    params = []
    for i in range(1000):
        params.append([class_name[i], indexes, weights])

    pool = MyPool()
    pool.map(getVal_feature, params)
    pool.close()
    pool.join()


class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.attention = nn.MultiheadAttention(num_heads=4, embed_dim=300)
    #     self.fc1 = nn.Linear(6000, 20)
    #
    # def forward(self, input):
    #     # return F.softmax(self.fc3(F.relu(self.fc2(F.relu(self.fc1(input))))))
    #     # return self.fc2(F.relu(F.dropout(self.fc1((input)), p=0.2, training=self.training)))
    #     #         # return F.softmax(self.fc2(F.relu(self.fc1(input))))
    #
    #     return self.fc1(F.relu(self.attention(F.relu(input))))

    def __init__(self):
        super(Net, self).__init__()
        self.attention = nn.MultiheadAttention(num_heads=10, embed_dim=300, dropout=0.5)
        self.fc = nn.Linear(300000, 1000)
        # self.fc2 = nn.Linear(3000, 1000)

    def forward(self, x):
        x = F.relu(x)
        x = x.view((-1, 1000, 300))
        x = F.relu(x)
        output, _ = self.attention(x, x, x)
        output = output.view(-1, 300000)
        output = self.fc(output)
        return output
        # x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # x = x.view(-1, 2048)
        # return self.fc2(F.relu(F.dropout(self.fc1(F.relu(x)), p=0.5)))
        # x, _ = self.attention(x, x, x)
        # return self.fc1(F.relu(x, inplace=False))


def train(is_cuda=False):
    # class_name = ["n01440764", "n01491361", "n01498041", "n01518878", "n01532829", "n01558993", "n01582220",
    #               "n01443537", "n01494475", "n01514668", "n01530575", "n01534433", "n01560419", "n01592084",
    #               "n01484850", "n01496331", "n01514859", "n01531178", "n01537544", "n01580077"
    #               ]
    #

    # model = torchvision.models.__dict__['resnet50'](pretrained=True)
    # data = np.zeros((1, 6000))
    # label = np.zeros((1,))
    # for dir in class_name:
    #     dataa = (np.load("/data/result/train_data_" + dir + ".npy", allow_pickle=True))
    #     labe = (np.load("/data/result/train_label_" + dir + ".npy", allow_pickle=True))
    #
    #     data = np.concatenate((data, dataa.T), axis=0)
    #     label = np.concatenate((label, labe), axis=0)
    # data = data[1:]
    # label = label[1:]

    # np.save("/data/result/data21.npy", data)
    # np.save("/data/result/label21.npy", label)

    # data = np.load("/data/result/data1.npy")
    # label = np.load("/data/result/label1.npy")
    #
    # data = np.load("/data/result/features.npy").T
    # label = np.arange(40000) // 2000

    # data = np.load("/data/all_changeaug_level1_20.npy")

    # data = np.load("/data/data.npy")
    # data = np.reshape(data, (40000, 2048, 7, 7))
    # label = np.arange(40000) // 2000

    # train_cls_label = {}
    # data = np.zeros((1000, 500, 300000))
    # for i in tqdm(range(1000)):
    #     data[i] = np.load("/data/result/feature_" + class_name[i] + ".npy")
    #     # train_cls_label[class_name[i]] = i
    # data = data.reshape((500000, 300000))
    # np.save("/data/result/data100.npy", data)
    # data = np.load("/data/result/data100.npy")
    # label = np.arange(500000) // 500

    # data = np.zeros((100, 2000, 30000))
    # for i in tqdm(range(100)):
    #     data[i] = np.load("/data/result/feature_" + class_name[i] + ".npy")
    # data = np.reshape(data, (200000, 30000))
    # np.save("/data/result/features_all.npy", data)
    # data = np.load("/data/result/data_all.npy")
    # label = np.arange(2000000) // 2000

    # val_data = np.zeros((1000, 50, 300000))
    # val_label = []
    # for i in tqdm(range(1000)):
    #     val_data[i] = np.load("/data/result/val_feature_" + class_name[i] + ".npy")
    #     val_label.extend([i for j in range(np.shape(val_data[i])[0])])
    # print(len(val_label))
    # val_data = val_data.reshape((50000, 300000))
    # np.save("/data/result/val_features_all.npy", val_data)
    # np.save("/data/result/val_label_all.npy", val_label)
    # val_data = np.load("/data/result/val_data_all.npy")
    # val_label = np.arange(50000) // 50

    # np.save("/data/result/train_cls_label.npy", train_cls_label)

    # val_data = np.zeros((1, 30000))

    # val_label = np.zeros((1,))
    # for cls in tqdm(class_name):
    #     # dataa = (np.load("/data/result/val_data_" + cls + ".npy", allow_pickle=True))
    #     labe = (np.load("/data/result/val_label_" + cls + ".npy", allow_pickle=True))
    #
    #     # val_data = np.concatenate((val_data, dataa.T), axis=0)
    #     val_label = np.concatenate((val_label, labe), axis=0)

    # val_data = val_data[1:]
    # np.save("/data/result/val_data100.npy", val_data)
    # val_data = np.load("/data/result/val_data100.npy")
    # val_label = np.arange(5000) // 50
    # val_label = val_label[1:]
    #
    # np.save("/data/result/val_data21.npy", val_data)
    # np.save("/data/result/val_label21.npy", val_label)

    # val_data = np.load("/data/result/val_data1.npy")
    # val_label = np.load("/data/result/val_label1.npy")

    # val_data = np.load("/data/val_change20.npy").T
    # val_label = np.arange(1000) // 50

    # val_data = torch.load("/data/ch5_change20_val.pt")
    # val_label = np.arange(1000) // 50
    #
    #
    # data = np.random.random((200, 2048, 7, 7))
    # label = [np.random.randint(2) for i in range(200)]
    #
    # val_data = np.random.random((20, 2048, 7, 7))
    # val_label = [np.random.randint(2) for i in range(20)]

    net = Net()
    net = net.cuda()
    torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
    net = nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)

    save_file = "./new_test.txt"

    print("数据准备完毕")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    for tratime in range(1):

        print('第', tratime, '次训练')

        data = np.zeros((1000, 20, 300000))
        val_data = np.zeros((1000, 50, 300000))
        for i in tqdm(range(1000)):
            data[i] = np.load("/data/result/feature_" + class_name[i] + ".npy", mmap_mode='r')
            val_data[i] = np.load("/data/result/val_feature_" + class_name[i] + ".npy", mmap_mode='r')
        data = data.reshape((20000, 300000))
        label = np.arange(20000) // 20
        dataset = TensorDataset(torch.Tensor(data), torch.LongTensor(label))
        sampler = DistributedSampler(dataset)
        train_loader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, sampler=sampler)

        val_label = np.arange(50000) // 50
        val_data = val_data.reshape((50000, 300000))
        val_dataset = TensorDataset(torch.Tensor(val_data), torch.LongTensor(val_label))
        val_loader = DataLoader(dataset=val_dataset, batch_size=50, num_workers=0)

        print_freq = 100
        print("开始训练")
        for epoch in range(100):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            # top5 = AverageMeter()

            # switch to train mode
            net.train()

            end = time.time()
            for i, (input, target) in enumerate(train_loader):
                # print(target)
                # measure data loading time
                # print(target)
                input = torch.where(torch.isnan(input), torch.full_like(input, 0), input)
                data_time.update(time.time() - end)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # print(input_var, target_var)
                # print(np.shape(input_var), np.shape(target_var))

                # if is_cuda:
                net = net.cuda()
                input_var = input_var.cuda()
                target_var = target_var.cuda()

                # compute output
                output = net(input_var)
                loss = criterion(output, target_var)
                # if loss.item() < 0.001:
                #     print(np.shape(output), loss, "///////////////////////////////")
                # measure accuracy and record loss
                acc1 = accuracy(output, target_var, topk=(1,))
                losses.update(loss.item(), input_var.size(0))
                top1.update(acc1[0], input_var.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Info log every args.print_freq
                if i % print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                          'Prec@1 {top1_val:.2f} ({top1_avg:.2f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses,
                        top1_val=np.asscalar(top1.val.cpu().numpy()),
                        top1_avg=np.asscalar(top1.avg.cpu().numpy())))
                    with open(save_file, 'a+', encoding='utf-8') as f:
                        f.write('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                                'Prec@1 {top1_val:.2f} ({top1_avg:.2f})\t\n'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses,
                            top1_val=np.asscalar(top1.val.cpu().numpy()),
                            top1_avg=np.asscalar(top1.avg.cpu().numpy())))

                    # top5_val=np.asscalar(top5.val.cpu().numpy()),
                    # top5_avg=np.asscalar(top5.avg.cpu().numpy())))
            # val_batch_time = AverageMeter()
            # val_losses = AverageMeter()
            # val_top1 = AverageMeter()
            # val_top5 = AverageMeter()
            #
            # im_dict = imagenet_class_index_dic()
            # # print(im_dict)
            #
            # val_print_freq = 100
            # end = time.time()

            # for j, (val_input, val_target) in enumerate(val_loader):
            #     net = net.eval()
            #     with torch.no_grad():
            #         # compute output
            #
            #         # print(val_target)
            #         # val_input = val_input.cuda()
            #         # val_target = val_target.cuda()
            #         # midres = model.avgpool(val_input).view(-1, 2048)
            #         # print(val_target)
            #         # print(target)
            #         #
            #         # val_output = model.fc(midres)
            #         val_output = net(val_input)
            #         # val_output = model(val_input)
            #         # print([np.argmax(val_output[i].numpy()) for i in range(np.shape(val_output)[0])])
            #         loss = criterion(val_output, val_target)
            #         # measure accuracy and record loss
            #         prec1, prec5 = accuracy(val_output, val_target, topk=(1, 5))
            #         val_losses.update(loss.item(), val_input.size(0))
            #         val_top1.update(prec1[0], val_input.size(0))
            #         # measure elapsed time
            #         val_batch_time.update(time.time() - end)
            #         end = time.time()
            #         if j % val_print_freq == 0:
            #             print('Test: [{0}/{1}]\t'
            #                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #                   'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'.format(
            #                 j, len(val_loader), batch_time=val_batch_time, loss=val_losses,
            #                 top1=val_top1))
            #
            # # train_result[epoch][0] = top1.avg
            # # train_result[epoch][1] = val_top1.avg
            # print(' * Loss@1 {loss.avg: .3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            #       .format(loss=val_losses, top1=val_top1, top5=val_top5))

            if epoch % 10 == 9:

                val_losses = AverageMeter()
                val_top1 = AverageMeter()
                net.eval()
                for i, (val_data, val_target) in enumerate(val_loader):
                    val_data = val_data.cuda()
                    val_data = torch.where(torch.isnan(val_data), torch.full_like(val_data, 0), val_data)
                    result = net(val_data)
                    target = val_target.cuda()
                    loss = criterion(result, target)
                    acc1 = accuracy(result, target, topk=(1,))
                    val_losses.update(loss.item(), val_data.size(0))
                    val_top1.update(acc1[0], val_data.size(0))
                print("验证集准确率为", val_top1.avg, '             loss:', val_losses.avg)

                with open(save_file, 'a+', encoding='utf-8') as f:
                    f.write("验证集准确率为" + str(val_top1.avg) + '             loss:' + str(val_losses.avg) + '\n')

    # np.save("/data/result/train_result.npy", train_result)
    print('Finished Training')

    # 验证集的准确率


def trans_tensor_from_image(dir, arch):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if arch == 'alexnet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # transform_data_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(dir, transforms.Compose([  # [1]
    #         transforms.Resize(256),  # [2]
    #         transforms.CenterCrop(227),  # [3]
    #         transforms.ToTensor(),  # [4]
    #     ])),
    #     batch_size=args.b, shuffle=False,
    #     num_workers=2, pin_memory=True)

    outputs = None

    with torch.no_grad():
        # print('2')
        for it_loader_batch_i, (image_input, target) in enumerate(transform_data_loader):
            image_input, target = image_input.to(device), target.to(device)

            # print('target is ', target)
            inputs = image_input.cpu()

            if it_loader_batch_i == 0:
                outputs = inputs
            else:
                outputs = torch.cat((outputs, inputs))

    return outputs


def test():
    ids = np.load("/data/result/ids.npy", allow_pickle=True)
    # print(ids)

    model = torchvision.models.__dict__['resnet50'](pretrained=True)
    result = []
    for x in range(100):
        input = trans_tensor_from_image("/data/data/val/" + class_name[x], " ")
        model.eval()
        val_output = model(input)
        # print(np.shape(input))
        target = [int(ids.item()[x]) for i in range(50)]
        pre1 = accuracy(val_output, torch.LongTensor(target), (1,))
        result.append(pre1[0].item())
        print(pre1[0].item())
    print(result, np.mean(result))
    np.save("/data/result/pre1.npy", result)


def mkdirsds():
    # class_name=[
    #     "n01440764", "n01491361", "n01498041", "n01518878", "n01532829", "n01558993", "n01582220",
    #     "n01443537", "n01494475", "n01514668", "n01530575", "n01534433", "n01560419", "n01592084",
    #     "n01484850", "n01496331", "n01514859", "n01531178", "n01537544", "n01580077"
    # ]
    for dir in class_name:
        os.makedirs("/data/data/train/" + dir + "/val/class/")
        os.makedirs("/data/data/val/" + dir + "/val/class/")

    for dir in class_name:
        # for file in os.listdir("/data/imagenet_2012/train/" + dir + "/" + dir):
        #     shutil.copy("/data/imagenet_2012/train/" + dir + "/" + dir + "/" + file,
        #                 "/data/data/train/" + dir + "/val/class/" + file)
        for file in os.listdir("/data/imagenet_2012/val/" + dir):
            shutil.copy("/data/imagenet_2012/val/" + dir + "/" + file,
                        "/data/data/val/" + dir + "/val/class/" + file)

    for ids in os.listdir("/data/data/val/"):
        if not os.path.exists("/data/data/val/" + ids + "/val/" + ids):
            os.makedirs("/data/data/val/" + ids + "/val/" + ids)
        try:
            for files in os.listdir("/data/data/val/" + ids):
                shutil.move("/data/data/val/" + ids + "/" + files,
                            "/data/data/val/" + ids + "/val/" + ids + "/" + files)
        except Exception as e:
            print(e)

    print("数据导入文件夹完毕！！！！！！！！！！！！！！！！！！！！！！")


def onedata(cls):
    try:
        # data = np.load("/data/result/feature_" + cls + ".npy")
        # data = pd.DataFrame(data)
        # data['label'] = class_name.index(cls)
        # # print(data.info())
        # data.to_csv("/data/result/data.csv", index=False, header=False, mode='a')
        if os.path.exists("/data/result/feature_" + cls + "_" + str(99) + ".npy"):
            print(cls)
            return
        data = np.load("/data/result/feature_" + cls + ".npy")

        if data.shape[0] != 500:
            data = data.T
        for i in (range(100)):
            if not os.path.exists("/data/result/feature_" + cls + "_" + str(i) + ".npy"):
                np.save("/data/result/feature_" + cls + "_" + str(i) + ".npy", data[5 * i:5 * (i + 1)])
        print(cls)
    except Exception as e:
        print(cls, e)


def getpdcsv():
    pool = MyPool(10)
    pool.map(onedata, class_name)
    pool.close()
    pool.join()
    # onedata(class_name[0])
    # for cls in class_name:
    #     onedata(cls)