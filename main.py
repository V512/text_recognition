import json
from collections import namedtuple, defaultdict
import cv2
from os import listdir
from os.path import join, isfile
import numpy as np
from time import time

SingleDigramStats = namedtuple("DigramStats", "frequencies probability")

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            ' ']


def load_frequencies(json_path):
    """
    Load frequeinces from json file and compute probability of digram

    :param json_path: path to json file
    :return: dict with SingleDigramStats objects
    """
    with open(json_path, "r") as json_file:
        frequencies = json.load(json_file)
    values_sum = sum(frequencies.values())
    statistic = defaultdict(SingleDigramStats)
    for key in frequencies.keys():
        digram_freq = frequencies[key]
        digram_prob = digram_freq / values_sum
        statistic[key] = SingleDigramStats(frequencies=digram_freq,
                                           probability=digram_prob)
    return statistic


def get_conditional_digram_probability(statistic_dict):
    """
    Compute conditional symbol probabilities

    :param statistic_dict: dict with freq and prob
    :return: dict with conditional symbol prob
    """
    conditional_prob_dict = {}
    for first_symbol in alphabet:
        common_prob = 0

        for second_symbol in alphabet:
            digram = first_symbol + second_symbol
            common_prob += statistic_dict[
                digram].probability if digram in statistic_dict else 0

        for second_symbol in alphabet:
            digram = first_symbol + second_symbol
            conditional_prob_dict[digram] = statistic_dict[
                digram].probability / common_prob if digram in statistic_dict else 0
    return conditional_prob_dict


def get_standard_images(path_to_images):
    """
    Return dict of standard images

    :param path_to_images: path to images
    :return: dict of images
    """
    standard_images = {}
    for image_name in listdir(path_to_images):
        image_path = join(path_to_images, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image[image == 255] = 1
        symbol = image_name.split(sep=".")[0]
        if symbol == "space":
            symbol = " "
        standard_images[symbol] = image
    return standard_images


def get_symbol_probability(a, b, p):
    """
    Get log probability of an unknown symbol under the assumption
    that it's a standard symbol

    :param a: noised image
    :param b: standard image
    :param p: noise probability
    :return: probability
    """

    assert a.shape == b.shape
    p1 = np.log(p ** (a ^ b)) + np.log(((1 - p) ** (a ^ b ^ 1)))
    # p2 = np.log(p1)
    p3 = np.sum(p1)
    return p3


def text_recognition(text, p, standard_images, conf_prob):
    cols = text.shape[1]
    print(cols)
    f = np.full(shape=(cols, len(alphabet)), fill_value=-np.inf)
    alphabet_len = len(alphabet)
    m = np.zeros((cols + 1, alphabet_len))
    for s in range(alphabet_len):
        orig_image = standard_images[alphabet[s]]
        orig_image_len = orig_image.shape[1]
        if orig_image_len <= cols:
            digram = " " + alphabet[s]
            noised_sample = text[::,0:orig_image_len]
            f[0][s] = np.log(conf_prob[digram]) + get_symbol_probability(orig_image, noised_sample, p)
            m[0][s] = 26
    for i in range(0, cols):
        if i % 100 == 0:
            print("It {}".format(i))
        for s1 in range(alphabet_len):
            if f[i][s1] == -np.inf:
                continue
            if i + standard_images[alphabet[s1]].shape[1] == cols:
                last_digram = alphabet[s1] + " "
                pr_last = np.log(conf_prob[last_digram]) + f[i][s1]
                f[-1][s1] = pr_last
            for s2 in range(alphabet_len):
                s2_start_index = i + standard_images[alphabet[s1]].shape[1]
                s2_end_index = s2_start_index + standard_images[alphabet[s2]].shape[1]
                if s2_end_index - 1 >= cols:
                    continue
                digram = alphabet[s1] + alphabet[s2]
                orig_s2_image = standard_images[alphabet[s2]]
                len_orig_s2_image = orig_s2_image.shape[1]
                noised_sample = text[::, s2_start_index:len_orig_s2_image + s2_start_index]
                pr = np.log(conf_prob[digram]) + \
                     get_symbol_probability(orig_s2_image, noised_sample, p) + f[i][s1]
                if pr > f[s2_start_index][s2]:
                    f[s2_start_index][s2] = pr
                    m[s2_start_index][s2] = s1

    last_symbol = int(np.argmax(f[-1]))
    denoised_text = alphabet[last_symbol]
    i = cols
    while i > 0:
        i -= standard_images[alphabet[last_symbol]].shape[1]
        last_symbol = int(m[i][last_symbol])
        denoised_text = alphabet[last_symbol] + denoised_text
    print(denoised_text[1:])
    print("Done")


if __name__ == '__main__':
    statistic = load_frequencies('frequencies.json')
    conf_prob = get_conditional_digram_probability(statistic)
    standard_images = get_standard_images('alphabet')
    input = cv2.imread('input/but thence i learn and find the lesson true drugs poison him that so feil sick of you_0.55.png', cv2.IMREAD_UNCHANGED)
    # input = cv2.imread('input/i am very glad to see you here stranger let us denoise some text_0.45.png', cv2.IMREAD_UNCHANGED)
    # input = cv2.imread('alphabet/a.png', cv2.IMREAD_UNCHANGED)
    input[input == 255] = 1
    start_time = time()
    text_recognition(input, 0.55, standard_images, conf_prob)
    print(time() - start_time)

