from collections import namedtuple, defaultdict
import cv2
import json
import numpy as np
from os import listdir
from os.path import join
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
    Compute conditional probabilities for symbol

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
            conditional_prob_dict[digram] = \
                statistic_dict[digram].probability \
                / common_prob if digram in statistic_dict else 0

    return conditional_prob_dict


def load_binary_image(image_path):
    """
    Load binary image

    :param image_path path to image:
    :return: image
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image[image == 255] = 1
    return image


def get_standard_images(path_to_images):
    """
    Return dict of standard images

    :param path_to_images: path to images
    :return: dict of images
    """
    standard_images = {}

    for image_name in listdir(path_to_images):
        image_path = join(path_to_images, image_name)
        image = load_binary_image(image_path)
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
    p = np.log(p ** (a ^ b)) + np.log(((1 - p) ** (a ^ b ^ 1)))
    p = np.sum(p)
    return p


def text_recognition(text, p, standard_images, conf_prob):
    cols = text.shape[1]
    f = np.full(shape=(cols, len(alphabet)), fill_value=-np.inf)
    alphabet_len = len(alphabet)
    m = np.zeros((cols + 1, alphabet_len))

    for s in range(alphabet_len):
        orig_image = standard_images[alphabet[s]]
        orig_image_len = orig_image.shape[1]

        if orig_image_len <= cols:
            digram = " " + alphabet[s]
            noised_sample = text[::, 0:orig_image_len]
            f[0][s] = np.log(conf_prob[digram]) + get_symbol_probability(
                orig_image, noised_sample, p)
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
                s2_end_index = \
                    s2_start_index + standard_images[alphabet[s2]].shape[1]

                if s2_end_index > cols:
                    continue

                digram = alphabet[s1] + alphabet[s2]
                orig_s2_image = standard_images[alphabet[s2]]
                len_orig_s2_image = orig_s2_image.shape[1]
                noised_sample = \
                    text[::, s2_start_index:len_orig_s2_image + s2_start_index]
                pr = np.log(conf_prob[digram]) + \
                     get_symbol_probability(orig_s2_image, noised_sample, p) \
                     + f[i][s1]

                if pr > f[s2_start_index][s2]:
                    f[s2_start_index][s2] = pr
                    m[s2_start_index][s2] = s1

    last_symbol = int(np.argmax(f[-1]))
    denoised_text = alphabet[last_symbol]
    i = cols - standard_images[alphabet[last_symbol]].shape[1]

    while i > 0:
        last_symbol = int(m[i][last_symbol])
        i -= standard_images[alphabet[last_symbol]].shape[1]
        denoised_text = alphabet[last_symbol] + denoised_text

    return denoised_text


def render_text(text, standard_images_path='alphabet', noise_probability=0.0):
    """
    Render text with given alphabet and level of noise.

    :param text: text to render
    :param standard_images_path: path to alphabets images
    :param noise_probability: level of noise
    :return: rendered text
    """
    images = []
    for s in text:
        if s == ' ':
            s = "space"
        image_name = "{}.png".format(s)
        image_path = join(standard_images_path, image_name)
        images.append(load_binary_image(image_path))
    image = np.concatenate(images, axis=1)
    eps = 1e-6
    if noise_probability > eps:
        noise = np.random.binomial(1, noise_probability, size=image.shape)
        image = image ^ noise
    return image.astype(np.uint8)


if __name__ == '__main__':
    noise_probability = 0.3
    input = render_text("its wednesday my dudes", noise_probability=noise_probability)
    # input = load_binary_image("input/hello sweety_0.2.png")
    image = input.copy()
    image[image == 1] = 255
    cv2.imshow('image', image)
    cv2.waitKey(0)
    statistic = load_frequencies('frequencies.json')
    conf_prob = get_conditional_digram_probability(statistic)
    standard_images = get_standard_images('alphabet')
    start_time = time()
    text = text_recognition(input, noise_probability, standard_images, conf_prob)
    print("Denoised text \'{}\'".format(text))
    print("Elapsed time {}".format(time() - start_time))
