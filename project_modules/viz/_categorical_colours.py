import colorcet as cc
import pandas as pd
import matplotlib.colors as mcolors

from ._colors import red_blue


def make_categorical_colours(df, palette):

    # make a list of colours
    colours = {feature: palette[i] for i, feature in enumerate(df.columns)}
    return colours


reduced_rb = red_blue.resampled(2)
binary_red_blue  = [mcolors.rgb2hex(reduced_rb(i)) for i in range(2)]
