"""Feature engineering for the Stellar Classification project."""

TARGET = "class"

# Differences between neighbouring filters: how astronomers describe an object's "color"
COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
COLOR_COLUMNS = ["u", "g", "r", "i", "z"] + [f"{a}_{b}" for a, b in COLOR_PAIRS]


def add_color_features(df):
    """Return a copy of df with the u_g, g_r, r_i and i_z color columns added."""
    data = df.copy()
    for a, b in COLOR_PAIRS:
        data[f"{a}_{b}"] = data[a] - data[b]
    return data


def split_features_target(df):
    """Split a cleaned DataFrame into features X and target y."""
    return df.drop(columns=[TARGET]), df[TARGET]
