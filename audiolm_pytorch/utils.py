def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :(data_len // mult * mult)]
