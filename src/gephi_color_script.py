# This script can be called from Gephi

def generate_color_scheme_red():
    # Create a monotonic red scheme
    # curr = [0xff, 0xD8, 0xE7]
    curr = [0xff, 0xff, 0xff]
    scheme = [[curr[0], curr[1], curr[2]]]

    curr = [0xff, 0xFa, 0xFb]
    scheme.append([curr[0], curr[1], curr[2]])

    for i in range(14):
        curr[1] -= 0x11
        curr[2] -= 0xA
        scheme.append([curr[0], curr[1], curr[2]])

    curr = [0xfa, 0x00, 0x64]
    scheme.append([curr[0], curr[1], curr[2]])

    for i in range(26 - 17):
        curr[0] -= 0x11
        curr[2] -= 0x7
        scheme.append([curr[0], curr[1], curr[2]])

    return scheme  # Total 26


def generate_color_scheme_green():
    # Create a monotonic green scheme
    curr = [0xff, 0xff, 0xff]
    scheme = [[curr[0], curr[1], curr[2]]]
    curr = [0xf1, 0xff, 0xff]
    scheme.append([curr[0], curr[1], curr[2]])

    for i in range(14):
        curr[0] -= 0x11
        scheme.append([curr[0], curr[1], curr[2]])

    curr = [0x00, 0xf1, 0xf1]
    scheme.append([curr[0], curr[1], curr[2]])

    for i in range(26-17):  # Need total 26
        curr[1] -= 0x1a
        curr[2] -= 0x1a
        scheme.append([curr[0], curr[1], curr[2]])

    return scheme
    

color_scheme_red = generate_color_scheme_red()
color_scheme_green = generate_color_scheme_green()

GRANULARITY = 200 / 25

for node in g.nodes:
    congestion = abs(node.congestion)

    idx_range = []
    # Create a range of indexes that are in steps of GRANULARITY that we can spread across. Currently using 200 % as
    # upper bound, but essentially will bin all percentages based on 0-GRANULARITY, GRANULARITY - 2*GRANULARITY, and
    # so on. Then, we find the first evaluation that is true and select that as the index.
    for i in range(25):
        idx_range.append(congestion <= int(GRANULARITY * i))
    idx_range.append(True)
    idx = idx_range.index(True)

    # Choose red for out congestion, green for in congestion
    tgt = color_scheme_red if node.congestion < 0 else color_scheme_green
    node.color = color(tgt[idx][0], tgt[idx][1], tgt[idx][2])

