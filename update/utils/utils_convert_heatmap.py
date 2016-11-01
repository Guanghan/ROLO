def coordinates_to_heatmap_vec(coord):
        heatmap_vec = np.zeros(1024)
        [x1, y1, x2, y2] = coord
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                index = y*32 + x
                heatmap_vec[index] = 1.0   #random.uniform(0.8, 1)#1.0
        return heatmap_vec


def heatmap_vec_to_heatmap(heatmap_vec):
    size = 32
    heatmap= np.zeros((size, size))
    for y in range(0, size):
        for x in range(0, size):
            index = y*size + x
            heatmap[y][x] = heatmap_vec[index]
    return heatmap
