from utils_io_folder import *

def load_unready_heatmap(tensorflow_x_path, batchsize, nsteps, id, w_img, h_img):
    lines = load_dataset_gt(tensorflow_x_path)
    [st, ed] = [id, id + batchsize * nsteps]

    heatmap_vec_batch= []
    for id in range(st, ed):
        location = find_gt_location(lines, id)
        location = locations_from_0_to_1(w_img, h_img, location)
        coords =  loc_to_coordinates(location)
        heatmap_vec = [coordinates_to_heatmap_vec(coords)]
        heatmap_vec_batch.append(heatmap_vec)
	return heatmap_vec_batch


def load_ready_heatmap(folder_path, params, id):
    batchsize = params['batchsize']
    nsteps = params['nsteps']
    vec_len = params['vec_len']

    heatmap_files_paths = get_immediate_childfile_paths(folder_path)
    [st, ed] = [id, id + batchsize * nsteps]
    paths_batch = heatmap_files_paths[st:ed]

    heatmap_vec_batch= []
    for path in paths_batch:
        heatmap_vec = np.load(path)
        heatmap_vec = np.reshape(heatmap_vec, vec_len)
        heatmap_vec_batch.append(heatmap_vec)
    heatmap_vec_batch = np.reshape(heatmap_vec_batch, [batchsize*nsteps, vec_len])
    return heatmap_vec_batch
