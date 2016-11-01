from utils_dataset import *
from utils_draw_coord import debug_decimal_coord
from utils_io_folder import *
from utils_io_coord import *

def get_batch_by_repeat(ndarray, batchsize):
    batch_ndarray = []
    for id in range(batchsize):
        batch_ndarray.append(ndarray)
    return batch_ndarray


def test(self, sess, loss, batch_pred_coords):
    print("\n\n\n--------------------------------------------TESTING OTB-50---------------------------------------------------------\n")
    num_videos = 50
    loss_dataset_total = 0
    OTB_folder_path = "/home/ngh/dev/ROLO-dev/benchmark/DATA/"

    for video_id in range(num_videos):
        if video_id in [1, 5, 16, 20, 21, 22, 23, 28, 30, 32, 36, 42, 43, 46]: continue

        [img_wid, img_ht, sequence_name, st_frame, self.training_iters] = choose_video_sequence_from_OTB50(video_id)
        print('testing sequence: ', sequence_name)

        x_path = os.path.join(OTB_folder_path, sequence_name, 'yolo_out/')
        y_path = os.path.join(OTB_folder_path, sequence_name, 'groundtruth_rect.txt')
        self.output_path = os.path.join(OTB_folder_path, sequence_name, 'rolo_loc_test/')
        create_folder(self.output_path)

        img_folder_path = os.path.join(OTB_folder_path, sequence_name, 'img/')
        img_paths = get_immediate_childfile_paths(img_folder_path)

        loss_seq_total = frame_id = 0
        offset_id = self.nsteps

        init_state_zeros = np.zeros((self.batchsize, 2*self.len_vec))

        while frame_id  < self.training_iters- self.nsteps:

            ''' The index start from zero, while the frame usually starts from one '''
            st_id = st_frame - 1
            if frame_id < st_id:
                frame_id += 1
                continue

            ''' Load input data & ground truth '''
            xs = load_vecs_of_stepsize_in_numpy_folder(x_path,
                                                       frame_id - st_id,
                                                       self.nsteps)
            ys = load_gt_decimal_coords_from_file(y_path,
                                                  frame_id - st_id + offset_id,
                                                  img_wid,
                                                  img_ht)

            batch_xs = get_batch_by_repeat(xs, self.batchsize)
            batch_ys = get_batch_by_repeat(ys, self.batchsize)

            batch_xs = np.reshape(batch_xs, [self.batchsize, self.nsteps, self.len_vec])
            batch_ys = np.reshape(batch_ys, [self.batchsize, 4])

            ''' Save pred_location to file '''
            #utils.save_rolo_output(self.output_path, pred_loc, id, self.nsteps, self.batchsize)

            init_state = init_state_zeros
            #init_state = sess.run(self.final_state,
            #                      feed_dict={self.x: batch_xs,
            #                                 self.y: batch_ys,
            #                                 self.istate: init_state_zeros})
            batch_loss = sess.run(loss,
                                  feed_dict={self.x: batch_xs,
                                             self.y: batch_ys,
                                             self.istate: init_state})
            loss_seq_total += batch_loss

            if self.display_validate is True:
                coord_decimal_gt = sess.run(self.y,
                                            feed_dict = {self.x: batch_xs,
                                                         self.y: batch_ys,
                                                         self.istate: init_state})
                coord_decimal_pred = sess.run(batch_pred_coords,
                                              feed_dict = {self.x: batch_xs,
                                                           self.y: batch_ys,
                                                           self.istate: init_state}
                                              )[0]

                img = cv2.imread(img_paths[frame_id])
                debug_decimal_coord(img, coord_decimal_pred)

            frame_id += 1

        loss_seq_avg = loss_seq_total / frame_id
        print "Avg loss for " + sequence_name + ": " + str(loss_seq_avg)
        loss_dataset_total += loss_seq_avg

    print('Total loss of Dataset: %f \n', loss_dataset_total)
    print("-----------------------------------------TESTING OTB-50 END---------------------------------------------------------\n\n\n")
    return loss_dataset_total
