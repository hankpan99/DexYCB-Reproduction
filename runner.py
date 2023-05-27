import os
import yaml
import torch
import numpy as np
import xml.etree.ElementTree as ET

from manopth.manolayer import ManoLayer
from optimizer import Optimizer

class Runner():
    def __init__(self, demo_id, device):
        self.root_pth = os.path.join('data', demo_id)
        self.device = device

        # read meta file
        with open(os.path.join(self.root_pth, 'meta.yml'), 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        
        self.camera_serials = meta['serials']
        self.num_frames = meta['num_frames']

        # read camera parameters
        self.extrinsic_pth = os.path.join('data', 'calibration', 'extrinsics_{}'.format(meta['extrinsics']), 'extrinsics.yml')
        self.intrinsic_pth = os.path.join('data', 'calibration', 'intrinsics', '{}_640x480.yml')

        self.proj_mats = self.compute_projection_matrix()

        # read kpts
        self.kpts = self.read_labels()

        # create mano layer
        self.manoLayer = ManoLayer(flat_hand_mean=False,
                                   ncomps=45,
                                   mano_root='manopth/mano_v1_2/models',
                                   use_pca=True).to(device)

    def read_labels(self):
        annot_root = ET.parse(os.path.join(self.root_pth, 'cvat_kpts', 'annotations.xml')).getroot()

        # define joint names
        # joint_order = ['"R_Wrist"', '"R_Thumb_Tip"', '"R_Index_PIP"', '"R_Index_Tip"',
        #                     '"R_Middle_PIP"', '"R_Middle_Tip"', '"R_Ring_PIP"', '"R_Ring_Tip"',
        #                     '"R_Pinky_PIP"', '"R_Pinky_Tip"']
        joint_order = ['"{}"'.format(str(i).zfill(2)) for i in range(21)]

        # create camera_id 2 task_id dict
        camera2task_dict = {}
        for task in annot_root.findall('./meta/project/tasks/task'):
            name = task.find('name').text
            id = task.find('id').text
            camera2task_dict[name] = '"' + id + '"'

        kpts_allview = []
        for camera_id in self.camera_serials:
            
            kpts_oneview = []
            for joint in joint_order:
                # find track where (label = joint) && (task_id = camera2task_dict[camera_id])
                condi_str = './track[@label=' + joint + '][@task_id=' + camera2task_dict[camera_id] + ']/'
                
                # get keypoints in each frame
                points = annot_root.findall(condi_str)
                sub_points = [points[i] for i in range(self.num_frames)]

                kpts_onejoint = []
                for point in sub_points:
                    position = point.attrib['points'].split(',')
                    position = [int(float(x)) for x in position]

                    # if occluded, fill in [-1, -1]
                    if not int(point.attrib['occluded']):
                        kpts_onejoint.append([position[0], position[1]])
                    else:
                        kpts_onejoint.append([-1, -1])

                kpts_oneview.append(kpts_onejoint)

            kpts_allview.append(kpts_oneview)

        kpts_arr = np.array(kpts_allview)
        kpts_arr = np.transpose(kpts_arr, (2, 0, 1, 3)) # [# frames, # cameras, # joints, xy-coordinates]
        kpts_arr = torch.from_numpy(kpts_arr).to(self.device, torch.float32)

        return kpts_arr
    
    def compute_projection_matrix(self):
        # load extrinsic
        with open(self.extrinsic_pth, 'r') as f:
            extrinsic_file = yaml.load(f, Loader=yaml.FullLoader)['extrinsics']
        extrinsic_list = []
        for camera_id in self.camera_serials:
            extrinsic_list.append(np.array(extrinsic_file[camera_id]).reshape(3, 4))
            extrinsic_list[-1] = np.linalg.inv(np.vstack((extrinsic_list[-1], np.array([0, 0, 0, 1]))))[:3, :]

        # load intrinsic
        intrinsic_list = []
        for camera_id in self.camera_serials:
            with open(self.intrinsic_pth.format(camera_id), 'r') as f:
                intrinsic_file = yaml.load(f, Loader=yaml.FullLoader)['color']
            intrinsic_list.append(np.array([[intrinsic_file['fx'], 0, intrinsic_file['ppx']],
                                            [0, intrinsic_file['fy'], intrinsic_file['ppy']],
                                            [0, 0, 1]]))

        # compute projection matrix = intrinsic @ inverse(extrinsic)
        proj_mat_list = []
        for intrinsic, extrinsic in zip(intrinsic_list, extrinsic_list):
            proj_mat_list.append(intrinsic @ extrinsic)
        
        proj_mats = np.array(proj_mat_list)
        proj_mats = torch.from_numpy(proj_mats).to(self.device, torch.float32)

        return proj_mats

    def run(self):
        # create optimizer
        optimizer = Optimizer(self.device, self.proj_mats, self.manoLayer)

        # set initial pose
        init_pose = torch.zeros((1, 51), dtype=torch.float32, device=self.device)
        init_pose[0, 0:3] = torch.tensor([-np.pi/2, -np.pi/2, 0], dtype=torch.float32, device=self.device)
        init_pose[0, 48:51] = torch.tensor([0.2, 0.2, 0.9], dtype=torch.float32, device=self.device)

        # output pose
        pose_m = np.zeros((self.num_frames, 1, 51), dtype=np.float32)

        # optimize all frames
        for i in range(self.num_frames):
            print('-' * 20)
            print('frame = {}'.format(i))

            num_steps = 3000 if (i == 0) else 100

            cur_pose, loss = optimizer.optimize(init_pose, self.kpts[i], num_steps)

            # set next-frame initial pose to curret pose
            init_pose = cur_pose.clone()

            # cache current pose
            pose_m[i] = cur_pose.detach().cpu().numpy()
            
            print('loss: {:.2f}'.format(loss))
            print()

        # save pose
        np.savez_compressed('{}/pose.npz'.format(self.root_pth), pose_m=pose_m.reshape((self.num_frames, 1, 51)))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    runner = Runner('20221001_171108', device)
    runner.run()
    