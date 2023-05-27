import torch
import torch.nn as nn
import torch.optim as optim

class Optimizer():
    def __init__(self, device, proj_mat, mano_layer):
        self.device = device
        self.proj_mat = proj_mat
        self.mano_layer = mano_layer
        
        self.mano_theta = torch.zeros((1, 51), dtype=torch.float32, device=self.device, requires_grad=True)
        self.mse_loss_sum = nn.MSELoss(reduction='sum').to(self.device)
        self.mse_loss_mean = nn.MSELoss(reduction='mean').to(self.device)
        self.optimizer = optim.Adam([self.mano_theta], lr=0.01)
    
    def keypoints_loss(self, points, kpts):
        # project 3d points into 2d camera coordinate
        # points_3d = points[[0, 4, 6, 8, 10, 12, 14, 16, 18, 20]].unsqueeze(0).repeat(self.proj_mat.shape[0], 1, 1)
        points_3d = points.unsqueeze(0).repeat(self.proj_mat.shape[0], 1, 1)
        ONE_TENSOR = torch.ones((points_3d.shape[0], points_3d.shape[1], 1), dtype=torch.float32, device=self.device)
        points_3d = torch.cat((points_3d, ONE_TENSOR), dim=2)
        points_3d = torch.transpose(points_3d, 1, 2)
        
        points_2d = torch.bmm(self.proj_mat, points_3d)
        points_2d = torch.transpose(points_2d, 1, 2)
        points_2d = points_2d[:, :, 0:2] / points_2d[:, :, [2]]

        # find visible keypoint index
        vis_mask = torch.all(kpts != -1, dim=2)
        points_2d, kpts = points_2d[vis_mask], kpts[vis_mask]
        
        # compute loss
        loss = self.mse_loss_mean(points_2d, kpts)

        return loss
    
    def regularization_loss(self):
        # compute loss
        joint_pose = self.mano_theta[0, 3:48]
        ZERO_TENSOR = torch.zeros_like(joint_pose, device=self.device)
        
        loss = self.mse_loss_sum(joint_pose, ZERO_TENSOR)
        
        return loss
    
    def optimize(self, init_pose, kpts, num_steps):
        self.mano_theta.data = init_pose.clone()
        
        for _ in range(num_steps):
            self.optimizer.zero_grad()
            
            # forward pass
            _, joints = self.mano_layer(th_pose_coeffs=self.mano_theta[:, :48], th_trans=self.mano_theta[:, 48:])
            joints = joints[0] / 1000
            
            # compute loss
            kpts_loss = self.keypoints_loss(joints, kpts)
            reg_loss = self.regularization_loss()
            
            loss = kpts_loss + reg_loss
            
            # backward
            loss.backward()
            
            # update parameters
            self.optimizer.step()
            
        return self.mano_theta, loss.detach()
        