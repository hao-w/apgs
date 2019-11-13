import torch
from torch.nn.functional import affine_grid, grid_sample

class Affine_Transformer():
    """
    z_where : samples of the coordinates, S * B * T * K * 2
    frames : S * B * H * W
    digit : S * B * K * H * W
    s_factor : the scaling parameters in the affine matrices
    t_factor : the translation parameters in the affine matrices

    Note :
    affine_grid : takes N * 2 * 3 affine matrices and torch.Size((N, C, H_out, W_out))
    grid_sample : takes N * H_in * W_in input images and the grid from affine_grid
    """
    def __init__(self, frame_pixels, digit_pixels, CUDA, DEVICE):
        super().__init__()
        self.digit_pixels =  digit_pixels
        self.frame_pixels = frame_pixels
        self.t1_factor = (self.frame_pixels - self.digit_pixels) / self.digit_pixels
        self.t2_factor = (self.frame_pixels - self.digit_pixels) / self.frame_pixels
        self.scale1 = torch.FloatTensor([[self.frame_pixels / self.digit_pixels, 0], [0, self.frame_pixels / self.digit_pixels]])
        self.scale2 = torch.FloatTensor([[self.digit_pixels / self.frame_pixels, 0], [0, self.digit_pixels / self.frame_pixels]])
        if CUDA:
            with torch.cuda.device(DEVICE):
                self.scale1 = self.scale1.cuda()
                self.scale2 = self.scale2.cuda()

    def digit_to_frame(self, digit, z_where):
        S, B, K, _ = z_where.shape
        affine_p1 = self.scale1.repeat(S, B, K, 1, 1)## S * B * K * 2 * 2
        affine_p2 = z_where.unsqueeze(-1) * self.t1_factor ## S * B * K * 2 * 1
        affine_p2[:, :, :, 0, :] = -1 * affine_p2[:, :, :, 0, :] ## flip the x-axis due to grid function
        grid = affine_grid(torch.cat((affine_p1, affine_p2), -1).view(S*B*K, 2, 3), torch.Size((S*B*K, 1, self.frame_pixels, self.frame_pixels)))
        frames = grid_sample(digit.view(S*B*K, self.digit_pixels, self.digit_pixels).unsqueeze(1), grid, mode='nearest')
        return frames.squeeze(1).view(S, B, K, self.frame_pixels, self.frame_pixels)
    #
    # def frame_to_digit(self, frames, z_where):
    #     S, B, K, _ = z_where.shape
    #     affine_p1 = self.scale2.repeat(S, B, K, 1, 1)## S * B * K * 2 * 2
    #     affine_p2 = z_where.unsqueeze(-1) * self.t2_factor ## S * B * K * 2 * 1
    #     affine_p2[:, :, :, 1, :] = -1 * affine_p2[:, :, :, 1, :] ## flip the y-axis due to grid function
    #     grid = affine_grid(torch.cat((affine_p1, affine_p2), -1).view(S*B*K, 2, 3), torch.Size((S*B*K, 1, self.digit_pixels, self.digit_pixels)))
    #     digit = grid_sample(frames.unsqueeze(-3).repeat(1, 1, K, 1, 1).view(S*B*K, self.frame_pixels, self.frame_pixels).unsqueeze(1), grid, mode='nearest')
    #     return digit.squeeze(1).view(S, B, K, self.digit_pixels, self.digit_pixels)

    def digit_to_frame_vec(self, digit, z_where):
        S, B, T, K, _ = z_where.shape
        affine_p1 = self.scale1.repeat(S, B, T, K, 1, 1)## S * B * T * K * 2 * 2
        affine_p2 = z_where.unsqueeze(-1) * self.t1_factor ## S * B * T * K * 2 * 1
        affine_p2[:, :, :, :, 0, :] = -1 * affine_p2[:, :, :, :, 0, :] ## flip the x-axis due to grid function
        grid = affine_grid(torch.cat((affine_p1, affine_p2), -1).view(S*B*T*K, 2, 3), torch.Size((S*B*T*K, 1, self.frame_pixels, self.frame_pixels)))
        frames = grid_sample(digit.unsqueeze(2).repeat(1,1,T,1,1,1).view(S*B*T*K, self.digit_pixels, self.digit_pixels).unsqueeze(1), grid, mode='nearest')
        return frames.squeeze(1).view(S, B, T, K, self.frame_pixels, self.frame_pixels)


    def frame_to_digit(self, frames, z_where):
        S, B, T, K, _ = z_where.shape
        affine_p1 = self.scale2.repeat(S, B, T, K, 1, 1)## S * B * T * K * 2 * 2
        affine_p2 = z_where.unsqueeze(-1) * self.t2_factor ## S * B * T * K * 2 * 1
        affine_p2[:, :, :, :, 1, :] = -1 * affine_p2[:, :, :, :, 1, :] ## flip the y-axis due to grid function
        grid = affine_grid(torch.cat((affine_p1, affine_p2), -1).view(S*B*T*K, 2, 3), torch.Size((S*B*T*K, 1, self.digit_pixels, self.digit_pixels)))
        digit = grid_sample(frames.unsqueeze(-3).repeat(1, 1, 1, K, 1, 1).view(S*B*T*K, self.frame_pixels, self.frame_pixels).unsqueeze(1), grid, mode='nearest')
        return digit.squeeze(1).view(S, B, T, K, self.digit_pixels, self.digit_pixels)
