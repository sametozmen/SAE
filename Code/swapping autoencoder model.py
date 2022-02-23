import numpy as np
import torch
import torch.nn.functional as F
import util
from models import BaseModel
import models.networks as networks
import models.networks.loss as loss
import sys

sys.path.insert(0, '/content')
import segmenter


class SwappingAutoencoderModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=8, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
        parser.add_argument("--lambda_L1", default=1.0, type=float)
        parser.add_argument("--lambda_GAN", default=1.0, type=float)
        parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)
        parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
        parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
        parser.add_argument("--patch_num_crops", default=8, type=int)
        parser.add_argument("--patch_use_aggregation",
                            type=util.str2bool, default=True)
        return parser

    def initialize(self):
        self.E = networks.create_network(self.opt, self.opt.netE, "encoder")
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator")
        if self.opt.lambda_PatchGAN > 0.0:
            self.Dpatch = networks.create_network(
                self.opt, self.opt.netPatchD, "patch_discriminator"
            )

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()

        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()

        if self.opt.num_gpus > 0:
            self.to("cuda:0")

    def per_gpu_initialize(self):
        pass

    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)

    def compute_image_discriminator_losses(self, real, rec, mix):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(real)
        pred_rec = self.D(rec)
        pred_mix = self.D(mix)

        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        losses["D_mix"] = loss.gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        return losses

    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops

    def apply_my_random_crop(self, x, target_size, scale_range, num_crops=1, return_rect=False):
        B = x.size(0) * num_crops
        flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
        unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :,
                      np.newaxis].repeat(
            B, target_size, 1, 1)
        unit_grid_y = unit_grid_x.transpose(1, 2)
        unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)
        x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
        scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
        sampling_grid = unit_grid * scale + offset
        crop = F.grid_sample(x, sampling_grid, align_corners=False)
        i = 0
        while not np.count_nonzero(crop[0, 0:3, :, :].permute(2, 1, 0).cpu().numpy() == 0) > 35_000:
            i += 1

            B = x.size(0) * num_crops
            flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
            unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :,
                          np.newaxis].repeat(
                B, target_size, 1, 1)
            unit_grid_y = unit_grid_x.transpose(1, 2)
            unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)
            x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
            scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
            offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
            sampling_grid = unit_grid * scale + offset
            crop = F.grid_sample(x, sampling_grid, align_corners=False)
            if i > 10:
                return crop
        else:
            crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))
            return crop

    def get_my_random_crops(self, x, crop_window=None):
        patch_size, patch_min_scale, patch_max_scale, patch_num_crops = 128, 0.125, 0.25, 8
        my_crops = []
        while len(my_crops) != patch_num_crops:
            crops = self.apply_my_random_crop(x, 128, (0.125, 0.25), num_crops=1)
            print(len(my_crops))
            if crops != None:
                xx = crops[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
                my_crops.append(xx)
                # cv2.imwrite(str(len(my_crops)) + ".png", xx.astype(np.uint8))
        tensors_stacked = torch.stack(tuple(torch.tensor(i).permute(0, 1, 2) for i in my_crops)).permute(0, 3, 1, 2)
        print(tensors_stacked.size())
        return crops

    def get_rand_crops(self, x):
        patch_size, patch_min_scale, patch_max_scale, patch_num_crops = 128, 0.125, 0.25, 8
        my_crops = []
        my_crops_2 = []
        # print("get_my_random_crops",x.shape)
        # print("get_my_random_crops_0",x[0][None,:].shape)
        # print("get_my_random_crops_1",x[1][None,:].shape)
        import cv2
        while len(my_crops) != patch_num_crops:
            crops = util.apply_random_crop(x[0][None, :], 128, (0.125, 0.25), num_crops=1)
            if crops != None:
                xx = crops[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
                # print(crops[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy())
                my_crops.append(xx)
        tensors_stacked = torch.stack(tuple(torch.tensor(i).permute(0, 1, 2) for i in my_crops)).permute(0, 3, 1, 2)
        while len(my_crops_2) != patch_num_crops:
            crops = self.apply_my_random_crop(x[1][None, :], 128, (0.125, 0.25), num_crops=1)
            if crops != None:
                xx = crops[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
                my_crops_2.append(xx)
        tensors_stacked_2 = torch.stack(tuple(torch.tensor(i).permute(0, 1, 2) for i in my_crops_2)).permute(0, 3, 1, 2)
        return torch.stack((tensors_stacked, tensors_stacked_2))

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        import cv2, numpy as np
        # img_real_mask_0 = segmenter.return_building_images_from_tensor(real[0])[0:256,256:512]
        # img_real_mask_0 = cv2.bitwise_not(img_real_mask_0)
        # img_real_mask_1 = segmenter.return_building_images_from_tensor(real[1])[0:256,256:512]
        # res_0 = torch.from_numpy(cv2.bitwise_and(real[0].permute(2,1,0).cpu().numpy().astype(np.uint8),img_real_mask_0))
        # res_1 = torch.from_numpy(cv2.bitwise_and(real[1].permute(2,1,0).cpu().numpy().astype(np.uint8),img_real_mask_1))
        # real = torch.stack((res_0.permute(2,1,0), res_1.permute(2,1,0))).type(torch.cuda.FloatTensor)
        # cv2.imwrite("111_111.png",cv2.normalize(real[0].permute(2,1,0).cpu().numpy(),None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        # cv2.imwrite("222_222.png",cv2.normalize(real[1].permute(2,1,0).cpu().numpy(),None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        # print(real[0].shape)
        # cv2.imwrite("example_1.jpg",segmenter.return_building_images_from_tensor(cv2.normalize(real[0].permute(1,2,0).cpu().numpy(),None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)))
        # cv2.imwrite("example_2.jpg",segmenter.return_building_images_from_tensor(cv2.normalize(real[2].permute(1,2,0).cpu().numpy(),None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)))
        img_real_mask_0 = segmenter.return_building_images_from_tensor(
            cv2.normalize(real[0].permute(1, 2, 0).cpu().numpy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        img_real_mask_0 = cv2.bitwise_not(img_real_mask_0)
        # print(cv2.normalize(real[0].permute(1, 2, 0).cpu().numpy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1).shape)
        # print(img_real_mask_0.shape)
        abc = torch.from_numpy(cv2.bitwise_and(
            cv2.normalize(real[0].permute(1, 2, 0).cpu().numpy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1),
            img_real_mask_0[0:256, 256:512])).permute(2, 1, 0)
        img_real_mask_1 = segmenter.return_building_images_from_tensor(
            cv2.normalize(real[1].permute(1, 2, 0).cpu().numpy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        abc2 = torch.from_numpy(cv2.bitwise_and(
            cv2.normalize(real[1].permute(1, 2, 0).cpu().numpy(), None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1),
            img_real_mask_1[0:256, 256:512])).permute(2, 1, 0)
        # cv2.imwrite("_1_.png",cv2.normalize(real[0].permute(2,1,0).cpu().numpy(),None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        # cv2.imwrite("_2_.png",cv2.normalize(real[1].permute(2,1,0).cpu().numpy(),None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1))
        # cv2.imwrite("_1_mask.png",img_real_mask_0)
        # cv2.imwrite("_2_mask.png",img_real_mask_1)

        # res_0 = torch.from_numpy(cv2.bitwise_and(real[0].permute(2,1,0).cpu().numpy().astype(np.uint8),img_real_mask_0))
        # res_1 = torch.from_numpy(cv2.bitwise_and(real[1].permute(2,1,0).cpu().numpy().astype(np.uint8),img_real_mask_1))
        real = torch.stack((abc, abc2)).type(torch.cuda.FloatTensor)

        real_feat = self.Dpatch.extract_features(self.get_rand_crops(real).cuda(),
                                                 aggregate=self.opt.patch_use_aggregation)
        target_feat = self.Dpatch.extract_features(self.get_rand_crops(real).cuda())

        mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = loss.gan_loss(
            self.Dpatch.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, real):
        self.num_discriminator_iters.add_(1)

        sp, gl = self.E(real)
        B = real.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        rec = self.G(sp[:B // 2], gl[:B // 2])
        mix = self.G(self.swap(sp), gl)

        losses = self.compute_image_discriminator_losses(real, rec, mix)

        if self.opt.lambda_PatchGAN > 0.0:
            patch_losses = self.compute_patch_discriminator_losses(real, mix)
            losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), gl.detach()

    def compute_R1_loss(self, real):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(real).detach()
            target_crop.requires_grad_()

            real_feat = self.Dpatch.extract_features(
                real_crop,
                aggregate=self.opt.patch_use_aggregation)
            target_feat = self.Dpatch.extract_features(target_crop)
            pred_real_patch = self.Dpatch.discriminate_features(
                real_feat, target_feat
            ).sum()

            grad_real, grad_target = torch.autograd.grad(
                outputs=pred_real_patch,
                inputs=[real_crop, target_crop],
                create_graph=True,
                retain_graph=True,
            )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + \
                                grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    def compute_generator_losses(self, real, sp_ma, gl_ma):
        losses, metrics = {}, {}
        B = real.size(0)

        sp, gl = self.E(real)
        rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)

        # record the error of the reconstructed images for monitoring purposes
        metrics["L1_dist"] = self.l1_loss(rec, real[:B // 2])

        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        if self.opt.crop_size >= 1024:
            # another momery-saving trick: reduce #outputs to save memory
            real = real[B // 2:]
            gl = gl[B // 2:]
            sp_mix = sp_mix[B // 2:]

        mix = self.G(sp_mix, gl)

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                self.D(rec),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

            losses["G_GAN_mix"] = loss.gan_loss(
                self.D(mix),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 1.0)

        if self.opt.lambda_PatchGAN > 0.0:
            real_feat = self.Dpatch.extract_features(
                self.get_random_crops(real),
                aggregate=self.opt.patch_use_aggregation).detach()
            mix_feat = self.Dpatch.extract_features(self.get_random_crops(mix))

            losses["G_mix"] = loss.gan_loss(
                self.Dpatch.discriminate_features(real_feat, mix_feat),
                should_be_classified_as_real=True,
            ) * self.opt.lambda_PatchGAN

        return losses, metrics

    def get_visuals_for_snapshot(self, real):
        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
        sp, gl = self.E(real)
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = self.G(sp, gl)
        mix = self.G(sp, self.swap(gl))

        visuals = {"real": real, "layout": layout, "rec": rec, "mix": mix}

        return visuals

    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.E(sample_image)
            self.G(sp, gl)
        noise_var = self.G.fix_and_gather_noise_parameters()
        return noise_var

    def encode(self, image, extract_features=False):
        return self.E(image, extract_features=extract_features)

    def decode(self, spatial_code, global_code):
        return self.G(spatial_code, global_code)

    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) + list(self.E.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams += list(self.D.parameters())
            if self.opt.lambda_PatchGAN > 0.0:
                Dparams += list(self.Dpatch.parameters())
            return Dparams
