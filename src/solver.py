import cv2
import imageio
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import optim

from hed import *
from model import G12, G21
from model import D1, D2


class Solver(object):
    def __init__(self, config, photo_loader, washink_loader):
        self.photo_loader = photo_loader
        self.washink_loader = washink_loader
        self.g12 = None
        self.g21 = None
        self.d1 = None
        self.d2 = None
        self.hed = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.rec_loss_weight = config.rec_loss_weight
        self.edge_loss_weight = config.edge_loss_weight
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        self.model_path = config.model_path
        self.sample_count = config.sample_count
        self.build_model()
        
    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g12 = G12(conv_dim=self.g_conv_dim)
        self.g21 = G21(conv_dim=self.g_conv_dim)
        self.d1 = D1(conv_dim=self.d_conv_dim)
        self.d2 = D2(conv_dim=self.d_conv_dim)
        self.hed = Hed()
        
        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())
        
        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        
        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.d1.cuda()
            self.d2.cuda()
            self.hed.cuda()
    
    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size * self.sample_count))
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1, 2, 0)
    
    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)
    
    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()
    
    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        photo_iter = iter(self.photo_loader)
        washink_iter = iter(self.washink_loader)
        iter_per_epoch = min(len(photo_iter), len(washink_iter))
        
        # fixed washink and photo for sampling
        fixed_photo = self.to_var(torch.cat([photo_iter.next()[0] for _ in range(self.sample_count)]))
        fixed_washink = self.to_var(torch.cat([washink_iter.next()[0] for _ in range(self.sample_count)]))
        
        for step in range(self.train_iters+1):
            # reset data_iter for each epoch
            if step % iter_per_epoch == 0:
                washink_iter = iter(self.washink_loader)
                photo_iter = iter(self.photo_loader)
            
            # load photo and washink dataset
            photo, p_labels = photo_iter.next() 
            photo, p_labels = self.to_var(photo), self.to_var(p_labels).long().squeeze()
            washink, w_labels = washink_iter.next() 
            washink, w_labels = self.to_var(washink), self.to_var(w_labels)
            
            #============ train D ============#
            
            # train with real images
            self.reset_grad()
            out = self.d1(washink)
            d1_loss = torch.mean((out-1)**2)
            
            out = self.d2(photo)
            d2_loss = torch.mean((out-1)**2)
            
            d_washink_loss = d1_loss
            d_photo_loss = d2_loss
            d_real_loss = d1_loss + d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()
            
            # train with fake images
            self.reset_grad()
            fake_photo = self.g12(washink)
            out = self.d2(fake_photo)
            d2_loss = torch.mean(out**2)
            
            fake_washink = self.g21(photo)
            out = self.d1(fake_washink)
            d1_loss = torch.mean(out**2)
            
            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()
            
            #============ train G ============#
            
            # train washink-photo-washink cycle
            self.reset_grad()
            fake_photo = self.g12(washink)
            out = self.d2(fake_photo)
            reconst_washink = self.g21(fake_photo)
            g_loss = torch.mean((out-1)**2)

            # reconstruction loss
            g_loss += self.rec_loss_weight * torch.mean((washink - reconst_washink)**2)

            g_loss.backward()
            self.g_optimizer.step()

            # train photo-washink-photo cycle
            self.reset_grad()
            fake_washink = self.g21(photo)
            out = self.d1(fake_washink)
            reconst_photo = self.g12(fake_washink)
            g_loss = torch.mean((out-1)**2)

            # reconstruction loss
            g_loss += self.rec_loss_weight * torch.mean((photo - reconst_photo)**2)

            # edge loss
            edge_real_A = torch.sigmoid(self.hed(photo).detach())
            edge_fake_B = torch.sigmoid(self.hed(fake_washink))
            g_loss += no_sigmoid_cross_entropy(edge_fake_B, edge_real_A) * self.edge_loss_weight

            g_loss.backward()
            self.g_optimizer.step()
            
            # print the log info
            if (step+1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_washink_loss: %.4f, d_photo_loss: %.4f, '
                      'd_fake_loss: %.4f, g_loss: %.4f'
                      % (step+1, self.train_iters, d_real_loss.data.item(), d_washink_loss.data.item(), 
                        d_photo_loss.item(), d_fake_loss.item(), g_loss.item()))

            # save the sampled images
            if (step+1) % self.sample_step == 0:
                fake_photo = self.g12(fixed_washink)
                fake_washink = self.g21(fixed_photo)
                
                washink, fake_washink = self.to_data(fixed_washink), self.to_data(fake_washink)
                photo, fake_photo = self.to_data(fixed_photo), self.to_data(fake_photo)
                
                merged = self.merge_images(washink, fake_photo)
                path = os.path.join(self.sample_path, 'sample-%d-w-p.png' %(step+1))
                imageio.imsave(path, merged)
                print ('saved %s' %path)
                
                merged = self.merge_images(photo, fake_washink)
                path = os.path.join(self.sample_path, 'sample-%d-p-w.png' %(step+1))
                imageio.imsave(path, merged)
                print ('saved %s' %path)
            
            if (step+1) % 5000 == 0:
                # save the model parameters for each epoch
                g12_path = os.path.join(self.model_path, 'g12-%d.pkl' %(step+1))
                g21_path = os.path.join(self.model_path, 'g21-%d.pkl' %(step+1))
                d1_path = os.path.join(self.model_path, 'd1-%d.pkl' %(step+1))
                d2_path = os.path.join(self.model_path, 'd2-%d.pkl' %(step+1))
                torch.save(self.g12.state_dict(), g12_path)
                torch.save(self.g21.state_dict(), g21_path)
                torch.save(self.d1.state_dict(), d1_path)
                torch.save(self.d2.state_dict(), d2_path)

    def sample(self):
        self.g21.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.g21.eval()
        for i, (image, _) in enumerate(self.photo_loader):
            imageio.imsave(os.path.join(self.sample_path, f'{i}_photo.png'), np.transpose(image[0], (1, 2, 0)))
            fake = np.transpose(self.to_data(self.g21(image))[0], (1, 2, 0))
            fake = cv2.bilateralFilter(fake, 3, sigmaSpace = 75, sigmaColor =75)
            imageio.imsave(os.path.join(self.sample_path, f'{i}.png'), fake)

    def gen_mobile_model(self):
        self.g21.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.g21.eval()
        example = torch.rand(1, 3, 256, 256)
        traced_script_module = torch.jit.trace(self.g21, example)
        from torch.utils.mobile_optimizer import optimize_for_mobile
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        traced_script_module_optimized.save("g21.pt")
