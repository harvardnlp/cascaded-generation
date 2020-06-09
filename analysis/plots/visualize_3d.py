import sys
import os
import pickle
import argparse
import copy
import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mplot3d_dragger import Dragger3D

import torch
import seaborn
import numpy as np

def plot_trigram_with_proj(plt_scores, plt_labels_x, plt_labels_y, plt_labels_z, plt_scores_xy=None, plt_scores_yz=None, save_path=None, beam_size=5, animate=False, use_ax=None, setaxis=True, use_norm=None):
    # invert x and z
    plt_labels_x = reversed(copy.deepcopy(plt_labels_x))
    plt_labels_z = reversed(copy.deepcopy(plt_labels_z))
    plt_scores = torch.flip(plt_scores.data.clone(), [0, 2])
    if plt_scores_xy is not None:
        plt_scores_xy = torch.flip(plt_scores_xy.data.clone(), [1])
    if plt_scores_yz is not None:
        plt_scores_yz = torch.flip(plt_scores_yz.data.clone(), [0])

    x_label_pad = 22
    y_label_pad = 30
    z_label_pad = 47
    proj_alpha = 1
    cube_alpha = 1

    mask_cube = plt_scores.ne(-float('inf'))
    mask_yz = mask_cube.any(0)
    mask_xy = mask_cube.any(-1)

    if not animate:
        seaborn.set_context("poster", font_scale=0.85)
        fig = plt.figure(figsize=(10,12), dpi=200)
        plt.rcParams['grid.linewidth'] = 0.
        plt.rcParams['figure.constrained_layout.use'] = False
        ax = fig.gca(projection='3d')

        ax.set_xticks(np.arange(0, beam_size, 1))
        ax.set_yticks(np.arange(0, beam_size, 1))
        ax.set_zticks(np.arange(0, beam_size, 1))
        ax.set_xlim(0, beam_size)
        ax.set_ylim(0, beam_size)
        ax.set_zlim(0, beam_size)
    else:
        assert use_ax is not None
        ax = use_ax

    if plt_scores_xy is not None:
        X = np.arange(0, beam_size+1, 1)
        Y = np.arange(0, beam_size+1, 1)
        X, Y = np.meshgrid(X, Y)
        Z = X * 0 
        plt_mask = plt_scores_xy.eq(-float('inf')) | (~mask_xy.transpose(-1, -2))

        #if (~plt_mask).float().sum().item() < 10:
        #  sys.exit(1)
        plt_scores_xy[plt_mask] = plt_scores_xy.max()
        v = plt_scores_xy.cpu().numpy()
        norm = plt.Normalize()
        colors = plt.cm.cool(norm(v))
        
        colors = torch.Tensor(colors)
        colors[:, :, -1] = proj_alpha
        colors[plt_mask] = 0
        colors = colors.numpy()
        colors[:,:,0]=0.173
        colors[:,:,1]=0.153
        colors[:,:,2]=0.118
        # Plot the surface.
        #surf = ax.plot_surface(X, Y, Z, facecolors=colors, #cmap=cm.coolwarm,
        #                    linewidth=0, zorder=-1e3, shade=True, edgecolors='none')
        

        for i in range(beam_size):
            for j in range(beam_size):
                if not plt_mask[j,i]:
                    #print (j,i)
                    color = np.zeros((1,1,4))
                    color = color + colors[j, i] * 0 
                    color[:,:,-1]=1
                    color[:,:,0]=0.5
                    color[:,:,1]=0.5
                    color[:,:,2]=0.5
                    ddd = 0
                    if j == 0:
                      linew = 0
                      #if i == 9:
                      #  ddd= 0.5
                      #  print ('here')
                    else:
                      linew = 0.5
                    X2 = np.arange(i-ddd, i+2, 1+ddd)
                    Y2 = np.arange(j, j+2, 1)
                    X2, Y2 = np.meshgrid(X2, Y2)
                    Z2 = X2 * 0 
                    surf = ax.plot_surface(X2, Y2, Z2, facecolors=color, #cmap=cm.coolwarm,
                                        linewidth=linew, zorder=-1e3, shade=False, edgecolors=color[0][0])
                    #      
                    #surf.set_edgecolor('k')
    
    if plt_scores_yz is not None:
        plt_mask = plt_scores_yz.eq(-float('inf')) | (~mask_yz.transpose(-1, -2))
        #print ('YZ:', (~plt_mask).float().sum())
        #if (~plt_mask).float().sum().item() < 10:
        #    sys.exit(1)
        plt_scores_yz[plt_mask] = plt_scores_yz.max()
        v = plt_scores_yz.cpu().numpy()
        norm = plt.Normalize()
        colors = plt.cm.cool(norm(v))
        colors = torch.Tensor(colors)
        colors[:, :, -1] = proj_alpha
        colors[plt_mask] = 0
        colors = colors.numpy()
        ##Plot the surface.
        #surf = ax.plot_surface(Z+10, X, Y, facecolors=colors, shade=False, #cmap=cm.coolwarm,
        #                       linewidth=0, edgecolors="none", zorder=-1e3)
        
        for i in range(beam_size):
            for j in range(beam_size):
                if not plt_mask[j,i]:
                    #print (j,i)
                    color = np.zeros((1,1,4))
                    color = color + colors[j, i] * 0
                    color[:,:,-1]=1
                    color[:,:,0]=0.5
                    color[:,:,1]=0.5
                    color[:,:,2]=0.5
                    X2 = np.arange(i, i+2, 1)
                    Y2 = np.arange(j, j+2, 1)
                    X2, Y2 = np.meshgrid(X2, Y2)
                    Z2 = X2 * 0 
                    surf = ax.plot_surface(Z2, X2, Y2, facecolors=color, #cmap=cm.coolwarm,
                                        linewidth=0.5, zorder=-1e3, shade=False, edgecolors=color[0][0], antialiased=True)
              
                #surf.set_edgecolor('k')
    
    v = plt_scores.cpu().numpy()
    voxels = v > -float('inf')
    if use_norm is None:
        norm = plt.Normalize()
    else:
        norm = use_norm
    last_norm = norm
    plt_scores[plt_scores.eq(-float('inf'))] = plt_scores.max()
    v = plt_scores.cpu().numpy()
    colors = plt.cm.cool(norm(v))
    colors = torch.Tensor(colors)
    colors[:, :, :, -1] = cube_alpha
    colors = colors.numpy()
    
    lightsource = None
    ax.voxels(voxels, facecolors=colors, edgecolors="black", shade=True, lightsource=lightsource)
    if setaxis:
        ax.set_xticks(np.arange(0, beam_size, 1)+1)
        ax.set_xticklabels(plt_labels_x, rotation=45)
        ax.set_yticklabels(plt_labels_y, rotation=-40)
        ax.set_zticklabels(plt_labels_z)
        #ax.set_yticks([], [])
        ax.xaxis.set_tick_params(pad=-11)
        ax.yaxis.set_tick_params(pad=24*0)
        ax.zaxis.set_tick_params(pad=-4)
        plt.setp( ax.zaxis.get_majorticklabels(), ha="left" )
        plt.setp( ax.yaxis.get_majorticklabels(), ha="left" )
        #import matplotlib.transforms
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=45) 

        import types # only works for Python 3
        SHIFT = 0.01 # Data coordinates
        for label in ax.xaxis.get_majorticklabels():
            label.customShiftValue = SHIFT
            label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x-self.customShiftValue), label)
            
        SHIFT = -0.005 # Data coordinates
        for label in ax.yaxis.get_majorticklabels():
            label.customShiftValue = SHIFT
            label.set_y = types.MethodType(lambda self, x: matplotlib.text.Text.set_y(self, x-self.customShiftValue), label)
            
        SHIFT = -0.0045 # Data coordinates
        for label in ax.zaxis.get_majorticklabels():
            label.customShiftValue = SHIFT
            label.set_y = types.MethodType(lambda self, x: matplotlib.text.Text.set_y(self, x-self.customShiftValue), label)
        
    ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    bbox = ax.figure.bbox_inches.from_bounds(1, 1, 8, 10)
    #plt.savefig(str(idx) + save_path, bbox_inches=bbox)
    if not animate:
        ax.dist = 13
        plt.savefig(save_path, bbox_inches='tight')
    return last_norm


# plot images
def create_imgs(d, dirname, vocab, pos):
    # plot unigrams
    unigram_scores = d['unigram']['scores']
    unigram_tokens = d['unigram']['tokens']
    plt_scores = unigram_scores[pos].cpu()
    plt_ids = unigram_tokens[pos].cpu().tolist()
    beam_size = plt_scores.size(-1)
    # all scores
    plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
    plt_scores_orig.fill_(-float('inf'))
    
    x_tokens = unigram_tokens[pos, :].cpu().tolist()
    y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
    z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
    plt_scores_x = unigram_scores[pos]
    plt_scores_y = unigram_scores[pos+1]
    plt_scores_z = unigram_scores[pos+2]
    plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
    plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
    plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
    for x_id, x_token in enumerate(x_tokens):
        for y_id, y_token in enumerate(y_tokens):
            for z_id, z_token in enumerate(z_tokens):
                plt_scores_orig[x_id, y_id, z_id] = plt_scores_x[x_id] + plt_scores_y[y_id] + plt_scores_z[z_id]
    plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, save_path=os.path.join(dirname, f'topk{beam_size}_unigram.png'), beam_size=beam_size)

    # plot bigram
    bigram_scores = d['bigram']['scores']
    bigram_tokens = d['bigram']['tokens']
    bigram_postfix_mm = d['bigram']['postfix_mm'] # l K K
    bigram_scores_mm = d['bigram']['scores_mm'] # l K K
    plt_scores = bigram_scores[pos] # beam_size
    plt_ids = bigram_tokens[pos] # beam_size
    
    # all scores
    plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
    plt_scores_orig.fill_(-float('inf'))
    
    x_tokens = unigram_tokens[pos, :].cpu().tolist()
    y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
    z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
    plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
    plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
    plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
    current_tokens = bigram_tokens[pos]
    next_tokens = bigram_tokens[pos+1]
    current_x_tokens = current_tokens[:, 0].cpu().tolist()
    current_y1_tokens = current_tokens[:, 1].cpu().tolist()
    plt_scores_xy = bigram_scores[pos] # beam_size
    current_y2_tokens = next_tokens[:, 0].cpu().tolist()
    current_z_tokens = next_tokens[:, 1].cpu().tolist()
    plt_scores_yz = bigram_scores[pos+1] # beam_size
    
    
    plt_scores_xy_full = plt_scores_xy.new(beam_size, beam_size)
    plt_scores_xy_full.fill_(-float('inf'))
    for plt_score_xy, topx_token, topy_token in zip(plt_scores_xy, current_x_tokens, current_y1_tokens):
        break_flag = False
        for x_id, x_token in enumerate(x_tokens):
            if break_flag:
                break
            for y_id, y_token in enumerate(y_tokens):
                if break_flag:
                    break
                if topx_token == x_token and topy_token == y_token:
                    plt_scores_xy_full[x_id, y_id] = plt_score_xy
                    break_flag = True
                    break
    plt_scores_xy_full = plt_scores_xy_full.transpose(-1,-2)
    
    plt_scores_yz_full = plt_scores_yz.new(beam_size, beam_size)
    plt_scores_yz_full.fill_(-float('inf'))
    for plt_score_yz, topy_token, topz_token in zip(plt_scores_yz, current_y2_tokens, current_z_tokens):
        break_flag = False
        for y_id, y_token in enumerate(y_tokens):
            if break_flag:
                break
            for z_id, z_token in enumerate(z_tokens):
                if break_flag:
                    break
                if topy_token == y_token and topz_token == z_token:
                    plt_scores_yz_full[y_id, z_id] = plt_score_yz
                    break_flag = True
                    break
    plt_scores_yz_full = plt_scores_yz_full.transpose(-1,-2)
    
    
    for plt_score_xy, topx_token, topy1_token in zip(plt_scores_xy, current_x_tokens, current_y1_tokens):
        for plt_score_yz, topy2_token, topz_token in zip(plt_scores_yz, current_y2_tokens, current_z_tokens):
            if topy1_token != topy2_token:
                continue
            topy_token = topy1_token
            break_flag = False
            for x_id, x_token in enumerate(x_tokens):
                if break_flag:
                    break
                for y_id, y_token in enumerate(y_tokens):
                    if break_flag:
                        break
                    for z_id, z_token in enumerate(z_tokens):
                        if topx_token == x_token and topy_token == y_token and topz_token == z_token:
                            plt_scores_orig[x_id, y_id, z_id] = bigram_scores_mm[pos][x_id][y_id] + bigram_scores_mm[pos+1][y_id][z_id] + bigram_postfix_mm[pos+1][:, z_id]
                            break_flag = True
                            break
    
    plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, None, None, os.path.join(dirname, f'topk{beam_size}_bigram.png'), beam_size=beam_size)
    plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, plt_scores_xy_full, plt_scores_yz_full, os.path.join(dirname, f'topk{beam_size}_bigram_2dproj.png'), beam_size=beam_size)

    # plot trigram
    trigram_scores = d['trigram']['scores']
    trigram_tokens = d['trigram']['tokens']
    plt_scores = trigram_scores[pos] # beam_size
    
    x_tokens = unigram_tokens[pos, :].cpu().tolist()
    y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
    z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
    plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
    plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
    plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
    current_tokens = trigram_tokens[pos]
    current_x_tokens = current_tokens[:, 0].cpu().tolist()
    current_y_tokens = current_tokens[:, 1].cpu().tolist()
    current_z_tokens = current_tokens[:, 2].cpu().tolist()
    
    plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
    plt_scores_orig.fill_(-float('inf'))
    
    for plt_score, topx_token, topy_token, topz_token in zip(plt_scores, current_x_tokens, current_y_tokens, current_z_tokens):
        break_flag = False
        for x_id, x_token in enumerate(x_tokens):
            if break_flag:
                break
            for y_id, y_token in enumerate(y_tokens):
                if break_flag:
                    break
                for z_id, z_token in enumerate(z_tokens):
                    if topx_token == x_token and topy_token == y_token and topz_token == z_token:
                        plt_scores_orig[x_id, y_id, z_id] = plt_score
                        break_flag = True
                        break
    plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, save_path=os.path.join(dirname, f'topk{beam_size}_trigram.png'), beam_size=beam_size)
    plt.close('all')


# plot gif
def create_gif(d, dirname, vocab, pos, interval):
    seaborn.set_context("poster", font_scale=0.85)
    fig = plt.figure(figsize=(10,12), dpi=200)
    plt.rcParams['grid.linewidth'] = 0.
    plt.rcParams['figure.constrained_layout.use'] = False
    fig.set_tight_layout(True)
    ax = fig.gca(projection='3d')
    
    unigram_scores = d['unigram']['scores']
    plt_scores = unigram_scores[pos].cpu()
    beam_size = plt_scores.size(-1)
    ax.set_xticks(np.arange(0, beam_size, 1))
    ax.set_yticks(np.arange(0, beam_size, 1))
    ax.set_zticks(np.arange(0, beam_size, 1))
    ax.set_xlim(0, beam_size)
    ax.set_ylim(0, beam_size)
    ax.set_zlim(0, beam_size)
    ax.dist = 13
    
    # plot unigrams
    unigram_scores = d['unigram']['scores']
    unigram_tokens = d['unigram']['tokens']
    plt_scores = unigram_scores[pos].cpu()
    plt_ids = unigram_tokens[pos].cpu().tolist()
    beam_size = plt_scores.size(-1)
    # all scores
    plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
    plt_scores_orig.fill_(-float('inf'))
    
    x_tokens = unigram_tokens[pos, :].cpu().tolist()
    y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
    z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
    plt_scores_x = unigram_scores[pos]
    plt_scores_y = unigram_scores[pos+1]
    plt_scores_z = unigram_scores[pos+2]
    plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
    plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
    plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
    for x_id, x_token in enumerate(x_tokens):
        for y_id, y_token in enumerate(y_tokens):
            for z_id, z_token in enumerate(z_tokens):
                plt_scores_orig[x_id, y_id, z_id] = plt_scores_x[x_id] + plt_scores_y[y_id] + plt_scores_z[z_id]
    
    plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, save_path=f'unigram_{beam_size}_all.png', beam_size=beam_size, animate=True, use_ax=ax)
    [p.remove() for p in reversed(ax.collections)]
    
    def init():
        #do nothing
        pass
    
    global lastnorm
    lastnorm = None
    def update(i):
        global lastnorm
        #print ('called', i)
        if i >= 0:
            if i == 0:
                [p.remove() for p in reversed(ax.collections)]
            # plot unigrams
            unigram_scores = d['unigram']['scores']
            unigram_tokens = d['unigram']['tokens']
            plt_scores = unigram_scores[pos].cpu()
            plt_ids = unigram_tokens[pos].cpu().tolist()
            beam_size = plt_scores.size(-1)
            # all scores
            plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
            plt_scores_orig.fill_(-float('inf'))
    
            x_tokens = unigram_tokens[pos, :].cpu().tolist()
            y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
            z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
            plt_scores_x = unigram_scores[pos]
            plt_scores_y = unigram_scores[pos+1]
            plt_scores_z = unigram_scores[pos+2]
            plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
            plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
            plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
            for x_id, x_token in enumerate(x_tokens):
                for y_id, y_token in enumerate(y_tokens):
                    for z_id, z_token in enumerate(z_tokens):
                        plt_scores_orig[x_id, y_id, z_id] = plt_scores_x[x_id] + plt_scores_y[y_id] + plt_scores_z[z_id]
            if i == 0:
                plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, save_path=f'unigram_{beam_size}_all.png', beam_size=beam_size, animate=True, use_ax=ax, setaxis=False)
        if i >= 1:
            if i == 1:
                [p.remove() for p in reversed(ax.collections)]
            bigram_marginals = d['bigram']['marginals'] # l -by K -by -K
            bigram_postfix_mm = d['bigram']['postfix_mm'] # l K K
            bigram_scores_mm = d['bigram']['scores_mm'] # l K K
            # all scores
            plt_scores_orig = bigram_marginals.new(beam_size, beam_size, beam_size) # beam_size, beam_size, beam_size
            plt_scores_orig.fill_(-float('inf'))
    
            for ii in range(beam_size):
                for jj in range(beam_size):
                    for kk in range(beam_size):
                      plt_scores_orig[ii, jj, kk] = bigram_scores_mm[pos][ii][jj] + bigram_scores_mm[pos+1][jj][kk] + bigram_postfix_mm[pos+1][:, kk]
            if i == 1:
                lastnorm = plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, None, None, f'change_bigram_{beam_size}_all.png', beam_size=beam_size, animate=True, use_ax=ax, setaxis=False)
        if i >= 2:
            if i == 2:
                [p.remove() for p in reversed(ax.collections)]
            # bigram
            bigram_scores = d['bigram']['scores']
            bigram_tokens = d['bigram']['tokens']
            plt_scores = bigram_scores[pos] # beam_size
            plt_ids = bigram_tokens[pos] # beam_size
    
            # all scores
            plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
            plt_scores_orig.fill_(-float('inf'))
    
            x_tokens = unigram_tokens[pos, :].cpu().tolist()
            y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
            z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
            plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
            plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
            plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
            current_tokens = bigram_tokens[pos]
            next_tokens = bigram_tokens[pos+1]
            current_x_tokens = current_tokens[:, 0].cpu().tolist()
            current_y1_tokens = current_tokens[:, 1].cpu().tolist()
            plt_scores_xy = bigram_scores[pos] # beam_size
            current_y2_tokens = next_tokens[:, 0].cpu().tolist()
            current_z_tokens = next_tokens[:, 1].cpu().tolist()
            plt_scores_yz = bigram_scores[pos+1] # beam_size
    
    
            plt_scores_xy_full = plt_scores_xy.new(beam_size, beam_size)
            plt_scores_xy_full.fill_(-float('inf'))
            for plt_score_xy, topx_token, topy_token in zip(plt_scores_xy, current_x_tokens, current_y1_tokens):
                break_flag = False
                for x_id, x_token in enumerate(x_tokens):
                    if break_flag:
                        break
                    for y_id, y_token in enumerate(y_tokens):
                        if break_flag:
                            break
                        if topx_token == x_token and topy_token == y_token:
                            plt_scores_xy_full[x_id, y_id] = plt_score_xy
                            break_flag = True
                            break
            plt_scores_xy_full = plt_scores_xy_full.transpose(-1,-2)
    
            plt_scores_yz_full = plt_scores_yz.new(beam_size, beam_size)
            plt_scores_yz_full.fill_(-float('inf'))
            for plt_score_yz, topy_token, topz_token in zip(plt_scores_yz, current_y2_tokens, current_z_tokens):
                break_flag = False
                for y_id, y_token in enumerate(y_tokens):
                    if break_flag:
                        break
                    for z_id, z_token in enumerate(z_tokens):
                        if break_flag:
                            break
                        if topy_token == y_token and topz_token == z_token:
                            plt_scores_yz_full[y_id, z_id] = plt_score_yz
                            break_flag = True
                            break
            plt_scores_yz_full = plt_scores_yz_full.transpose(-1,-2)
    
    
            for plt_score_xy, topx_token, topy1_token in zip(plt_scores_xy, current_x_tokens, current_y1_tokens):
                for plt_score_yz, topy2_token, topz_token in zip(plt_scores_yz, current_y2_tokens, current_z_tokens):
                    if topy1_token != topy2_token:
                        continue
                    topy_token = topy1_token
                    break_flag = False
                    for x_id, x_token in enumerate(x_tokens):
                        if break_flag:
                            break
                        for y_id, y_token in enumerate(y_tokens):
                            if break_flag:
                                break
                            for z_id, z_token in enumerate(z_tokens):
                                if topx_token == x_token and topy_token == y_token and topz_token == z_token:
                                    plt_scores_orig[x_id, y_id, z_id] = bigram_scores_mm[pos][x_id][y_id] + bigram_scores_mm[pos+1][y_id][z_id] + bigram_postfix_mm[pos+1][:, z_id]
                                    break_flag = True
                                    break
    
            if i == 2:
                plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, None, None, f'bigram_{beam_size}_all.png', beam_size=beam_size, animate=True, use_ax=ax, setaxis=False, use_norm=lastnorm)
        if i >= 3:
            if i == 3:
                [p.remove() for p in reversed(ax.collections)]
            # bigram
            bigram_scores = d['bigram']['scores']
            bigram_tokens = d['bigram']['tokens']
            trigram_marginals = d['trigram']['marginals']
            plt_scores = bigram_scores[pos] # beam_size
            plt_ids = bigram_tokens[pos] # beam_size
    
            # all scores
            plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
            plt_scores_orig.fill_(-float('inf'))
    
            x_tokens = unigram_tokens[pos, :].cpu().tolist()
            y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
            z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
            plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
            plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
            plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
            current_tokens = bigram_tokens[pos]
            next_tokens = bigram_tokens[pos+1]
            current_x_tokens = current_tokens[:, 0].cpu().tolist()
            current_y1_tokens = current_tokens[:, 1].cpu().tolist()
            #plt_scores_xy = bigram_scores[pos] # beam_size
            current_y2_tokens = next_tokens[:, 0].cpu().tolist()
            current_z_tokens = next_tokens[:, 1].cpu().tolist()
            #plt_scores_yz = bigram_scores[pos+1] # beam_size
    
            for idx1, (topx_token, topy1_token) in enumerate(zip(current_x_tokens, current_y1_tokens)):
                for idx2, (topy2_token, topz_token) in enumerate(zip(current_y2_tokens, current_z_tokens)):
                    if topy1_token != topy2_token:
                        continue
                    topy_token = topy1_token
                    break_flag = False
                    for x_id, x_token in enumerate(x_tokens):
                        if break_flag:
                            break
                        for y_id, y_token in enumerate(y_tokens):
                            if break_flag:
                                break
                            for z_id, z_token in enumerate(z_tokens):
                                if topx_token == x_token and topy_token == y_token and topz_token == z_token:
                                    plt_scores_orig[x_id, y_id, z_id] = trigram_marginals[pos][idx1, idx2]
                                    break_flag = True
                                    break
    
            if i == 3:
                lastnorm = plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, None, None, f'bigram_{beam_size}_all.png', beam_size=beam_size, animate=True, use_ax=ax, setaxis=False)
        if i >= 4:
            if i == 4:
                [p.remove() for p in reversed(ax.collections)]
            trigram_scores = d['trigram']['scores']
            trigram_tokens = d['trigram']['tokens']
            plt_scores = trigram_scores[pos] # beam_size
    
            x_tokens = unigram_tokens[pos, :].cpu().tolist()
            y_tokens = unigram_tokens[pos+1, :].cpu().tolist()
            z_tokens = unigram_tokens[pos+2, :].cpu().tolist()
            plt_labels_x = [vocab[item].replace('&apos;', "'") for item in x_tokens]
            plt_labels_y = [vocab[item].replace('&apos;', "'") for item in y_tokens]
            plt_labels_z = [vocab[item].replace('&apos;', "'") for item in z_tokens]
    
            current_tokens = trigram_tokens[pos]
            current_x_tokens = current_tokens[:, 0].cpu().tolist()
            current_y_tokens = current_tokens[:, 1].cpu().tolist()
            current_z_tokens = current_tokens[:, 2].cpu().tolist()
    
            plt_scores_orig = plt_scores.new(plt_scores.size(-1), plt_scores.size(-1), plt_scores.size(-1)) # beam_size, beam_size, beam_size
            plt_scores_orig.fill_(-float('inf'))
            
            for plt_score, topx_token, topy_token, topz_token in zip(plt_scores, current_x_tokens, current_y_tokens, current_z_tokens):
                break_flag = False
                for x_id, x_token in enumerate(x_tokens):
                    if break_flag:
                        break
                    for y_id, y_token in enumerate(y_tokens):
                        if break_flag:
                            break
                        for z_id, z_token in enumerate(z_tokens):
                            if topx_token == x_token and topy_token == y_token and topz_token == z_token:
                                plt_scores_orig[x_id, y_id, z_id] = plt_score
                                break_flag = True
                                break
            if i == 4:
                plot_trigram_with_proj(plt_scores_orig, plt_labels_x, plt_labels_y, plt_labels_z, save_path=f'trigram_{beam_size}.png', beam_size=beam_size, animate=True, use_ax=ax, setaxis=False, use_norm=lastnorm)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        fig.canvas.draw_idle()
        #print ('here', i, len(ax.collections))
        return ax
    
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    bbox = fig.bbox_inches.from_bounds(1, 1, 8, 10)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 5), interval=interval, blit=False, init_func=init)
    filename_tmp = os.path.join(dirname, '_cascade.gif')
    anim.save(filename_tmp, dpi=80, writer='imagemagick')
    filename = os.path.join(dirname, f'topk{beam_size}_cascade.gif')
    os.system(f'convert {filename_tmp} -coalesce -repage 0x0 -crop 600x600+130+210 +repage {filename}')
    os.remove(filename_tmp)
    plt.close('all')


#!convert -coalesce output2.gif xx2_%01d.png

def parse_args(args):
    parser = argparse.ArgumentParser(description='visualize_3d')
    parser.add_argument('--dump-vis-path', type=str, required=True,
                        help=('Dump data for visualization purposes'))
    parser.add_argument('--output-dir', type=str, required=True,
                        help=('Output directory.'))
    parser.add_argument('--start-position', type=int, default=0,
                        help=('Output directory.'))
    parser.add_argument('--frame-interval', type=int, default=1500,
                        help=('Output directory.'))
    return parser.parse_args(args)

def main(args):
    parameters = parse_args(args)
    checkpoint = pickle.load(open(parameters.dump_vis_path, 'rb'))
    vocab = checkpoint['vocab']
    data = checkpoint['data']

    print (f'Processing {len(data)} sentences')
    for d in tqdm.tqdm(data):
        idx = d['id']
        dirname = os.path.join(parameters.output_dir, str(idx.item()))
        os.makedirs(dirname, exist_ok=True)

        create_gif(d, dirname, vocab, parameters.start_position, parameters.frame_interval)
        create_imgs(d, dirname, vocab, parameters.start_position)

if __name__ == '__main__':
    main(sys.argv[1:])
