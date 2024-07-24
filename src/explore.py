import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
import os
from .data import compile_data
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    gen_dx_bx, get_nusc_maps, plot_nusc_map, add_ego, cam_to_ego, get_val_info, SimpleLoss)
from .models import compile_model

def lidar_check(version,
                dataroot='/data/nuscenes',
                show_lidar=True,
                viz_train=False,
                nepochs=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=1,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='vizdata')

    loader = trainloader if viz_train else valloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to('cpu')  # Ensure the model is on the CPU

    rat = H / W
    val = 10.1
    fig = plt.figure(figsize=(val + val/3*2*rat*3, val/3*2*rat))
    gs = mpl.gridspec.GridSpec(2, 6, width_ratios=(1, 1, 1, 2*rat, 2*rat, 2*rat))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    for epoch in range(nepochs):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, pts, binimgs) in enumerate(loader):

            img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)

            for si in range(imgs.shape[0]):
                plt.clf()
                final_ax = plt.subplot(gs[:, 5:6])
                for imgi, img in enumerate(imgs[si]):
                    ego_pts = ego_to_cam(pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                    mask = get_only_in_img_mask(ego_pts, H, W)
                    plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    plt.imshow(showimg)
                    if show_lidar:
                        plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask],
                                s=5, alpha=0.1, cmap='jet')
                    plt.axis('off')

                    plt.sca(final_ax)
                    plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.', label=cams[imgi].replace('_', ' '))

                plt.legend(loc='upper right')
                final_ax.set_aspect('equal')
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))

                ax = plt.subplot(gs[:, 3:4])
                plt.scatter(pts[si, 0], pts[si, 1], c=pts[si, 2], vmin=-5, vmax=5, s=5)
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))
                ax.set_aspect('equal')

                save_dir = 'results/Lidar_Viz_Preds'
                os.makedirs(save_dir, exist_ok=True)

                ax = plt.subplot(gs[:, 4:5])
                plt.imshow(binimgs[si].squeeze(0).T, origin='lower', cmap='Greys', vmin=0, vmax=1)
                imname = os.path.join(save_dir, f'lcheck{epoch:03}_{batchi:05}_{si:02}.jpg')

                print('saving', imname)
                plt.savefig(imname)


# def eval_model_iou(version,
#                 modelf,
#                 dataroot='/data/nuscenes',
#
#                 H=900, W=1600,
#                 resize_lim=(0.193, 0.225),
#                 final_dim=(128, 352),
#                 bot_pct_lim=(0.0, 0.22),
#                 rot_lim=(-5.4, 5.4),
#                 rand_flip=True,
#
#                 xbound=[-50.0, 50.0, 0.5],
#                 ybound=[-50.0, 50.0, 0.5],
#                 zbound=[-10.0, 10.0, 20.0],
#                 dbound=[4.0, 45.0, 1.0],
#
#                 bsz=4,
#                 nworkers=10,
#                 ):
#     grid_conf = {
#         'xbound': xbound,
#         'ybound': ybound,
#         'zbound': zbound,
#         'dbound': dbound,
#     }
#     data_aug_conf = {
#                     'resize_lim': resize_lim,
#                     'final_dim': final_dim,
#                     'rot_lim': rot_lim,
#                     'H': H, 'W': W,
#                     'rand_flip': rand_flip,
#                     'bot_pct_lim': bot_pct_lim,
#                     'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
#                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
#                     'Ncams': 5,
#                 }
#     trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
#                                           grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
#                                           parser_name='segmentationdata')
#
#     device = torch.device('cpu')  # Ensure using CPU
#
#     model = compile_model(grid_conf, data_aug_conf, outC=1)
#     print('loading', modelf)
#     model.load_state_dict(torch.load(modelf, map_location=device))
#     model.to(device)
#
#     loss_fn = SimpleLoss(1.0).cpu()  # Ensure loss function is on CPU
#
#     model.eval()
#     val_info = get_val_info(model, valloader, loss_fn, device)
#     print(val_info)


def eval_model_iou(version,
                   modelf,
                   dataroot='/data/nuscenes',
                   H=900, W=1600,
                   resize_lim=(0.193, 0.225),
                   final_dim=(128, 352),
                   bot_pct_lim=(0.0, 0.22),
                   rot_lim=(-5.4, 5.4),
                   rand_flip=True,
                   xbound=[-50.0, 50.0, 0.5],
                   ybound=[-50.0, 50.0, 0.5],
                   zbound=[-10.0, 10.0, 20.0],
                   dbound=[4.0, 45.0, 1.0],
                   bsz=4,
                   nworkers=10):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 5,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu')  # Ensure using CPU

    model = compile_model(grid_conf, data_aug_conf, outC=3)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf, map_location=device))
    model.to(device)

    loss_fn = SimpleLoss().cpu()  # Ensure loss function is on CPU

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, device)
    print(val_info)


# def viz_model_preds(version,
#                     modelf,
#                     dataroot='/data/nuscenes',
#                     map_folder='/data/nuscenes/mini',
#
#                     viz_train=False,
#
#                     H=900, W=1600,
#                     resize_lim=(0.193, 0.225),
#                     final_dim=(128, 352),
#                     bot_pct_lim=(0.0, 0.22),
#                     rot_lim=(-5.4, 5.4),
#                     rand_flip=True,
#
#                     xbound=[-50.0, 50.0, 0.5],
#                     ybound=[-50.0, 50.0, 0.5],
#                     zbound=[-10.0, 10.0, 20.0],
#                     dbound=[4.0, 45.0, 1.0],
#
#                     bsz=4,
#                     nworkers=10,
#                     ):
#     grid_conf = {
#         'xbound': xbound,
#         'ybound': ybound,
#         'zbound': zbound,
#         'dbound': dbound,
#     }
#     cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
#             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
#     data_aug_conf = {
#                     'resize_lim': resize_lim,
#                     'final_dim': final_dim,
#                     'rot_lim': rot_lim,
#                     'H': H, 'W': W,
#                     'rand_flip': rand_flip,
#                     'bot_pct_lim': bot_pct_lim,
#                     'cams': cams,
#                     'Ncams': 5,
#                 }
#     trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
#                                           grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
#                                           parser_name='segmentationdata')
#     loader = trainloader if viz_train else valloader
#     nusc_maps = get_nusc_maps(map_folder)
#
#     device = torch.device('cpu')  # Ensure using CPU
#
#     model = compile_model(grid_conf, data_aug_conf, outC=1)
#     print('loading', modelf)
#     model.load_state_dict(torch.load(modelf, map_location=device))
#     model.to(device)
#
#     dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
#     dx, bx = dx[:2].numpy(), bx[:2].numpy()
#
#     scene2map = {}
#     for rec in loader.dataset.nusc.scene:
#         log = loader.dataset.nusc.get('log', rec['log_token'])
#         scene2map[rec['name']] = log['location']
#
#     val = 0.01
#     fH, fW = final_dim
#     fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
#     gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
#     gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
#
#     model.eval()
#     counter = 0
#     with torch.no_grad():
#         for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
#             out = model(imgs.to(device),
#                     rots.to(device),
#                     trans.to(device),
#                     intrins.to(device),
#                     post_rots.to(device),
#                     post_trans.to(device),
#                     )
#             out = out.sigmoid().cpu()
#
#             for si in range(imgs.shape[0]):
#                 plt.clf()
#                 for imgi, img in enumerate(imgs[si]):
#                     ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
#                     showimg = denormalize_img(img)
#                     # flip the bottom images
#                     if imgi > 2:
#                         showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
#                     plt.imshow(showimg)
#                     plt.axis('off')
#                     plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')
#
#                 ax = plt.subplot(gs[0, :])
#                 ax.get_xaxis().set_ticks([])
#                 ax.get_yaxis().set_ticks([])
#                 plt.setp(ax.spines.values(), color='b', linewidth=2)
#                 plt.legend(handles=[
#                     mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
#                     mpatches.Patch(color='#76b900', label='Ego Vehicle'),
#                     mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
#                 ], loc=(0.01, 0.86))
#                 plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')
#
#                 # Ensure the directory exists
#                 save_dir = 'results/viz_model_preds'
#                 os.makedirs(save_dir, exist_ok=True)
#
#                 # plot static map (improves visualization)
#                 rec = loader.dataset.ixes[counter]
#                 plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
#                 plt.xlim((out.shape[3], 0))
#                 plt.ylim((0, out.shape[3]))
#                 add_ego(bx, dx)
#
#                 imname = os.path.join(save_dir, f'eval{batchi:06}_{si:03}.jpg')
#                 print('saving', imname)
#                 plt.savefig(imname)
#                 counter += 1

def viz_model_preds(version,
                    modelf,
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    viz_train=False,
                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,
                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],
                    bsz=4,
                    nworkers=10):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': cams,
        'Ncams': 5,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device('cpu')  # Ensure using CPU

    model = compile_model(grid_conf, data_aug_conf, outC=3)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf, map_location=device))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            out = out.sigmoid().cpu()

            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

                # Ensure the directory exists
                save_dir = 'results/viz_model_preds'
                os.makedirs(save_dir, exist_ok=True)

                # plot static map (improves visualization)
                rec = loader.dataset.ixes[counter]
                plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                imname = os.path.join(save_dir, f'eval{batchi:06}_{si:03}.jpg')
                print('saving', imname)
                plt.savefig(imname)
                counter += 1


def cumsum_check(version,
                dataroot='/data/nuscenes',

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=4,
                nworkers=10,
                ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': 6,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu')  # Ensure using CPU
    loader = trainloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.eval()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):

        model.use_quickcumsum = False
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('autograd:    ', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())

        model.use_quickcumsum = True
        model.zero_grad()
        out = model(imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
                )
        out.mean().backward()
        print('quick cumsum:', out.mean().detach().item(), model.camencode.depthnet.weight.grad.mean().item())
        print()



def visualize_grid_feature_map(version,
                               modelf,
                               dataroot='/data/nuscenes',
                               map_folder='/data/nuscenes/mini',
                               H=900, W=1600,
                               resize_lim=(0.193, 0.225),
                               final_dim=(128, 352),
                               bot_pct_lim=(0.0, 0.22),
                               rot_lim=(-5.4, 5.4),
                               rand_flip=True,
                               xbound=[-50.0, 50.0, 0.5],
                               ybound=[-50.0, 50.0, 0.5],
                               zbound=[-10.0, 10.0, 20.0],
                               dbound=[4.0, 45.0, 1.0],
                               bsz=4,
                               nworkers=10,
                               save_dir='results/visualizations'):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define grid and data augmentation configurations
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 5,
    }

    # Compile data
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='vizdata')
    loader = valloader
    nusc_maps = get_nusc_maps(map_folder)

    # Load the model
    device = torch.device('cpu')
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    model.load_state_dict(torch.load(modelf, map_location=device))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.ixes:
        log = loader.dataset.nusc.get('log', loader.dataset.nusc.get('scene', rec['scene_token'])['log_token'])
        scene2map[loader.dataset.nusc.get('scene', rec['scene_token'])['name']] = log['location']

    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3 * fW * val, (1.5 * fW + 2 * fH) * val))
    gs = mpl.gridspec.GridSpec(3, 3, height_ratios=(1.5 * fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            # Get the grid map before BEV encoder
            geom_feats = model.get_geometry(rots, trans, intrins, post_rots, post_trans)
            x = model.get_cam_feats(imgs.to(device))
            grid_map = model.voxel_pooling(geom_feats, x)

            # Forward pass through BEV encoder
            out = model.bevencode(grid_map)
            out = out.sigmoid().cpu()

            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # flip the bottom images
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(data_aug_conf['cams'][imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                # Plot the intermediate grid map
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0), label='Intermediate Grid Map'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8), label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                plt.imshow(grid_map[si].sum(dim=0).cpu().numpy(), vmin=0, vmax=1, cmap='Blues')

                # plot static map (improves visualization)
                rec = loader.dataset.ixes[counter]
                plot_nusc_map(rec, nusc_maps, loader.dataset.nusc, scene2map, dx, bx)
                plt.xlim((grid_map.shape[-1], 0))
                plt.ylim((0, grid_map.shape[-1]))
                add_ego(bx, dx)

                # Save intermediate grid map
                imname = os.path.join(save_dir, f'GMap_{batchi:06}_{si:03}.jpg')
                os.makedirs(os.path.dirname(imname), exist_ok=True)
                print('saving', imname)
                plt.savefig(imname)

                # Plot the final output map
                ax = plt.subplot(gs[2, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='r', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(1.0, 0.0, 0.0, 1.0), label='Final Segmentation Result')
                ], loc=(0.01, 0.86))
                plt.imshow(out[si].squeeze(0).cpu().numpy(), vmin=0, vmax=1, cmap='Reds')

                # Save final output map
                imname = os.path.join(save_dir, f'Out_{batchi:06}_{si:03}.jpg')
                os.makedirs(os.path.dirname(imname), exist_ok=True)
                print('saving', imname)
                plt.savefig(imname)

                # Project the segmentation results back onto the original images
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    if imgi > 2:
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)

                    # Project grid map back to the camera view
                    geom = geom_feats[si, imgi].reshape(-1, 3).T  # Reshape geom_feats to 3xN
                    points = cam_to_ego(geom, rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                    mask = get_only_in_img_mask(points, H, W)

                    # Adjust the shape of the mask and the indexed tensor from grid_map
                    flat_grid_map = grid_map[si].sum(dim=0).view(-1)
                    valid_points = points[:, mask]

                    if valid_points.shape[1] > flat_grid_map.shape[0]:
                        valid_points = valid_points[:, :flat_grid_map.shape[0]]
                        mask = mask[:flat_grid_map.shape[0]]

                    plt.scatter(valid_points[0], valid_points[1], c=flat_grid_map[mask].cpu().numpy(), s=1, cmap='Reds',
                                alpha=0.5)
                    plt.axis('off')
                    plt.annotate(data_aug_conf['cams'][imgi].replace('_', ' '), (0.01, 0.92), xycoords='axes fraction')

                imname = os.path.join(save_dir, f'Proj_{batchi:06}_{si:03}.jpg')
                os.makedirs(os.path.dirname(imname), exist_ok=True)
                print('saving', imname)
                plt.savefig(imname)

                counter += 1


# def evaluate_with_masks(version, modelf, dataroot, mask_generator, map_folder=None):
#     grid_conf = {
#         'xbound': [-50.0, 50.0, 0.5],
#         'ybound': [-50.0, 50.0, 0.5],
#         'zbound': [-10.0, 10.0, 20.0],
#         'dbound': [4.0, 45.0, 1.0],
#     }
#     data_aug_conf = {
#         'resize_lim': (0.193, 0.225),
#         'final_dim': (128, 352),
#         'rot_lim': (-5.4, 5.4),
#         'H': 900, 'W': 1600,
#         'rand_flip': True,
#         'bot_pct_lim': (0.0, 0.22),
#         'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
#         'Ncams': 5,
#     }
#
#     trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf, grid_conf=grid_conf, bsz=4,
#                                           nworkers=10, parser_name='segmentationdata')
#     device = torch.device('cpu')
#
#     model = compile_model(grid_conf, data_aug_conf, outC=1)
#     print('loading', modelf)
#     model.load_state_dict(torch.load(modelf, map_location=device))
#     model.to(device)
#
#     loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])).cpu()
#     model.eval()
#
#     # Generate masks
#     masks = mask_generator(valloader)
#
#     val_info = get_val_info(model, valloader, loss_fn, device, masks=masks)
#     print(val_info)


def evaluate_with_masks(version, modelf, dataroot, mask_generator, map_folder=None):
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {
        'resize_lim': (0.193, 0.225),
        'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4),
        'H': 900, 'W': 1600,
        'rand_flip': True,
        'bot_pct_lim': (0.0, 0.22),
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 5,
    }

    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf, grid_conf=grid_conf, bsz=4,
                                          nworkers=10, parser_name='segmentationdata')
    device = torch.device('cpu')

    model = compile_model(grid_conf, data_aug_conf, outC=3)  # Initialize with outC=3
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss().cpu()

    # Generate masks
    masks = mask_generator(valloader)

    val_info = get_val_info(model, valloader, loss_fn, device, masks=masks)
    print(val_info)




def mask_generator(dataloader):
    """
    Generate masks for the vehicles to be patched out.
    :param dataloader: DataLoader for the dataset
    :return: Tensor of masks
    """
    masks = []
    for batch in dataloader:
        _, _, _, _, _, _, binimgs = batch
        mask = (binimgs == 0).float()  # Adjust to create masks for the unknown class
        masks.append(mask)
    masks = torch.cat(masks, dim=0)
    return masks


# def mask_generator(dataloader):
#     """
#     Generate masks for the vehicles to be patched out.
#     :param dataloader: DataLoader for the dataset
#     :return: Tensor of masks
#     """
#     masks = []
#     for batch in dataloader:
#         _, _, _, _, _, _, binimgs = batch
#         mask = (binimgs == 0).float()  # Original mask where vehicles are present
#         masks.append(mask)
#     masks = torch.cat(masks, dim=0)
#     return masks

