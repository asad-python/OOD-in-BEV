import src.train
import src.explore
from fire import Fire

def main():
    Fire({
        'train': src.train.train,
        'eval_model_iou': src.explore.eval_model_iou,
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,
        'viz_model_preds': src.explore.viz_model_preds,
        'visualize_grid_feature_map': src.explore.visualize_grid_feature_map,
        'evaluate_with_masks': lambda version, modelf, dataroot, map_folder=None: src.explore.evaluate_with_masks(
            version, modelf, dataroot, src.explore.mask_generator, map_folder)
    })

if __name__ == '__main__':
    main()
