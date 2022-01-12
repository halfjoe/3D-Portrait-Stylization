# PyTorch implementation of Style Transfer by Relaxed Optimal Transport and Self-Similarity (STROTSS) with improvements

Implements [STROTSS](https://arxiv.org/abs/1904.12785) with sinkhorn EMD as introduced in the paper [Interactive Neural Style Transfer with artists](https://arxiv.org/pdf/2003.06659).

This code is inspired by [the original implementation](https://github.com/nkolkin13/STROTSS) released by the authors of STROTSS.


## Dependencies:
* python3 >= 3.6
* pytorch >= 1.0
* torchvision >= 0.4
* imageio >= 2.2
* numpy >= 1.1

## Usage:

  * standard
    ```
    python test.py -c images/content_im.jpg -s images/style_im.jpg
    ```
  * sinkhorn earth movers distance
    ```
    python test.py -c images/content_im.jpg -s images/style_im.jpg --use_sinkhorn
    ```
  * guidance masks
    ```
    python test.py -c images/content_im.jpg -s images/style_im.jpg --content_guidance images/content_guidance.jpg --style_guidance images/style_guidance
    ```
General usage
```
python test.py
    --content CONTENT
    --style STYLE
    [--output OUTPUT]
    [--content_weight CONTENT_WEIGHT]
    [--max_scale MAX_SCALE]
    [--seed SEED]
    [--content_guidance CONTENT_GUIDANCE]
    [--style_guidance STYLE_GUIDANCE]
    [--print_freq PRINT_FREQ]
    [--use_sinkhorn]
    [--sinkhorn_reg SINKHORN_REG]
    [--sinkhorn_maxiter SINKHORN_MAXITER]
```

## Citation

If you use this code, please cite [the original STROTSS paper](https://arxiv.org/abs/1904.12785) and
```
@article{kerdreux2020interactive,
  title={Interactive Neural Style Transfer with Artists},
  author={Kerdreux, Thomas and Thiry, Louis and Kerdreux, Erwan},
  journal={arXiv preprint arXiv:2003.06659},
  year={2020}
}
```
