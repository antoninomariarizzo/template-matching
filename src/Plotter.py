import os
import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from src.utils import read_rgb_img
from typing import List, Tuple


class Plotter:
    def __init__(self,
                 dpi: int = 100,
                 figsize: Tuple[int, int] = None,
                 base_path: str = None,
                 format: str = 'pdf'):
        self.dpi = dpi
        self.base_path = base_path
        self.format = format
        self.figsize = figsize

    def draw_single_template_square(self,
                                    img_template: np.ndarray,
                                    img_scene: np.ndarray,
                                    H: np.ndarray,
                                    fname: str = None
                                    ) -> None:
        height, width = img_template.shape[:2]

        template_corners = np.asarray([[0, 0],
                                       [width, 0],
                                       [width, height],
                                       [0, height]],
                                      dtype='float32')

        # Convert to homogeneous coordinates
        template_corners_homogeneous = np.hstack((template_corners,
                                                  np.ones((4, 1))))

        projected_template_corners_homogeneous = H @ template_corners_homogeneous.T
        projected_template_corners_homogeneous = projected_template_corners_homogeneous.T

        # Convert to Cartesian coordinates
        projected_template_corners = projected_template_corners_homogeneous[:,
                                                                            :2] / projected_template_corners_homogeneous[:, 2:]

        # Draw the quadrilateral on the scene image
        projected_template_corners = projected_template_corners.astype(int)
        img_scene_with_projection = cv.cvtColor(img_scene, cv.COLOR_RGB2GRAY)
        img_scene_with_projection = cv.cvtColor(img_scene_with_projection,
                                                cv.COLOR_GRAY2RGB)
        cv.polylines(img_scene_with_projection,
                     [projected_template_corners.reshape(-1, 1, 2)],
                     isClosed=True,
                     color=(0, 255, 0),
                     thickness=12)

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax1.imshow(img_scene_with_projection)
        ax1.axis('off')
        ax2.imshow(img_template)
        ax2.axis('off')

        if fname is None:
            plt.show()
        else:
            plt.savefig(self.base_path + fname + '.' + self.format,
                        format=self.format,
                        dpi=self.dpi,
                        bbox_inches='tight', 
                        transparent=True)
            plt.close()

    def draw_template_squares(self,
                              scene_path: str,
                              Hs: List[np.ndarray],
                              matched_template_paths: List[str],
                              fname: str = None
                              ) -> None:
        img_scene = read_rgb_img(scene_path)
        for i, (template_path, H) in enumerate(zip(matched_template_paths, Hs)):
            img_template = read_rgb_img(template_path)
            tname = os.path.splitext(os.path.basename(template_path))[0]
            print(f"{i}. {tname}")

            fname_cur = f"{fname}{i}" if fname is not None else None
            self.draw_single_template_square(img_template, 
                                             img_scene, 
                                             H, 
                                             fname_cur)

    def draw_templates(self,
                       scene_path: str,
                       Hs: List[np.ndarray],
                       matched_template_paths: List[str],
                       fname: str = None
                       ) -> None:
        img_scene = read_rgb_img(scene_path)
        scene_height, scene_width = img_scene.shape[:2]
        img_scene = (img_scene / 2).astype(img_scene.dtype)

        for template_path, H in zip(matched_template_paths, Hs):
            img_template = read_rgb_img(template_path)

            warped_template = cv.warpPerspective(img_template, H,
                                                 (scene_width,
                                                  scene_height))

            mask = cv.cvtColor(warped_template, cv.COLOR_RGB2GRAY) > 0
            img_scene[mask] = warped_template[mask]

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.imshow(img_scene)
        plt.axis('off')

        if fname is None:
            plt.show()
        else:
            plt.savefig(self.base_path + fname + '.' + self.format,
                        format=self.format,
                        dpi=self.dpi,
                        bbox_inches='tight',
                        transparent=True)
            plt.close()

    def show_matched_region(self,
                            template_path: str,
                            H: np.ndarray,
                            scene_path: str,
                            fname: str = None
                            ) -> None:

        img_scene = read_rgb_img(scene_path)
        img_template = read_rgb_img(template_path)
        template_height, template_width = img_template.shape[:2]

        #  Warp scene part to the template
        H_inv = np.linalg.inv(H)
        img_scene_rect = cv.warpPerspective(img_scene, H_inv,
                                            (template_width,
                                             template_height))

        ssim_idx, _ = ssim(img_scene_rect, img_template,
                           multichannel=True, full=True,
                           channel_axis=-1)
        print(f"SSIM Index: {ssim_idx}")

        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        axes[0].imshow(img_template)
        axes[0].axis('off')
        axes[1].imshow(img_scene_rect)
        axes[1].axis('off')

        if fname is None:
            plt.show()
        else:
            plt.savefig(self.base_path + fname + '.' + self.format,
                        format=self.format,
                        dpi=self.dpi,
                        bbox_inches='tight',
                        transparent=True)
            plt.close()

    def scene(self,
             img_path: str,
             fname: str = None
             ) -> None:

        img = read_rgb_img(img_path)

        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.imshow(img)
        plt.axis('off')

        if fname is None:
            plt.show()
        else:
            plt.savefig(self.base_path + fname + '.' + self.format,
                        format=self.format,
                        dpi=self.dpi,
                        bbox_inches='tight',
                        transparent=True)
            plt.close()
    
    def templates(self,
                  img_paths: str,
                  fname: str = None
                  ) -> None:
        fig, axes = plt.subplots(1, len(img_paths),
                                 figsize=self.figsize, 
                                 dpi=self.dpi)

        for i, img_path in enumerate(img_paths):
            img = read_rgb_img(img_path)

            axes[i].imshow(img)
            axes[i].axis('off')

        if fname is None:
            plt.show()
        else:
            plt.savefig(self.base_path + fname + '.' + self.format,
                        format=self.format,
                        dpi=self.dpi,
                        bbox_inches='tight',
                        transparent=True)
            plt.close()
