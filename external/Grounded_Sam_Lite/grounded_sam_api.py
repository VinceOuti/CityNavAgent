import sys
sys.path.append(".GroundingDINO")

from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T

from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

import os
import cv2
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


class GroundedSam:
    def __init__(
            self,
            dino_checkpoint_path,
            sam_checkpoint_path,
            dino_config_path="external/Grounded_Sam_Lite/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            device='cuda'
    ):
        self.device = device
        self.dino = self.load_dino_model(dino_config_path, dino_checkpoint_path, bert_base_uncased_path=None, device=device)
        self.sam = SamPredictor(sam_model_registry['vit_h'](checkpoint=sam_checkpoint_path).to(device))

        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_dino_model(self, model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        args.bert_base_uncased_path = bert_base_uncased_path
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


    def get_dino_output(self, image, caption, box_threshold, text_threshold, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = self.dino.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=5))
        ax.text(x0, y0, label)

    def greedy_mask_predict(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25, visualize=False):
        seg_success = True

        h, w = image.shape[:2]
        in_phrases = text_prompt.split(".")
        in_phrases = [inp.strip(" ") for inp in in_phrases]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil, _ = self.transform(image_pil, None)

        pred_boxes, pred_phrases = self.get_dino_output(image_pil, text_prompt, box_threshold, text_threshold, with_logits=False)
        pred_boxes = pred_boxes.cpu()

        if len(pred_phrases) == 0:
            seg_success = False
            return np.zeros_like((h, w), dtype=np.bool_), seg_success

        best_phrase = "<inf>"
        best_bboxes = []

        def _soft_contain(ele, ele_list):
            res = False
            for e in ele_list:
                if ele in e:
                    res = True
                    break
            return res

        for inp in in_phrases:
            if _soft_contain(inp, pred_phrases):
                best_phrase = inp
                break

        # print(f"{best_phrase}, {in_phrases}, {pred_phrases}")
        # no object is segmented
        if best_phrase not in in_phrases:
            seg_success = False
            return np.zeros_like((h, w), dtype=np.bool_), seg_success

        # collect boxes
        for i, pp in enumerate(pred_phrases):
            if best_phrase in pp:
                best_bboxes.append(pred_boxes[i:i+1, :])

        boxes_filt = torch.cat(best_bboxes, dim=0)
        self.sam.set_image(image)
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([w, h, w, h])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        if visualize:
            print("visualizing image ...")
            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            # for mask in masks:
            #     self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                self.show_box(box.numpy(), plt.gca(), label)

            plt.axis('off')
            # plt.savefig(
            #     os.path.join(output_dir, "grounded_sam_output.jpg"),
            #     bbox_inches="tight", dpi=300, pad_inches=0.0
            # )
            plt.show()

        # print(masks.shape)      # [1, C, H, W]
        masks = masks.cpu()
        final_mask = torch.any(masks[0], dim=0).numpy()
        return final_mask, seg_success        # [H, W] numpy array

    def predict(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25, visualize=False):
        '''
        :param image:   ndarray, in BGR format
        :param text_prompt:
        :param box_threshold:
        :param text_threshold:
        :return:
        '''
        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil, _ = self.transform(image_pil, None)

        boxes_filt, pred_phrases = self.get_dino_output(image_pil, text_prompt, box_threshold, text_threshold)

        self.sam.set_image(image)
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([w, h, w, h])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        if visualize:
            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                self.show_box(box.numpy(), plt.gca(), label)

            plt.axis('off')
            # plt.savefig(
            #     os.path.join(output_dir, "grounded_sam_output.jpg"),
            #     bbox_inches="tight", dpi=300, pad_inches=0.0
            # )
            plt.show()


if __name__ == "__main__":
    predictor = GroundedSam(dino_config_path='groundingdino/config/GroundingDINO_SwinT_OGC.py',
                            dino_checkpoint_path='weights/groundingdino_swint_ogc.pth',
                            sam_checkpoint_path='weights/sam_vit_h_4b8939.pth',
                            device='cuda')

    # img = cv2.imread("assets/demo9.jpg")
    # predictor.predict(img, "bear", visualize=True)
    # predictor.greedy_mask_predict(img, "bear.painting on the wall.dog", visualize=True)
    img_path = "/media/vincent/Seagate Expansion Drive/airvln/files/traj_obs/3/val_seen/3VELCLL3GTHB2UJSCD7IC4U05DK1F0/obs/front_16.png"

    img = cv2.imread(img_path)
    predictor.predict(img, "smallest glass building", visualize=True)
