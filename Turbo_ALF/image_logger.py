from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import numpy as np
import textwrap

class ImageLogger:
    def __init__(self, log_path):
        self.log_path = log_path

    def step_log(self, update_num, step_num, obs, output, infos, task, reward, action_history, correction):
        if output.startswith('<s>') and output.endswith('</s>'):
            raw = output[3:-4].strip()
            wrapped_string = textwrap.wrap(raw, width=50)
            action = '\n'.join(wrapped_string)
        else:
            action = output
        wrapped_infos = textwrap.wrap(str(infos), width=50)
        wrapped_history = textwrap.wrap(str(action_history), width=50)
        wrapped_correction = textwrap.wrap(str(correction), width=50)
        image = self.get_image(obs, action, '\n'.join(wrapped_infos), '\n'.join(wrapped_history), '\n'.join(wrapped_correction),  f"Task: {task}  Reward: {format(reward, '.3f')}")
        cv2.imwrite(f'{self.log_path}/update{update_num}_step{step_num}.jpg', image)
        print(f"step-log: update{update_num}_step{step_num}.jpg")

    def get_image(
        self,
        obs: np.ndarray,
        output: str,
        infos: str,
        history: str,
        correction: str,
        other: str,
    ) -> np.array:
        obs_height, obs_width, ch = obs.shape

        font_to_use = "./Arial.ttf"
        full_font_load = ImageFont.truetype(font_to_use, 14)

        IMAGE_BORDER = 15
        TEXT_OFFSET_H = 30
        TEXT_OFFSET_V = 30
        OTHER_H = 40

        draw = ImageDraw.Draw(Image.fromarray(obs))
        text1_width, text1_height = draw.textsize(str(infos), ImageFont.truetype(font_to_use, 12))
        text2_width, text2_height = draw.textsize(str(history), ImageFont.truetype(font_to_use, 12))
        text3_width, text3_height = draw.textsize(output, ImageFont.truetype(font_to_use, 12))
        text4_width, text4_height = draw.textsize(str(correction), ImageFont.truetype(font_to_use, 12))

        image_dims = (
            max(max(max(max(obs_height + 2 * IMAGE_BORDER, text1_height), text2_height), text3_height), text4_height) + TEXT_OFFSET_H + OTHER_H,
            obs_width + 2 * IMAGE_BORDER + text1_width + text2_width + text3_width + text4_width + 4 * TEXT_OFFSET_V,
            ch,
        )
        image = np.full(image_dims, 255, dtype=np.uint8)

        image[
            IMAGE_BORDER : IMAGE_BORDER + obs_height, IMAGE_BORDER : IMAGE_BORDER + obs_width, :
        ] = obs

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)

        img_draw.text(
            (
                obs_width + 2 * IMAGE_BORDER + TEXT_OFFSET_V,
                TEXT_OFFSET_H,
            ),
            str(infos),
            font=ImageFont.truetype(font_to_use, 12),
            fill="black",
            align="left",
        )

        img_draw.text(
            (
                obs_width + 2 * IMAGE_BORDER + 2 * TEXT_OFFSET_V + text1_width,
                TEXT_OFFSET_H,
            ),
            str(history),
            font=ImageFont.truetype(font_to_use, 12),
            fill="brown",
            align="left",
        )

        img_draw.text(
            (
                obs_width + 2 * IMAGE_BORDER + 3 * TEXT_OFFSET_V + text1_width + text2_width,
                TEXT_OFFSET_H,
            ),
            output,
            font=ImageFont.truetype(font_to_use, 12),
            fill="red",
            align="left",
        )

        img_draw.text(
            (
                obs_width + 2 * IMAGE_BORDER + 4 * TEXT_OFFSET_V + text1_width + text2_width + text3_width,
                TEXT_OFFSET_H,
            ),
            correction,
            font=ImageFont.truetype(font_to_use, 12),
            fill="darkblue",
            align="left",
        )

        img_draw.text(
            (
                IMAGE_BORDER,
                obs_height + 2 * IMAGE_BORDER + TEXT_OFFSET_H,
            ),
            other,
            font=ImageFont.truetype(font_to_use, 12),
            fill="darkblue",
            align="left",
        )

        return np.array(text_image)
