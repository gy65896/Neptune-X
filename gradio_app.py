import json
import os
import time
from types import SimpleNamespace

import gradio as gr
import torch

from inference import Processer
from ldm.data.base import BaseData

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_JSON_PATH = os.path.join(ROOT_DIR, "sample", "1.json")
DEFAULT_CKPT_PATH = os.path.join(ROOT_DIR, "ckpt", "X2Mari.pth")
DEFAULT_OUT_BASE = os.path.join(ROOT_DIR, "out", "gradio")
SAMPLE_DIR = os.path.join(ROOT_DIR, "sample")

_PROCESSER = None
_PROCESSER_SIG = None


def _build_cond_from_data(data, batch_size):
    return {
        "caption": data["caption"] * batch_size,
        "boxes": data["boxes"] * batch_size,
        "classes": data["classes"] * batch_size,
        "water_boxes": data["water_boxes"] * batch_size,
        "water_masks": data["water_masks"] * batch_size,
        "water_caption": data["water_caption"] * batch_size,
        "file_name": [f"img_{str(i)}.jpg" for i in range(batch_size)],
        "size": tuple(data["size"]),
    }


def _decode_water_mask_if_needed(cond):
    if isinstance(cond["water_masks"][0], str):
        first_box = cond["water_boxes"][0]
        first_mask = cond["water_masks"][0]
        cvat_rle = dict(
            rle=first_mask,
            top=first_box[1],
            left=first_box[0],
            width=first_box[2] - first_box[0],
        )
        img_w, img_h = cond["size"]
        decoded_mask = BaseData._rle2mask(None, cvat_rle, img_h, img_w)
        cond["water_masks"] = [decoded_mask] * len(cond["caption"])
    return cond


def _load_sample_data(sample_name):
    path = os.path.join(SAMPLE_DIR, sample_name)
    if not os.path.isfile(path):
        raise gr.Error(f"Sample not found: {sample_name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _data_to_fields(data):
    caption_text = ""
    water_caption_text = ""
    if isinstance(data.get("caption"), list) and data["caption"]:
        caption_text = data["caption"][0]
    elif isinstance(data.get("caption"), str):
        caption_text = data["caption"]
    if isinstance(data.get("water_caption"), list) and data["water_caption"]:
        water_caption_text = data["water_caption"][0]
    elif isinstance(data.get("water_caption"), str):
        water_caption_text = data["water_caption"]
    boxes_text = json.dumps(data.get("boxes", []), ensure_ascii=True)
    classes_text = json.dumps(data.get("classes", []), ensure_ascii=True)
    water_boxes_text = json.dumps(data.get("water_boxes", []), ensure_ascii=True)
    water_masks_text = json.dumps(data.get("water_masks", []), ensure_ascii=True)
    size_text = json.dumps(data.get("size", []), ensure_ascii=True)
    return (
        caption_text,
        boxes_text,
        classes_text,
        water_boxes_text,
        water_masks_text,
        water_caption_text,
        size_text,
    )


def _parse_json_list(name, text):
    if text is None or str(text).strip() == "":
        raise gr.Error(f"{name} is required.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"{name} JSON parse failed: {exc}") from exc
    if not isinstance(value, list):
        raise gr.Error(f"{name} must be a JSON list.")
    return value


def _parse_caption(text, name):
    if text is None or str(text).strip() == "":
        raise gr.Error(f"{name} is required.")
    text = str(text).strip()
    if text.startswith("["):
        return _parse_json_list(name, text)
    return [text]


def _build_data_from_fields(
    caption_text,
    boxes_text,
    classes_text,
    water_boxes_text,
    water_masks_text,
    water_caption_text,
    size_text,
):
    data = {
        "caption": _parse_caption(caption_text, "caption"),
        "boxes": _parse_json_list("boxes", boxes_text),
        "classes": _parse_json_list("classes", classes_text),
        "water_boxes": _parse_json_list("water_boxes", water_boxes_text),
        "water_masks": _parse_json_list("water_masks", water_masks_text),
        "water_caption": _parse_caption(water_caption_text, "water_caption"),
        "size": _parse_json_list("size", size_text),
    }
    return data


def _load_processer(args):
    global _PROCESSER, _PROCESSER_SIG
    sig = (args.sd_ckpt, args.fp16)
    if _PROCESSER is None or _PROCESSER_SIG != sig:
        _PROCESSER = Processer(args)
        _PROCESSER_SIG = sig
    return _PROCESSER


def _configure_processer(args):
    if _PROCESSER is None or _PROCESSER_SIG != (args.sd_ckpt, args.fp16):
        raise gr.Error("Model not loaded. Please click 'Load Model' first.")
    _PROCESSER.batch_size = args.batch_size
    _PROCESSER.guidance_scale = args.guidance_scale
    _PROCESSER.negative_prompt = args.negative_prompt
    _PROCESSER.steps = args.steps
    _PROCESSER.draw_text = False
    _PROCESSER.uc = _PROCESSER.text_encoder.encode(
        _PROCESSER.batch_size * [args.negative_prompt]
    )
    os.makedirs(args.out_path, exist_ok=True)
    _PROCESSER.image_dir = os.path.join(args.out_path, "image")
    _PROCESSER.image_w_cond_dir = os.path.join(args.out_path, "img_w_cond")
    os.makedirs(_PROCESSER.image_dir, exist_ok=True)
    os.makedirs(_PROCESSER.image_w_cond_dir, exist_ok=True)
    return _PROCESSER


def _collect_images(image_dir):
    if not os.path.isdir(image_dir):
        return []
    files = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    return files


def _clear_cuda_cache():
    # Keep model weights in GPU; only clear temporary cached allocations.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model(sd_ckpt, fp16):
    args = SimpleNamespace(
        json_path=SAMPLE_JSON_PATH,
        out_path=DEFAULT_OUT_BASE,
        rand_seed=None,
        batch_size=1,
        num_workers=8,
        steps=1,
        guidance_scale=1.0,
        negative_prompt="",
        sd_ckpt=sd_ckpt,
        fp16=bool(fp16),
    )
    _load_processer(args)
    _clear_cuda_cache()
    return f"Model loaded: {sd_ckpt} (fp16={bool(fp16)})"


def run_inference(
    caption_text,
    boxes_text,
    classes_text,
    water_boxes_text,
    water_masks_text,
    water_caption_text,
    size_text,
    batch_size,
    steps,
    guidance_scale,
    rand_seed,
    negative_prompt,
    sd_ckpt,
    out_base,
    fp16,
):
    data = _build_data_from_fields(
        caption_text,
        boxes_text,
        classes_text,
        water_boxes_text,
        water_masks_text,
        water_caption_text,
        size_text,
    )
    cond = _build_cond_from_data(data, int(batch_size))

    cond = _decode_water_mask_if_needed(cond)

    run_dir = os.path.join(out_base, time.strftime("%Y%m%d_%H%M%S"))
    args = SimpleNamespace(
        json_path=SAMPLE_JSON_PATH,
        out_path=run_dir,
        rand_seed=int(rand_seed) if rand_seed is not None else None,
        batch_size=int(batch_size),
        num_workers=8,
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        negative_prompt=negative_prompt or "",
        sd_ckpt=sd_ckpt,
        fp16=bool(fp16),
    )

    if args.rand_seed is not None and args.rand_seed >= 0:
        torch.manual_seed(args.rand_seed)

    process = _configure_processer(args)
    process.generator(cond)
    _clear_cuda_cache()

    image_dir = os.path.join(run_dir, "image")
    image_w_cond_dir = os.path.join(run_dir, "img_w_cond")
    images = _collect_images(image_dir)
    images_w_cond = _collect_images(image_w_cond_dir)
    status = f"Done: {run_dir}"
    return images, images_w_cond, run_dir, status


def build_ui():
    default_data = _load_sample_data("1.json")
    (
        default_caption,
        default_boxes,
        default_classes,
        default_water_boxes,
        default_water_masks,
        default_water_caption,
        default_size,
    ) = _data_to_fields(default_data)
    sample_files = sorted(
        [name for name in os.listdir(SAMPLE_DIR) if name.endswith(".json")]
    )
    with gr.Blocks(title="Neptune-X Inference") as demo:
        gr.Markdown(
            "Neptune-X inference UI (default FP16). Use sample configs or edit fields."
        )

        with gr.Row():
            sample_name = gr.Dropdown(
                choices=sample_files,
                value="1.json",
                label="Sample JSON",
            )
            json_file = gr.File(label="Upload JSON file", file_types=[".json"])

        with gr.Row():
            load_sample_btn = gr.Button("Load Sample")
            load_json_btn = gr.Button("Load JSON File")

        caption_text = gr.Textbox(
            label="caption",
            value=default_caption,
            lines=3,
            placeholder="Single text or JSON list of strings",
        )
        boxes_text = gr.Textbox(
            label="boxes",
            value=default_boxes,
            lines=3,
            placeholder='JSON list like [[[x1,y1,x2,y2], ...]]',
        )
        classes_text = gr.Textbox(
            label="classes",
            value=default_classes,
            lines=3,
            placeholder='JSON list like [["ship", "boat"]]',
        )
        water_boxes_text = gr.Textbox(
            label="water_boxes",
            value=default_water_boxes,
            lines=2,
            placeholder="JSON list like [[x1,y1,x2,y2]]",
        )
        water_masks_text = gr.Textbox(
            label="water_masks",
            value=default_water_masks,
            lines=4,
            placeholder='JSON list like ["RLE_STRING"] or masks',
        )
        water_caption_text = gr.Textbox(
            label="water_caption",
            value=default_water_caption,
            lines=2,
            placeholder="Single text or JSON list of strings",
        )
        size_text = gr.Textbox(
            label="size",
            value=default_size,
            lines=1,
            placeholder="JSON list like [512, 512]",
        )

        with gr.Row():
            batch_size = gr.Slider(1, 8, value=4, step=1, label="batch_size")
            steps = gr.Slider(1, 100, value=50, step=1, label="steps")
            guidance_scale = gr.Slider(1, 20, value=7.5, step=0.1, label="guidance_scale")

        with gr.Row():
            rand_seed = gr.Number(value=64, label="rand_seed (-1 means no fixed seed)")
            negative_prompt = gr.Textbox(value="", label="negative_prompt")
            fp16 = gr.Checkbox(value=True, label="fp16")

        with gr.Row():
            sd_ckpt = gr.Textbox(value=DEFAULT_CKPT_PATH, label="sd_ckpt")
            out_base = gr.Textbox(value=DEFAULT_OUT_BASE, label="out_path (directory)")

        load_btn = gr.Button("Load Model")
        run_btn = gr.Button("Generate")

        with gr.Row():
            out_gallery = gr.Gallery(label="Generated Images", show_label=True)
            out_gallery_cond = gr.Gallery(label="Condition Visualization", show_label=True)

        out_dir = gr.Textbox(label="Output Directory", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)

        def _load_sample_fields(name):
            data = _load_sample_data(name)
            return _data_to_fields(data)

        def _load_json_fields(upload):
            if upload is None or not hasattr(upload, "name"):
                raise gr.Error("Please upload a JSON file first.")
            with open(upload.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _data_to_fields(data)

        load_sample_btn.click(
            _load_sample_fields,
            inputs=[sample_name],
            outputs=[
                caption_text,
                boxes_text,
                classes_text,
                water_boxes_text,
                water_masks_text,
                water_caption_text,
                size_text,
            ],
        )

        load_json_btn.click(
            _load_json_fields,
            inputs=[json_file],
            outputs=[
                caption_text,
                boxes_text,
                classes_text,
                water_boxes_text,
                water_masks_text,
                water_caption_text,
                size_text,
            ],
        )

        load_btn.click(
            load_model,
            inputs=[sd_ckpt, fp16],
            outputs=[status],
        )

        run_btn.click(
            run_inference,
            inputs=[
                caption_text,
                boxes_text,
                classes_text,
                water_boxes_text,
                water_masks_text,
                water_caption_text,
                size_text,
                batch_size,
                steps,
                guidance_scale,
                rand_seed,
                negative_prompt,
                sd_ckpt,
                out_base,
                fp16,
            ],
            outputs=[out_gallery, out_gallery_cond, out_dir, status],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()

