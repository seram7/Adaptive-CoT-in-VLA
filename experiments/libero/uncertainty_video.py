from pathlib import Path
import textwrap

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _as_float_array(values):
    if values is None:
        return np.asarray([], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _safe_text(value, default="n/a"):
    if value is None:
        return default
    try:
        if not np.isfinite(float(value)):
            return default
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _scale_y(value, y_min, y_max, top, height):
    if y_max <= y_min:
        return top + height // 2
    frac = (float(value) - y_min) / (y_max - y_min)
    return int(round(top + height - frac * height))


def _draw_metric_panel(
    draw,
    x0,
    y0,
    width,
    height,
    series,
    step_ids,
    frame_idx,
    metric_name,
    threshold,
    high_score_means_uncertain,
    uncertain_decision,
):
    plot_left = x0 + 54
    plot_top = y0 + 36
    plot_width = width - 78
    plot_height = height - 112
    plot_bottom = plot_top + plot_height

    values = series[np.isfinite(series)]
    y_candidates = values.tolist()
    if threshold is not None and np.isfinite(threshold):
        y_candidates.append(float(threshold))
    if y_candidates:
        y_min = min(y_candidates)
        y_max = max(y_candidates)
        pad = max((y_max - y_min) * 0.08, 1e-4)
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = 0.0, 1.0

    draw.rounded_rectangle(
        [x0, y0, x0 + width, y0 + height],
        radius=8,
        fill=(248, 249, 251),
        outline=(205, 212, 222),
        width=1,
    )
    draw.text((x0 + 16, y0 + 10), metric_name, fill=(20, 28, 38))
    draw.text((x0 + width - 142, y0 + 10), f"step {int(step_ids[frame_idx])}", fill=(20, 28, 38))

    if threshold is not None and np.isfinite(threshold):
        ty = _scale_y(threshold, y_min, y_max, plot_top, plot_height)
        if high_score_means_uncertain:
            shade = [plot_left, plot_top, plot_left + plot_width, max(plot_top, ty)]
        else:
            shade = [plot_left, min(plot_bottom, ty), plot_left + plot_width, plot_bottom]
        draw.rectangle(shade, fill=(255, 231, 231))
        draw.line([plot_left, ty, plot_left + plot_width, ty], fill=(220, 75, 75), width=2)
        draw.text((plot_left + 4, max(plot_top, ty - 18)), f"thr {_safe_text(threshold)}", fill=(160, 42, 42))

    draw.rectangle([plot_left, plot_top, plot_left + plot_width, plot_bottom], outline=(155, 165, 178))
    for frac in (0.25, 0.5, 0.75):
        gy = int(round(plot_top + plot_height * frac))
        draw.line([plot_left, gy, plot_left + plot_width, gy], fill=(225, 229, 235), width=1)

    n = len(series)
    if n > 1:
        xs = np.linspace(plot_left, plot_left + plot_width, n)
        ys = np.asarray([_scale_y(v, y_min, y_max, plot_top, plot_height) for v in series])
        full_points = [
            (int(x), int(y))
            for x, y, v in zip(xs, ys, series)
            if np.isfinite(v)
        ]
        if len(full_points) > 1:
            draw.line(full_points, fill=(166, 176, 190), width=2)
        past_points = [
            (int(xs[i]), int(ys[i]))
            for i in range(frame_idx + 1)
            if np.isfinite(series[i])
        ]
        if len(past_points) > 1:
            draw.line(past_points, fill=(25, 105, 215), width=3)

        cx = int(xs[frame_idx])
        cy = int(ys[frame_idx])
        draw.line([cx, plot_top, cx, plot_bottom], fill=(28, 36, 48), width=2)
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(28, 36, 48))
    else:
        cx = plot_left

    draw.text((x0 + 14, plot_top - 4), _safe_text(y_max), fill=(90, 99, 112))
    draw.text((x0 + 14, plot_bottom - 12), _safe_text(y_min), fill=(90, 99, 112))

    def draw_band(values, band_y, label, active_color):
        draw.text((plot_left, band_y - 14), label, fill=(65, 73, 86))
        for i, value in enumerate(values):
            x_a = int(round(plot_left + plot_width * i / max(n, 1)))
            x_b = int(round(plot_left + plot_width * (i + 1) / max(n, 1)))
            color = active_color if float(value) > 0.5 else (214, 219, 226)
            draw.rectangle([x_a, band_y, max(x_a + 1, x_b), band_y + 8], fill=color)

    band_y = plot_bottom + 18
    if len(uncertain_decision) == n and n > 0:
        draw_band(
            uncertain_decision,
            band_y,
            "CoT step",
            (245, 145, 65),
        )

    value_text = f"value {_safe_text(series[frame_idx] if len(series) else None)}"
    draw.text((plot_left + plot_width - 128, plot_bottom + 10), value_text, fill=(20, 28, 38))


def _clean_reasoning_text(value):
    if value is None:
        return "No reasoning text recorded for this step."

    text = str(value)
    if "##E##" in text:
        text = text.split("##E##", 1)[1]
    text = text.replace("##S##", "").replace("##E##", "\n")
    text = text.replace("<s>", "").replace("</s>", "")
    text = text.replace("USER:", "\nUSER:")
    text = text.replace("ASSISTANT:", "\nASSISTANT:")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text if text else "No reasoning text recorded for this step."


def _draw_reasoning_panel(draw, x0, y0, width, height, reasoning_text, font):
    draw.rounded_rectangle(
        [x0, y0, x0 + width, y0 + height],
        radius=8,
        fill=(248, 249, 251),
        outline=(205, 212, 222),
        width=1,
    )
    draw.text((x0 + 14, y0 + 12), "Reasoning", fill=(20, 28, 38), font=font)

    try:
        char_w = max(6, font.getbbox("M")[2])
        line_h = max(12, font.getbbox("Mg")[3] + 4)
    except AttributeError:
        char_w = 7
        line_h = 14

    wrap_width = max(18, (width - 28) // char_w)
    text = _clean_reasoning_text(reasoning_text)
    y = y0 + 34
    bottom = y0 + height - 14

    for raw_line in text.splitlines():
        wrapped = textwrap.wrap(
            raw_line,
            width=wrap_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for line in wrapped:
            if y + line_h > bottom:
                draw.text((x0 + 14, bottom - line_h), "...", fill=(90, 99, 112), font=font)
                return
            draw.text((x0 + 14, y), line, fill=(44, 52, 64), font=font)
            y += line_h
        y += 2


def save_uncertainty_rollout_video(
    rollout_images,
    step_ids,
    selected_metric_series,
    cot_used_series,
    save_dir,
    episode_idx,
    success,
    task_description,
    metric_name,
    threshold=None,
    high_score_means_uncertain=True,
    fps=10,
    uncertain_decision_series=None,
    rollout_reasoning=None,
):
    series = _as_float_array(selected_metric_series)
    uncertain_decision = _as_float_array(uncertain_decision_series)
    step_ids = np.asarray(step_ids, dtype=np.int32)

    n = min(len(rollout_images), len(series), len(step_ids))
    if n == 0:
        return None

    series = series[:n]
    uncertain_decision = (
        uncertain_decision[:n]
        if len(uncertain_decision) >= n
        else np.asarray([], dtype=np.float32)
    )
    step_ids = step_ids[:n]
    reasoning = list(rollout_reasoning) if rollout_reasoning is not None else []

    first = Image.fromarray(np.asarray(rollout_images[0]).astype(np.uint8)).convert("RGB")
    video_scale = 2
    video_w = first.width * video_scale
    video_h = first.height * video_scale
    margin = 16
    gap = 16
    base_reason_w = 560
    canvas_w = max(880, margin + video_w + gap + base_reason_w + margin)
    canvas_w = ((canvas_w + 15) // 16) * 16
    reason_w = canvas_w - margin * 2 - gap - video_w
    panel_h = 260
    meta_h = 48
    canvas_h = panel_h + video_h + meta_h
    canvas_h = ((canvas_h + 15) // 16) * 16

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    processed_task = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    metric_slug = metric_name.lower().replace(" ", "_").replace("/", "_")
    mp4_path = (
        save_dir
        / f"uncertainty--episode={episode_idx}--success={int(success)}--metric={metric_slug}"
        f"--task={processed_task}.mp4"
    )

    font = ImageFont.load_default()
    with imageio.get_writer(str(mp4_path), fps=fps) as writer:
        for i in range(n):
            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            _draw_metric_panel(
                draw=draw,
                x0=12,
                y0=10,
                width=canvas_w - 24,
                height=panel_h - 20,
                series=series,
                step_ids=step_ids,
                frame_idx=i,
                metric_name=metric_name,
                threshold=threshold,
                high_score_means_uncertain=high_score_means_uncertain,
                uncertain_decision=uncertain_decision,
            )

            frame = Image.fromarray(np.asarray(rollout_images[i]).astype(np.uint8)).convert("RGB")
            frame = frame.resize((video_w, video_h), Image.Resampling.BILINEAR)
            video_x = margin
            video_y = panel_h
            canvas.paste(frame, (video_x, video_y))
            reasoning_text = reasoning[i] if i < len(reasoning) else None
            _draw_reasoning_panel(
                draw=draw,
                x0=video_x + video_w + gap,
                y0=video_y,
                width=reason_w,
                height=video_h,
                reasoning_text=reasoning_text,
                font=font,
            )

            status = "success" if success else "failure"
            meta = (
                f"{processed_task} | episode {episode_idx} | {status} | "
                f"step {int(step_ids[i])} | {metric_name}={_safe_text(series[i])}"
            )
            draw.rectangle([0, canvas_h - meta_h, canvas_w, canvas_h], fill=(22, 28, 38))
            draw.text((14, canvas_h - 32), meta, fill=(245, 247, 250), font=font)

            writer.append_data(np.asarray(canvas))

    print(f"Saved uncertainty rollout MP4 at path {mp4_path}")
    return str(mp4_path)
