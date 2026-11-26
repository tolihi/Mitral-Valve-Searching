"""realtime.py —— 实时（在线）检测：RTSP / HTTP 流、摄像头、本地视频回放。

与 predict.py 的分工：
    predict.py    离线批处理，读文件 -> 写标注图 / 视频 / JSON，追求完整可复现的记录
    realtime.py   在线流式，后台线程持续取帧，主线程永远只处理"最新一帧"，
                  处理慢时自动丢旧帧（延迟不累积），支持预览、边录边存、JSONL、断流重连

支持的输入：
    rtsp://... / rtsps://... / http(s)://... / rtmp://... / udp://...   网络流
    0 / 1 / cam:0 / camera:1                                            本机摄像头（序号）
    path\\to\\video.avi（配 --loop）                                     本地视频回放，用来离线演练实时链路

用法示例：
    python realtime.py --weights best.pt --source rtsp://192.168.1.10:8554/echo --show
    python realtime.py --weights best.pt --source 0 --show --record runs/realtime/cam.mp4
    python realtime.py --weights best.pt --source demo.avi --loop --show --no-guided
    python realtime.py --weights best.pt --source rtsp://... --no-show --record out.mp4 --jsonl out.jsonl

实时性说明（重要，别把它当"流畅 30fps"）：
    * 纯 CPU 单帧约 146 ms（imgsz 768）；imgsz 640 会快一些，--guided 还可能多一次前向。
      所以 CPU 上大致 5-10 fps 量级，够看清方向，不够流畅播放。要实时帧率必须上 GPU（--device 0）。
    * 本脚本保证的是"预览/录制里永远是最新画面、延迟不越拖越长"，**不是**"每一帧都被处理"。
      源端 25 fps、模型 8 fps 时，你会得到 8 fps 的预览，但延迟稳定在百毫秒级。
    * 丢帧率会打印在结尾汇总里，不要忽略它。

数据合规：接院内超声机/采集卡的实时流等于把患者影像实时送进模型，需要院内授权与合规评估；
本脚本只做技术链路，不做脱敏。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from common import save_json
from inference import EchoSegmenter, FrameResult, InferenceConfig, analyze_frame
from predict import create_video_writer, discover_relation
from relation import RelationPredictor
from visualize import render_result

STREAM_PREFIXES = (
    "rtsp://",
    "rtsps://",
    "http://",
    "https://",
    "rtmp://",
    "udp://",
    "tcp://",
    "srt://",
)

# 连续读失败多少次算"流断了"（摄像头偶发 False 不算断）
READ_FAILURE_LIMIT = 25


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
@dataclass
class Source:
    """解析后的输入源。kind: stream / device / file。"""

    kind: str
    text: str
    index: int = -1

    def describe(self) -> str:
        label = {"stream": "网络流", "device": "摄像头", "file": "本地文件"}[self.kind]
        return f"{label} {self.text}"


def parse_source(text: str) -> Source:
    raw = (text or "").strip()
    if not raw:
        raise SystemExit("[error] --source 不能为空")
    low = raw.lower()
    if low.startswith(STREAM_PREFIXES):
        return Source("stream", raw)
    camera = re.fullmatch(r"(?:cam|camera)\s*:\s*(\d+)", low) or re.fullmatch(r"(\d+)", low)
    if camera:
        return Source("device", raw, int(camera.group(1)))
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(
            f"[error] 既不是流地址/摄像头序号，也找不到文件：{raw}\n"
            f"        流地址形如 rtsp://host:554/stream、http://host/live.mjpg；"
            f"摄像头用 0/1 或 cam:0；本地视频要给真实存在的文件。"
        )
    return Source("file", str(path))


def _needs_ascii_copy(target: str) -> bool:
    try:
        target.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _prepare_ffmpeg_options(source: Source) -> None:
    """给 FFmpeg 后端压低延迟（只影响本次进程；用户自己设了就不覆盖）。"""
    if source.kind != "stream":
        return
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
        return
    options = ["fflags;nobuffer", "flags;low_delay", "max_delay;500000"]
    if source.text.lower().startswith(("rtsp://", "rtsps://")):
        options.insert(0, "rtsp_transport;tcp")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(options)


def open_capture(kind: str, target: str, backend: str = "auto") -> cv2.VideoCapture:
    """按输入类型和后端打开 VideoCapture。target 必须是 ASCII 安全的可打开字符串。"""
    if kind == "device":
        index = int(target)
        if backend in ("auto", "any"):
            for flag in _device_backends():
                cap = cv2.VideoCapture(index, flag) if flag is not None else cv2.VideoCapture(index)
                if cap.isOpened():
                    return cap
                cap.release()
            return cv2.VideoCapture(index)
        flag = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "v4l2": "CAP_V4L2"}.get(backend)
        flag_value = getattr(cv2, flag, None) if flag else None
        if flag_value is None:
            return cv2.VideoCapture(index)
        return cv2.VideoCapture(index, flag_value)

    if backend in ("auto", "any", "ffmpeg"):
        cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
        if cap.isOpened() or backend == "ffmpeg":
            return cap
        cap.release()
    if backend == "gstreamer":
        return cv2.VideoCapture(target, cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(target)


def _device_backends() -> list[int | None]:
    """Windows 上默认后端有时打不开摄像头，按 DSHOW -> MSMF -> 默认 依次试。"""
    options: list[int | None] = []
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_V4L2"):
        flag = getattr(cv2, name, None)
        if flag is not None:
            options.append(flag)
    options.append(None)
    return options


# ---------------------------------------------------------------------------
# 后台取帧线程
# ---------------------------------------------------------------------------
class FrameGrabber(threading.Thread):
    """持续 cap.read()，只保留最新一帧；主线程慢就自动丢中间帧，延迟因此不累积。"""

    def __init__(
        self,
        source: Source,
        loop_file: bool = False,
        reconnect: int = 0,
        reconnect_delay: float = 2.0,
        backend: str = "auto",
        fps_cap: float = 0.0,
    ) -> None:
        super().__init__(daemon=True, name="frame-grabber")
        self.source = source
        self.loop_file = loop_file
        self.reconnect = max(0, int(reconnect))
        self.reconnect_delay = max(0.0, float(reconnect_delay))
        self.backend = backend
        self.fps_cap = max(0.0, float(fps_cap))

        self._cond = threading.Condition()
        self._halt = threading.Event()
        self._seq = 0                     # 已发布帧序号（主线程用它判断有没有新帧）
        self._frame: np.ndarray | None = None
        self._stamp = 0.0                 # 该帧进入队列的 perf_counter 时间
        self._captured = 0
        self._fail_streak = 0
        self._eof = False
        self._error: str | None = None
        self._opens = 0
        self.capture_fps = 0.0
        self.size = (0, 0)

        # 中文路径：FFmpeg 后端在 Windows 上打不开，先复制到 ASCII 临时路径
        self._temp_dir: Path | None = None
        self._target = source.text
        if source.kind == "device":
            self._target = str(source.index)
        elif source.kind == "file" and _needs_ascii_copy(source.text):
            self._temp_dir = Path(tempfile.mkdtemp(prefix="echo_stream_"))
            temp_path = self._temp_dir / ("input" + (Path(source.text).suffix or ".mp4"))
            shutil.copyfile(source.text, temp_path)
            self._target = str(temp_path)

    # -- 线程主体 -------------------------------------------------------
    def run(self) -> None:
        reconnects_left = self.reconnect
        first_attempt = True
        while not self._halt.is_set():
            cap = open_capture(self.source.kind, self._target, self.backend)
            if not cap.isOpened():
                cap.release()
                if first_attempt or reconnects_left <= 0:
                    self._fail("打不开输入：地址不可达、摄像头被占用，或缺少 FFmpeg 解码支持")
                    return
                reconnects_left -= 1
                self._warn(f"重连中…（剩余 {reconnects_left} 次，{self.reconnect_delay:.0f}s 后重试）")
                self._halt.wait(self.reconnect_delay)
                continue
            first_attempt = False
            self._opens += 1
            self.capture_fps = _positive(cap.get(cv2.CAP_PROP_FPS))
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            reason = self._read_loop(cap)
            cap.release()
            if self._halt.is_set():
                return
            if reason == "eof":
                self._mark_eof("")
                return
            if reason == "error":
                self._mark_eof("连续读取失败，流疑似断开")
                return
            # reason == "disconnect"：给重连机会
            if reconnects_left <= 0:
                self._mark_eof("流已断开（可用 --reconnect 打开自动重连）")
                return
            reconnects_left -= 1
            self._warn(f"流断开，重连中…（剩余 {reconnects_left} 次）")
            self._halt.wait(self.reconnect_delay)

    def _read_loop(self, cap: cv2.VideoCapture) -> str:
        # 文件回放默认按源端 fps 限速，否则取帧线程满速空转会抢光 CPU，把推理拖慢
        fps = self.fps_cap
        if fps <= 0 and self.source.kind == "file":
            fps = self.capture_fps
        interval = 1.0 / fps if fps > 0 else 0.0
        while not self._halt.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                self._fail_streak = 0
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    self.size = (int(frame.shape[1]), int(frame.shape[0]))
                self._publish(frame)
                if interval > 0:
                    self._halt.wait(interval)
                continue
            self._fail_streak += 1
            if self.source.kind == "file":
                if self.loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._halt.wait(0.01)
                    continue
                return "eof"
            if self._fail_streak >= READ_FAILURE_LIMIT:
                return "error"
            self._halt.wait(0.02)
        return "stop"

    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._seq += 1
            self._frame = frame
            self._stamp = time.perf_counter()
            self._captured += 1
            self._cond.notify_all()

    # -- 状态 -----------------------------------------------------------
    def _fail(self, message: str) -> None:
        with self._cond:
            self._error = message
            self._eof = True
            self._cond.notify_all()

    def _mark_eof(self, note: str) -> None:
        with self._cond:
            if note:
                self._error = note
            self._eof = True
            self._cond.notify_all()

    @staticmethod
    def _warn(message: str) -> None:
        print(f"[warn] {message}", file=sys.stderr)

    # -- 主线程接口 -----------------------------------------------------
    def wait_new(self, last_seq: int, timeout: float) -> tuple[int, np.ndarray, float] | None:
        """等到比 last_seq 新的一帧；超时返回 None。已结束但还有未取走的帧时仍会返回该帧。"""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._seq <= last_seq and not self._eof and self._error is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._seq > last_seq and self._frame is not None:
                return self._seq, self._frame, self._stamp
            return None

    def done(self) -> bool:
        """取帧线程已结束（正常结束或出错），且不会再产出新帧。"""
        with self._cond:
            return self._eof or self._error is not None

    @property
    def captured(self) -> int:
        with self._cond:
            return self._captured

    @property
    def seq(self) -> int:
        with self._cond:
            return self._seq

    @property
    def error(self) -> str | None:
        with self._cond:
            return self._error

    @property
    def opens(self) -> int:
        return self._opens

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def stop(self) -> None:
        self._halt.set()
        with self._cond:
            self._cond.notify_all()


def _positive(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if 1.0 <= number <= 240.0 else 0.0


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class SessionStats:
    processed: int = 0
    skipped_by_stride: int = 0
    both: int = 0
    guided: int = 0
    implausible: int = 0
    processing_ms: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)

    def add(self, frame: FrameResult, processing_ms: float, latency_ms: float) -> None:
        self.processed += 1
        self.processing_ms.append(processing_ms)
        self.latency_ms.append(latency_ms)
        if not frame.missing:
            self.both += 1
        if any(link.guided for link in frame.links):
            self.guided += 1
        for link in frame.links:
            if link.consistency and not link.consistency.get("plausible", True):
                self.implausible += 1
                break

    def summary(self) -> dict:
        return {
            "processing_ms": {
                "p50": round(_percentile(self.processing_ms, 0.50), 1),
                "p90": round(_percentile(self.processing_ms, 0.90), 1),
            },
            "latency_ms": {
                "p50": round(_percentile(self.latency_ms, 0.50), 1),
                "p90": round(_percentile(self.latency_ms, 0.90), 1),
            },
            "frames_processed": self.processed,
            "frames_with_both": self.both,
            "frames_with_guided_recovery": self.guided,
            "frames_implausible_pair": self.implausible,
        }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _config_payload(cfg: InferenceConfig) -> dict:
    return {
        "conf": cfg.conf,
        "iou": cfg.iou,
        "imgsz": cfg.imgsz,
        "guided": cfg.guided,
        "guide_conf": cfg.guide_conf,
        "guide_max_angle": cfg.guide_max_angle,
        "guide_upscale": cfg.guide_upscale,
        "min_area_px": cfg.min_area_px,
    }


def _warmup(segmenter: EchoSegmenter, cfg: InferenceConfig) -> float:
    """跑一次空图像前向，把 torch/后端的首次初始化开销从首帧里挪走。"""
    size = max(320, int(cfg.imgsz) if int(cfg.imgsz) > 0 else 640)
    dummy = np.zeros((size, size, 3), dtype=np.uint8)
    started = time.perf_counter()
    try:
        segmenter.detect(dummy, cfg)
    except Exception as exc:  # 预热失败不影响正式运行，只提示
        print(f"[warn] 预热失败（{exc}），首帧可能偏慢", file=sys.stderr)
    return time.perf_counter() - started


def _frame_line(frame: FrameResult, seq: int, processing_ms: float, latency_ms: float) -> str:
    parts = [
        f"#{seq}",
        f"mitral={len(frame.by_class(0))}",
        f"fossa={len(frame.by_class(1))}",
    ]
    guided = sum(1 for link in frame.links if link.guided)
    if guided:
        parts.append(f"guided={guided}")
    parts.append(f"{processing_ms:.0f}ms")
    parts.append(f"lag={latency_ms:.0f}ms")
    return " ".join(parts)


def run_realtime(args: argparse.Namespace) -> int:
    source = parse_source(args.source)
    weights = Path(args.weights)
    if not weights.is_file():
        print(f"[error] 找不到权重：{weights}", file=sys.stderr)
        return 1
    if args.max_frames < 0:
        print("[error] --max-frames 不能为负", file=sys.stderr)
        return 1
    if args.frame_stride < 1:
        print("[error] --frame-stride 必须 >= 1", file=sys.stderr)
        return 1

    relation_path = None
    if not args.relation_disabled:
        relation_path = discover_relation(weights, args.relation)
    predictor = RelationPredictor.load(relation_path) if relation_path else None
    print(f"输入: {source.describe()}")
    print(f"关系模型: {relation_path if relation_path else '未使用（只做分割）'}")

    cfg = InferenceConfig(
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        guided=args.guided,
        guide_conf=args.guide_conf,
        guide_max_angle=args.guide_max_angle,
        guide_upscale=args.guide_upscale,
        min_area_px=args.min_area,
    )
    segmenter = EchoSegmenter(weights, imgsz=args.imgsz, device=args.device)
    # 先跑一次空图像预热：首次前向要初始化 torch 线程池/后端，不预热的话首帧延迟会非常难看
    warmup_s = _warmup(segmenter, cfg)
    print(f"模型预热: {warmup_s:.1f} s（首帧不再被初始化拖慢）")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    record_path: Path | None = None
    if args.record:
        record_path = Path(args.record)
        if not record_path.suffix:
            record_path = record_path.with_suffix(".mp4")
        record_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_handle = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")

    grabber = FrameGrabber(
        source,
        loop_file=bool(args.loop),
        reconnect=args.reconnect,
        reconnect_delay=args.reconnect_delay,
        backend=args.backend,
        fps_cap=args.fps_cap,
    )
    grabber.start()

    stats = SessionStats()
    writer: cv2.VideoWriter | None = None
    temp_output: Path | None = None
    temp_output_dir: Path | None = None
    record_failed = False
    show = args.show
    last_seq = 0
    written = 0
    started = time.perf_counter()
    last_progress = started
    stopped_by_user = False
    idle_timeout_hit = False

    try:
        while True:
            if args.max_frames and stats.processed >= args.max_frames:
                break
            item = grabber.wait_new(last_seq, args.wait_timeout)
            if item is None:
                if grabber.done():
                    break
                # 空闲计时按"最后拿到新帧"算，流中途卡住也能退出
                if args.idle_timeout > 0 and time.perf_counter() - last_progress > args.idle_timeout:
                    idle_timeout_hit = True
                    break
                continue
            seq, frame, stamp = item
            last_seq = seq
            last_progress = time.perf_counter()

            if args.frame_stride > 1 and (seq - 1) % args.frame_stride != 0:
                stats.skipped_by_stride += 1
                continue

            t0 = time.perf_counter()
            result = analyze_frame(segmenter, predictor, frame, cfg)
            canvas = render_result(
                frame,
                result,
                show_masks=args.show_masks,
                show_cone=args.show_cone,
                show_roi=args.show_roi,
            )
            now = time.perf_counter()
            processing_ms = (now - t0) * 1000.0
            latency_ms = (now - stamp) * 1000.0
            stats.add(result, processing_ms, latency_ms)

            if jsonl_handle is not None:
                record = result.to_json(f"frame_{seq:06d}")
                record["capture_seq"] = seq
                record["processing_ms"] = round(processing_ms, 1)
                record["latency_ms"] = round(latency_ms, 1)
                record["elapsed_s"] = round(now - started, 3)
                jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_handle.flush()

            if record_path is not None:
                written += 1
                if written % max(1, args.record_every) == 0:
                    if writer is None:
                        fps = (args.record_fps or grabber.capture_fps or 25.0) / max(1, args.record_every)
                        writer, temp_output, temp_output_dir = create_video_writer(
                            record_path, fps, (canvas.shape[1], canvas.shape[0])
                        )
                        if not writer.isOpened():
                            print("[error] 无法创建视频写入器（缺少 mp4v 编码器）", file=sys.stderr)
                            record_failed = True
                            record_path = None
                        else:
                            print(f"录制: {record_path}（{fps:.1f} fps）")
                    if writer is not None:
                        writer.write(canvas)

            if show:
                try:
                    cv2.imshow(args.window_title, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("\n收到退出键，停止。")
                        stopped_by_user = True
                        break
                except cv2.error as exc:
                    print(f"[warn] 无法开预览窗口（{exc}），已自动关闭 --show", file=sys.stderr)
                    show = False

            if stats.processed % max(1, args.print_every) == 0:
                elapsed = max(1e-6, time.perf_counter() - started)
                print(
                    f"  {_frame_line(result, seq, processing_ms, latency_ms)}"
                    f"  |  已处理 {stats.processed}，采集 {grabber.captured}，"
                    f"平均 {stats.processed / elapsed:.2f} fps"
                )
    except KeyboardInterrupt:
        stopped_by_user = True
        print("\n收到 Ctrl+C，收尾中…")
    finally:
        grabber.stop()
        grabber.join(timeout=3.0)
        if writer is not None:
            writer.release()
        if jsonl_handle is not None:
            jsonl_handle.close()
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if record_failed and temp_output_dir is not None:
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            temp_output = None

    final_record: Path | None = None
    if record_path is not None and temp_output is not None:
        final_record = record_path
        if temp_output != record_path:
            shutil.move(str(temp_output), str(record_path))
            if temp_output_dir is not None:
                shutil.rmtree(temp_output_dir, ignore_errors=True)

    elapsed = max(1e-6, time.perf_counter() - started)
    captured = grabber.captured
    dropped = max(0, captured - stats.processed - stats.skipped_by_stride)
    summary = {
        "source": source.text,
        "source_kind": source.kind,
        "weights": str(weights),
        "relation": str(relation_path) if relation_path else None,
        "config": _config_payload(cfg),
        "duration_s": round(elapsed, 2),
        "fps_average": round(stats.processed / elapsed, 2),
        "capture_fps_reported": round(grabber.capture_fps, 2),
        "frames_captured": captured,
        "frames_processed": stats.processed,
        "frames_dropped": dropped,
        "drop_ratio": round(dropped / captured, 4) if captured else 0.0,
        "reconnects": max(0, grabber.opens - 1),
        "stopped_by_user": stopped_by_user,
        "idle_timeout_hit": idle_timeout_hit,
        "error": grabber.error,
        "record": str(final_record) if final_record else None,
        **stats.summary(),
    }
    save_json(out_dir / "realtime_summary.json", summary)
    grabber.cleanup()

    print("\n实时会话结束")
    print(f"  输入          {source.describe()}")
    print(
        f"  处理 / 采集    {stats.processed} / {captured} 帧"
        f"（丢弃 {dropped} 帧，丢帧率 {summary['drop_ratio'] * 100:.1f}%）"
    )
    print(f"  时长 / 帧率   {elapsed:.1f} s，平均 {summary['fps_average']:.2f} fps")
    if stats.processing_ms:
        print(
            f"  单帧耗时      p50 {summary['processing_ms']['p50']:.0f} ms"
            f" / p90 {summary['processing_ms']['p90']:.0f} ms"
        )
        print(
            f"  端到端延迟    p50 {summary['latency_ms']['p50']:.0f} ms"
            f" / p90 {summary['latency_ms']['p90']:.0f} ms"
        )
    print(f"  双结构帧      {stats.both}（引导找回 {stats.guided} 帧，不合理配对 {stats.implausible} 帧）")
    if final_record:
        print(f"  录制          {final_record}")
    if grabber.error:
        print(f"  结束原因      {grabber.error}", file=sys.stderr)
    if idle_timeout_hit:
        print(f"  结束原因      {args.idle_timeout:.0f}s 内没有新帧（--idle-timeout）", file=sys.stderr)
    print(f"  汇总 JSON     {out_dir / 'realtime_summary.json'}")

    if (grabber.error or idle_timeout_hit) and stats.processed == 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="二尖瓣/卵圆孔 实时检测（RTSP / HTTP 流 / 摄像头 / 视频回放）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="YOLO-seg 权重，如 runs/segment/.../best.pt")
    parser.add_argument(
        "--source",
        required=True,
        help="rtsp://... / http://... / rtmp://... 流地址；摄像头序号 0/1/cam:0；或本地视频文件",
    )
    parser.add_argument("--out", default="runs/realtime", help="汇总 JSON 输出目录")
    parser.add_argument("--relation", default="auto",
                        help="relation.json 路径；auto=自动查找；none=不做关系预测")
    parser.add_argument("--no-relation", dest="relation_disabled", action="store_true",
                        help="完全不加载关系模型（只做分割，最快）")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸；实时场景别用太大")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--device", default=None, help="如 0 / cpu；实时场景建议 GPU")
    parser.add_argument("--guided", dest="guided", action="store_true", default=False,
                        help="开启引导式二次检测（找回漏检，但每帧可能多一次前向）")
    parser.add_argument("--no-guided", dest="guided", action="store_false")
    parser.add_argument("--guide-conf", type=float, default=0.25)
    parser.add_argument("--guide-max-angle", type=float, default=45.0)
    parser.add_argument("--guide-upscale", type=float, default=2.0)
    parser.add_argument("--min-area", type=float, default=150.0)

    parser.add_argument("--show", dest="show", action="store_true", default=True,
                        help="弹出预览窗口（按 q / ESC 退出）")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--window-title", default="echo realtime", help="预览窗口标题（建议 ASCII）")
    parser.add_argument("--record", default="", help="把标注结果录成 mp4；留空不录")
    parser.add_argument("--record-fps", type=float, default=0.0,
                        help="录制帧率；0=用源端 fps，取不到则 25")
    parser.add_argument("--record-every", type=int, default=1,
                        help="每处理 N 帧写 1 帧到视频，长会话可调大以省磁盘")
    parser.add_argument("--jsonl", default="", help="逐帧结果写 JSONL（体积大，默认不写）")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧（0=一直跑）")
    parser.add_argument("--frame-stride", type=int, default=1, help="每 N 帧处理 1 帧")
    parser.add_argument("--fps-cap", type=float, default=0.0,
                        help="取帧限速；本地视频回放想接近实时可设 25")
    parser.add_argument("--loop", dest="loop", action="store_true", default=False,
                        help="本地视频循环播放（模拟直播源）")
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--wait-timeout", type=float, default=5.0, help="等新帧的超时秒数")
    parser.add_argument("--idle-timeout", type=float, default=0.0,
                        help="超过该秒数没有新帧就退出；0=不限")
    parser.add_argument("--reconnect", type=int, default=0, help="流断开后重连次数；0=不重连")
    parser.add_argument("--reconnect-delay", type=float, default=2.0, help="重连间隔秒数")
    parser.add_argument("--backend", default="auto", help="auto / ffmpeg / dshow / msmf / gstreamer")
    parser.add_argument("--print-every", type=int, default=25, help="每 N 帧打印一次状态")
    parser.add_argument("--show-masks", dest="show_masks", action="store_true", default=True)
    parser.add_argument("--no-masks", dest="show_masks", action="store_false")
    parser.add_argument("--show-cone", dest="show_cone", action="store_true", default=True)
    parser.add_argument("--no-cone", dest="show_cone", action="store_false")
    parser.add_argument("--show-roi", dest="show_roi", action="store_true", default=True)
    parser.add_argument("--no-roi", dest="show_roi", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_realtime(args)


if __name__ == "__main__":
    raise SystemExit(main())
