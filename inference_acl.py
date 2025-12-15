import argparse
from abc import abstractmethod, ABC
import logging
import numpy as np
import cv2
import acl
import time
import torch
import torchvision

# =========================
# 日志配置（中文）
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# =========================
# 全局常量
# =========================
SUCCESS = 0  # 成功状态值
FAILED = 1   # 失败状态值
ACL_MEM_MALLOC_NORMAL_ONLY = 2  # 申请内存策略, 仅申请普通页


# =========================
# ACL 初始化 / 去初始化
# =========================
def init_acl(device_id: int):
    """初始化 ACL 资源"""
    logging.info(f"正在初始化 ACL，设备 ID: {device_id} ...")
    ret = acl.init()
    if ret != SUCCESS:
        logging.error(f"ACL 初始化失败，错误码: {ret}")
        raise RuntimeError(ret)

    ret = acl.rt.set_device(device_id)
    if ret != SUCCESS:
        logging.error(f"设置设备 {device_id} 失败，错误码: {ret}")
        raise RuntimeError(ret)

    context, ret = acl.rt.create_context(device_id)
    if ret != SUCCESS:
        logging.error(f"创建 ACL 上下文失败，错误码: {ret}")
        raise RuntimeError(ret)

    logging.info("ACL 初始化成功。")
    return context


def deinit_acl(context, device_id: int):
    """去初始化 ACL 资源"""
    logging.info("正在释放 ACL 资源...")
    if context is not None:
        ret = acl.rt.destroy_context(context)
        if ret != SUCCESS:
            logging.error(f"销毁 ACL 上下文失败，错误码: {ret}")
            raise RuntimeError(ret)

    ret = acl.rt.reset_device(device_id)
    if ret != SUCCESS:
        logging.error(f"重置设备 {device_id} 失败，错误码: {ret}")
        raise RuntimeError(ret)

    ret = acl.finalize()
    if ret != SUCCESS:
        logging.error(f"ACL 去初始化失败，错误码: {ret}")
        raise RuntimeError(ret)

    logging.info("ACL 资源释放完成。")


# =========================
# 通用 ACL 模型基类
# =========================
class Model(ABC):
    def __init__(self, model_path: str):
        logging.info(f"正在从路径加载离线模型 (.om)：{model_path}")
        self.model_path = model_path
        self.model_id = None
        self.input_dataset = None
        self.output_dataset = None
        self.model_desc = None
        self._input_num = 0
        self._output_num = 0
        self._output_info = []
        self._is_released = False
        self._init_resource()

    def _init_resource(self):
        """初始化模型、输出相关资源（aclmdlDesc / aclDataBuffer / aclmdlDataset）"""
        logging.info("正在初始化模型相关资源...")
        # 加载模型文件
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != SUCCESS:
            logging.error(f"从文件加载模型失败：{self.model_path}，错误码: {ret}")
            raise RuntimeError(ret)

        # 创建并获取模型描述信息
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != SUCCESS:
            logging.error(f"获取模型描述信息失败，错误码: {ret}")
            raise RuntimeError(ret)

        logging.info("模型描述信息获取成功。")
        # 创建输出 dataset 结构
        self._gen_output_dataset()

    def _gen_output_dataset(self):
        """根据模型输出信息，申请 Device 内存并组织输出 dataset"""
        self._output_num = acl.mdl.get_num_outputs(self.model_desc)
        self.output_dataset = acl.mdl.create_dataset()
        logging.info(f"模型输出数量：{self._output_num}")

        for i in range(self._output_num):
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(
                temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY
            )
            if ret != SUCCESS:
                logging.error(f"为第 {i} 个输出申请 Device 内存失败，大小：{temp_buffer_size} 字节，错误码: {ret}")
                self._release_dataset(self.output_dataset)
                raise RuntimeError(ret)

            dataset_buffer = acl.create_data_buffer(temp_buffer, temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, dataset_buffer)
            if ret != SUCCESS:
                logging.error(f"为第 {i} 个输出添加 dataset buffer 失败，错误码: {ret}")
                self._release_dataset(self.output_dataset)
                raise RuntimeError(ret)

        logging.info("模型输出 dataset 创建成功。")

    def _gen_input_dataset(self, input_list):
        """根据 numpy 输入数据构造输入 dataset"""
        self._input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.input_dataset = acl.mdl.create_dataset()
        logging.info(f"模型输入数量：{self._input_num}")

        for i in range(self._input_num):
            item = input_list[i]
            data_ptr = acl.util.bytes_to_ptr(item.tobytes())
            size = item.size * item.itemsize
            dataset_buffer = acl.create_data_buffer(data_ptr, size)
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, dataset_buffer)
            if ret != SUCCESS:
                logging.error(f"添加第 {i} 个输入 dataset buffer 失败，错误码: {ret}")
                self._release_dataset(self.input_dataset)
                raise RuntimeError(ret)

        logging.info("模型输入 dataset 创建成功。")

    def _unpack_bytes_array(self, byte_array, shape, datatype):
        """将 Device 输出内存转为指定 dtype 的 numpy 数组"""
        np_type = None
        if datatype == 0:        # ACL_FLOAT
            np_type = np.float32
        elif datatype == 1:      # ACL_FLOAT16
            np_type = np.float16
        elif datatype == 3:      # ACL_INT32
            np_type = np.int32
        elif datatype == 8:      # ACL_UINT32
            np_type = np.uint32
        else:
            logging.warning(f"不支持的数据类型（datatype={datatype}），无法解码输出。")
            return None

        return np.frombuffer(byte_array, dtype=np_type).reshape(shape)

    def _output_dataset_to_numpy(self):
        """将模型输出 dataset 中的数据解码为 numpy 数组列表"""
        dataset = []
        for i in range(self._output_num):
            buffer = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            data_ptr = acl.get_data_buffer_addr(buffer)
            size = acl.get_data_buffer_size(buffer)
            narray = acl.util.ptr_to_bytes(data_ptr, size)

            dims_info = acl.mdl.get_output_dims(self.model_desc, i)
            if not dims_info or "dims" not in dims_info[0]:
                logging.error(f"获取第 {i} 个输出维度信息失败。")
                continue
            dims = dims_info[0]["dims"]
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)

            output_nparray = self._unpack_bytes_array(narray, tuple(dims), datatype)
            dataset.append(output_nparray)

        return dataset

    def execute(self, input_list):
        """执行一次推理：构造输入 dataset -> 执行模型 -> 输出转 numpy"""
        self._gen_input_dataset(input_list)
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != SUCCESS:
            logging.error(f"模型推理执行失败，错误码: {ret}")
            return None

        out_numpy = self._output_dataset_to_numpy()
        return out_numpy

    def release(self):
        """释放模型相关资源"""
        if self._is_released:
            return

        logging.info("正在释放模型相关资源...")
        self._release_dataset(self.input_dataset)
        self.input_dataset = None

        self._release_dataset(self.output_dataset)
        self.output_dataset = None

        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            if ret != SUCCESS:
                logging.warning(f"卸载模型失败，错误码: {ret}")
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            if ret != SUCCESS:
                logging.warning(f"销毁模型描述信息失败，错误码: {ret}")

        self._is_released = True
        logging.info("模型资源释放完成。")

    def _release_dataset(self, dataset):
        """释放 aclmdlDataset 类型数据"""
        if not dataset:
            return
        num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buf:
                ret = acl.destroy_data_buffer(data_buf)
                if ret != SUCCESS:
                    logging.warning(f"销毁第 {i} 个 data buffer 失败，错误码: {ret}")
        ret = acl.mdl.destroy_dataset(dataset)
        if ret != SUCCESS:
            logging.warning(f"销毁 dataset 失败，错误码: {ret}")

    @abstractmethod
    def infer(self, *args, **kwargs):
        """抽象推理接口，子类必须实现"""
        pass


# =========================
# YOLO 工具函数
# =========================
def wh2xy(x):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(outputs, conf_threshold, iou_threshold, nc):
    """YOLO 风格的后处理 NMS"""
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    bs = outputs.shape[0]
    nc = nc or (outputs.shape[1] - 4)
    nm = outputs.shape[1] - nc - 4
    mi = 4 + nc
    xc = outputs[:, 4:mi].amax(1) > conf_threshold

    time_limit = 0.5 + 0.05 * bs
    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=outputs.device)] * bs

    for index, x in enumerate(outputs):
        x = x.transpose(0, -1)[xc[index]]

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, nm), 1)
        box = wh2xy(box)

        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_threshold]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_threshold)
        i = i[:max_det]

        output[index] = x[i]
        if (time.time() - t) > time_limit:
            logging.warning("NMS 处理时间超过限制，提前结束。")
            break

    return output


# =========================
# 姿态识别模型类（继承 Model）
# =========================
class Pose(Model):
    def __init__(self, modelpath, input_w, input_h):
        super().__init__(modelpath)
        self.input_w = input_w
        self.input_h = input_h
        logging.info(f"模型输入尺寸设置为：宽={self.input_w}, 高={self.input_h}")

    @torch.no_grad()
    def infer(self, video_path, output_path):
        logging.info(f"开始对视频进行推理，输入视频路径：{video_path}")

        palette = np.array(
            [[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
             [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
             [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
             [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
             [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
            dtype=np.uint8
        )
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

        camera = cv2.VideoCapture(video_path)
        if not camera.isOpened():
            logging.error(f"无法打开视频文件或视频流：{video_path}")
            return

        frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        logging.info(f"视频读取与写入初始化完成。输出视频路径：{output_path}")

        frame_count = 0
        total_infer_time = 0.0

        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            frame_count += 1
            image = frame.copy()
            shape = image.shape[:2]

            # 预处理：resize + 归一化 + CHW + batch
            image_resized = cv2.resize(image, (self.input_w, self.input_h))
            image_blob = cv2.dnn.blobFromImage(
                image_resized, scalefactor=1 / 255.0, swapRB=True
            )

            # 推理
            t1 = time.time()
            outputs_list = self.execute([image_blob])
            if outputs_list is None or len(outputs_list) == 0:
                logging.error("模型推理返回为空，跳过当前帧。")
                continue
            outputs = outputs_list[0]
            t2 = time.time()
            infer_time = t2 - t1
            total_infer_time += infer_time
            logging.info(f"第 {frame_count} 帧推理耗时：{infer_time:.4f} 秒")

            # 后处理 + NMS
            outputs = torch.from_numpy(np.array(outputs, copy=True))
            outputs = non_max_suppression(outputs, 0.25, 0.7, 1)

            for output in outputs:
                output = output.clone()
                if len(output):
                    box_output = output[:, :6]
                    kps_output = output[:, 6:].view(len(output), 17, 3)
                else:
                    continue

                # 坐标还原到原始分辨率
                r = min(self.input_h / shape[0], self.input_w / shape[1])
                pad_x = (self.input_w - shape[1] * r) / 2
                pad_y = (self.input_h - shape[0] * r) / 2

                box_output[:, [0, 2]] -= pad_x
                box_output[:, [1, 3]] -= pad_y
                box_output[:, :4] /= r
                box_output[:, 0].clamp_(0, shape[1])
                box_output[:, 1].clamp_(0, shape[0])
                box_output[:, 2].clamp_(0, shape[1])
                box_output[:, 3].clamp_(0, shape[0])

                kps_output[..., 0] -= pad_x
                kps_output[..., 1] -= pad_y
                kps_output[..., :2] /= r
                kps_output[..., 0].clamp_(0, shape[1])
                kps_output[..., 1].clamp_(0, shape[0])

                # 画框
                for box in box_output:
                    x1, y1, x2, y2, _, _ = box.cpu().numpy()
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                # 画关键点 + 骨架
                for kpt in reversed(kps_output):
                    for i, k in enumerate(kpt):
                        color_k = [int(x) for x in kpt_color[i]]
                        x_coord, y_coord = k[0], k[1]
                        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                            if len(k) == 3 and k[2] < 0.5:
                                continue
                            cv2.circle(
                                frame,
                                (int(x_coord), int(y_coord)),
                                5,
                                color_k,
                                -1,
                                lineType=cv2.LINE_AA
                            )

                    for i, sk in enumerate(skeleton):
                        pos1 = (int(kpt[sk[0] - 1, 0]), int(kpt[sk[0] - 1, 1]))
                        pos2 = (int(kpt[sk[1] - 1, 0]), int(kpt[sk[1] - 1, 1]))

                        if kpt.shape[-1] == 3:
                            conf1 = kpt[sk[0] - 1, 2]
                            conf2 = kpt[sk[1] - 1, 2]
                            if conf1 < 0.5 or conf2 < 0.5:
                                continue

                        if pos1[0] <= 0 or pos1[1] <= 0 or pos2[0] <= 0 or pos2[1] <= 0:
                            continue

                        cv2.line(
                            frame,
                            pos1,
                            pos2,
                            [int(x) for x in limb_color[i]],
                            thickness=2,
                            lineType=cv2.LINE_AA
                        )

            out.write(frame)

        camera.release()
        out.release()

        logging.info(f"视频推理完成，输出已保存到：{output_path}")
        if frame_count > 0:
            avg_time = total_infer_time / frame_count
            logging.info(f"共处理帧数：{frame_count}，总推理耗时：{total_infer_time:.4f} 秒，平均每帧：{avg_time:.4f} 秒")
        else:
            logging.info("未成功处理任何帧。")


# =========================
# 命令行入口
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".om 模型推理程序")
    parser.add_argument(
        "--model",
        type=str,
        default="weights/yolov8n-pose.om",
        help="离线模型 (.om) 文件路径，例如：yolov8n-pose.om"
    )
    parser.add_argument(
        "--input-w",
        type=int,
        default=640,
        help="模型输入宽度（默认：640）"
    )
    parser.add_argument(
        "--input-h",
        type=int,
        default=640,
        help="模型输入高度（默认：640）"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/test2.mp4",
        help="输入视频路径，例如：./test_video/input.mp4"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/test2_.mp4",
        help="输出视频路径，例如：./test_video/output.mp4"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Ascend 设备 ID（默认：0）"
    )

    args = parser.parse_args()

    context = None
    pose_model = None
    try:
        # 1. 初始化 ACL
        context = init_acl(args.device)

        # 2. 加载姿态模型
        pose_model = Pose(
            modelpath=args.model,
            input_w=args.input_w,
            input_h=args.input_h
        )

        # 3. 执行视频推理
        pose_model.infer(
            video_path=args.video,
            output_path=args.output
        )

    except Exception as e:
        logging.error(f"程序运行过程中发生异常：{e}")
    finally:
        # 4. 释放资源
        if pose_model:
            pose_model.release()
        if context:
            deinit_acl(context, args.device)
        logging.info("程序结束。")

