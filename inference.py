import os

import acl
import numpy as np
import cv2, torch
import torch_npu
# 预处理：LetterBox（官方等比缩放+padding，支持 pose/det/seg）
from ultralytics.data.augment import LetterBox  # class LetterBox :contentReference[oaicite:0]{index=0}
# __call__ 支持 image=... 直接传图返回图 :contentReference[oaicite:1]{index=1}

# 后处理：NMS（官方 non_max_suppression）
from ultralytics.utils.nms import non_max_suppression  # function non_max_suppression :contentReference[oaicite:2]{index=2}

# 坐标还原：scale_boxes / xywh2xyxy 等（官方 ops）
from ultralytics.utils.ops import scale_boxes, xywh2xyxy  # scale_boxes / xywh2xyxy :contentReference[oaicite:3]{index=3}

# 可视化：Annotator 画框/画关键点骨架 + Colors 自带 pose palette
from ultralytics.utils.plotting import Annotator, Colors  # Annotator import 示例 :contentReference[oaicite:4]{index=4}
# Annotator.kpts() 用来画 keypoints :contentReference[oaicite:5]{index=5}


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


class Model:
    def __init__(self, model_path):
        # 初始化函数
        self.device_id = 0

        # step1: 初始化
        ret = acl.init()
        # 指定运算的Device
        ret = acl.rt.set_device(self.device_id)

        # step2: 加载模型，本示例为ResNet-50模型
        # 加载离线模型文件，返回标识模型的ID
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        # 创建空白模型描述信息，获取模型描述信息的指针地址
        self.model_desc = acl.mdl.create_desc()
        # 通过模型的ID，将模型的描述信息填充到model_desc
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)

        # step3：创建输入输出数据集
        # 创建输入数据集
        self.input_dataset, self.input_data = self.prepare_dataset('input')
        # 创建输出数据集
        self.output_dataset, self.output_data = self.prepare_dataset('output')
    """
    在调用pyACL接口进行模型推理时，模型推理有输入、输出数据，输入、输出数据需要按照pyACL规定的数据类型存放。相关数据类型如下：

    使用aclmdlDesc类型的数据描述模型基本信息（例如输入/输出的个数、名称、数据类型、Format、维度信息等）。

    模型加载成功后，用户可根据模型的ID，调用该数据类型下的操作接口获取该模型的描述信息，进而从模型的描述信息中获取模型输入/输出的个数、内存大小、维度信息、Format、数据类型等信息。

    使用aclDataBuffer类型的数据来描述每个输入/输出的内存地址、内存大小。

    调用aclDataBuffer类型下的操作接口获取内存地址、内存大小等，便于向内存中存放输入数据、获取输出数据。

    使用aclmdlDataset类型的数据描述模型的输入/输出数据。

    模型可能存在多个输入、多个输出，调用aclmdlDataset类型的操作接口添加多个aclDataBuffer类型的数据。
    """
    def prepare_dataset(self, io_type):
        # 准备数据集
        if io_type == "input":
            # 获得模型输入的个数
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            # 获得模型输出的个数
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index
            # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        dataset = acl.mdl.create_dataset()
        datas = []
        for i in range(io_num):
            # 获取所需的buffer内存大小
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            # 申请buffer内存
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            # 从内存创建buffer数据
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            # 将buffer数据添加到数据集
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, datas
    # 推理
    def forward(self, inputs):
        # 执行推理任务
        # 遍历所有输入，拷贝到对应的buffer内存中
        input_num = len(inputs)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            # 将图片数据从Host传输到Device。
            ret = acl.rt.memcpy(self.input_data[i]["buffer"],  # 目标地址 device
                                self.input_data[i]["size"],  # 目标地址大小
                                bytes_ptr,  # 源地址 host
                                len(bytes_data),  # 源地址大小
                                ACL_MEMCPY_HOST_TO_DEVICE)  # 模式:从host到device
        # 执行模型推理。
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # 处理模型推理的输出数据，输出top5置信度的类别编号。
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # 将推理输出数据从Device传输到Host。
            ret = acl.rt.memcpy(buffer_host,  # 目标地址 host
                                self.output_data[i]["size"],  # 目标地址大小
                                self.output_data[i]["buffer"],  # 源地址 device
                                self.output_data[i]["size"],  # 源地址大小
                                ACL_MEMCPY_DEVICE_TO_HOST)  # 模式：从device到host
            # 从内存地址获取bytes对象
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            # 按照float32格式将数据转为numpy数组
            data = np.frombuffer(bytes_out, dtype=np.float32)
            inference_result.append(data)
        vals = np.array(inference_result).flatten()

        return vals
    # 析构函数
    def __del__(self):
        # 析构函数 按照初始化资源的相反顺序释放资源。
         # 销毁输入输出数据集
         for dataset in [self.input_data, self.output_data]:
             while dataset:
                 item = dataset.pop()
                 ret = acl.destroy_data_buffer(item["data"])    # 销毁buffer数据
                 ret = acl.rt.free(item["buffer"])              # 释放buffer内存
         ret = acl.mdl.destroy_dataset(self.input_dataset)      # 销毁输入数据集
         ret = acl.mdl.destroy_dataset(self.output_dataset)     # 销毁输出数据集
         # 销毁模型描述
         ret = acl.mdl.destroy_desc(self.model_desc)
         # 卸载模型
         ret = acl.mdl.unload(self.model_id)
         # 释放device
         ret = acl.rt.reset_device(self.device_id)
         # acl去初始化
         ret = acl.finalize()
