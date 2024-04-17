#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
# from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,#导入hooks模块
    launch,
)

def setup_cfg(args):
    # cfg = get_cfg()
    # # cuda context is initialized before creating dataloader, so we don't fork anymore
    # cfg.DATALOADER.NUM_WORKERS = 0
    # add_pointrend_config(cfg)
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    cfg = LazyConfig.load(args.config_file)#从配置文件加载训练配置，args.config_file 包含了训练的配置信息。
    #cfg = LazyConfig.apply_overrides(cfg, args.opts)：根据命令行参数中的覆盖选项，修改配置文件中的配置项。
    # 这可以用于在命令行中修改配置，例如更改学习率、批大小等。
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    print(cfg)
    #增加start=========
    # cfg.DATASETS.TRAIN=instantiate(cfg.DATASETS.TRAIN)
    # #print("cfg.DATASETS.TRAIN:",cfg.DATASETS.TRAIN)
    # cfg.DATASETS.TEST=instantiate(cfg.DATASETS.TEST)
    #print("cfg.DATASETS.TEST:",cfg.DATASETS.TEST)
    #assert cfg.DATASETS.TRAIN == "mydata_train"
    #assert cfg.DATASETS.TEST == "mydata_val"
    # cfg.DATASETS.TRAIN = ("oil_train")
    # cfg.DATASETS.TEST = ("oil_val",)  # 没有不用填
    #cfg.DATALOADER.NUM_WORKERS = 2
    # # 预训练模型文件,可自行提前下载
    # cfg.MODEL.WEIGHTS = "/home/arina/repos/EVA/EVA-02/det/outputs/bpla_winter_oil_2048/model_0001999.pth"

    # cfg.MODEL.WEIGHTS = instantiate(cfg.MODEL.WEIGHTS)
    # cfg.MODEL.META_ARCHITECTURE=instantiate(cfg.MODEL.META_ARCHITECTURE)
    # cfg.MODEL.DEVICE=instantiate(cfg.MODEL.DEVICE)
    # cfg.MODEL.PIXEL_MEAN=instantiate(cfg.MODEL.PIXEL_MEAN)
    # 或者使用自己的预训练模型
    #cfg.SOLVER.IMS_PER_BATCH = 2
    #========end
    #default_setup(cfg, args)：初始化 Detectron2 库，包括 GPU 设置、分布式训练等。
    default_setup(cfg, args)
    return cfg


def export_caffe2_tracing(cfg, torch_model, inputs):
    from detectron2.export import Caffe2Tracer

    tracer = Caffe2Tracer(cfg, torch_model, inputs)
    if args.format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=inputs)
        return caffe2_model
    elif args.format == "onnx":
        import onnx

        onnx_model = tracer.export_onnx()
        onnx.save(onnx_model, os.path.join(args.output, "model.onnx"))
    elif args.format == "torchscript":
        ts_model = tracer.export_torchscript()
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)


# experimental. API not yet final
def export_scripting(torch_model):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
    assert args.format == "torchscript", "Scripting only supports torchscript format."

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, args.output)
    # TODO inference in Python now missing postprocessing glue code
    return None


# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f, opset_version=STABLE_ONNX_OPSET_VERSION)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


def get_sample_inputs(args):

    if args.sample_image is None:
        # get a first batch from dataset
        #data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        data_loader = instantiate(cfg.dataloader.test)
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(args.sample_image, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["caffe2_tracing", "tracing", "scripting"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(args)

    # create a torch model
    #torch_model = build_model(cfg)
    torch_model = instantiate(cfg.model)  # 创建了目标检测模型，cfg.model 包含了用于构建模型的配置信息。
    torch_model.to(cfg.train.device)  # 模型移动到指定的计算设备，通常是 GPU
    DetectionCheckpointer(torch_model).resume_or_load("/home/arina/repos/EVA/EVA-02/det/outputs/bpla_winter_oil_2048/model_0001999.pth")
    torch_model.eval()

    # get sample data
    import glob, cv2

    for i in glob.glob('/home/arina/projs/spils/BPLA_winter/yolo_01/images/val/*'):
        img = cv2.imread(i)
        height, width = img.shape[:2]
        factor = max(height, width)/2048
        dim = (int(height//factor), int(width//factor))
        # print(dim, type(dim[1]))
        img2 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (dim[1], dim[0]))
        with torch.no_grad():

            height, width = dim

            image = torch.as_tensor(img2.astype("float32").transpose(2, 0, 1), device='cuda')
            sample_inputs = [{"image": image, "height": torch.tensor(height), "width": torch.tensor(width)}]

            break
    # convert and save model
    if args.export_method == "caffe2_tracing":
        exported_model = export_caffe2_tracing(cfg, torch_model, sample_inputs)
    elif args.export_method == "scripting":
        exported_model = export_scripting(torch_model)
    elif args.export_method == "tracing":
        exported_model = export_tracing(torch_model, sample_inputs)

    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        dataset = cfg.dataloader
        #data_loader = build_detection_test_loader(cfg, dataset)
        data_loader = instantiate(cfg.dataloader.test)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, output_dir=args.output)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
    logger.info("Success.")