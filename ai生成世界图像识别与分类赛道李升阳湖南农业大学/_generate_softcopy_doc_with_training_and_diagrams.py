from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


def set_run_font(run, font_name: str, size_pt: int | None = None, bold: bool | None = None):
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.bold = bold


def add_title(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run_font(r, "宋体", size_pt=22, bold=True)


def add_subtitle_center(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run_font(r, "宋体", size_pt=12)


def add_h1(doc: Document, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, "黑体", size_pt=16, bold=True)


def add_h2(doc: Document, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, "黑体", size_pt=14, bold=True)


def add_h3(doc: Document, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, "黑体", size_pt=12, bold=True)


def add_para(doc: Document, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(24)
    r = p.add_run(text)
    set_run_font(r, "宋体", size_pt=12)


def add_list_item(doc: Document, text: str):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    set_run_font(r, "宋体", size_pt=12)


def add_code_block(doc: Document, lines: list[str]):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(24)
        r = p.add_run(line)
        set_run_font(r, "Consolas", size_pt=10)


def _setup_mpl_fonts():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _draw_box(ax, x, y, w, h, text, fc="#E8F3FF", ec="#2E6FBE"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    return box


def _arrow(ax, x1, y1, x2, y2, text: str | None = None):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.2,
        color="#333333",
    )
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", va="bottom", fontsize=9, color="#333333")


def draw_system_architecture(out_png: Path):
    _setup_mpl_fonts()
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    b1 = _draw_box(ax, 0.05, 0.62, 0.20, 0.22, "用户浏览器\n(Web UI)\nindex.html", fc="#FFF7E6", ec="#CC7A00")
    b2 = _draw_box(ax, 0.32, 0.62, 0.26, 0.22, "后端服务\nFlask API\napp.py", fc="#E8F3FF", ec="#2E6FBE")
    b3 = _draw_box(ax, 0.65, 0.62, 0.30, 0.22, "推理引擎\n检测/分割\nYOLO / EfficientNet", fc="#EAF7EA", ec="#2E8B57")
    b4 = _draw_box(ax, 0.32, 0.18, 0.26, 0.22, "文件系统\nuploads/ results/\nmodels/", fc="#F2F2F2", ec="#666666")

    _arrow(ax, 0.25, 0.73, 0.32, 0.73, "HTTP请求")
    _arrow(ax, 0.58, 0.73, 0.65, 0.73, "推理调用")
    _arrow(ax, 0.45, 0.62, 0.45, 0.40, "读写文件")
    _arrow(ax, 0.65, 0.62, 0.58, 0.40, "读取权重/输出结果")
    _arrow(ax, 0.32, 0.68, 0.25, 0.68, "JSON响应")

    ax.text(0.5, 0.95, "系统总体架构（Web + 服务 + 推理）", ha="center", va="center", fontsize=12, fontweight="bold")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def draw_module_diagram(out_png: Path):
    _setup_mpl_fonts()
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    core = _draw_box(ax, 0.38, 0.74, 0.24, 0.18, "核心系统", fc="#E8F3FF", ec="#2E6FBE")

    ui = _draw_box(ax, 0.05, 0.62, 0.25, 0.16, "Web交互模块\ntemplates/index.html", fc="#FFF7E6", ec="#CC7A00")
    api = _draw_box(ax, 0.38, 0.50, 0.24, 0.16, "后端服务模块\napp.py", fc="#E8F3FF", ec="#2E6FBE")
    infer = _draw_box(ax, 0.70, 0.62, 0.25, 0.16, "推理与后处理\nsrc/inference.py\nsrc/cam.py", fc="#EAF7EA", ec="#2E8B57")

    seg = _draw_box(ax, 0.70, 0.38, 0.25, 0.16, "分割模块\nsrc/yolov8_wrapper.py", fc="#EAF7EA", ec="#2E8B57")
    batch = _draw_box(ax, 0.05, 0.38, 0.25, 0.16, "批量处理模块\nbatch_process.py", fc="#F7EAFE", ec="#7A3DB8")
    cfg = _draw_box(ax, 0.38, 0.26, 0.24, 0.16, "配置与启动自检\nconfig.py / run.py", fc="#F2F2F2", ec="#666666")
    train = _draw_box(ax, 0.70, 0.14, 0.25, 0.16, "离线训练模块\nsrc/train.py\nsrc/dataset.py", fc="#FFE8F0", ec="#B83D6B")

    _arrow(ax, 0.50, 0.74, 0.18, 0.70)
    _arrow(ax, 0.50, 0.74, 0.50, 0.66)
    _arrow(ax, 0.50, 0.74, 0.82, 0.70)
    _arrow(ax, 0.50, 0.50, 0.82, 0.62)
    _arrow(ax, 0.50, 0.50, 0.18, 0.46)
    _arrow(ax, 0.50, 0.50, 0.50, 0.42)
    _arrow(ax, 0.82, 0.38, 0.82, 0.30)

    ax.text(0.5, 0.95, "关键模块与代码映射关系", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def draw_training_pipeline(out_png: Path):
    _setup_mpl_fonts()
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    b1 = _draw_box(ax, 0.05, 0.62, 0.20, 0.22, "数据集准备\n按文件名/目录标注\nY=1, N=0", fc="#F2F2F2", ec="#666666")
    b2 = _draw_box(ax, 0.28, 0.62, 0.22, 0.22, "数据增强\n翻转/旋转/抖动\ndataset.py", fc="#FFF7E6", ec="#CC7A00")
    b3 = _draw_box(ax, 0.53, 0.62, 0.20, 0.22, "K折训练\nStratifiedKFold\ntrain.py", fc="#E8F3FF", ec="#2E6FBE")
    b4 = _draw_box(ax, 0.76, 0.62, 0.19, 0.22, "模型选择\nEfficientNetV2\ntimm", fc="#EAF7EA", ec="#2E8B57")

    b5 = _draw_box(ax, 0.20, 0.18, 0.28, 0.22, "两阶段优化\n冻结骨干训练Head\n再全量微调", fc="#E8F3FF", ec="#2E6FBE")
    b6 = _draw_box(ax, 0.52, 0.18, 0.22, 0.22, "阈值选择\n0.3~0.7网格\nbest_threshold()", fc="#F7EAFE", ec="#7A3DB8")
    b7 = _draw_box(ax, 0.77, 0.18, 0.18, 0.22, "导出产物\nbest.pt\nmeta.json", fc="#F2F2F2", ec="#666666")

    _arrow(ax, 0.25, 0.73, 0.28, 0.73)
    _arrow(ax, 0.50, 0.73, 0.53, 0.73)
    _arrow(ax, 0.73, 0.73, 0.76, 0.73)
    _arrow(ax, 0.63, 0.62, 0.34, 0.40, "训练循环")
    _arrow(ax, 0.48, 0.29, 0.52, 0.29)
    _arrow(ax, 0.74, 0.29, 0.77, 0.29)

    ax.text(0.5, 0.95, "检测模型训练流程（离线训练）", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def draw_yolov8_architecture(out_png: Path):
    _setup_mpl_fonts()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Backbone
    _draw_box(ax, 0.05, 0.35, 0.22, 0.3, "Backbone\nCSPDarknet53\n特征提取", fc="#E8F3FF", ec="#2E6FBE")
    
    # Neck
    _draw_box(ax, 0.37, 0.35, 0.22, 0.3, "Neck\nPANet\n多尺度特征融合", fc="#EAF7EA", ec="#2E8B57")
    
    # Head
    _draw_box(ax, 0.69, 0.35, 0.26, 0.3, "Head\nDecoupled Head\n解耦分类与回归", fc="#FFF7E6", ec="#CC7A00")

    # Arrows
    _arrow(ax, 0.27, 0.5, 0.37, 0.5, "C3/C4/C5特征")
    _arrow(ax, 0.59, 0.5, 0.69, 0.5, "融合特征")
    
    # Output Arrows
    _arrow(ax, 0.95, 0.55, 0.99, 0.55, "")
    ax.text(0.97, 0.56, "Class", ha="center", va="bottom", fontsize=8)
    _arrow(ax, 0.95, 0.45, 0.99, 0.45, "")
    ax.text(0.97, 0.46, "Bbox", ha="center", va="bottom", fontsize=8)

    # Loss Info
    ax.text(0.5, 0.15, "Loss Function: CIoU Loss (边界框回归) + DFL (类别不平衡处理)", 
            ha="center", va="center", fontsize=10, 
            bbox=dict(facecolor='#F2F2F2', edgecolor='#666666', boxstyle='round,pad=0.5'))

    ax.text(0.5, 0.9, "YOLOv8m 目标检测模型架构", ha="center", va="center", fontsize=12, fontweight="bold")
    
    fig.tight_layout(pad=0.3)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def add_picture_with_caption(doc: Document, image_path: Path, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    doc.add_picture(str(image_path), width=Inches(6.6))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = cap.add_run(caption)
    set_run_font(rr, "宋体", size_pt=11)


def build_doc(out_path: Path) -> Path:
    doc = Document()

    add_title(doc, "脑肿瘤检测系统")
    add_title(doc, "软件著作权登记用设计说明报告")
    add_subtitle_center(doc, "版本：V1.1")
    add_subtitle_center(doc, f"编写日期：{datetime.now().strftime('%Y-%m-%d')}")
    doc.add_page_break()

    add_h1(doc, "目录")
    add_para(doc, "1 简介")
    add_para(doc, "  1.1 编写目的")
    add_para(doc, "  1.2 使用对象")
    add_para(doc, "  1.3 产品范围")
    add_para(doc, "  1.4 适用与非适用边界")
    add_para(doc, "  1.5 典型应用场景")
    add_para(doc, "2 产品概述")
    add_para(doc, "  2.1 总体框架")
    add_para(doc, "  2.2 系统架构")
    add_para(doc, "  2.3 模块描述")
    add_para(doc, "3 使用说明")
    doc.add_page_break()

    add_h1(doc, "1 简介")

    add_h2(doc, "1.1 编写目的")
    add_para(
        doc,
        "本设计说明报告用于软件著作权登记材料编制，描述“脑肿瘤检测系统”的产品定位、总体框架、系统架构、核心模块、接口与使用方式。"
        "报告从工程实现角度说明软件的组成、数据流与关键处理逻辑，便于第三方理解软件实现范围与工作原理。",
    )

    add_h2(doc, "1.2 使用对象")
    add_para(doc, "本系统面向以下使用对象：")
    add_list_item(doc, "医学影像AI辅助诊断研究人员：用于科研验证与算法对比。")
    add_list_item(doc, "医疗信息化/工程技术人员：用于部署、集成与功能验证。")
    add_list_item(doc, "教学与竞赛使用者：用于课程实验、模型推理演示与批量评测。")
    add_list_item(doc, "非专业终端用户：通过浏览器界面上传影像，查看推理与可视化结果。")

    add_h2(doc, "1.3 产品范围")
    add_para(doc, "系统提供从影像上传、模型推理到结果展示的一体化流程，范围包括：")
    add_list_item(doc, "Web交互：浏览器端上传单张医学影像，查看预测结论、置信度与可视化图。")
    add_list_item(doc, "检测推理：对影像进行肿瘤存在性判断与定位，输出分类与边界框。")
    add_list_item(doc, "分割推理：分割模型输出肿瘤区域掩码（可选）。")
    add_list_item(doc, "结果导出：支持导出检测结果（Web端JSON、批处理CSV）。")
    add_list_item(doc, "批量处理：对目录内图像批量推理并输出报告与可视化结果。")
    add_list_item(doc, "启动自检：检查Python版本、依赖包、模型文件与目录结构。")

    add_h2(doc, "1.4 适用与非适用边界")
    add_para(doc, "适用边界：")
    add_list_item(doc, "适用于在本地计算机或工作站上进行医学影像肿瘤筛查的科研验证与演示。")
    add_list_item(doc, "适用于单机或局域网环境部署，通过HTTP接口提供推理服务。")
    add_list_item(doc, "适用于常见图像格式（PNG、JPG/JPEG、BMP、TIF/TIFF）的二维影像输入。")
    add_para(doc, "非适用边界：")
    add_list_item(doc, "系统输出仅供参考，不替代医师诊断或临床确诊。")
    add_list_item(doc, "不包含生产级用户权限、审计、电子病历集成与数据治理能力。")
    add_list_item(doc, "不提供完整DICOM全流程（标签解析、序列管理、脱敏）能力。")

    add_h2(doc, "1.5 典型应用场景")
    add_list_item(doc, "教学演示：通过浏览器上传样例影像，展示AI推理与可视化结果。")
    add_list_item(doc, "算法验证：更换模型权重后快速验证推理效果与接口字段一致性。")
    add_list_item(doc, "批量评测：对目录影像执行批处理，生成CSV报告与可视化产物。")
    add_list_item(doc, "工程集成：第三方系统通过REST接口调用检测服务并获取JSON结果。")

    doc.add_page_break()

    add_h1(doc, "2 产品概述")

    add_h2(doc, "2.1 总体框架")
    add_para(
        doc,
        "系统总体采用“前端展示 + 后端服务 + 模型推理”的三层框架。前端负责文件选择、上传与结果可视化展示；"
        "后端基于Flask提供HTTP接口与页面渲染，完成文件校验、推理调度、结果组织与返回；"
        "推理层调用训练好的检测/分割模型权重，输出分类/定位/掩码等结果并生成可视化图。",
    )
    add_para(doc, "关键数据对象包括：上传影像文件、预测标签、概率/置信度、可视化图像（Base64编码）、边界框与分割掩码等。")

    add_h2(doc, "2.2 系统架构")
    add_para(doc, "从逻辑架构角度，系统划分为表示层、服务层、算法层与数据层：")
    add_list_item(doc, "表示层（Web UI）：HTML/CSS/JavaScript页面，负责交互与结果展示。")
    add_list_item(doc, "服务层（Flask API）：路由控制、上传管理、参数校验、统一JSON响应。")
    add_list_item(doc, "算法层（推理引擎）：加载模型权重，执行检测/分割推理与后处理。")
    add_list_item(doc, "数据层（文件系统）：模型文件、上传文件、结果文件、批处理输出与可视化产物。")

    add_para(doc, "典型调用链如下：")
    add_code_block(
        doc,
        [
            "浏览器 -> POST /api/detect (multipart/form-data: file)",
            "  Flask 保存 uploads/ 临时文件",
            "  推理模块加载模型并执行预测",
            "  可视化模块生成 bbox/热力图/掩码（按结果）",
            "Flask -> JSON 响应（含 base64 图片与结构化字段） -> 浏览器渲染",
        ],
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        sys_png = tmpdir / "arch.png"
        mod_png = tmpdir / "modules.png"
        train_png = tmpdir / "train.png"
        yolo_png = tmpdir / "yolo_arch.png"
        draw_system_architecture(sys_png)
        draw_module_diagram(mod_png)
        draw_training_pipeline(train_png)
        draw_yolov8_architecture(yolo_png)
        add_picture_with_caption(doc, sys_png, "图2-1 系统总体架构图")
        add_picture_with_caption(doc, mod_png, "图2-2 关键模块与架构分解图")

        add_h2(doc, "2.3 模块描述")

        add_h3(doc, "2.3.1 Web交互模块")
        add_para(doc, "模块职责：提供网页界面，支持拖拽/选择上传影像，展示预测结果、置信度与可视化图，支持结果导出。")
        add_para(doc, "对应实现：templates/index.html（页面与脚本），通过fetch调用后端接口。")

        add_h3(doc, "2.3.2 后端服务模块")
        add_para(doc, "模块职责：提供HTTP路由、文件校验与保存、推理调度、结果封装与错误处理。")
        add_para(doc, "对应实现：app.py。主要接口包括：/、/api/detect、/api/batch_detect、/api/model_info。")
        add_para(doc, "关键约束：限制文件大小（默认16MB）、限制文件扩展名、使用安全文件名，避免不安全路径访问。")

        add_h3(doc, "2.3.3 检测推理模块（Object Detection Module）")
        add_para(doc, "模块职责：对输入影像进行肿瘤检测，输出预测标签与置信度，并在需要时输出定位信息。")
        add_para(doc, "对应实现：app.py中的推理逻辑，以及src/inference.py的批处理推理逻辑。")

        add_para(doc, "模型选择：YOLOv8m (Medium version)。")
        add_para(doc, "选择理由：YOLOv8m 在精度（mAP）和推理速度之间取得了良好的平衡，适合在桌面端进行快速筛查。")

        add_para(doc, "算法原理：")
        add_list_item(doc, "Backbone：采用 CSPDarknet53 结构进行特征提取。")
        add_list_item(doc, "Neck：使用 PANet (Path Aggregation Network) 进行多尺度特征融合，增强对小目标的检测能力。")
        add_list_item(doc, "Head：Decoupled Head 结构，将分类和回归任务解耦，提高收敛速度。")
        add_list_item(doc, "Loss Function：采用 CIoU Loss 进行边界框回归，DFL (Distribution Focal Loss) 处理类别不平衡。")

        add_para(doc, "推理参数：Confidence Threshold = 0.25 (保证较高的召回率)。")
        add_picture_with_caption(doc, yolo_png, "图2-3 YOLOv8m模型架构图")

        add_h3(doc, "2.3.4 可视化与后处理模块")
        add_para(doc, "模块职责：提供模型解释性呈现，包括热力图、掩码与边界框等。")
        add_para(doc, "对应实现：src/cam.py（Grad-CAM、cam_to_mask、mask_to_bbox等），app.py负责编码并返回前端。")

        add_h3(doc, "2.3.5 分割推理模块")
        add_para(doc, "模块职责：对疑似肿瘤影像执行实例分割，输出肿瘤区域掩码与定位框（可选）。")
        add_para(doc, "对应实现：src/yolov8_wrapper.py，基于ultralytics YOLO加载seg_best.pt并执行predict。")
        add_list_item(doc, "默认推理参数：imgsz=1024，conf=0.35，iou=0.5，max_det=3。")

        add_h3(doc, "2.3.6 批量处理模块")
        add_para(doc, "模块职责：对指定目录批量推理并生成CSV报告，可选输出可视化文件。")
        add_para(doc, "对应实现：batch_process.py调用src/inference.py的run方法。")

        add_h3(doc, "2.3.7 配置与启动自检模块")
        add_para(doc, "模块职责：管理路径、阈值、端口等配置，并在启动时完成环境自检。")
        add_para(doc, "对应实现：config.py提供配置常量；run.py检查版本/依赖/模型文件/目录后启动Flask服务。")

        add_h3(doc, "2.3.8 模型训练模块（离线）")
        add_para(doc, "模块职责：用于离线训练检测模型并导出权重文件，供在线推理服务加载。")
        add_para(doc, "对应实现：src/train.py、src/dataset.py、src/utils.py。")
        add_para(doc, "训练数据与标注约定：通过文件名/目录规则自动推断二分类标签，示例规则为：文件名以Y开头表示阳性(1)，以N开头或包含no/normal表示阴性(0)。")
        add_para(doc, "数据增强与预处理：训练阶段包含随机水平/垂直翻转、±10°旋转、亮度对比度抖动；验证阶段仅缩放与归一化。")
        add_para(
            doc,
            "训练策略：采用分层K折交叉验证(StratifiedKFold)保证正负样本比例稳定；每折训练采用两阶段优化，先冻结骨干网络仅训练分类头(AdamW, lr=1e-3)，"
            "再解冻全网络进行微调(骨干lr=1e-4，head lr=5e-4)，并使用余弦退火调度器。为处理类别不平衡，损失函数支持pos_weight加权。"
            "每轮在验证集计算概率并搜索最佳阈值(best_threshold)，以获得最佳准确率对应的权重快照。",
        )
        add_para(doc, "训练产物：每折保存best_{model}_fold{idx}.pt，最终选择最佳折导出best.pt，并生成meta.json记录模型名与阈值。")
        add_picture_with_caption(doc, train_png, "图2-4 检测模型训练流程图")

    doc.add_page_break()

    add_h1(doc, "3 使用说明")
    add_h2(doc, "3.1 运行环境")
    add_para(doc, "建议运行环境：Windows/Linux，Python 3.7及以上；支持CPU或GPU推理。")
    add_para(doc, "核心依赖：torch、torchvision、flask、flask-cors、Pillow、opencv-python、ultralytics、timm、numpy等。")

    add_h2(doc, "3.2 部署与启动")
    add_para(doc, "步骤如下：")
    add_list_item(doc, "进入项目根目录。")
    add_list_item(doc, "安装依赖：pip install -r requirements.txt。")
    add_list_item(doc, "确认模型文件存在：models/det_best.pt、models/seg_best.pt。")
    add_list_item(doc, "启动服务：python run.py（或直接运行python app.py）。")
    add_para(doc, "启动成功后在浏览器访问：http://localhost:5000。")
    add_code_block(doc, ["python run.py", "浏览器打开 http://localhost:5000"])

    add_h2(doc, "3.3 Web端使用")
    add_list_item(doc, "点击选择文件或拖拽影像到上传区域。")
    add_list_item(doc, "系统自动上传并调用检测接口，显示检测结论与置信度。")
    add_list_item(doc, "当检测到肿瘤时，页面额外展示边界框图（以及热力图/掩码图，如已生成）。")
    add_list_item(doc, "可通过导出功能下载检测结果JSON。")

    add_h2(doc, "3.4 API接口说明")
    add_para(doc, "（1）单张图像检测接口")
    add_list_item(doc, "URL：POST /api/detect")
    add_list_item(doc, "请求：multipart/form-data，字段file为图像文件")
    add_list_item(doc, "响应：JSON，包含prediction、confidence、probability、original_image等字段")
    add_code_block(
        doc,
        [
            "示例响应字段（节选）：",
            "{",
            '  \"filename\": \"image.jpg\",',
            '  \"prediction\": \"tumor_detected\" | \"no_tumor\",',
            '  \"confidence\": 0.95,',
            '  \"probability\": 0.82,',
            '  \"original_image\": \"<base64>\",',
            '  \"bbox_image\": \"<base64>\",',
            '  \"bbox\": [x1, y1, x2, y2]',
            "}",
        ],
    )

    add_para(doc, "（2）批量检测接口")
    add_list_item(doc, "URL：POST /api/batch_detect")
    add_list_item(doc, "请求：application/json，字段image_paths为图像路径数组")
    add_list_item(doc, "响应：JSON，字段results为逐图结果列表")

    add_para(doc, "（3）模型信息接口")
    add_list_item(doc, "URL：GET /api/model_info")
    add_list_item(doc, "响应：JSON，包含device、模型加载状态与模型类型信息")

    add_h2(doc, "3.5 批量处理")
    add_para(doc, "可使用命令行脚本对目录内影像进行批量推理并生成报告，例如：")
    add_code_block(doc, ["python batch_process.py --input 测试数据集 --output 批量结果 --model both --report"])

    add_h2(doc, "3.6 训练复现实例（可选）")
    add_para(doc, "如需复现检测模型训练，可使用训练脚本，示例命令：")
    add_code_block(doc, ["python -m src.train --data \"训练数据目录\" --out \"./weights\" --model efficientnetv2_s --img-size 224 --folds 5"])
    add_para(doc, "训练完成后将best.pt拷贝/重命名为models/det_best.pt即可被推理服务加载。")

    add_h2(doc, "3.7 输出文件与目录说明")
    add_list_item(doc, "models/：模型权重文件（det_best.pt、seg_best.pt）。")
    add_list_item(doc, "uploads/：Web上传的临时文件目录。")
    add_list_item(doc, "results/：Web服务输出与下载目录。")
    add_list_item(doc, "输出结果/：批处理可视化与CSV输出目录。")

    try:
        doc.save(out_path)
        return out_path
    except PermissionError:
        fallback = out_path.with_name(out_path.stem + f"_更新_{datetime.now().strftime('%Y%m%d_%H%M%S')}" + out_path.suffix)
        doc.save(fallback)
        return fallback


def main():
    out_path = Path("脑肿瘤检测系统_设计说明报告_软著.docx").resolve()
    saved = build_doc(out_path)
    print(str(saved.resolve()))


if __name__ == "__main__":
    main()

