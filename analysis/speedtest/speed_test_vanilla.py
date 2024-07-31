import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.utils.benchmark as benchmark

import models.vision_transformer as vits
from utils.dino import MultiCropWrapper, DINOHead, DINOLoss
from minsim import MinSim


if __name__ == "__main__":
    bs = 32
    out_dim = 65536
    local_crops_number = 8

    student = vits.vit_small(
        image_size=224,
        patch_size=16,
        drop_path_rate=0.1,
    )
    teacher = vits.vit_small(
        image_size=224,
        patch_size=16,
    )

    student = MultiCropWrapper(
        student,
        DINOHead(student.embed_dim, out_dim),
    ).cuda()
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(teacher.embed_dim, out_dim, False),
    ).cuda()
    for p in teacher.parameters():
        p.requires_grad = False
    
    dino_loss = DINOLoss(
        out_dim,
        2 + local_crops_number,  # total number of crops = 2 global crops + local_crops_number
        0.04,
        0.04,
        0,
        100,
    ).cuda()

    optimizer = torch.optim.AdamW(student.parameters())
    fp16_scaler = GradScaler()

    # select_fn = MinSim("cross", student, teacher, dino_loss, fp16_scaler,
    #                    2, cfg.num_local_crops_loader,
    #                 cfg.local_crops_number, cfg.limit_comparisons)

    images = [torch.randn(bs, 3, 224, 224).cuda() for _ in range(2)]
    images += [torch.randn(bs, 3, 96, 96).cuda() for _ in range(local_crops_number)]

    def test_fn(images):
        for _ in range(100):

            # inputs, _, _ = select_fn(images, 0)

            with autocast():
                teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = student(images)
                loss = dino_loss(student_output, teacher_output, 0)

            optimizer.zero_grad()
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

    
    t = benchmark.Timer(
        stmt="test_fn(images)",
        globals={"test_fn": test_fn, "images": images},
    )

    print(t.timeit(1))
