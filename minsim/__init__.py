import itertools as it

import numpy as np
import torch


class MinSim(object):
    def __init__(self,
                 select_fn,
                 student,
                 teacher,
                 criterion,
                 fp16,
                 num_global_crops_loader,
                 num_local_crops_loader,
                 local_crops_number,
                 limit):
        self.select_fn = select_fn
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.fp16 = fp16
        self.num_global_crops_loader = num_global_crops_loader
        self.num_local_crops_loader = num_local_crops_loader
        self.local_crops_number = local_crops_number
        self.limit = limit

        # list of all valid combinations for global crops
        # equal to list of tuples with even and odd numbers
        self.global_combinations = it.product(range(0, num_global_crops_loader, 2),
                                              range(1, num_global_crops_loader, 2))
        # list of all valid combinations for local crops
        self.local_combinations = list(
            it.combinations(range(num_global_crops_loader, num_global_crops_loader + num_local_crops_loader),
                            local_crops_number))
        self.all_combinations = it.product(self.global_combinations, self.local_combinations)
        self.all_combinations = [g + l for g, l in self.all_combinations]

    def __call__(self, images, epoch):
        if self.select_fn == 'identity':
            return images, torch.zeros(1), torch.zeros(1)
        return self.cross(images, epoch)

    @staticmethod
    def identity(images, *args, **kwargs):
        return images, torch.zeros(1), torch.zeros(1)

    @torch.no_grad()
    def cross(self, images, epoch):
        bs = images[0].size(0)
        device = self.student.device

        score = torch.zeros(bs, device=device)
        selected = torch.zeros((2 + self.local_crops_number, bs), dtype=torch.uint8, device=device)
        out = [torch.empty_like(images[0]) for _ in range(2)] + \
              [torch.empty_like(images[-1]) for _ in range(self.local_crops_number)]

        with torch.cuda.amp.autocast(self.fp16 is not None):
            teacher_output = self.teacher(images[:self.num_global_crops_loader])
            student_output = self.student(images)
            student_output, teacher_output = self.criterion.prepare_outputs(student_output, teacher_output, epoch)
            student_output, teacher_output = student_output.chunk(len(images)), teacher_output.chunk(
                self.num_global_crops_loader)

        if self.limit:
            combinations = [self.all_combinations[i] for i in torch.randperm(len(self.all_combinations))[:self.limit]]
        else:
            combinations = self.all_combinations

        for idx in combinations:
            _teacher_out = [teacher_output[x] for x in idx[:2]]
            _student_out = [student_output[x] for x in idx]
            with torch.cuda.amp.autocast(self.fp16 is not None):
                sim = self.criterion.select_forward(_student_out, _teacher_out)  # sample-loss
                score, indices = torch.stack((score, sim)).max(dim=0)
                indices = indices.type(torch.bool)

            for n, ids in enumerate(idx):
                selected[n][indices] = ids
                # slower than below implementation
                # out[n] = torch.where(indices[:, None, None, None], images[ids], out[n])

        for n in range(2):
            for m in range(self.num_global_crops_loader):
                out[n] = torch.where((selected[n] == m)[:, None, None, None], images[m], out[n])
        for n in range(2, len(out)):
            for m in range(self.num_global_crops_loader, len(images)):
                out[n] = torch.where((selected[n] == m)[:, None, None, None], images[m], out[n])

        # check that all images are selected correctly
        # for n, ids in enumerate(selected):
        #     for m, idx in enumerate(ids):
        #         assert torch.equal(out[n][m], images[idx][m])

        return out, selected, score