import torch
import torch.nn.functional as nnf


@torch.no_grad()
def select_crops_identity(images, model, fp16):
    return images[0], images[1], torch.zeros(1), torch.zeros(1)


@torch.no_grad()
def select_crops_cross(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.single_forward(torch.cat(images, dim=0))
    p1s, z1s = model_out[0].chunk(len(images)), model_out[1].chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)
    selected = torch.zeros((2, b), dtype=torch.uint8)

    for n in range(len(images)):
        p1, z1 = p1s[n], z1s[n]
        for m in range(n + 1, len(images)):
            p2, z2 = p1s[m], z1s[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = (nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1)) * 0.5
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)
            selected[0][indices] = n
            selected[1][indices] = m
            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2, selected, score


@torch.no_grad()
def select_crops_avgpool(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.avgpool_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_conv1_layer(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.conv1_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_first_layer(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.first_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_second_layer(images, model, fp16):
    b, c, h, w = images[0].shape
    device = images[0].device

    with torch.cuda.amp.autocast(fp16 is not None):
        model_out = model.module.second_layer(torch.cat(images, dim=0))
    activations = model_out.chunk(len(images))

    out1 = torch.zeros_like(images[0])
    out2 = torch.zeros_like(images[0])
    score = torch.full([b], torch.inf, device=device)

    for n, x in enumerate(activations):
        e1 = activations[n]
        for m in range(n + 1, len(activations)):
            e2 = activations[m]

            with torch.cuda.amp.autocast(fp16 is not None):
                sim = nnf.cosine_similarity(e1, e2)
                score, indices = torch.stack((score, sim)).min(dim=0)
                indices = indices.type(torch.bool)

            out1 = torch.where(indices[:, None, None, None], images[n], out1)
            out2 = torch.where(indices[:, None, None, None], images[m], out2)

    return out1, out2


@torch.no_grad()
def select_crops_anchor(images, model, fp16):
    with torch.cuda.amp.autocast(fp16 is not None):
        embeds = [model.module.simple_forward(img) for img in images]
    p1, z1 = embeds[0]
    scores = [nnf.cosine_similarity(p1, z2) + nnf.cosine_similarity(p2, z1) for p2, z2 in embeds[1:]]
    stacked = torch.stack(scores)
    values, indic = stacked.min(dim=0)
    a = nnf.one_hot(indic, len(scores)).T
    out = torch.zeros_like(images[0])
    for n, img in enumerate(images[1:]):
        out += a[n].view(-1, 1, 1, 1) * img
    return images[0], out


names = {
    "anchor": select_crops_anchor,
    "cross": select_crops_cross,
    "conv1layer": select_crops_conv1_layer,
    "firstlayer": select_crops_first_layer,
    "secondlayer": select_crops_second_layer,
    "avgpool": select_crops_avgpool,
    "identity": select_crops_identity,
}
