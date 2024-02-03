import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import gc
import os
import random
import math
import torch
import pandas as pd
import numpy as np
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import nn, cos, sin
from collections import OrderedDict


class TrainData(Dataset):
    """
    A Torch Dataset class to import point cloud data
    """

    def __init__(
        self,
        input_dir,
        query_dir,
        patch_k,
        device,
        num_sample=None,
        num_query=2048,
        num_op=None,
        seed=0,
    ):
        self.device = device
        self.patch_k = patch_k

        self.labels = None
        self.query_vectors = None
        self.knn_coords = None

        # Reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        if input_dir == query_dir:
            first = False
        else:
            first = True

        file_li = os.listdir(input_dir)
        random.shuffle(file_li)

        print(f"{len(file_li)} point clouds in the dataset")
        i = 0
        for filename in tqdm(file_li):
            if num_sample != None and i == num_sample:
                break

            input_df = pd.read_csv(
                f"{input_dir}/{filename}", sep=" ", names=["x", "y", "z"]
            )
            if num_op == None:
                num_op = len(input_df)

            input_pc = (
                torch.tensor(input_df.sample(frac=1, random_state=seed).values)
                .double()
                .to("cpu")
            )

            query_df = pd.read_csv(
                f"{query_dir}/{filename}", sep=" ", names=["x", "y", "z"]
            )
            query_pc = (
                torch.tensor(query_df.sample(frac=1, random_state=seed).values)
                .double()
                .to("cpu")
            )
            op = (
                generate_op(
                    farthest_point_sampling(input_df, num_op),
                    input_df,
                    "cpu",
                    k=patch_k,
                    calculate_mean=False,
                    perturb=False,
                )
                .double()
                .to("cpu")
            )

            op_xyz, op_indices = KNN(op, query_pc, 1, include_nearest=True)
            op_xyz = op_xyz.squeeze()
            op_indices = op_indices.squeeze()

            knn_coords, _ = KNN(
                input_pc,
                query_pc,
                self.patch_k,
                include_nearest=first,
            )

            # relative positioning
            knn_coords -= op_xyz.unsqueeze(1)
            knn_depths = torch.norm(knn_coords, dim=-1, keepdim=True)
            query_pc -= op_xyz

            # Downsample
            if num_sample != None:
                knn_coords = knn_coords[: num_query // num_sample + 1]
                knn_depths = knn_depths[: num_query // num_sample + 1]
                query_pc = query_pc[: num_query // num_sample + 1]
            else:
                knn_coords = knn_coords[: num_query // len(file_li) + 1]
                knn_depths = knn_depths[: num_query // len(file_li) + 1]
                query_pc = query_pc[: num_query // len(file_li) + 1]

            # query vector and grouth truth depth
            labels = torch.norm(query_pc, dim=1)
            scaling_factor = knn_depths.max(1, keepdims=True).values
            query_vectors = query_pc / labels.unsqueeze(1)
            knn_coords = knn_coords / scaling_factor

            # Data augmentation with random rotation
            rotation_matrices = (
                torch.cat([random_rotation() for _ in range(len(labels))], 0)
                .double()
                .to("cpu")
            )
            query_vectors = (query_vectors.unsqueeze(1) @ rotation_matrices).squeeze()
            knn_coords = knn_coords @ rotation_matrices

            if self.labels == None:
                self.labels = labels / scaling_factor.squeeze()
                self.query_vectors = query_vectors
                self.knn_coords = knn_coords
            else:
                self.labels = torch.cat(
                    [self.labels, labels / scaling_factor.squeeze()], 0
                )
                self.query_vectors = torch.cat([self.query_vectors, query_vectors], 0)
                self.knn_coords = torch.cat(
                    [
                        self.knn_coords,
                        knn_coords,
                    ],
                    0,
                )

            i += 1

        indices = torch.randperm(len(self.labels))[:num_query]
        self.query_vectors = self.query_vectors[indices]
        self.knn_coords = self.knn_coords[indices]
        self.labels = self.labels[indices]

        garbage_collect(
            [
                input_pc,
                query_pc,
                op,
                knn_coords,
                knn_depths,
                labels,
                query_vectors,
            ]
        )

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.query_vectors[idx].to(self.device),
            self.knn_coords[idx].to(self.device),
            self.labels[idx].to(self.device),
        )

    def __len__(self):
        return len(self.labels)


class UpsampleData(Dataset):
    """
    A Torch Dataset class to import point cloud data
    """

    def __init__(
        self,
        input_pc,
        patch_k,
        query_k,
        device,
        output_size,
        num_op=None,
        real_scanned=False,
    ):
        self.device = device
        self.patch_k = patch_k

        input_pc = input_pc.to(self.device)
        input_df = pd.DataFrame(input_pc.cpu().numpy(), columns=["x", "y", "z"])

        if num_op == None:
            num_op = len(input_df) // 32 + 1

        query_pc = self.novel_queries(
            input_df,
            input_df,
            query_k,
            self.device,
            output_size,
            real_scanned=real_scanned,
        )

        query_pc = query_pc.double().to(self.device)

        # if real_scanned:
        #     op = torch.tensor([[0, 0, 0]])
        # else:
        op = (
            generate_op(
                farthest_point_sampling(input_df, num_op),
                input_df,
                self.device,
                k=self.patch_k,
                calculate_mean=False,
                perturb=False,
                real_scanned=real_scanned,
            )
            .double()
            .to(self.device)
        )

        knn_coords, _ = KNN(
            input_pc,
            query_pc,
            self.patch_k,
            include_nearest=True,
            cossim=False,
            device=self.device,
        )
        step = (knn_coords.shape[1]) // 16
        if step < 1:
            step = 1

        knn_coords = knn_coords[
            :,
            torch.arange(start=0, end=knn_coords.shape[1], step=step),
        ]

        op_xyz, _ = KNN(op, query_pc, 1, include_nearest=True, device=self.device)
        op_xyz = op_xyz.squeeze()

        # relative positioning
        self.op_pos = op_xyz
        knn_coords -= self.op_pos.unsqueeze(1)
        query_pc -= op_xyz

        knn_depths = torch.norm(knn_coords, dim=-1, keepdim=True)
        query_depths = torch.norm(query_pc, dim=1)
        query_vectors = query_pc / query_depths.unsqueeze(1)

        self.scaling_factor = knn_depths.max(1, keepdims=True).values
        self.knn_coords = knn_coords / self.scaling_factor
        self.query_vectors = query_vectors

        garbage_collect(
            [
                input_pc,
                query_pc,
                op,
                knn_coords,
                knn_depths,
                query_depths,
                query_vectors,
            ]
        )

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.query_vectors[idx].to(self.device),
            self.knn_coords[idx].to(self.device),
            self.op_pos[idx].to(self.device),
            self.scaling_factor[idx].to(self.device),
        )

    def __len__(self):
        return len(self.query_vectors)

    def novel_queries(
        self, target, reference, k, device, output_size, real_scanned=False
    ):
        target = torch.tensor(target[["x", "y", "z"]].values).to(device)
        reference = torch.tensor(reference[["x", "y", "z"]].values).to(device)

        if real_scanned:
            num_chunks = 1

            try:
                while True:
                    try:
                        input_chunks = torch.chunk(target, num_chunks)

                        input_knn_li = []
                        for chunk in input_chunks:
                            input_knn_li.append(
                                KNN(
                                    target,
                                    chunk,
                                    k,
                                    include_nearest=True,
                                    cossim=True,
                                    device=device,
                                )[0]
                            )
                        break
                    except torch.cuda.OutOfMemoryError:
                        num_chunks *= 2
                input_knn = torch.cat(input_knn_li, 0)

                knn_dists = torch.norm(input_knn - target.unsqueeze(1), dim=-1)

                max_dist, _ = knn_dists.topk(2, largest=True)
                max_dist = max_dist[:, -1].unsqueeze(1)
                min_dist, _ = knn_dists.topk(2, largest=False)
                min_dist = min_dist[:, -1].unsqueeze(1)

                valid_idx = knn_dists < max_dist
                valid_idx *= knn_dists > min_dist

                valid_dist, _ = knn_dists.flatten().topk(
                    len(knn_dists.flatten()) // 2, largest=True
                )
                valid_dist = valid_dist[-1]

                valid_idx *= knn_dists > valid_dist
            except RuntimeError:
                raise IndexError

            target = target.unsqueeze(1).repeat(1, valid_idx.shape[1], 1)[valid_idx]

            valid_neighbour = input_knn[valid_idx]
            # valid_dists = knn_dists[valid_idx]
            # valid_dist_mean = torch.mean(knn_dists[valid_idx])

            try:
                mult = output_size // len(target) + 1
            except ZeroDivisionError:
                raise IndexError
            while True:
                queries = torch.cat(
                    [
                        (
                            target * (i + 1) / (mult + 1)
                            + valid_neighbour * (mult - i) / (mult + 1)
                        )
                        for i in range(0, mult)
                    ],
                    0,
                )

                # queries = []
                # for idx, point in enumerate(target):
                #     dynamic_mult = int(mult * valid_dists[idx] / valid_dist_mean)
                #     for i in range(0, dynamic_mult):
                #         queries.append(
                #             (
                #                 point * (i + 1) / (dynamic_mult + 1)
                #                 + valid_neighbour[idx]
                #                 * (dynamic_mult - i)
                #                 / (dynamic_mult + 1)
                #             ).unsqueeze(0)
                #         )
                # queries = torch.cat(queries, 0)

                queries = torch.unique(queries, dim=0)

                queries, _ = select_aoi_points(queries, reference, device, num_chunks=1)

                if len(queries) == 0:
                    raise IndexError
                if len(queries) < output_size:
                    mult += 1
                    continue

                if output_size > len(queries):
                    mult += 1
                else:
                    break

        else:
            queries = []
            for point in target:
                rel_pos = reference - point.unsqueeze(0)
                rel_dist = rel_pos.norm(dim=-1)
                rel_vectors = rel_pos / rel_dist.unsqueeze(-1)

                criteria = -rel_dist

                _, knn_indices = criteria.topk(k + 1, largest=True)

                queries += [
                    (point + reference[knn_indices[i]]).unsqueeze(0) / 2
                    for i in range(1, k + 1)
                    if torch.sum(
                        torch.sum(
                            rel_vectors[knn_indices[1 : i + 1]]
                            * rel_vectors[knn_indices[i]],
                            -1,
                        )
                        > math.cos(math.pi / 6)
                    )
                    == 1
                ][:6]

            queries = torch.cat(queries, 0)

        garbage_collect([target, reference])

        if real_scanned:
            query_pc = queries
        else:
            query_df = pd.DataFrame(
                torch.unique(queries, dim=0).cpu().numpy(),
                columns=["x", "y", "z"],
            )
            query_pc = torch.tensor(
                farthest_point_sampling(query_df, output_size - len(target))[
                    ["x", "y", "z"]
                ].values
            )

        return query_pc


def random_rotation():
    roll = torch.randn(1)
    yaw = torch.randn(1)
    pitch = torch.randn(1)

    one = torch.ones(1)
    zero = torch.zeros(1)

    rotation_x = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos(roll), -sin(roll)]),
            torch.stack([zero, sin(roll), cos(roll)]),
        ]
    ).reshape(3, 3)

    rotation_y = torch.stack(
        [
            torch.stack([cos(pitch), zero, sin(pitch)]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin(pitch), zero, cos(pitch)]),
        ]
    ).reshape(3, 3)

    rotation_z = torch.stack(
        [
            torch.stack([cos(yaw), -sin(yaw), zero]),
            torch.stack([sin(yaw), cos(yaw), zero]),
            torch.stack([zero, zero, one]),
        ]
    ).reshape(3, 3)

    return torch.mm(torch.mm(rotation_z, rotation_y), rotation_x).unsqueeze(0)


def read_test_file(filename):
    with open(filename, "r") as test_file:
        if "OFF" != test_file.readline().strip():
            raise ("Not a valid OFF header")
        n_verts, n_faces, _ = tuple(
            [int(s) for s in test_file.readline().strip().split(" ")]
        )
        verts = np.array(
            [
                [float(s) for s in test_file.readline().strip().split(" ")]
                for _ in range(n_verts)
            ]
        )
        faces = np.array(
            [
                [int(s) for s in test_file.readline().strip().split(" ")][1:]
                for _ in range(n_faces)
            ]
        )

    return verts, faces


def KNN(
    references, xyz, k, include_nearest=False, cossim=False, device="cpu", num_chunks=1
):
    if cossim:
        reference_vectors = references.to(device) / references.to(device).norm(
            dim=-1, keepdim=True
        )

        # Cosine similarities
        while True:
            try:
                chunks = torch.chunk(xyz, num_chunks)

                criteria_li = []
                for chunk in chunks:
                    query_vector = chunk.to(device) / chunk.to(device).norm(
                        dim=-1, keepdim=True
                    )
                    cossim = torch.sum(
                        reference_vectors.unsqueeze(0).repeat(
                            [query_vector.shape[0], 1, 1]
                        )
                        * query_vector.unsqueeze(1).repeat(
                            [1, reference_vectors.shape[0], 1]
                        ),
                        dim=-1,
                    )
                    criteria_li.append(cossim)
                break
            except torch.cuda.OutOfMemoryError:
                num_chunks *= 2

    else:
        # Distances between observation points and input points
        num_chunks = 1
        while True:
            try:
                chunks = torch.chunk(xyz, num_chunks)

                criteria_li = []
                for chunk in chunks:
                    dist = torch.norm(
                        references.unsqueeze(0).repeat([chunk.shape[0], 1, 1])
                        - chunk.unsqueeze(1).repeat([1, references.shape[0], 1]),
                        dim=2,
                    )
                    criteria_li.append(-dist)
                break
            except torch.cuda.OutOfMemoryError:
                num_chunks *= 2

    # first == False if input and query point clouds are the same
    topk_idx_li = []
    knn_li = []
    for criteria in criteria_li:
        if include_nearest == True:
            if criteria.shape[-1] < k:
                k = criteria.shape[-1]
            topk_indices = torch.topk(
                criteria, k, largest=True, sorted=True, dim=1
            ).indices
            knn = references[topk_indices]
        else:
            if criteria.shape[-1] < k + 1:
                k = criteria.shape[-1] - 1
            topk_indices = torch.topk(
                criteria, k + 1, largest=True, sorted=True, dim=1
            ).indices
            topk_indices = topk_indices[:, 1:]
            knn = references[topk_indices]

        topk_idx_li.append(topk_indices)
        knn_li.append(knn)

    return torch.cat(knn_li, 0), torch.cat(topk_idx_li, 0)


def farthest_point_sampling(pc, num_sample, device="cuda"):
    # Farthest point sampling using implementation by
    #  @article{open3d,
    #    author  = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    #    title   = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    #    journal = {arXiv:1801.09847},
    #    year    = {2018},
    # }
    vertices = np.array(
        pd.DataFrame(
            [
                pc["x"],
                pc["y"],
                pc["z"],
            ]
        ).T
    )
    np.random.shuffle(vertices)
    pcd = o3d.t.geometry.PointCloud(
        o3c.Tensor(vertices, o3c.float64, o3c.Device(f"{device}:0"))
    )
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(vertices)

    try:
        downsampled = pcd.farthest_point_down_sample(num_sample)
    except RuntimeError:
        print("The point cloud has lesser points than the desired number.")
        downsampled = pcd
    # downsampled = pd.DataFrame(np.asarray(downsampled.points), columns=["x", "y", "z"])
    downsampled = pd.DataFrame(
        downsampled.point.positions.cpu().numpy(), columns=["x", "y", "z"]
    )

    return downsampled


def generate_op(
    target,
    reference,
    device,
    k=16,
    calculate_mean=True,
    include_nearest=True,
    perturb=False,
    real_scanned=False,
):
    target = torch.tensor(target[["x", "y", "z"]].values).to(device)
    reference = torch.tensor(reference[["x", "y", "z"]].values).to(device)

    cov, mean = covariance(
        reference,
        target,
        k,
        include_nearest=include_nearest,
        calculate_mean=calculate_mean,
        cossim=real_scanned,
        device=device,
    )

    eigvals = torch.linalg.eigvals(cov)

    offset = torch.sqrt(eigvals)
    offset *= 1 if random.random() < 0.5 else -1

    if perturb:
        offset *= (torch.randn(1)).to(device)

    op = mean + offset

    return op


def covariance(
    reference,
    target,
    k,
    device,
    include_nearest=True,
    calculate_mean=True,
    cossim=False,
):
    _, knn_indices = KNN(
        reference,
        target,
        k,
        include_nearest=include_nearest,
        cossim=cossim,
        device=device,
        num_chunks=1,
    )
    # Find local of the target point in the reference point cloud
    step = knn_indices.shape[1] // 16
    if step < 1:
        step = 1
    knn_indices = knn_indices[
        :,
        torch.arange(start=0, end=knn_indices.shape[1], step=step),
    ]

    if calculate_mean:
        mean = torch.mean(reference[knn_indices], 0).unsqueeze(1)
    else:
        mean = target.unsqueeze(1)

    cov = (
        torch.matmul(
            (reference[knn_indices] - mean).transpose(-2, -1),
            (reference[knn_indices] - mean),
        )
        / k
    )

    gc.collect()
    torch.cuda.empty_cache()

    return cov, mean.squeeze()


def load_model(model_file, model_class, device, marching_steps):
    # Loading the saved model
    model = model_class(device=device, steps=marching_steps)
    state_dict = torch.load(f"./models/{model_file}.pt", map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    del state_dict
    del new_state_dict
    model.eval()
    model = model.double().to(device)

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    # https://discuss.pytorch.org/t/finding-model-size/130275/2
    # param_size = 0
    # for param in model.parameters():
    #     if param.requires_grad:
    #         param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # # for buffer in model.buffers():
    # #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_kb = (param_size + buffer_size) / 1024
    # print("model size: {:.3f}KB".format(size_all_kb))

    return model


class RayMarchingLoss(nn.Module):
    """
    The loss function for ray marching
    """

    def __init__(self, device, steps, batch_size, ray_marching_loss):
        super().__init__()
        self.device = device
        self.steps = steps
        self.batch_size = batch_size
        self.ray_marching_loss = ray_marching_loss
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def gaussian_weights(self, criteria):
        return torch.exp(
            -(criteria**2 / (2 * torch.mean(criteria**2, dim=-1)).unsqueeze(-1))
        )

    def forward(
        self,
        knn_depths,
        march_depths,
        cossims,
        epsilon,
        output_depth,
        label,
    ):
        if self.steps == 0:
            # Loss is MAE of the final depths if no ray marching is performed
            return self.mae(output_depth, label)
        elif self.steps > 0:
            # Projection depths of the neighbouring points to the tangent plane
            # Required for tangent loss and marching step loss
            march_depths = march_depths.squeeze(2)
            projection_depths = knn_depths * cossims
            local_weights = self.gaussian_weights(knn_depths)

            march_depths_gt = projection_depths.gather(
                -1, knn_depths.min(-1, keepdim=True).indices
            ).squeeze()

            weighted_projection_depth_mean = torch.sum(
                projection_depths * local_weights, -1, keepdim=True
            ) / torch.sum(local_weights, -1, keepdim=True)

            # Step surplus for big step loss
            epsilon[epsilon > 0] = 0

            # Losses are defined as below
            tangent_loss = torch.mean(
                torch.sqrt(
                    torch.sum(
                        (projection_depths - torch.abs(weighted_projection_depth_mean))
                        ** 2
                        * local_weights,
                        -1,
                    )
                    / torch.sum(local_weights, -1)
                )
            )
            marching_step_loss = torch.mean(torch.abs((march_depths - march_depths_gt)))
            epsilon_loss = torch.mean(-epsilon)

            # return combined loss
            return (
                self.mae(output_depth, label)
                + torch.sqrt(self.mse(output_depth, label))
                + tangent_loss * self.ray_marching_loss
                + marching_step_loss * self.ray_marching_loss
                + epsilon_loss
            )


def evaluate(pc1, pc2, device="cpu"):
    pc1 = pc1.to(device)
    pc2 = pc2.to(device)

    num_chunks = 1
    while True:
        try:
            pc1_chunks = torch.chunk(pc1, num_chunks)

            dist1_li = []
            for chunk in pc1_chunks:
                dist1_li.append(
                    torch.min(
                        torch.norm(
                            pc2.unsqueeze(0).repeat([chunk.shape[0], 1, 1])
                            - chunk.unsqueeze(1).repeat([1, pc2.shape[0], 1]),
                            p=2,
                            dim=2,
                        ),
                        1,
                    )[0]
                )
            break
        except torch.cuda.OutOfMemoryError:
            num_chunks *= 2

    dist1 = torch.cat(dist1_li, 0)

    while True:
        try:
            pc2_chunks = torch.chunk(pc2, num_chunks)

            dist2_li = []
            for chunk in pc2_chunks:
                dist2_li.append(
                    torch.min(
                        torch.norm(
                            pc1.unsqueeze(0).repeat([chunk.shape[0], 1, 1])
                            - chunk.unsqueeze(1).repeat([1, pc1.shape[0], 1]),
                            p=2,
                            dim=2,
                        ),
                        1,
                    )[0]
                )
            break
        except torch.cuda.OutOfMemoryError:
            num_chunks *= 2

    dist2 = torch.cat(dist2_li, 0)

    cd = torch.mean(dist1) + torch.mean(dist2)
    hd = max(torch.max(dist1), torch.max(dist2))

    return cd, hd


def select_aoi_points(self_pc, aoi_pc, device, num_chunks=1):
    self_pc = self_pc.to(device)
    aoi_pc = aoi_pc.to(device)

    while True:
        try:
            reference_chunks = torch.chunk(self_pc, num_chunks)
            # knn_std = []
            # knn_mean = []
            knn_max = []
            knn_min = []
            for chunk in reference_chunks:
                knn_coords, _ = KNN(
                    aoi_pc,
                    chunk,
                    # 6,
                    10,
                    include_nearest=True,
                    cossim=True,
                    device=device,
                )
                # knn_std.append(torch.std(knn_coords, 1))
                # knn_mean.append(torch.mean(knn_coords, 1))
                knn_max.append(
                    torch.topk(knn_coords, 2, dim=1, largest=True, sorted=True)[0][
                        :, -1, :
                    ]
                )
                knn_min.append(
                    torch.topk(knn_coords, 2, dim=1, largest=False, sorted=True)[0][
                        :, -1, :
                    ]
                )
            break
        except torch.cuda.OutOfMemoryError:
            num_chunks *= 2

    # knn_std = torch.cat(knn_std, 0)
    # knn_mean = torch.cat(knn_mean, 0)
    # valid_idx = torch.sum(torch.abs(self_pc - knn_mean) < knn_std, 1) == 3

    knn_max = torch.cat(knn_max, 0)
    knn_min = torch.cat(knn_min, 0)
    knn_max = torch.nextafter(knn_max, torch.ones(knn_max.shape).to(device) * np.inf)
    knn_min = torch.nextafter(knn_min, -torch.ones(knn_max.shape).to(device) * np.inf)
    valid_idx = torch.sum(self_pc < knn_max, 1) == 3
    valid_idx *= torch.sum(self_pc > knn_min, 1) == 3

    gt_points = self_pc[valid_idx]

    return gt_points, valid_idx


def garbage_collect(items):
    for item in items:
        try:
            item.detach().cpu()
            del item
        except AttributeError:
            continue

    gc.collect()
    torch.cuda.empty_cache()
