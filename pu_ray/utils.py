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


class QueryPointsData(Dataset):
    def __init__(
        self,
        input_dir,
        reference_dir,
        gt_dir,
        device,
        r=4,
    ):
        self.device = device

        file_li = os.listdir(input_dir)

        self.input_li = []
        self.gt_li = []
        for _, filename in tqdm(enumerate(file_li)):
            try:
                input_df = pd.read_csv(f"{input_dir}/{filename}", names=["x", "y", "z"])
                input_df = input_df[
                    (
                        (input_df["x"] ** 2 + input_df["y"] ** 2 + input_df["z"] ** 2)
                        ** (1 / 2)
                        > 50000
                    )
                    & (input_df["x"] > 0)
                ]
                input_pc = torch.tensor(input_df.sample(frac=1).values).to(device)

                num_chunks = 1
                while True:
                    try:
                        input_chunks = torch.chunk(input_pc, num_chunks)

                        input_knn_li = []
                        for chunk in input_chunks:
                            input_knn_li.append(
                                KNN(
                                    input_pc,
                                    chunk,
                                    16,
                                    include_nearest=True,
                                    cossim=True,
                                    device=device,
                                )[0]
                            )
                        break
                    except torch.cuda.OutOfMemoryError:
                        num_chunks *= 2

                input_knn = torch.cat(input_knn_li, 0)

                knn_std = torch.std(input_knn, 1)
                knn_mean = torch.mean(input_knn, 1)
                valid_idx = (
                    torch.sum(torch.abs(input_pc - knn_mean) < knn_std * 1.5, 1) == 3
                )

                valid_input = input_pc[valid_idx]
                knn_std = knn_std[valid_idx]

                std_avg = torch.mean(knn_std, 0)
                std_std = torch.std(knn_std, 0)
                valid_idx = (
                    torch.sum(torch.abs(knn_std - std_avg) < std_std * 1.5, 1) == 3
                )
                valid_input = valid_input[valid_idx]

                gt_df = pd.read_csv(
                    f"{reference_dir}/{filename}", names=["x", "y", "z"]
                )
                gt_df = gt_df[
                    (
                        (gt_df["x"] ** 2 + gt_df["y"] ** 2 + gt_df["z"] ** 2) ** (1 / 2)
                        > 50000
                    )
                    & (gt_df["x"] > 0)
                ]
                gt_pc = torch.tensor(gt_df.sample(frac=1).values).to(device)
            except IsADirectoryError:
                continue
            print(filename)

            try:
                os.mkdir(f"{gt_dir}")
            except FileExistsError:
                None

            if os.path.isfile(f"{gt_dir}/{filename}"):
                None
            else:
                num_chunks = 256

                gt_chunks = torch.chunk(gt_pc, num_chunks)
                knn_std = []
                knn_mean = []
                for chunk in gt_chunks:
                    knn_coords, _ = KNN(
                        input_pc,
                        chunk,
                        16,
                        include_nearest=True,
                        cossim=True,
                        device=device,
                    )
                    knn_std.append(torch.std(knn_coords, 1))
                    knn_mean.append(torch.mean(knn_coords, 1))

                knn_std = torch.cat(knn_std, 0)
                knn_mean = torch.cat(knn_mean, 0)
                valid_idx = (
                    torch.sum(torch.abs(gt_pc - knn_mean) < knn_std * 1.5, 1) == 3
                )

                gt_pc = gt_pc[valid_idx]
                knn_std = knn_std[valid_idx]

                std_avg = torch.mean(knn_std, 0)
                std_std = torch.std(knn_std, 0)
                valid_idx = (
                    torch.sum(torch.abs(knn_std - std_avg) < std_std * 1.5, 1) == 3
                )

                query_points = gt_pc[valid_idx]
                query_points = pd.DataFrame(
                    query_points.cpu().numpy(),
                    columns=["x", "y", "z"],
                )
                query_points = farthest_point_sampling(
                    query_points, len(valid_input) * r
                )[["x", "y", "z"]]

                np.savetxt(f"{gt_dir}/{filename}", query_points, delimiter=",")

            scaling_factor = random.random() * 0.2 + 0.9
            rotation_matrix = random_rotation().double().to(device)

            valid_input /= 120000
            self.input_li.append(
                valid_input.to(device) @ rotation_matrix * scaling_factor
            )

            query_points = pd.read_csv(f"{gt_dir}/{filename}", names=["x", "y", "z"])
            query_points /= 120000
            self.gt_li.append(
                torch.tensor(query_points.sample(frac=1).values).double().to(device)
                @ rotation_matrix
                * scaling_factor
            )

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.input_li[idx].to(self.device),
            self.gt_li[idx].to(self.device),
        )

    def __len__(self):
        return len(self.input_li)


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
                    k=16,
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
        updating_pc,
        query_pc,
        patch_k,
        query_k,
        device,
        output_size,
        iter,
        num_op=None,
        real_scanned=False,
    ):
        self.device = device
        self.patch_k = patch_k

        input_pc = input_pc.to("cpu")
        input_df = pd.DataFrame(input_pc.cpu().numpy(), columns=["x", "y", "z"])
        updating_pc = updating_pc.to("cpu")
        updating_df = pd.DataFrame(updating_pc.cpu().numpy(), columns=["x", "y", "z"])

        if num_op == None:
            num_op = len(updating_df) // 32

        if query_pc == None:
            query_pc = self.novel_queries(
                updating_df,
                updating_df,
                query_k,
                "cpu",
                output_size,
                real_scanned=real_scanned,
            )

        query_pc = query_pc.double().to("cpu")

        # if real_scanned:
        #     op = torch.tensor([[0, 0, 0]])
        # else:
        op = (
            generate_op(
                farthest_point_sampling(updating_df, num_op),
                updating_df,
                "cpu",
                k=16,
                calculate_mean=False,
                perturb=False,
                real_scanned=real_scanned,
            )
            .double()
            .to("cpu")
        )

        knn_coords, _ = KNN(
            updating_pc,
            query_pc,
            self.patch_k,
            include_nearest=True,
            cossim=False,
        )
        if real_scanned:
            knn_std = torch.std(knn_coords, 1)
            std_avg = torch.mean(knn_std, 0)
            std_std = torch.std(knn_std, 0)
            valid_idx = torch.mean(knn_std, 1) > torch.mean(std_avg, 0) * 0.999**iter

            std_std[0] *= 5
            std_std[1] *= 3
            std_std[2] *= 1
            valid_idx *= torch.sum(torch.abs(knn_std - std_avg) < std_std, 1) == 3

            query_pc = query_pc[valid_idx]
            knn_coords = knn_coords[valid_idx]

        op_xyz, _ = KNN(op, query_pc, 1, include_nearest=True)
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
            # point_avg = torch.mean(reference, 0)
            # point_std = torch.std(reference, 0)
            # point_std[0] *= 2
            # point_std[1] *= 2
            # point_std[2] *= 2
            # valid_idx = torch.sum(torch.abs(target - point_avg) < point_std, 1) == 3
            # target = target[valid_idx]

            perm = torch.randperm(target.size(0))
            target = target[perm]

            target_df = pd.DataFrame(
                target.cpu().numpy(),
                columns=["x", "y", "z"],
            )
            target = torch.tensor(
                farthest_point_sampling(target_df, output_size // 24)[
                    ["x", "y", "z"]
                ].values
            )

        queries = []
        for point in target:
            rel_pos = reference - point.unsqueeze(0)
            rel_dist = rel_pos.norm(dim=-1)
            rel_vectors = rel_pos / rel_dist.unsqueeze(-1)

            if real_scanned:
                reference_vectors = reference / reference.norm(dim=-1, keepdim=True)
                point_vector = point / point.norm(dim=-1)
                # Cosine similarities
                cossims = torch.sum(
                    reference_vectors * point_vector.unsqueeze(0),
                    dim=-1,
                )
                criteria = cossims

            else:
                criteria = -rel_dist

            _, knn_indices = criteria.topk(k + 1, largest=True)

            # queries.append((point.unsqueeze(0) + reference[knn_indices[1:]]) / 2)
            if real_scanned:
                queries += [
                    (point + reference[knn_indices[i]]).unsqueeze(0) / 2
                    for i in range(k, 0, -1)
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
            else:
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

        garbage_collect([target, reference])

        if real_scanned:
            query_pc = torch.cat(queries, 0)
        else:
            query_df = pd.DataFrame(
                torch.unique(torch.cat(queries, 0), dim=0).cpu().numpy(),
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


def KNN(references, xyz, k, include_nearest=False, cossim=False, device="cpu"):
    if cossim:
        query_vector = xyz.to(device) / xyz.to(device).norm(dim=-1, keepdim=True)
        reference_vectors = references.to(device) / references.to(device).norm(
            dim=-1, keepdim=True
        )

        # Cosine similarities
        num_chunks = 1
        while True:
            try:
                chunks = torch.chunk(query_vector, num_chunks)

                cossim_li = []
                for chunk in chunks:
                    cossim_li.append(
                        torch.sum(
                            reference_vectors.unsqueeze(0).repeat(
                                [chunk.shape[0], 1, 1]
                            )
                            * chunk.unsqueeze(1).repeat(
                                [1, reference_vectors.shape[0], 1]
                            ),
                            dim=-1,
                        )
                    )
                break
            except torch.cuda.OutOfMemoryError:
                num_chunks *= 2

        cossims = torch.cat(cossim_li, 0)
        criteria = cossims

    else:
        # Distances between observation points and input points
        num_chunks = 1
        while True:
            try:
                chunks = torch.chunk(xyz, num_chunks)

                dist_li = []
                for chunk in chunks:
                    dist_li.append(
                        torch.norm(
                            references.unsqueeze(0).repeat([chunk.shape[0], 1, 1])
                            - chunk.unsqueeze(1).repeat([1, references.shape[0], 1]),
                            dim=2,
                        )
                    )
                break
            except torch.cuda.OutOfMemoryError:
                num_chunks *= 2

        dists = torch.cat(dist_li, 0)
        criteria = -dists

    # first == False if input and query point clouds are the same
    if include_nearest == True:
        topk_indices = torch.topk(criteria, k, largest=True, sorted=True, dim=1).indices
        knn = references[topk_indices]
    else:
        topk_indices = torch.topk(
            criteria, k + 1, largest=True, sorted=True, dim=1
        ).indices
        topk_indices = topk_indices[:, 1:]
        knn = references[topk_indices]

    return knn, topk_indices


def farthest_point_sampling(pc, num_sample):
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
    # pcd = o3d.t.geometry.PointCloud(
    #     o3c.Tensor(vertices, o3c.float64, o3c.Device("cuda:0"))
    # )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    try:
        downsampled = pcd.farthest_point_down_sample(num_sample)
    except RuntimeError:
        downsampled = pcd
    downsampled = pd.DataFrame(np.asarray(downsampled.points), columns=["x", "y", "z"])

    return downsampled


def noise_removal(points, input_pc, updating_pc):
    updating_knn = KNN(updating_pc, points, 16, include_nearest=True, cossim=True)[0]

    knn_avg = torch.mean(updating_knn, 1)
    knn_std = torch.std(updating_knn, 1)
    knn_std[:, 0] *= 5
    knn_std[:, 1] *= 3
    knn_std[:, 2] *= 1
    valid_idx = torch.sum(torch.abs(points - knn_avg) < knn_std, 1) == 3

    return valid_idx


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

    op_list = []
    for point in target:
        # Find local of the target point in the reference point cloud
        cov, mean = covariance(
            reference,
            point,
            k,
            include_nearest=include_nearest,
            calculate_mean=calculate_mean,
            cossim=real_scanned,
        )

        offset = torch.sqrt(torch.linalg.eig(cov)[0])
        offset *= 1 if random.random() < 0.5 else -1

        if perturb:
            offset *= (torch.randn(1)).to(device)

        op = mean + offset
        op_list.append(op.unsqueeze(0))

    return torch.cat(op_list, 0)


def covariance(
    reference, point, k, include_nearest=True, calculate_mean=True, cossim=False
):
    # Find local of the target point in the reference point cloud
    if cossim:
        reference_vectors = reference / reference.norm(dim=-1, keepdim=True)
        point_vector = point / point.norm(dim=-1, keepdim=True)
        # Cosine similarities
        cossims = torch.sum(
            reference_vectors * point_vector.unsqueeze(0),
            dim=-1,
        )
        criteria = cossims
    else:
        rel_pos = reference - point.unsqueeze(0)
        rel_dist = rel_pos.norm(dim=-1)
        criteria = -rel_dist

    if include_nearest == True:
        _, knn_indices = criteria.topk(k, largest=True)
    else:
        _, knn_indices = rel_dist.topk(k + 1, largest=True)
        knn_indices = knn_indices[1:]

    if calculate_mean:
        mean = torch.mean(reference[knn_indices], 0)
    else:
        mean = point

    cov = (
        torch.matmul(
            (reference[knn_indices] - mean).T,
            (reference[knn_indices] - mean),
        )
        / k
    )

    return cov, mean


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


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_pc, gt_pc, k=-1, device="cpu"):
        pc1 = output_pc.squeeze().to(device)
        pc2 = gt_pc.squeeze().to(device)

        num_chunks = len(pc1) // 2048
        while True:
            try:
                pc1_chunks = torch.chunk(pc1, num_chunks)

                dist1_li = []
                for chunk in pc1_chunks:
                    dist1_li.append(
                        torch.min(
                            torch.norm(
                                chunk.unsqueeze(0).repeat([pc2.shape[0], 1, 1])
                                - pc2.unsqueeze(1).repeat([1, chunk.shape[0], 1]),
                                dim=2,
                            )
                            ** 2,
                            1,
                        )[0]
                    )
                    del chunk
                    gc.collect()
                break
            except torch.cuda.OutOfMemoryError:
                num_chunks *= 2

        dist1 = torch.cat(dist1_li, 0)

        num_chunks = len(pc2) // 2048
        while True:
            try:
                pc2_chunks = torch.chunk(pc2, num_chunks)

                dist2_li = []
                for chunk in pc2_chunks:
                    dist2_li.append(
                        torch.min(
                            torch.norm(
                                chunk.unsqueeze(0).repeat([pc1.shape[0], 1, 1])
                                - pc1.unsqueeze(1).repeat([1, chunk.shape[0], 1]),
                                dim=2,
                            )
                            ** 2,
                            1,
                        )[0]
                    )
                    del chunk
                    gc.collect()
                break
            except torch.cuda.OutOfMemoryError:
                num_chunks *= 2

        dist2 = torch.cat(dist2_li, 0)

        chamfer_distance = torch.mean(dist1) + torch.mean(dist2)

        pc1 = pc1.detach().cpu()
        pc2 = pc2.detach().cpu()
        dist1 = dist1.detach().cpu()
        dist2 = dist2.detach().cpu()
        del (
            pc1,
            pc2,
            dist1,
            dist2,
        )
        gc.collect()
        torch.cuda.empty_cache()

        return chamfer_distance


def garbage_collect(items):
    for item in items:
        try:
            item.detach().cpu()
            del item
        except AttributeError:
            continue

    gc.collect()
    torch.cuda.empty_cache()
