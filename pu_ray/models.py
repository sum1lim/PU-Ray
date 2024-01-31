import torch
import torch.nn.functional as F
from torch import nn
from pu_ray.utils import KNN
from point_transformer_pytorch import PointTransformerLayer


class CrossAttention(nn.Module):
    """
    Cross attention is an ablated and modifie version of Point Tranformer by
    @inproceedings{zhao2021point,
        title={Point transformer},
        author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
        booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
        pages={16259--16268},
        year={2021}
    }

    The module calculates the attentions of neighbouring points to the op point
    """

    def __init__(self, *, device, hidden_size=32, mult=4):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.key_mlp = nn.Linear(hidden_size, hidden_size).double().to(device)
        self.value_mlp = nn.Linear(hidden_size, hidden_size).double().to(device)
        self.query_mlp = nn.Linear(hidden_size, hidden_size).double().to(device)
        self.pos_mlp = self.mlp(3, mult)
        self.attention_mlp = self.mlp(hidden_size, mult)

    def mlp(self, input_size, mult):
        return (
            nn.Sequential(
                nn.Linear(input_size, self.hidden_size * mult),
                nn.ReLU(),
                nn.Linear(self.hidden_size * mult, self.hidden_size),
            )
            .double()
            .to(self.device)
        )

    def forward(self, op, feats, rel_pos):
        # Relative positioning of the neighouring points wrt op
        position_embedding = self.pos_mlp(rel_pos)

        #  qkv
        query = self.query_mlp(op.unsqueeze(1))
        key = self.key_mlp(feats)
        value = self.value_mlp(feats) + position_embedding

        # Cacluate op attention and transformation
        attention = self.attention_mlp(query - key + position_embedding).softmax(dim=-2)
        transformed = torch.sum(attention * value, dim=1)

        return transformed


class PUray(nn.Module):
    """
    Preliminary model with a point transformer layer followed by MLP
    """

    def __init__(self, *, device, hidden_size=32, steps=8):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.steps = steps
        self.mult = 4

        # Feature extraction from 3D coordinates
        self.feat_mlp = (
            nn.Sequential(
                nn.Linear(3, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            .double()
            .to(device)
        )

        # Feature transformation with Point Transformer
        # Using the reimplementation by
        # @misc{
        #     lucidrains,
        #     title={Lucidrains/point-transformer-pytorch: Implementation of the point transformer layer, in Pytorch},
        #     url={https://github.com/lucidrains/point-transformer-pytorch},
        #     journal={GitHub},
        #     author={Lucidrains}
        # }
        self.transformer = (
            PointTransformerLayer(
                dim=self.hidden_size,
                pos_mlp_hidden_dim=self.hidden_size * self.mult,
                attn_mlp_hidden_mult=self.mult,
            )
            .double()
            .to(device)
        )

        # Multi-head op Attention
        self.cross_attention = CrossAttention(
            device=self.device, hidden_size=self.hidden_size, mult=self.mult
        )

        # MLP for outputing implicit points
        self.implicit_mlps = (
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 3),
            )
            .double()
            .to(device)
        )

        # MLP for epsilon estimation
        self.epsilon_mlp = (
            nn.Sequential(
                nn.Linear(self.hidden_size + 3, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
            )
            .double()
            .to(device)
        )

        self.apply(self._init_weights)

    def marching(self, knn_coords, output_coords):
        """
        Calcuated patch's relative vectors and depths wrt updated op at every marching step
        """
        march_knn_coords = knn_coords - output_coords.unsqueeze(1)
        march_knn_depths = torch.norm(march_knn_coords, dim=2)
        march_knn_vectors = march_knn_coords / torch.unsqueeze(march_knn_depths, dim=-1)

        return march_knn_vectors, march_knn_depths

    def marching_step(
        self,
        feats,
        rel_vectors,
        rel_depths,
        query,
        cumulative_depth,
        ca_li,
    ):
        """
        This function is called iteratively for every marching step
        """
        # Update op
        op_coords = query * cumulative_depth
        # Patch centered at updated op
        rel_coords = rel_vectors * rel_depths.unsqueeze(-1)

        # Calculate op attention in the smaller neighbourhood
        op_feats = self.feat_mlp(op_coords)
        ca = self.cross_attention(
            op_feats,
            feats,
            rel_coords,
        )

        # Implicit point generation
        implicit_coords = self.implicit_mlps(ca)
        implicit_point = op_coords + implicit_coords

        # Marching step
        march_step = torch.norm(implicit_coords, dim=-1)
        march_step = march_step.unsqueeze(-1)

        # Tangent vector
        tangent_vector = implicit_coords / march_step

        # Update op
        output_coords = op_coords + query * march_step

        # Calculate cosine similaries of the small neighbourhood wrt tanget vector
        cossim = torch.sum(
            tangent_vector.unsqueeze(1) * rel_vectors,
            dim=-1,
        )

        return (
            march_step,
            output_coords,
            implicit_point,
            cossim,
        )

    def forward(self, knn_coords, query):
        KNN_depths = knn_coords.norm(dim=-1)
        KNN_vectors = knn_coords / KNN_depths.unsqueeze(-1)

        # Feature extraction
        feats = self.feat_mlp(knn_coords)
        feats = self.transformer(feats, knn_coords)

        # Initialize variables
        local_depths = torch.empty(0).to(self.device)
        march_steps = torch.empty([query.shape[0], 0, 1]).to(self.device)
        cossims = torch.empty(0).to(self.device)
        implicit_points = torch.empty(0).to(self.device)
        cumulative_depth = torch.zeros([query.shape[0], 1]).to(self.device)
        op = torch.zeros([query.shape[0], 3]).double().to(self.device)
        rel_vectors = KNN_vectors
        rel_depths = KNN_depths

        ca_li = []
        # Marching steps
        for step in range(self.steps):
            (
                rel_vectors,
                rel_depths,
            ) = self.marching(knn_coords, op)
            (
                march_step,
                op,
                implicit_point,
                cossim,
            ) = self.marching_step(
                feats, rel_vectors, rel_depths, query, cumulative_depth, ca_li
            )

            # Update variables
            local_depths = torch.cat([local_depths, rel_depths.unsqueeze(1)], 1)
            march_steps = torch.cat([march_steps, march_step.unsqueeze(1)], 1)
            cossims = torch.cat([cossims, cossim.unsqueeze(1)], 1)
            implicit_points = torch.cat(
                [implicit_points, implicit_point.unsqueeze(1)], 1
            )

            # Update cumulative depth
            cumulative_depth = cumulative_depth + march_step

        # Epsilon estimation
        rel_coords = rel_vectors * rel_depths.unsqueeze(-1)
        op_feats = self.feat_mlp(op)
        ca = self.cross_attention(
            op_feats,
            feats,
            rel_coords,
        ).unsqueeze(2)

        epsilon = self.epsilon_mlp(torch.cat([ca.squeeze(2), query], dim=1))

        return (
            local_depths,
            march_steps,
            cossims,
            implicit_points,
            cumulative_depth,
            epsilon,
        )

    def _init_weights(self, module):
        import math

        if isinstance(module, nn.Linear):
            stdv = 1.0 / math.sqrt(module.weight.size(1))
            module.weight.data.uniform_(-stdv, stdv)
            if module.bias is not None:
                module.bias.data.uniform_(-stdv, stdv)
