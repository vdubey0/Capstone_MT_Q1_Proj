import torch
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SAGEConvCat(MessagePassing):
    """
    *Note: Source function taken from PyTorch Geometric and modified such that
    embeddings are first concatenated and then reduced to out_channel size as
    per the original GraphSAGE paper.
    
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    PyTorch Geometric citation:
    @inproceedings{Fey/Lenssen/2019,
      title={Fast Graph Representation Learning with {PyTorch Geometric}},
      author={Fey, Matthias and Lenssen, Jan E.},
      booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
      year={2019},
    }
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        super(SAGEConvCat, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weights_init = False
        #self.inp_size = 10

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0]*2, out_channels, bias=bias)
        
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()

        #changed
        if self.weights_init:
            torch.nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

        #changed
        if self.weights_init == False:
            self.edge_weights = torch.nn.Parameter(torch.rand(x.shape[0], dtype=torch.float32))
            self.weights_init = True
        
        out = self.propagate(edge_index, x=x, size=size)

        ### Concatenation
        out = torch.cat([x, out], dim=-1)
        out = self.lin_l(out)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        #changed
        coo = adj_t.coo()
        row_indices = coo[0]
        col_indices = coo[1]

        row_sum = torch.zeros(4, dtype=torch.float32).scatter_add(0, row_indices, edge_weights)
        row_sum = row_sum + 1e-8
        normalized_weights = self.edge_weights / row_sum[row_indices]
        adj_t = adj_t.set_value(normalized_weights)
        
        #adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
