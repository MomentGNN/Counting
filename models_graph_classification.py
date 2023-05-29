import torch
import torch.nn as nn
import torch.nn.functional as F
    

from graph_filters.MPNN_sparse import MPNN_sparse

from models_misc import mlp, choose_activation
from utils_graph_learning import global_add_pool_sparse, global_mean_pool_sparse, DiscreteEmbedding


class GNNSubstructures(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_features, 
                 d_in_id,
                 d_in_node_encoder=None,
                 d_degree=None, 
                 **kwargs):

        super(GNNSubstructures, self).__init__()
        
        seed = kwargs['seed']
        

        #-------------- Initializations

        self.readout = kwargs['readout'] if kwargs['readout'] is not None else 'sum'
        self.dropout_features = kwargs['dropout_features']
        self.bn = kwargs['bn']
        self.final_projection = kwargs['final_projection']
        self.inject_ids = kwargs['inject_ids']
        self.random_features = kwargs['random_features']

        d_msg = kwargs['d_msg']
        d_out = kwargs['d_out']
        d_h = kwargs['d_h']
        aggr = kwargs['aggr'] if kwargs['aggr'] is not None else 'add'
        flow = kwargs['flow'] if kwargs['flow'] is not None else 'target_to_source'
        msg_kind = kwargs['msg_kind'] if kwargs['msg_kind'] is not None else 'general'
        train_eps = kwargs['train_eps'] if kwargs['train_eps'] is not None else [False for _ in range(len(d_out))]
        activation_mlp = kwargs['activation_mlp']
        bn_mlp = kwargs['bn_mlp']
        jk_mlp = kwargs['jk_mlp']
        degree_embedding = 'None'
        
        encoders_kwargs = {'seed':seed,
                           'activation_mlp': activation_mlp,
                           'bn_mlp': bn_mlp,
                           'aggr': kwargs['multi_embedding_aggr']}


        #-------------- Input node embedding
        self.input_node_encoder = DiscreteEmbedding(kwargs['input_node_encoder'], 
                                                    in_features,
                                                    d_in_node_encoder,
                                                    kwargs['d_out_node_encoder'],
                                                    **encoders_kwargs)
        d_in = self.input_node_encoder.d_out
        if self.random_features:
            self.r_d_out = d_out[0]
            d_in = d_in + self.r_d_out
            


          
        #-------------- Identifier embedding (for each GNN layer)
        self.id_encoder = []
        d_id = []
        num_id_encoders = len(d_out) if kwargs['inject_ids'] else 1
        for i in range(num_id_encoders):
            id_encoder_layer = DiscreteEmbedding('None', 
                                                 len(d_in_id),
                                                 d_in_id,
                                                 None,
                                                 **encoders_kwargs)
            self.id_encoder.append(id_encoder_layer)
            d_id.append(id_encoder_layer.d_out)
        
        self.id_encoder  = nn.ModuleList(self.id_encoder)  
            

        #-------------- Degree embedding            
        self.degree_encoder = DiscreteEmbedding(degree_embedding,
                                                1,
                                                d_degree,
                                                kwargs['d_out_degree_embedding'],
                                                **encoders_kwargs)
        d_degree = self.degree_encoder.d_out

        
        #-------------- GNN layers w/ bn and jk
        self.conv = []
        self.batch_norms = []
        self.lin_proj = []
        for i in range(len(d_out)):
            
            kwargs_filter = {
                 'd_in': d_in,
                 'd_msg': d_msg[i],
                 'd_up': d_out[i],
                 'd_h': d_h[i],
                 'seed': seed,
                 'activation_name': activation_mlp,
                 'bn': bn_mlp,
                 'aggr': aggr,
                 'msg_kind': msg_kind,
                 'eps': 0,
                 'train_eps': train_eps[i],
                 'flow': flow,
                 'extend_dims': kwargs['extend_dims']}

            filter_fn = MPNN_sparse
            self.conv.append(filter_fn(**kwargs_filter))

            if self.final_projection[i]:
                # if desired, jk projections can be performed
                # by an mlp instead of a simple linear layer;
                if jk_mlp:
                    jk_layer = mlp(d_in, out_features, d_h[i], seed, activation_mlp, bn_mlp)
                else:
                    jk_layer = nn.Linear(d_in, out_features)
            else:
                jk_layer = None
            self.lin_proj.append(jk_layer)

            bn_layer = nn.BatchNorm1d(d_out[i]) if self.bn[i] else None
            self.batch_norms.append(bn_layer)

            d_in = d_out[i]

        if self.final_projection[-1]:
                # if desired, jk projections can be performed
                # by an mlp instead of a simple linear layer;
                if jk_mlp:
                    final_jk_layer = mlp(d_in, out_features, d_h[-1], seed, activation_mlp, bn_mlp)
                else:
                    final_jk_layer = nn.Linear(d_in, out_features)
        else:
            final_jk_layer = None
        self.lin_proj.append(final_jk_layer)

        self.conv = nn.ModuleList(self.conv)
        self.lin_proj = nn.ModuleList(self.lin_proj)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        
        
        #-------------- Readout
        if self.readout == 'sum':
            self.global_pool = global_add_pool_sparse
        elif self.readout == 'mean': 
            self.global_pool = global_mean_pool_sparse
        else:
            raise ValueError("Invalid graph pooling type.")
                
                
        #-------------- Activation fn (same across the network)
        self.activation = choose_activation(kwargs['activation'])
                
        return
        

    def forward(self, data, print_flag=False, return_intermediate=False):
        
        kwargs = {}
        kwargs['degrees'] = self.degree_encoder(data.degrees)
            
        #-------------- edge index and initial node features encoding
        edge_index = data.edge_index                                  
        x = self.input_node_encoder(data.x)  
        # if self.random_features:
        #     r = torch.rand(size=(x.shape[0], self.r_d_out), device=x.device).float()
        #     x = torch.cat((x,r), 1)
            
        #-------------- NOTE: the node features are first encoded and then passed to the jk layer 
        x_interm = [x]
        
        for i in range(0, len(self.conv)):
            
            #-------------- encode ids (different for each layer)    
            kwargs['identifiers'] = self.id_encoder[i](data.identifiers) if self.inject_ids else self.id_encoder[0](data.identifiers)
            

            kwargs['edge_features'] = None
                
            x = self.conv[i](x, edge_index, **kwargs)
            if self.bn[i]:
                x = self.batch_norms[i](x)
            x = self.activation(x)          
            x_interm.append(x)
            
        prediction = 0
        for i in range(0, len(self.conv) + 1):
            if self.final_projection[i]:
                x_global = self.global_pool(x_interm[i], data.batch)
                prediction += F.dropout(self.lin_proj[i](x_global), p=self.dropout_features[i], training=self.training)
            else:
                pass  # NB: the last final project is always constrained to be True
                
        if return_intermediate:
            return prediction, x_interm
        else:
            return prediction
