import torch
import torch.nn.functional as F


def get_degree_mat(adj_mat, pow=1, degree_version='v1'):
    degree_mat = torch.eye(adj_mat.size()[0]).to(adj_mat.device)

    if degree_version == 'v1':
        degree_list = torch.sum((adj_mat > 0), dim=1).float()
    elif degree_version == 'v2':
        adj_mat_hat = F.relu(adj_mat)
        degree_list = torch.sum(adj_mat_hat, dim=1).float()
    elif degree_version == 'v3':
        degree_list = torch.sum(adj_mat, dim=1).float()
        degree_list = F.relu(degree_list)
    else:
        exit('error degree_version ' + degree_version)
    degree_list = torch.pow(degree_list, pow)
    degree_mat = degree_mat * degree_list
    return degree_mat


def get_laplace_mat(adj_mat, type='sym', add_i=False, degree_version='v1'):
    if type == 'sym':
        # Symmetric normalized Laplacian
        if add_i is True:
            adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        else:
            adj_mat_hat = adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-0.5, degree_version=degree_version)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        laplace_mat = torch.mm(laplace_mat, degree_mat_hat)
        return laplace_mat
    elif type == 'rw':
        # Random walk normalized Laplacian
        adj_mat_hat = torch.eye(adj_mat.size()[0]).to(adj_mat.device) + adj_mat
        degree_mat_hat = get_degree_mat(adj_mat_hat, pow=-1)
        laplace_mat = torch.mm(degree_mat_hat, adj_mat_hat)
        return laplace_mat


def adj_2_bias(adj):
    node_size = adj.size()[0]
    new_mat = ((torch.eye(node_size).to(adj.device) + adj) >= 1).float()
    new_mat = torch.tensor(-1e9) * (1 - new_mat)
    return new_mat


def adj_2_bias_without_self_loop(adj):
    # node_size = adj.size()[0]
    new_mat = (adj >= 1).float()
    new_mat = torch.tensor(-1e9).to(adj.device) * (1 - new_mat)
    return new_mat


if __name__ == '__main__':
    adj_mat = torch.FloatTensor([
        [1, 0],
        [-1, 1]
    ])
    print(get_degree_mat(adj_mat, degree_version='v3'))
    print(get_laplace_mat(adj_mat, degree_version='v1'))