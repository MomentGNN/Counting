import torch
from torch_geometric.utils.sparse import to_torch_coo_tensor

def compute_diagonal(S):
    n = S.shape[0]
    diag = torch.zeros(n)
    for i in range(n):
        diag[i] = S[i,i] 
    return diag

def add_attributes(edges, K):
    K = 10
    S = to_torch_coo_tensor(edges)
    N = S.shape[0]
    deg_k = torch.zeros(N,K)
    diag_k = torch.zeros(N,K)
    x = S
    for k in range(K):
        deg_k[:,k] = torch.sparse.sum(x,1).to_dense()
        # x = torch.sparse.mm(S, x)/math.factorial(k+1)
        x = torch.sparse.mm(S, x)
        if k > 0:
            # diag_k[:,k-1] = compute_diagonal(x)/math.factorial(k+1)
            diag_k[:,k-1] = compute_diagonal(x)

    ###### new part ######
    new = torch.zeros(N,10)
    # new = torch.zeros(N,10)
    I = torch.eye(N).to_sparse()
    S2 = torch.sparse.mm(S, S)
    S3 = torch.sparse.mm(S, S2)

    if torch.sparse.sum(I * S2 * S2) != 0:
        new[:,0] = torch.sparse.sum(I * S2 * S2,1).to_dense()
    # print(i)
    if torch.sparse.sum(I * S2 * S3) != 0:
        new[:,1] = torch.sparse.sum(I * S2 * S3,1).to_dense()

    if torch.sparse.sum(I * S3 * S3) != 0:
        new[:,2] = torch.sparse.sum(I * S3 * S3,1).to_dense()


    if torch.sparse.sum(S * S2) != 0:
        new[:,3] = torch.sparse.sum(S * S2 * S2,1).to_dense()
    if torch.sparse.sum(S * S2 * S3) != 0:
        new[:,4] = torch.sparse.sum(S * S2 * S3,1).to_dense()

    new[:,5] = torch.sparse.sum(S * S3 * S3,1).to_dense()


    new[:,6] = torch.sparse.sum(S2 * S2 * S2,1).to_dense()
    if torch.sparse.sum(S2 * S3) != 0:
        new[:,7] = torch.sparse.sum(S2 * S2 * S3,1).to_dense()
        new[:,8] = torch.sparse.sum(S2 * S3 * S3,1).to_dense()


    new[:,9] = torch.sparse.sum(S3 * S3 * S3,1).to_dense()

    y = torch.cat((diag_k, deg_k, new), 1)
    y[y==0] = 0.5
    y = torch.log(y)
    return y



def add_attributes2(edges, K):
    S = to_torch_coo_tensor(edges)
    N = S.shape[0]
    deg_k = torch.zeros(N,K)
    diag_k = torch.zeros(N,K-2)
    x = S
    for k in range(K):
        deg_k[:,k] = torch.sparse.sum(x,1).to_dense()
        # x = torch.sparse.mm(S, x)/math.factorial(k+1)
        if k > 1:
            # diag_k[:,k-1] = compute_diagonal(x)/math.factorial(k+1)
            diag_k[:,k-2] = compute_diagonal(x)
        x = torch.sparse.mm(S, x)

    if K > 9:
        new = torch.zeros(N,27)
    elif K > 8:
        new = torch.zeros(N,22)
    elif K > 7:
        new = torch.zeros(N,16)
    elif K > 5:
        new = torch.zeros(N,7)

    if K > 5:
        I = torch.eye(N).to_sparse()
        S2 = torch.sparse.mm(S, S)
        S3 = torch.sparse.mm(S, S2)

        if torch.sparse.sum(I * S2 * S2) != 0:
            new[:,0] = torch.sparse.sum(I * S2 * S2,1).to_dense()
        if torch.sparse.sum(I * S2 * S3) != 0:
            new[:,1] = torch.sparse.sum(I * S2 * S3,1).to_dense()

        if torch.sparse.sum(I * S3 * S3) != 0:
            new[:,2] = torch.sparse.sum(I * S3 * S3,1).to_dense()


        if torch.sparse.sum(S * S2) != 0:
            new[:,3] = torch.sparse.sum(S * S2 * S2,1).to_dense()
        if torch.sparse.sum(S * S2 * S3) != 0:
            new[:,4] = torch.sparse.sum(S * S2 * S3,1).to_dense()

        if torch.sparse.sum(I * S2) != 0:
                    new[:,5] = torch.sparse.sum(I * S2 * S2 * S2,1).to_dense()
        new[:,6] = torch.sparse.sum(S2 * S2 * S2,1).to_dense()


        if K > 7:
            new[:,7] = torch.sparse.sum(S * S3 * S3,1).to_dense()
            if torch.sparse.sum(S2 * S3) != 0:
                new[:,8] = torch.sparse.sum(S2 * S2 * S3,1).to_dense()
                new[:,9] = torch.sparse.sum(S2 * S3 * S3,1).to_dense()
            if torch.sparse.sum(S * S2) != 0:
                new[:,10] = torch.sparse.sum(S * S2 * S2 * S2,1).to_dense()
            if torch.sparse.sum(S * S2 * S3) != 0:
                new[:,11] = torch.sparse.sum(S * S2 * S2 * S3,1).to_dense()

            if torch.sparse.sum(I * S2 * S3) != 0:
                new[:,12] = torch.sparse.sum(I * S2 * S2 * S3,1).to_dense()

            if torch.sparse.sum(I * S2 * S3) != 0:
                new[:,13] = torch.sparse.sum(I * S2 * S3 * S3,1).to_dense()

            if torch.sparse.sum(S2 * I) != 0:
                new[:,14] = torch.sparse.sum(I * S2 * S2 * S2 * S2,1).to_dense()
            new[:,15] = torch.sparse.sum(S2 * S2 * S2 * S2,1).to_dense()




            if K > 8:

                new[:,16] = torch.sparse.sum(S3 * S3 * S3,1).to_dense()
                if torch.sparse.sum(S * S2 * S3) != 0:
                    new[:,17] = torch.sparse.sum(S * S2 * S3 * S3,1).to_dense()
                if torch.sparse.sum(I * S3) != 0:
                    new[:,18] = torch.sparse.sum(I * S3 * S3 * S3,1).to_dense()
                if torch.sparse.sum(S2 * S3) != 0:
                    new[:,19] = torch.sparse.sum(S2 * S2 * S2 * S3,1).to_dense()
                if torch.sparse.sum(I * S2 * S3) != 0:
                    new[:,20] = torch.sparse.sum(I * S3 * S2 * S2 * S2,1).to_dense()
                if torch.sparse.sum(S2 * S) != 0:
                    new[:,21] = torch.sparse.sum(S * S2 * S2 * S2 * S2,1).to_dense()


                if K > 9:
                    if torch.sparse.sum(S * S3) != 0:
                        new[:,22] = torch.sparse.sum(S * S3 * S3 * S3,1).to_dense()

                    if torch.sparse.sum(S2 * S3) != 0:
                        new[:,23] = torch.sparse.sum(S2 * S2 * S3 * S3,1).to_dense()
                    if torch.sparse.sum(I * S2 * S3) != 0:
                        new[:,24] = torch.sparse.sum(I * S3 * S3 * S2 * S2,1).to_dense()
                    if torch.sparse.sum(S * S2 * S3) != 0:
                        new[:,25] = torch.sparse.sum(S * S2 * S2 * S2 * S3,1).to_dense()

                    new[:,26] = torch.sparse.sum(S2 * S2 * S2 * S2 * S2,1).to_dense()
        

    if K > 5:
        y = torch.cat((diag_k, deg_k, new), 1)
        y[y==0] = 0.5
        y = torch.log(y)
    else:
        y = torch.cat((diag_k, deg_k), 1)
        y[y==0] = 0.5
        y = torch.log(y)
    return y






def add_attributes_hexa(edges):
    S = to_torch_coo_tensor(edges)
    N = S.shape[0]
    # S = S_dense.to_sparse()


    ###### new part ######
    new = torch.zeros(N,9)
    # new = torch.zeros(N,10)
    I = torch.eye(N).to_sparse()
    S2 = torch.sparse.mm(S, S)
    S3 = torch.sparse.mm(S, S2)
    S4 = torch.sparse.mm(S, S3)
    S5 = torch.sparse.mm(S, S4)
    S6 = torch.sparse.mm(S, S5)

    # print(torch.sparse.sum(I * S2 * S2,1))
    if torch.sparse.sum(I * S2 * S2) != 0:
        new[:,0] = torch.sparse.sum(I * S2 * S2,1).to_dense()
    # print(i)

    if torch.sparse.sum(I * S3 * S3) != 0:
        new[:,1] = torch.sparse.sum(I * S3 * S3,1).to_dense()

    if torch.sparse.sum(S * S2) != 0:
        new[:,2] = torch.sparse.sum(S * S2 * S2,1).to_dense()


    new[:,3] = compute_diagonal(S6) 
    new[:,4] = compute_diagonal(S4)
    new[:,5] = compute_diagonal(S3)
    new[:,6] = torch.sparse.sum(S,1).to_dense() 
    new[:,7] = torch.sparse.sum(S2,1).to_dense() 
    new[:,8] = torch.sparse.sum(S3,1).to_dense() 

    y = new/12
    return y


def add_attributes_penta(edges):
    S = to_torch_coo_tensor(edges)
    N = S.shape[0]
    new = torch.zeros(N,4)
    I = torch.eye(N).to_sparse()
    S2 = torch.sparse.mm(S, S)
    S3 = torch.sparse.mm(S, S2)
    S4 = torch.sparse.mm(S, S3)
    S5 = torch.sparse.mm(S, S4)


    new[:,0] = compute_diagonal(S2)
    new[:,1] = compute_diagonal(S3)
    new[:,2] = compute_diagonal(S5)
    new[:,3] = compute_diagonal(S4)


    y = new/10

    return y