import torch


def batch_matmul(A, b):
    if A.is_sparse:
        return torch.squeeze(batch_mm(A, torch.unsqueeze(b, dim=-1)))
    else:
        return torch.squeeze(torch.matmul(A, torch.unsqueeze(b, dim=-1)), dim=-1)


def batch_mm(matrix_batch, matrix):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    # matrix_batch can be not batched, create dummy dim if necessary
    if len(matrix_batch.shape) == 1:
        matrix_batch = torch.unsqueeze(torch.unsqueeze(matrix_batch, dim=-1), dim=0)
    elif len(matrix_batch.shape) == 2:
        matrix_batch = torch.unsqueeze(matrix_batch, dim=0)

    batch_size = matrix_batch.shape[0]
    vectors = matrix_batch.to_dense().transpose(0, 1).reshape(matrix.shape[0], -1).to_sparse()

    return torch.sparse.mm(vectors, matrix).reshape(matrix.shape[1], batch_size, -1).transpose(1, 0)


def batch_quadratic(A, x):
    return torch.sum(batch_matmul(A, x) * x, dim=-1)


def auto_gradient(fn):
    def func(x, **kwargs):
        xc = x.requires_grad_(True)
        df = torch.autograd.grad(inputs=xc, outputs=fn(xc, **kwargs), create_graph=True)[0]
        return fn(x, **kwargs), df

    return func


def auto_gradient_batch(fn):
    def func(x, **kwargs):
        with torch.enable_grad():
            # check if batch
            if len(x.size()) == 1:
                xc = x.requires_grad_(True)
                df = torch.autograd.grad(inputs=xc, outputs=fn(xc, **kwargs), create_graph=True)[0]
            else:
                quantil = torch.tensor(1.0, dtype=torch.float32).repeat(x.shape[0]).to(x.device)
                xin = x.requires_grad_(True)
                if x.shape[0] == 1:
                    out = fn(xin, **kwargs).unsqueeze(0)
                    df = torch.stack(
                        torch.autograd.grad(inputs=xin, outputs=out, grad_outputs=quantil.unsqueeze(0),
                                            create_graph=True)).squeeze(0)
                else:
                    out = fn(xin, **kwargs)
                    df = torch.stack(
                        torch.autograd.grad(inputs=xin, outputs=out, grad_outputs=quantil,
                                            create_graph=True)).squeeze()
        return fn(x, **kwargs), df
    return func


def slice_instance(param_batch, indx):
    #  sliceIstance function: sliced_params is dict with inherited keys from kwargs (depends on fn)
    #  but only contains parameters for one sample from given batch
    sliced_params = {}
    for key in param_batch.keys():
        sliced_params[key] = param_batch[key][indx]
    return sliced_params


def batch_apply(fns, xs, shape, *args):
    results = torch.zeros(shape, dtype=xs.dtype).to(xs.device)
    num_args = len(args)
    for fn in set(fns):
        mask = torch.tensor([f == fn for f in fns], dtype=torch.bool)
        if num_args == 1:
            results[mask] = fn(xs[mask], args[0][mask])
        elif num_args == 2:
            results[mask] = fn(xs[mask], args[0][mask], args[1][mask])
        elif num_args == 3:
            results[mask] = fn(xs[mask], args[0][mask], args[1][mask], args[2][mask])
    return results
