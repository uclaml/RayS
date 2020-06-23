import time
import numpy as np
import torch
import torch.nn.functional as F
import json

# np.set_printoptions(precision=2)

from pgbar import progress_bar


class RaySBatch(object):
    def __init__(self, model, ord=np.inf, epsilon=0.3, early_stopping=True, args=None):
        self.model = model
        self.ord = ord
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}
        self.early_stopping = early_stopping
        self.args = args

        assert self.early_stopping == False

    def get_xadv(self, x, v, d, lb=0., ub=1.):
        if isinstance(d, int):
            d = torch.tensor(d).repeat(len(x)).cuda()
        out = x + d.view(len(x), 1, 1, 1) * v
        out = torch.clamp(out, lb, ub)
        return out

    def attack_hard_label(self, x, y, target=None, query_limit=10000, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        self.queries = torch.zeros_like(y).cuda()
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.d_t = torch.ones_like(y).float().fill_(float("Inf")).cuda()

        working_ind = (self.d_t > self.epsilon).nonzero().flatten()
        # correct_mask = self.search_succ(x, y, target, working_ind)
        # self.d_t[correct_mask] = 0

        stop_queries = self.queries.clone()
        dist = self.d_t.clone()
        stop_dist = self.d_t.clone()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)

        # self.binary_search(x, y, target, self.sgn_t, (dist > self.epsilon))

        if self.args is not None:
            args = self.args
            name = args.dataset + '_' + args.alg + '_' + args.norm + '_query' + str(args.query_limit) + '_eps' + str(
                args.epsilon) + '_early' + args.early + '_batch'
            adbr_traj = []
            robacc_traj = []



        block_level = 0
        block_ind = 0
        for i in range(query_limit):

            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            valid_mask = (self.queries < query_limit)
            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[valid_mask.nonzero().flatten(), start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt, valid_mask)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm((self.x_final - x).view(shape[0], -1), self.ord, 1)
            stop_dist[working_ind] = dist[working_ind]
            stop_queries[working_ind] = self.queries[working_ind]

            working_ind = (dist > self.epsilon).nonzero().flatten()

            if torch.sum(self.queries >= query_limit) == shape[0]:
                print('out of queries')
                break

            # if i % 1 == 0:
            #     print("Iter %3d d_t %.6f dist %.6f queries %d" % (i + 1, self.d_t, dist, self.queries))
            #     print(["{0: 0.3f}".format(i) for i in dist.cpu().numpy().tolist()])
            #     print(["{0: 0.3f}".format(i) for i in self.queries.float().cpu().numpy().tolist()])
            #     print(valid_mask)

            progress_bar(torch.min(self.queries.float()), query_limit,
                         'd_t: %.4f | dist: %.4f | queries: %.4f | succ: %.4f | iter: %d'
                         % (torch.mean(self.d_t), torch.mean(dist), torch.mean(self.queries.float()),
                            1 - len(working_ind) / len(x), i + 1))

            if self.args is not None:
                adbr_traj.append(torch.mean(dist).item())
                robacc_traj.append(1 - len(working_ind) / len(x))
                with open(name + '_adbr_traj' + '.txt', 'w') as f:
                    json.dump(adbr_traj, f)
                with open(name + '_robacc_traj' + '.txt', 'w') as f:
                    json.dump(robacc_traj, f)

        stop_queries = torch.clamp(stop_queries, 0, query_limit)
        print('last iter', i + 1, 'd_t', torch.mean(self.d_t), 'dist', torch.mean(dist), 'query',
              torch.mean(self.queries.float()))
        return self.x_final, stop_queries, stop_dist, dist, (dist <= self.epsilon)

    def search_succ(self, x, y, target, mask):
        self.queries[mask] += 1
        if target:
            return self.model.predict_label(x[mask]) == target[mask]
        else:
            return self.model.predict_label(x[mask]) != y[mask]

    def lin_search(self, x, y, target, sgn):
        d_end = self.d_t.clone()
        lin_search_ind = (d_end == float("Inf")).nonzero().flatten()
        for d in range(1, self.lin_search_rad + 1):
            lin_succ_mask = self.search_succ(self.get_xadv(x, sgn, d), y, target, lin_search_ind)
            d_end[lin_search_ind[lin_succ_mask]] = d
            lin_search_ind = lin_search_ind[~lin_succ_mask]
            if len(lin_search_ind) == 0:
                break
        return d_end

    def binary_search(self, x, y, target, sgn, valid_mask, tol=1e-3):
        sgn_norm = torch.norm(sgn.view(len(x), -1), 2, 1)
        sgn_unit = sgn / sgn_norm.view(len(x), 1, 1, 1)

        d_start = torch.zeros_like(y).float().cuda()
        d_end = self.d_t.clone()

        # inited_mask = (float("Inf") > self.d_t)
        # not_inited_mask = ~inited_mask
        # if torch.sum(not_inited_mask) > 0:
        #     d_end = self.lin_search(x, y, target, sgn / self.lin_search_rad) * sgn_norm / self.lin_search_rad
        # else:
        #     d_end = self.d_t.clone()
        #     skip_ind = ~self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, inited_mask & valid_mask)
        #     d_end[(inited_mask & valid_mask).nonzero().flatten()[skip_ind]] = float("Inf")

        # inited_mask = (float("Inf") > self.d_t)
        # not_inited_mask = ~inited_mask
        # if torch.sum(not_inited_mask) > 0:
        #     initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target,
        #                                          not_inited_mask & valid_mask)
        #     d_end[(not_inited_mask & valid_mask).nonzero().flatten()[initial_succ_mask]] = sgn_norm[
        #         (not_inited_mask & valid_mask).nonzero().flatten()[initial_succ_mask]]
        # else:
        #     skip_ind = ~self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        #     d_end[valid_mask.nonzero().flatten()[skip_ind]] = float("Inf")

        # to_search_ind = ((d_end < float("Inf")) & valid_mask).nonzero().flatten()

        initial_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target, valid_mask)
        to_search_ind = valid_mask.nonzero().flatten()[initial_succ_mask]
        d_end[to_search_ind] = torch.min(self.d_t, sgn_norm)[to_search_ind]

        while len(to_search_ind) > 0:
            d_mid = (d_start + d_end) / 2.0
            search_succ_mask = self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target, to_search_ind)
            d_end[to_search_ind[search_succ_mask]] = d_mid[to_search_ind[search_succ_mask]]
            d_start[to_search_ind[~search_succ_mask]] = d_mid[to_search_ind[~search_succ_mask]]
            to_search_ind = to_search_ind[((d_end - d_start)[to_search_ind] > tol)]

        to_update_ind = (d_end < self.d_t).nonzero().flatten()
        if len(to_update_ind) > 0:
            self.d_t[to_update_ind] = d_end[to_update_ind]
            self.x_final[to_update_ind] = self.get_xadv(x, sgn_unit, d_end)[to_update_ind]
            self.sgn_t[to_update_ind] = sgn[to_update_ind]

    def __call__(self, input, label, target=None, seed=None, query_limit=10000):
        return self.attack_hard_label(input, label, target=target, seed=seed, query_limit=query_limit)
