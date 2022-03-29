import bisect
import torch
import torch.nn as nn

class E2EQA(nn.Module):
    def init(self, N_R, kb_info, n_hop_max):
        """Initialize the model
        Args:
            N_R(int): number of relations
            kb_info: M_subj, M_rel, M_obj matrixes where
                M_subj(matrix): dim (N_T, N_E) where N_T is the number of
                                 triples in the KB
                M_rel(matrix): dim (N_T, N_R)
                M_obj(matrix): dim (N_T, N_E)
            n_hop_max(int): the maximum number of hops the model handles
        """
        self.M_subj, self.M_rel, self.M_obj = kb_info # store the kb info
        self.n_hop_max = n_hop_max
        self.dense_inf = nn.Linear(768 + (n_hop_max - 1) * N_R, N_R, bias=False)
        self.dense_att = nn.Linear(768 + (n_hop_max - 1) * N_R, 1, bias=False)
        self.softmax = nn.Softmax()

    def _one_step(self, x, r):
        """One step follow
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            r(matrix): batched relation embeddings corresponding to
                        the x with dim (batch_size, N_R)

        Return:
            x_new(matrix): batched k-hot vector with dim (batch_size, N_E)
                            where N_E is the number of relations
        """
        # vector x * M_subj^T
        x_t = torch.dot(x, self.M_subj.T)
        # vector r * M_subj^T
        r_t = torch.dot(r, self.M_rel.T)
        # (x_t * r_t) * M_obj
        x_new = torch.dot(x_t * r_t, self.M_obj)
        return x_new

    def _calculate_r(self, h_q, rs, n_hop):
        """Calculate r based on the history
        Args:
            h_q(matrix): batched query vectors with dim (batch_size, 768)
            rs(list): list of batched history relations with dim (batch_size, N_R)
            n_hop(int): range from [0, n_hop_max - 1]
        """
        # concat all history relations
        if n_hop:
            rs = rs[::-1]
            concat_r = torch.concat(rs, dim=1) # (batch, n_hop*N_R)
            concat_r = torch.concat([h_q, concat_r], dim=1) # (batch, 768+n_hop*N_R)
        else:
            concat_r = h_q
        # assume others non showed relations are 0
        pad_zeros = torch.zeros(h_q.size(0), (n_hop_max - n_hop) * N_R)
        concat_r = torch.concat([concat_r, pad_zeros], dim=1) # (batch, 768+(n_hop_max-1)*N_R)

        # calculate new relation vector
        r = self.dense_inf(concat_r) # (batch, N_R)
        c = self.dense_att(concat_r) # (batch, 1)

        return r, c

    def _attention(self, cs):
        """Calculation attention score for all steps
        Args:
            cs(list): list of batched attention score of all steps with dim (batch, 1)
        """
        c = torch.cat(cs, dim=1) # (batch, n_hop_max)
        a = self.softmax(c, dim=-1).unsqueeze(-1) # (batch, n_hop_max, 1)
        return a

    def foward(self, x, h_q, n_hop):
        """Forward process
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            h_q(matrix): batched question embeddings corresponding to
                          the x with dim (batch_size, N_W2V)
            n_hop(int): the number of hop in this batch
        """
        rs = [] # record history relations vector
        cs = [] # record history attention score for each step
        xs = [] # record history x vector
        for i in n_hop:
            r, c = self._calculate_r(h_q, rs, i)
            x = self._one_step(x, r) # (batch, N_E)
            rs.append(r)
            cs.append(c)
            xs.append(x.unsqueeze(1)) # (batch, 1, N_E)
        rs.append(h_q)
        xs = torch.cat(xs, dim=1) # (batch, n_hop_max, N_E)

        a = self._attention(cs) # (batch, n_hop_max, 1)
        y = torch.sum(torch.bmm(a.transpose(-1, -2), xs), dim=1).squeeze(-1) # (batch, N_E)

        return y

class EntityExtraction(nn.Module):
    def init(self, dim, N_PO, N_E):
        """Initialize the model
        Args:
            dim(int): the output dimension of the RoBERTa model
            N_PO(int): the number of distinct tokens (split by space)
                        in both relations and entities.
            N_E(int): the number of entities
        """
        self.dim = dim
        self.dense_ws = nn.Linear(dim, 1, bias=False)
        self.softmax = nn.Softmax()
        self.embedding_po = nn.EmbeddingBag(N_PO, dim) # 1 for padding (by default)
        self.N_E = N_E

    def _get_sij(self, qs):
        """Calculate sij for each possible entry
        Implement the formula 8 and 9 in the section 3.3 in the paper

        Args:
            qs(list): batched querys' token embeddings output by RoBERTa
                       each token embedding has shape (length, N_D)

        Return:
            sij(list): batched tuple (sij vector, dictionary for query)
                        having length batch size
        """
        sijs = [] # for batch
        for q in qs:
            qij = []
            length = q.size(1)
            q_sum = torch.cumsum(q, dim=0) # to save time
            cnt = 0
            pos = {}
            for i in range(length):
                for j in range(i + 1, length):
                    if i:
                        qij_avg = (q_sum[j] - q_sum[i - 1]) / (j - i + 1)
                    else:
                        qij_avg = q_sum[j] / (j + 1)
                    qij.append(qij_avg) # (1, N_D)

                    # record the position in the list, save time for extracting
                    # sij given span
                    pos[(i, j)] = cnt
                    cnt += 1
            sij = torch.concat(qij, dim=0) # (O(length^2), N_D)
            sij = self.dense_ws(sij) # (O(length^2), 1)
            sij = self.softamx(sij) # (O(length^2), 1)
            sijs.append((sij, pos))
        return sijs

    def _get_zij(self, batched_entities_info):
        """Calculate zij for each entry
        Implement the formula 10 and 11 in the section 3.3 in the paper

        Args:
            batched_entities_info(list): batched entity span, k candidates list
                                          including token list of corresponding
                                          concatenated relation and object text,
                                          and index of each candidate (assume)
                in each entry of list, the structure looks like:
                    [[(i, j), [[p|o, p|o], [], ...], [[idx1, idx2], [], ...]], [], ...]
                where p|o is list of tokens (split by space) after concating p and o

        Return:
            zijs(list): batched entity zij embeddings list, each has number of entities
                         entries, each has dim with (number of candidates, dim). It looks
                         like:
                        [[(k, dim), (k, dim)], [], ...]
        """
        zijs = []
        for entities_info in batched_entities_info:
            zij = []
            for entity_info in entities_info:
                ij, kpos, _ = entity_info # (i, j), k list of p|o (k candidates)
                # calculate z for each candidate
                zijk = []
                for pos in kpos:
                    input_pos = []
                    input_offset = []
                    # concatenate all p|o
                    for po in pos:
                        input_offset.append(len(input_list))
                        input_pos.extend(po)
                    embeddings = self.embedding_po(input_pos, input_offset) # (len(pos), dim)
                    zijk.append(torch.mean(embeddings, dim=0, keepdim=True)) # (1, dim)
                zijk = torch.concate(zij, dim=0) # (len(kpos), dim)
                zij.append(zijk)
            zijs.append(zij)

        return zijs

    def _get_x0(self, sijs, zijs, qs, ijs, idxs):
        """Calculate initial x for each sentence
        Args:
            sij(list): batched tuple (sij vector, dictionary for query)
                        having length batch size
            zijs(list): batched entity zij embeddings list, each has number of entities
                         entries, each has dim with (number of candidates, dim). It looks
                         like:
                        [[(k, dim), (k, dim)], [], ...]
            qs(list): batched querys' token embeddings output by RoBERTa
                       each token embedding has shape (length, N_D)
            ijs(list): batched list of spans for each sentence
            idxs(list): batched list of candidate index for each sentence

        Return:
            x0s(list): list of x0 vectors with dim (1, dim)
        """
        x0s = []
        for sij, zij, q, ij_list, idx_list in zip(sijs, zijs, qs, ijs, idxs):
            # calculate qij for each span (i, j)
            q_sum = torch.cumsum(q, 0)
            qij = []
            for ij in ij_list:
                i, j = ij
                if i:
                    qij_avg = (q_sum[j] - q_sum[i - 1]) / (j - i + 1)
                else:
                    qij_avg = q_sum[j] / (j + 1)
                qij.append(qij_avg.unsqueeze(0)) # (1, dim)

            # calculate e
            eij = []
            offset = []
            cnt = 0
            for s, z, q in zip(sij, zij, qij):
                e = torch.mm(z, q.T) # (k, 1) z_{ij}^k * q_{ij}
                e = self.softmax(e) # (k, 1)
                e = s * e # (k, 1)
                eij.append(e)

                # record the position of each entity
                offset.append(cnt)
                cnt += z.size(0) # k
            eij = torch.concat(eij, dim=0) # (sum of k, 1)
            xij = self.softmax(eij) # (sum of k, 1)

            # put value to x vector based on index
            x = torch.zeros(self.N_E) # (dim)
            x = x.index_put_(idx_list, xij) # (dim)
            x0s.append(x.unsqueeze(0)) # (1, dim)

        return x0s

    def forward(self, qs, batched_entities_info):
        """Forward process
        Args:
            qs(list): batched querys' token embeddings output by RoBERTa
            batched_entities_info(list): batched entity span and corresponding relation
                                          and object text tokenized by space (assume)
                in each entry of list, the structure looks like:
                    [[(i, j), [[p|o, p|o], [], ...], [[idx1, idx2], [], ...]], [], ...]
                where p|o is list of tokens (split by space) after concating p and o
        """
        # get all sij for all sentences
        sijs = self._get_sij(qs)

        # filter useful sij for all sentences
        sijs_filtered = []
        ijs = [] # extract all span (i, j) for furture use
        idxs = [] # extract candidate index in order
        for entities_info, sij in zip(bachted_entities_info, sijs):
            index = [] # get index of each span
            entity_idx = [] # record candidates' index
            entities_ij = []
            sij, pos = sij 
            for entity_info in entities_info:
                ij, _, idx = entity_info
                index.append(pos[ij])
                entities_ij.append(ij)
                entity_idx.extend(idx)
            sij = torch.index_select(sij, 0, index)
            sijs_filtered.append(sij)
            ijs.append(entities_ij)
            idxs.append(eneity_idx)
        sijs = sijs_filtered

        # calculate zij for all sentences
        zijs = self._get_zij(batched_entities_info)

        # calculate xij (x likelihood) for all sentences
        x0s = self._get_x0(sijs, zijs, qs, ijs, idxs)

        x0s = torch.concat(x0s, dim=0)
        return x0s

if __name__ == "__main__":
    E2EQA(None, None, None)
