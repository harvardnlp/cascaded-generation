import torch

has_cuda = False
try:
    import _genbmm

    has_cuda = True
except ImportError:
    pass


def banddiag(orig_x, lu, ld, fill=0):
    s1 = list(orig_x.shape)
    s2 = list(orig_x.shape)
    x = orig_x
    s1[-2] = lu
    s2[-2] = ld
    x = torch.cat(
        [
            torch.zeros(*s1, device=x.device, dtype=x.dtype),
            x,
            torch.zeros(*s2, device=x.device, dtype=x.dtype),
        ],
        dim=-2,
    )
    unf = x.unfold(-2, lu + ld + 1, 1)
    return (
        torch.diagonal(unf, 0, -3, -2).transpose(-2, -1),
        x.narrow(-2, lu, orig_x.shape[-2]),
    )


def repdiag(x, lu, ld):
    s1, s2 = list(x.shape), list(x.shape)
    s1[-2] = ld
    s2[-2] = lu
    x = torch.cat(
        [
            torch.zeros(*s1, device=x.device, dtype=x.dtype),
            x,
            torch.zeros(*s2, device=x.device, dtype=x.dtype),
        ],
        dim=-2,
    )
    unf = x.unfold(-2, lu + ld + 1, 1)
    return torch.diagonal(unf, 0, -2, -1)


class Transpose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, val, lu, ld):
        ctx.save_for_backward(torch.tensor([lu, ld]))
        return repdiag(val.flip(-1), lu, ld)

    @staticmethod
    def backward(ctx, grad_output):
        val, = ctx.saved_tensors
        lu, ld = val.tolist()
        return repdiag(grad_output.flip(-1), ld, lu), None, None


class BandedMatrix:
    def __init__(self, data, lu, ld, fill=0):
        batch, n, off = data.shape
        assert off == lu + ld + 1, "Offsets need to add up."
        self.data = data
        self.fill = fill
        self.lu, self.ld = lu, ld
        self.width = lu + ld + 1

    def _new(self, lu, ld):
        batch, n, off = self.data.shape
        data = torch.zeros(
            batch, n, ld + lu + 1, dtype=self.data.dtype, device=self.data.device
        ).fill_(self.fill)
        return data

    def band_shift(self, t):
        if t == 0:
            return self
        batch, n, off = self.data.shape

        pad = torch.zeros(
            batch, n, abs(t), dtype=self.data.dtype, device=self.data.device
        ).fill_(self.fill)
        if t > 0:
            v = torch.cat([self.data[:, :, t:], pad], 2)
        else:
            v = torch.cat([pad, self.data[:, :, :t]], 2)

        return BandedMatrix(v, self.lu + t, self.ld - t, self.fill)

    # def band_shift(self):
    #     batch, n, off = self.data.shape
    #     return BandedMatrix(
    #         torch.cat(
    #             [self.data[:, :, 1:],
    #              torch.zeros(batch, n, 1, dtype=self.data.dtype, device=self.data.device).fill_(self.fill)], 2
    #         ),
    #         self.lu - 1,
    #         self.ld + 1,
    #         self.fill,
    #     )

    def band_unshift(self):
        batch, n, off = self.data.shape
        return BandedMatrix(
            torch.cat(
                [
                    torch.zeros(
                        batch, n, 1, dtype=self.data.dtype, device=self.data.device
                    ).fill_(self.fill),
                    self.data[:, :, :-1],
                ],
                2,
            ),
            self.lu - 1,
            self.ld + 1,
            self.fill,
        )

    def col_shift(self, t):
        if t == 0:
            return self
        batch, n, off = self.data.shape
        pad = torch.zeros(
            batch, abs(t), off, dtype=self.data.dtype, device=self.data.device
        ).fill_(self.fill)
        if t > 0:
            v = torch.cat([self.data[:, t:, :], pad], 1)
        else:
            v = torch.cat([pad, self.data[:, :t, :]], 1)
        return BandedMatrix(v, self.lu - t, self.ld + t, self.fill)

    def col_unshift(self):
        batch, n, off = self.data.shape
        return BandedMatrix(
            torch.cat(
                [
                    torch.zeros(
                        batch, 1, off, dtype=self.data.dtype, device=self.data.device
                    ).fill_(self.fill),
                    self.data[:, :-1, :],
                ],
                1,
            ),
            self.lu + 1,
            self.ld - 1,
            self.fill,
        )

    def to_dense(self):
        batch, n, off = self.data.shape
        full = torch.zeros(batch, n, n, dtype=self.data.dtype, device=self.data.device)
        full.fill_(self.fill)
        x2, x = banddiag(full, self.lu, self.ld)
        x2[:] = self.data
        return x

    def _expand(self, lu, ld):
        batch, n, off = self.data.shape
        data = self._new(lu, ld)
        s = lu - self.lu
        data[:, :, s : s + self.width] = self.data
        return BandedMatrix(data, lu, ld, self.fill)

    def op(self, other, op, zero=0):
        batch, n, off = self.data.shape
        lu = max(self.lu, other.lu)
        ld = max(self.ld, other.ld)
        data = self._new(lu, ld).fill_(zero)

        s1 = lu - self.lu
        data[:, :, s1 : s1 + self.width] = self.data

        s2 = lu - other.lu
        data[:, :, s2 : s2 + other.width] = op(
            data[:, :, s2 : s2 + other.width], other.data
        )
        return BandedMatrix(data, lu, ld, self.fill)

    def transpose(self):
        batch, n, off = self.data.shape
        y2 = Transpose.apply(self.data, self.lu, self.ld)
        assert y2.shape[1] == n
        return BandedMatrix(y2, self.ld, self.lu, self.fill)

    # def multiply(self, other):
    #     batch, n, off = self.data.shape
    #     assert other.data.shape[1] == n
    #     lu = self.lu + other.ld
    #     ld = self.ld + other.lu
    #     out, = _genbmm.forward_band(self.data, self.lu, self.ld,
    #                                 other.data, other.lu, other.ld, 3)
    #     return BandedMatrix(out, lu, ld, self.fill)

    def multiply(self, other):
        if has_cuda:
            batch, n, off = self.data.shape
            assert other.data.shape[1] == n
            lu = self.lu + other.ld
            ld = self.ld + other.lu
            out = bandedbmm(
                self.data, self.lu, self.ld, other.data, other.lu, other.ld, lu, ld
            )
            return BandedMatrix(out, lu, ld, self.fill)
        else:
            return self.multiply_simple(other)

    def multiply_log(self, other):
        if has_cuda:
            batch, n, off = self.data.shape
            assert other.data.shape[1] == n
            lu = self.lu + other.ld
            ld = self.ld + other.lu
            out = bandedlogbmm(
                self.data, self.lu, self.ld, other.data, other.lu, other.ld, lu, ld
            )
            return BandedMatrix(out, lu, ld, self.fill)
        else:
            return self.multiply_log_simple(other)

    def multiply_max(self, other):
        if has_cuda and other.data.is_cuda:
            batch, n, off = self.data.shape
            assert other.data.shape[1] == n
            lu = self.lu + other.ld
            ld = self.ld + other.lu
            out = bandedmaxbmm(
                self.data, self.lu, self.ld, other.data, other.lu, other.ld, lu, ld
            )
            return BandedMatrix(out, lu, ld, self.fill)
        else:
            return self.multiply_max_simple(other)

    def multiply_simple(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n

        lu = self.lu + other.ld
        ld = self.ld + other.lu
        data = self._new(lu, ld)
        result = BandedMatrix(data, lu, ld, self.fill)

        for i in range(n):
            for j in range(result.width):
                o = i + (j - result.lu)
                if o < 0 or o >= n:
                    continue

                val = torch.zeros(batch)
                for k in range(self.width):
                    pos = i + (k - self.lu)
                    if pos < 0 or pos >= n:
                        continue

                    k2 = (pos - o) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    val += self.data[:, i, k] * other.data[:, o, k2]
                data[:, i, j] = val
        return result

    def multiply_max_simple(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n

        lu = self.lu + other.ld
        ld = self.ld + other.lu
        data = self._new(lu, ld)
        result = BandedMatrix(data, lu, ld, self.fill)

        for i in range(n):
            for j in range(result.width):
                o = i + (j - result.lu)
                if o < 0 or o >= n:
                    continue

                m = torch.zeros(batch).fill_(-1e9)
                for k in range(self.width):
                    pos = i + (k - self.lu)
                    if pos < 0 or pos >= n:
                        continue

                    k2 = (pos - o) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    m = torch.max(m, self.data[:, i, k] + other.data[:, o, k2])

                data[:, i, j] = m
        return result

    def multiply_log_simple(self, other):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n

        lu = self.lu + other.ld
        ld = self.ld + other.lu
        data = self._new(lu, ld)
        result = BandedMatrix(data, lu, ld, self.fill)

        for i in range(n):
            for j in range(result.width):
                o = i + (j - result.lu)
                if o < 0 or o >= n:
                    continue

                val = torch.zeros(batch)
                m = torch.zeros(batch).fill_(-1e9)
                for k in range(self.width):
                    pos = i + (k - self.lu)
                    if pos < 0 or pos >= n:
                        continue

                    k2 = (pos - o) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    m = torch.max(m, self.data[:, i, k] + other.data[:, o, k2])

                for k in range(self.width):
                    pos = i + (k - self.lu)
                    if pos < 0 or pos >= n:
                        continue

                    k2 = (pos - o) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    val += torch.exp(self.data[:, i, k] + other.data[:, o, k2] - m)

                data[:, i, j] = torch.log(val) + m
        return result

    def multiply_back(self, other, out, grad_out):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n
        grad_a, = _genbmm.backward_band(
            self.data,
            self.lu,
            self.ld,
            other.data,
            other.lu,
            other.ld,
            grad_out,
            grad_out,
            3,
        )
        grad_a = BandedMatrix(grad_a, self.lu, self.ld, self.fill)
        return grad_a

    def multiply_back_simple(self, other, grad_out):
        batch, n, off = self.data.shape
        assert other.data.shape[1] == n
        data = self._new(self.lu, self.ld)
        result = BandedMatrix(data, self.lu, self.ld, self.fill)

        for i in range(n):
            for j in range(self.width):
                o = i + (j - self.lu)
                val = torch.zeros(batch)
                for k in range(grad_out.width):
                    pos = i + (k - grad_out.lu)
                    if pos < 0 or pos >= n:
                        continue
                    k2 = (o - pos) + other.lu
                    if k2 < 0 or k2 >= other.width:
                        continue
                    val += other.data[:, pos, k2] * grad_out.data[:, i, k]
                data[:, i, j] = val
        return result.transpose()


class BandedMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, a_lu, a_ld, b, b_lu, b_ld, o_lu, o_ld):
        a = a.contiguous()
        b = b.contiguous()
        out, _ = _genbmm.forward_band(a, a_lu, a_ld, b, b_lu, b_ld, 3)
        ctx.save_for_backward(
            a, b, out, torch.LongTensor([a_lu, a_ld, b_lu, b_ld, o_lu, o_ld])
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches, bands = ctx.saved_tensors
        a_lu, a_ld, b_lu, b_ld, o_lu, o_ld = bands.tolist()
        a = BandedMatrix(a, a_lu, a_ld, 0)
        b = BandedMatrix(b, b_lu, b_ld, 0)
        grad_output = BandedMatrix(grad_output, o_lu, o_ld, 0)
        switches = BandedMatrix(switches.float(), o_lu, o_ld, 0)

        grad_a, = _genbmm.backward_band(
            a.data,
            a.lu,
            a.ld,
            b.data,
            b.lu,
            b.ld,
            grad_output.data.contiguous(),
            switches.data,
            3,
        )

        grad_b, = _genbmm.backward_band(
            b.data.contiguous(),
            b.lu,
            b.ld,
            a.data.contiguous(),
            a.lu,
            a.ld,
            grad_output.transpose().data.contiguous(),
            switches.transpose().data.contiguous(),
            3,
        )
        return grad_a, None, None, grad_b, None, None, None, None


class BandedLogMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, a_lu, a_ld, b, b_lu, b_ld, o_lu, o_ld):
        a = a.contiguous()
        b = b.contiguous()
        out, _ = _genbmm.forward_band(a, a_lu, a_ld, b, b_lu, b_ld, 0)
        ctx.save_for_backward(
            a, b, out, torch.LongTensor([a_lu, a_ld, b_lu, b_ld, o_lu, o_ld])
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches, bands = ctx.saved_tensors
        a_lu, a_ld, b_lu, b_ld, o_lu, o_ld = bands.tolist()
        a = BandedMatrix(a, a_lu, a_ld, -1e9)
        b = BandedMatrix(b, b_lu, b_ld, -1e9)
        grad_output = BandedMatrix(grad_output, o_lu, o_ld, -1e9)
        switches = BandedMatrix(switches.float(), o_lu, o_ld, -1e9)

        grad_a, = _genbmm.backward_band(
            a.data,
            a.lu,
            a.ld,
            b.data,
            b.lu,
            b.ld,
            grad_output.data.contiguous(),
            switches.data,
            0,
        )

        grad_b, = _genbmm.backward_band(
            b.data.contiguous(),
            b.lu,
            b.ld,
            a.data.contiguous(),
            a.lu,
            a.ld,
            grad_output.transpose().data.contiguous(),
            switches.transpose().data.contiguous(),
            0,
        )
        return grad_a, None, None, grad_b, None, None, None, None


class BandedMaxMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, a_lu, a_ld, b, b_lu, b_ld, o_lu, o_ld):
        a = a.contiguous()
        b = b.contiguous()

        out, indices = _genbmm.forward_band(a, a_lu, a_ld, b, b_lu, b_ld, 1)

        at = BandedMatrix(a, a_lu, a_ld, -1e9)
        bt = BandedMatrix(b, b_lu, b_ld, -1e9)

        _, indices2 = _genbmm.forward_band(
            bt.data.contiguous(), bt.lu, bt.ld, at.data.contiguous(), at.lu, at.ld, 1
        )

        ctx.save_for_backward(
            a,
            b,
            indices,
            indices2,
            torch.LongTensor([a_lu, a_ld, b_lu, b_ld, o_lu, o_ld]),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches, switches2, bands = ctx.saved_tensors
        a_lu, a_ld, b_lu, b_ld, o_lu, o_ld = bands.tolist()
        a = BandedMatrix(a, a_lu, a_ld, -1e9)
        b = BandedMatrix(b, b_lu, b_ld, -1e9)
        grad_output = BandedMatrix(grad_output, o_lu, o_ld, -1e9)
        switches = BandedMatrix(switches.float(), o_lu, o_ld, -1e9)
        switches2 = BandedMatrix(switches2.float(), o_lu, o_ld, -1e9)

        grad_a, = _genbmm.backward_band(
            a.data,
            a.lu,
            a.ld,
            b.data,
            b.lu,
            b.ld,
            grad_output.data.contiguous(),
            switches.data,
            1,
        )

        grad_b, = _genbmm.backward_band(
            b.data.contiguous(),
            b.lu,
            b.ld,
            a.data.contiguous(),
            a.lu,
            a.ld,
            grad_output.transpose().data.contiguous(),
            switches2.data.contiguous(),
            1,
        )

        return grad_a, None, None, grad_b, None, None, None, None


bandedbmm = BandedMul.apply
bandedlogbmm = BandedLogMul.apply
bandedmaxbmm = BandedMaxMul.apply
