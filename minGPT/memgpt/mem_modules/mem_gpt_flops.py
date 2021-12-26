from typing import List

import numpy as np

class DNNLayer:
    def __init__(self, out_shape, depends_on: List["DNNLayer"] = tuple(), param_count=0):
        assert out_shape is not None  # get around varargs restriction
        self.extra_repr_params = {}
        self.unique_idx = "{}{:02d}".format(self.__class__.__name__, id(self) % 100)
        self.out_shape = out_shape
        self.depends_on = depends_on
        self.param_count = param_count

    def __repr__(self):
        args = self.extra_repr_params
        args["out_shape"] = self.out_shape
        args["param_count"] = self.param_count
        args["depends_on"] = "[{}]".format(", ".join([x.unique_idx for x in self.depends_on]))
        return "{}({})".format(self.unique_idx, ",".join(["{}={}".format(k, v) for k, v in args.items()]))

class QueryKeyValueMatrix(DNNLayer):
	# Fusing Query, Key, And Value into 1
	def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
		super().__init__(
			out_shape=(3 * SEQ_LEN,I,ATTN_HEADS), # [seq_lean X intermediate_vector_dim] for 12 heads 
			depends_on=[input] if input is not None else [],
			param_count=3 * HIDDEN_DIM*I*ATTN_HEADS)
		self.flop = 3 * SEQ_LEN*HIDDEN_DIM*I*ATTN_HEADS

class QKTMatrix(DNNLayer):
	# Fusing Masking and Dropout
	def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
		super().__init__(
			out_shape=(SEQ_LEN,I,ATTN_HEADS),
			depends_on=[input] if input is not None else [], # Different to accept a list
			param_count=0)
		self.flop = SEQ_LEN*HIDDEN_DIM*I*ATTN_HEADS + np.prod(self.out_shape) + np.prod(self.out_shape) # QKT + mask + dropout

class Mask(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(
        	out_shape=input.out_shape, 
        	depends_on=[input] if input is not None else [],
        	param_count=0)
        self.flop = np.prod(self.out_shape)

class QKTVMatrix(DNNLayer):
	# QKTV + Concat
	def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
		super().__init__(
			out_shape=(SEQ_LEN,I * ATTN_HEADS),
			depends_on=[input] if input is not None else [],
			param_count=0)
		self.flop = SEQ_LEN*HIDDEN_DIM*I*ATTN_HEADS + SEQ_LEN*HIDDEN_DIM*I*ATTN_HEADS # QKTVMatrix + Concat

class Concat(DNNLayer):
	def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
		super().__init__(
			out_shape=(SEQ_LEN,I * ATTN_HEADS),
			depends_on=[input] if input is not None else [],
			param_count=HIDDEN_DIM*I*ATTN_HEADS)
		self.flop = SEQ_LEN*HIDDEN_DIM*I*ATTN_HEADS

class LinearLayerReLU(DNNLayer):
    def __init__(self, in_features: int, out_features: int, input: DNNLayer):
        super().__init__(
            self.find_outshape(in_features, out_features, input),
            [input] if input is not None else [],
            param_count=((in_features + 1) * out_features),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.flop = 2 * self.param_count + self.out_features + np.prod(self.out_shape) # (Linear) + ReLU

    def find_outshape(self, in_features, out_features, input):
        assert len(input.out_shape) == 2 and input.out_shape[1] == in_features, f"{input.out_shape}, {in_features}"
        return (input.out_shape[0], out_features)

def selfattn_flop(B, H, K, Tc, Tg, cache=True):
		if cache:
			#### Mem_selfattn
			# 0th forward pass
			mem_selfattn_flop = 0
			x = DNNLayer(out_shape=(B, Tc, H))
			qkt = QKTMatrix(SEQ_LEN=Tc, HIDDEN_DIM=H, I=1, ATTN_HEADS=K, input=x)
			mask = Mask(input=x)
			# print(qkt.flop)
			# print(mask.flop)
			mem_selfattn_flop = qkt.flop + mask.flop

			# 1th-(n-1)th forward pass
			for i in range(1, Tg):
				x = DNNLayer(out_shape=(B, Tc + i, H))
				x_row = DNNLayer(out_shape=(B, 1, H))
				qkt = QKTMatrix(SEQ_LEN=Tc+i, HIDDEN_DIM=H, I=1, ATTN_HEADS=K, input=x_row)
				mask = Mask(input=x)
				cat = Concat(SEQ_LEN=Tc+i, HIDDEN_DIM=H, I=1, ATTN_HEADS=K, input=x)
				mem_selfattn_flop += qkt.flop + mask.flop + cat.flop

			print(f"mem_selfattn_flop: {mem_selfattn_flop}")
		else:
			#### Mim_selfattn
			# 0th-(n-1)th forward pass
			min_selfattn_flop = 0
			for i in range(0, Tg):
				x = DNNLayer(out_shape=(B, Tc + i, H))
				qkt = QKTMatrix(SEQ_LEN=Tc, HIDDEN_DIM=H, I=1, ATTN_HEADS=K, input=x)
				mask = Mask(input=x)
				min_selfattn_flop += qkt.flop + mask.flop

			print(f"min_selfattn_flop: {min_selfattn_flop}")

if __name__ == "__main__":
	
	hparams = {"117M": (12, 768), "345M": (24, 1024), "762M": (36, 1280), "1542M": (48, 1600)}
	K = 4
	B, H = hparams["117M"]
	Tc = 128
	Tg = 128
    # layer = CachedSelfAttn(K, H).cuda()
    # x = torch.randn((B, T, H)).cuda()

	selfattn_flop(B=B, H=H, K=K, Tc=Tc, Tg=Tg, cache=True)
	selfattn_flop(B=B, H=H, K=K, Tc=Tc, Tg=Tg, cache=False)



