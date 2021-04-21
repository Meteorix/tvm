import math
import tvm
from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm.relay import transform, expr
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from tvm.relay.function import Function


def fuse2(mod):
    mod = relay.transform.InferType()(mod)
    return relay.transform.FuseOps(fuse_opt_level=2)(mod)


class MarkFnType(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.dense_fns = []
        self.shuffle_fns = []
        self.dense_ops = []

    # def visit(self, expr):
    #     ret = super().visit(expr)
    #     # self.memo_map = {}
    #     return ret

    def visit_op(self, op):
        # import pdb; pdb.set_trace()
        print("got op:", op.name, id(op))
        if op.name in ["nn.dense", "nn.conv2d", "nn.batch_matmul"]:
            self.dense_ops.append(op)

    def visit_function(self, fn):
        if int(getattr(fn.attrs, "Primitive", 0)) == 0:
            self.visit(fn.body)
            return fn
        # import pdb; pdb.set_trace()
        self.dense_ops = []
        self.visit(fn.body)
        if self.dense_ops:
            self.dense_fns.append(fn)
        else:
            self.shuffle_fns.append(fn)


@transform.function_pass(opt_level=0)
class TensorViewPass:
    """An explicit pass wrapper around LiftConst."""

    def transform_function(self, func, mod, _):
        pass0 = MarkFnType()
        pass0.visit(func)
        for k, i in enumerate(pass0.dense_fns):
            print("=====dense: %d=====" % k)
            print(i)
        for k, i in enumerate(pass0.shuffle_fns):
            print("=====shuffle: %d=====" % k)
            print(i)
        print("=====end======")
        return func


def test_easy_case():
    """Test fusion case involving concat and gather_nd"""

    def before():
        shape = (tvm.tir.const(10, "int64"), tvm.tir.const(1, "int64"))
        x = relay.var("x", shape=shape, dtype="int64")
        x1 = relay.gather_nd(x, indices=relay.expr.const([[0, 1], [1, 0]], dtype="int64"))
        x2 = relay.nn.dense(x, relay.expr.const([[3], [4]], dtype="int64"))
        x1 = relay.nn.relu(x1)
        x1 = relay.reshape(x1, [-1, 1])
        x1 = relay.nn.dense(x1, relay.expr.const([[5], [6]], dtype="int64"))
        concat = relay.concatenate([x1, x2], axis=0)
        y = relay.add(concat, relay.expr.const(3, dtype="int64"))
        out = relay.Tuple([x2, y])
        return relay.Function(relay.analysis.free_vars(out), out)

    def expected():
        shape1 = (tvm.tir.const(10, "int64"), tvm.tir.const(1, "int64"))
        shape2 = (tvm.tir.const(2, "int64"), tvm.tir.const(2, "int64"))
        x = relay.var("x", shape=shape1)
        p0 = relay.var("p0", shape=shape1)
        p1 = relay.var("p1", shape=shape2, dtype="int64")
        c = relay.const([[0, 1], [1, 0]], dtype="int64")
        concat = relay.concatenate([p0, p0], axis=-1)
        out = relay.gather_nd(concat, indices=p1)

        f0 = relay.Function([p0, p1], out)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        y = relay.Call(f0, [x, c])
        return relay.Function([x], y)

    orig = before()
    print(run_opt_pass(orig, transform.InferType()))
    m = fuse2(tvm.IRModule.from_expr(orig))
    print(m)

    pass0 = TensorViewPass()
    m = pass0(m)
    print(m)
    # opt = run_opt_pass(m, pass0)

    # relay.build(m, "llvm")
    # after = run_opt_pass(expected(), transform.InferType())
    # assert tvm.ir.structural_equal(m["main"], after)
    # print(after)


def test_transformer_case():
    """Test fusion case involving concat and gather_nd"""

    def before():
        bs = 4
        seq_len = 32
        hidden_size = 768
        intermediate_size = 3072
        heads = 8
        size_per_head = hidden_size / heads
        input_tensor = relay.var("input_tensor", shape=(bs, seq_len, hidden_size), dtype="float32")
        q_weight = relay.var("q_weight", shape=(hidden_size, hidden_size), dtype="float32")
        k_weight = relay.var("k_weight", shape=(hidden_size, hidden_size), dtype="float32")
        v_weight = relay.var("v_weight", shape=(hidden_size, hidden_size), dtype="float32")
        gamma = relay.var("gamma", shape=(hidden_size, ), dtype="float32")
        beta = relay.var("beta", shape=(hidden_size, ), dtype="float32")
        q = relay.nn.dense(input_tensor, q_weight)
        k = relay.nn.dense(input_tensor, k_weight)
        v = relay.nn.dense(input_tensor, v_weight)

        def tranpose_for_scores(x):
            y = relay.reshape(x, [bs, seq_len, heads, size_per_head])
            y = relay.transpose(y, [0, 2, 1, 3])
            y = relay.reshape(y, [bs * heads, seq_len, size_per_head])
            return y

        q = tranpose_for_scores(q)
        k = tranpose_for_scores(k)
        v = tranpose_for_scores(v)

        # k = relay.transpose(k, [0, 2, 1])
        v = relay.transpose(v, [0, 2, 1])
        attention_scores = relay.nn.batch_matmul(q, k)
        attention_scores = relay.multiply(attention_scores, relay.const(1.0 / math.sqrt(float(size_per_head))))
        attention_probs = relay.nn.softmax(attention_scores)
        context_layer = relay.nn.batch_matmul(attention_probs, v)
        context_layer = relay.reshape(context_layer, [bs, heads, seq_len, size_per_head])
        context_layer = relay.transpose(context_layer, [0, 2, 1, 3])
        context_layer = relay.reshape(context_layer, [bs, seq_len, hidden_size])

        attention_dense_weight = relay.var("attention_dense_weight", shape=(hidden_size, hidden_size))
        attention_output = relay.nn.dense(context_layer, attention_dense_weight)
        attention_output = relay.nn.relu(attention_output)
        attention_output = relay.add(attention_output, input_tensor)
        attention_output = relay.nn.layer_norm(attention_output, gamma=gamma, beta=beta)

        intermediate_dense_weight = relay.var("intermediate_dense_weight", shape=(intermediate_size, hidden_size))
        intermediate_output = relay.nn.dense(attention_output, intermediate_dense_weight)
        intermediate_output = relay.nn.relu(intermediate_output)

        output_dense_weight = relay.var("output_dense_weight", shape=(hidden_size, intermediate_size))
        output = relay.nn.dense(intermediate_output, output_dense_weight)
        output = relay.nn.relu(output)
        output = relay.add(output, attention_output)
        output = relay.nn.layer_norm(output, gamma=gamma, beta=beta)

        return relay.Function(relay.analysis.free_vars(output), output)

    def expected():
        shape1 = (tvm.tir.const(10, "int64"), tvm.tir.const(1, "int64"))
        shape2 = (tvm.tir.const(2, "int64"), tvm.tir.const(2, "int64"))
        x = relay.var("x", shape=shape1)
        p0 = relay.var("p0", shape=shape1)
        p1 = relay.var("p1", shape=shape2, dtype="int64")
        c = relay.const([[0, 1], [1, 0]], dtype="int64")
        concat = relay.concatenate([p0, p0], axis=-1)
        out = relay.gather_nd(concat, indices=p1)

        f0 = relay.Function([p0, p1], out)
        f0 = f0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

        y = relay.Call(f0, [x, c])
        return relay.Function([x], y)

    orig = before()
    opt = run_opt_pass(orig, transform.SimplifyInference())
    opt = run_opt_pass(opt, transform.InferType())
    print(opt)
    opt = run_opt_pass(opt, transform.CombineParallelDense(to_batch=False))
    print(opt)
    m = fuse2(tvm.IRModule.from_expr(opt))
    print(m)
    # relay.build(m, "llvm")
    # after = run_opt_pass(expected(), transform.InferType())
    # assert tvm.ir.structural_equal(m["main"], after)
    # print(after)


if __name__ == "__main__":
    test_easy_case()
    # test_transformer_case()
