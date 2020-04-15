/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/arith/int_solver.h>
#include <tvm/ir/op.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/te/operation.h>
#include <tvm/te/autodiff.h>

namespace tvm {
namespace te {

#define FILE_POSITION \
  __FILE__ ":L" TVM_STRINGIZE(__LINE__)

namespace vjp {

class NotImplementedError : public std::runtime_error {
public:
  NotImplementedError(const std::string& where)
    : std::runtime_error("The operation is unimplemented. (" + where + ")") {}
};

#define NOT_IMPLEMENTED \
  { throw NotImplementedError(FILE_POSITION); }

class TensorArgsCollector : public ExprFunctor<void(const PrimExpr&)> {
 public:
  TensorArgsCollector(const Tensor& input)
    : input_(input) {}

  Array<Array<PrimExpr>> Collect(const PrimExpr& expr) {
    VisitExpr(expr);
    return std::move(args_);
  }

  void VisitExpr_(const CallNode *op) override {
    switch(op->call_type)
    {
    case CallNode::Halide:
      if (op->func.same_as(input_->op))
        args_.push_back(op->args);
      break;

    case CallNode::PureIntrinsic:
      for (const PrimExpr& arg : op->args)
        VisitExpr(arg);
      break;

    default:
      throw NotImplementedError(FILE_POSITION);
    }
  }

  void VisitExpr_(const AddNode* op) override {
    VisitExpr(op->a);
    VisitExpr(op->b);
  }

  void VisitExpr_(const SubNode* op) override {
    VisitExpr(op->a);
    VisitExpr(op->b);
  }

  void VisitExpr_(const MulNode* op) override {
    VisitExpr(op->a);
    VisitExpr(op->b);
  }

  void VisitExpr_(const DivNode* op) override {
    VisitExpr(op->a);
    VisitExpr(op->b);
  }

  void VisitExpr_(const MinNode* op) override {
    VisitExpr(op->a);
    VisitExpr(op->b);
  }

  void VisitExpr_(const MaxNode* op) override {
    VisitExpr(op->a);
    VisitExpr(op->b);
  }

  void VisitExpr_(const CastNode* op) override {
    VisitExpr(op->value);
  }

  void VisitExpr_(const SelectNode* op) override {
    VisitExpr(op->true_value);
    VisitExpr(op->false_value);
  }

  void VisitExpr_(const IntImmNode* op) override {}

  void VisitExpr_(const FloatImmNode* op) override {}

  void VisitExpr_(const VarNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const SizeVarNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const LoadNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const BufferLoadNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const LetNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const ModNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const FloorDivNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const FloorModNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const EQNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const NENode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const LTNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const LENode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const GTNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const GENode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const AndNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const OrNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const ReduceNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const NotNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const RampNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const BroadcastNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const ShuffleNode* op) override NOT_IMPLEMENTED;
  void VisitExpr_(const StringImmNode* op) override NOT_IMPLEMENTED;

 private:
  Tensor input_;
  Array<Array<PrimExpr>> args_;
};

class TensorArgsReplacer : public ExprFunctor<PrimExpr(const PrimExpr&)> {
 public:
  TensorArgsReplacer(const Tensor& input,
                     const Array<PrimExpr>& args,
                     const Map<Var, PrimExpr>& vmap)
    : input_(input), args_(args), vmap_(vmap) {}

  PrimExpr Replace(const PrimExpr& expr) {
    auto new_expr = VisitExpr(expr);
    return Substitute(new_expr, vmap_);
  }

  PrimExpr VisitExpr(const PrimExpr& expr) {
    if (expr.dtype().is_int() || expr.dtype().is_uint())
      throw NotImplementedError(FILE_POSITION);
    return ExprFunctor::VisitExpr(expr);
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    if (op->call_type == CallNode::Halide) {
      if (op->func.same_as(input_->op) && op->args.same_as(args_))
        return FloatImm(op->dtype, 1.0);
      else
        return FloatImm(op->dtype, 0.0);
    } else if (op->call_type == CallNode::PureIntrinsic) {
      PrimExpr expr = GetRef<PrimExpr>(op);
      if (op->name == "log") {
        return DivNode::make(VisitExpr(op->args[0]), op->args[0]);
      } else if (op->name == "pow") {
        PrimExpr x = op->args[0], y = op->args[1];
        PrimExpr mid = AddNode::make(
            MulNode::make(VisitExpr(y),
              CallNode::make(x.dtype(), "log", {x}, CallNode::PureIntrinsic)),
            MulNode::make(VisitExpr(x), DivNode::make(y, x)));
        return MulNode::make(expr, mid);
      } else if (op->name == intrinsic::tvm_if_then_else) {
        Array<PrimExpr> args = {op->args[0], VisitExpr(op->args[1]), VisitExpr(op->args[2])};
        return CallNode::make(op->dtype, intrinsic::tvm_if_then_else, args,
                              CallNode::PureIntrinsic, op->func, op->value_index);
      }

      PrimExpr result;
      if (op->name == "exp")
        result = expr;
      else if (op->name == "sigmoid")
        result = MulNode::make(expr, SubNode::make(FloatImm(op->dtype, 1.0), expr));
      else if (op->name == "sqrt")
        result = DivNode::make(FloatImm(op->dtype, 0.5), expr);
      else if (op->name == "tanh")
        result = SubNode::make(FloatImm(op->dtype, 1.0), MulNode::make(expr, expr));
      else if (op->name == "fabs")
        result = SelectNode::make(GENode::make(op->args[0], FloatImm(op->dtype, 0.0)),
                                  FloatImm(op->dtype, 1.0), FloatImm(op->dtype, -1.0));
      else
        throw NotImplementedError(FILE_POSITION);

      return MulNode::make(VisitExpr(op->args[0]), result);
    }

    throw NotImplementedError(FILE_POSITION);
  }

  PrimExpr VisitExpr_(const AddNode* op) override {
    return AddNode::make(VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const SubNode* op) override {
    return SubNode::make(VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const MulNode* op) override {
    return AddNode::make(MulNode::make(VisitExpr(op->a), op->b),
                         MulNode::make(op->a, VisitExpr(op->b)));
  }

  PrimExpr VisitExpr_(const DivNode* op) override {
    return DivNode::make(SubNode::make(MulNode::make(VisitExpr(op->a), op->b),
          MulNode::make(op->a, VisitExpr(op->b))), MulNode::make(op->b, op->b));
  }

  PrimExpr VisitExpr_(const MinNode* op) override {
    return SelectNode::make(LENode::make(op->a, op->b),
                            VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const MaxNode* op) override {
    return SelectNode::make(GENode::make(op->a, op->b),
                            VisitExpr(op->a), VisitExpr(op->b));
  }

  PrimExpr VisitExpr_(const CastNode* op) override {
    CHECK(op->dtype.is_float());
    return CastNode::make(op->dtype, VisitExpr(op->value));
  }

  PrimExpr VisitExpr_(const SelectNode* op) override {
    return SelectNode::make(op->condition,
                            VisitExpr(op->true_value), VisitExpr(op->false_value));
  }

  PrimExpr VisitExpr_(const IntImmNode* op) override {
    return FloatImm(op->dtype, 0.0);
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) override {
    return FloatImm(op->dtype, 0.0);
  }

  PrimExpr VisitExpr_(const VarNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const SizeVarNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const LoadNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const BufferLoadNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const LetNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const ModNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const FloorDivNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const FloorModNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const EQNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const NENode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const LTNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const LENode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const GTNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const GENode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const AndNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const OrNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const ReduceNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const NotNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const RampNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const BroadcastNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const ShuffleNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const StringImmNode* op) override NOT_IMPLEMENTED;

 private:
  Tensor input_;
  Array<PrimExpr> args_;
  Map<Var, PrimExpr> vmap_;
};

#undef NOT_IMPLEMENTED

} // namespace vjp

Tensor VectorJacobianProductOptimized(const Tensor& output,
                                      const Tensor& input,
                                      const Tensor& head)
{
  const ComputeOpNode *op = output->op.as<ComputeOpNode>();
  if (!op) // not implemented
    return Tensor();

  CHECK_GE(head->shape.size(), output->shape.size())
      << "head->shape shouldn't have less elements than output->shape";

  Array<PrimExpr> shape;
  Array<IterVar> axis;
  Array<PrimExpr> input_indices, head_indices;
  size_t output_offset = head->shape.size() - output->shape.size();

  for (size_t i = 0; i < output_offset; ++i) {
    PrimExpr ext = head->shape[i];
    IterVar iv = IterVarNode::make(Range(0, ext),
        Var(head->op->name + "_i" + std::to_string(i)), IterVarType::kDataPar);

    shape.push_back(ext);
    axis.push_back(iv);
    head_indices.push_back(iv);
  }
  for (size_t i = 0; i < output->shape.size(); ++i)
    head_indices.push_back(PrimExpr());

  for (size_t i = 0; i < input->shape.size(); ++i) {
    PrimExpr ext = input->shape[i];
    IterVar iv = IterVarNode::make(Range(0, ext),
        Var(input->op->name + "_i" + std::to_string(i)), IterVarType::kDataPar);

    shape.push_back(ext);
    axis.push_back(iv);
    input_indices.push_back(iv);
  }

  PrimExpr expr = op->body[output->value_index];
  PrimExpr result;

  try {
    vjp::TensorArgsCollector collector(input);
    Array<Array<PrimExpr>> args = collector.Collect(expr);

    Array<Var> vars;
    Map<Var, Range> ranges;
    for (const IterVar& it : op->axis) {
      vars.push_back(it->var);
      ranges.Set(it->var, it->dom);
    }

    for (const Array<PrimExpr>& arg : args) {
      CHECK_EQ(arg.size(), input_indices.size());

      Array<PrimExpr> eqs;
      for (size_t i = 0; i < arg.size(); ++i)
        eqs.push_back(EQNode::make(arg[i], input_indices[i]));

      arith::IntConstraints constraints(vars, ranges, eqs);
      arith::IntConstraintsTransform tf = arith::SolveLinearEquations(constraints);

      vjp::TensorArgsReplacer replacer(input, arg, tf->src_to_dst);
      PrimExpr e = replacer.Replace(expr);

      for (size_t i = 0; i < vars.size(); ++i)
        head_indices.Set(output_offset + i, tf->src_to_dst[vars[i]]);
      e = MulNode::make(e, CallNode::make(head->dtype,
            head->op->name, head_indices, CallNode::Halide, head->op, head->value_index));

      if (tf->dst->relations.size() != 0) {
        PrimExpr cond;
        for (const auto& rel : tf->dst->relations)
          cond = cond.get() ? AndNode::make(cond, rel) : rel;
        e = if_then_else(cond, e, make_zero(e.dtype()));
      }
      if (tf->dst->variables.size() != 0) {
        Array<IterVar> sum_axis;
        Map<Var, PrimExpr> vmap;
        for (const auto& it : tf->dst->ranges) {
          IterVar var = reduce_axis(it.second, it.first->name_hint);
          sum_axis.push_back(var);
          vmap.Set(it.first, var->var);
        }
        e = sum(Substitute(e, vmap), sum_axis);
      }

      result = result.get() ? AddNode::make(result, e) : e;
    }
    result = Simplify(result);
  } catch (vjp::NotImplementedError &err) {
    LOG(INFO) << err.what();
    return Tensor();
  }

  Operation grad_op = ComputeOpNode::make(
      "grad." + input->op->name, op->tag, op->attrs, axis, {result});
  Tensor grad = TensorNode::make(shape, input->dtype, grad_op, 0);
  return grad;
}

} // namespace te
} // namespace tvm
