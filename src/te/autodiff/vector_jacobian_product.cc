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
#include "ad_util.h"

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

  void VisitExpr_(const CallNode* op) override {
    switch(op->call_type)
    {
    case CallNode::Halide:
      if (op->func.same_as(input_->op))
        args_.push_back(op->args);
      break;

    case CallNode::PureIntrinsic:
      for (const PrimExpr& arg : op->args)
        if (!arg.dtype().is_bool())
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

  void VisitExpr_(const ReduceNode* op) override {
    for (const auto& expr : op->source)
      VisitExpr(expr);
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
  enum ReduceType {
    kReduceNone,
    kReduceDerivative,
    kReducePartially,
    kReduceSkip
  };

  TensorArgsReplacer(const Tensor& input,
                     const Array<PrimExpr>& args,
                     const Map<Var, PrimExpr>& vmap,
                     ReduceType reduce_type = kReduceNone,
                     const PrimExpr& reduce_cond = PrimExpr(),
                     const PrimExpr& head_val = PrimExpr())
    : input_(input), args_(args), vmap_(vmap), reduce_type_(reduce_type),
      reduce_cond_(reduce_cond), head_val_(head_val) {}

  PrimExpr Replace(const PrimExpr& expr) {
    auto new_expr = VisitExpr(expr);
    return Simplify(Substitute(new_expr, vmap_));
  }

  PrimExpr VisitExpr(const PrimExpr& expr) {
    if (expr.dtype().is_int() || expr.dtype().is_uint())
      throw NotImplementedError(FILE_POSITION);
    return ExprFunctor::VisitExpr(expr);
  }

  PrimExpr VisitExpr_(const CallNode* op) override {
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

  PrimExpr VisitExpr_(const ReduceNode* op) override {
    CHECK(reduce_type_ != kReduceNone);

    if (reduce_type_ == kReduceSkip)
      return VisitExpr(op->source[op->value_index]);

    CommReducer combiner = op->combiner;
    if (reduce_type_ == kReduceDerivative) {
      Array<Var> lhs, rhs;
      Array<PrimExpr> result, identity;

      for (const Var& v : combiner->lhs)
        lhs.push_back(v.copy_with_suffix(".grad"));
      for (const Var& v : combiner->lhs)
        lhs.push_back(v);
      for (const Var& v : combiner->rhs)
        rhs.push_back(v.copy_with_suffix(".grad"));
      for (const Var& v : combiner->rhs)
        rhs.push_back(v);

      for (const auto& expr : combiner->result) {
        PrimExpr grad;
        for (size_t i = 0; i < combiner->lhs.size(); ++i) {
          PrimExpr d = Derivative(expr, combiner->lhs[i]);
          d = MulNode::make(d, lhs[i]);
          grad = grad.get() ? AddNode::make(d, grad) : d;
        }
        for (size_t i = 0; i < combiner->rhs.size(); ++i) {
          PrimExpr d = Derivative(expr, combiner->rhs[i]);
          d = MulNode::make(d, rhs[i]);
          grad = grad.get() ? AddNode::make(d, grad) : d;
        }
        result.push_back(grad);
      }
      for (const auto& expr : combiner->result)
        result.push_back(expr);

      for (const auto& id : combiner->identity_element)
        identity.push_back(make_zero(id.dtype()));
      for (const auto& id : combiner->identity_element)
        identity.push_back(id);

      combiner = CommReducerNode::make(lhs, rhs, result, identity);
    }

    Array<IterVar> axis;
    Map<Var, PrimExpr> vmap;
    for (const auto& iv : op->axis) {
      IterVar new_iv = reduce_axis(iv->dom, iv->var->name_hint);
      axis.push_back(new_iv);
      vmap.Set(iv->var, new_iv->var);
    }

    Array<PrimExpr> source;
    if (reduce_type_ == kReduceDerivative) {
      for (PrimExpr expr : op->source) {
        expr = VisitExpr(expr);
        expr = MulNode::make(expr, head_val_);
        if (reduce_cond_.get())
          expr = if_then_else(reduce_cond_, expr, make_zero(expr.dtype()));
        source.push_back(expr);
      }
    }
    for (const auto& expr : op->source)
      source.push_back(expr);

    for (size_t i = 0; i < source.size(); ++i)
      source.Set(i, Substitute(source[i], vmap));
    PrimExpr cond = reduce_type_ == kReducePartially ? reduce_cond_ : PrimExpr();
    if (cond.get())
      cond = Substitute(cond, vmap);

    return ReduceNode::make(combiner, source, axis, cond, op->value_index);
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
  PrimExpr VisitExpr_(const NotNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const RampNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const BroadcastNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const ShuffleNode* op) override NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const StringImmNode* op) override NOT_IMPLEMENTED;

 private:
  Tensor input_;
  Array<PrimExpr> args_;
  Map<Var, PrimExpr> vmap_;
  ReduceType reduce_type_;
  PrimExpr reduce_cond_;
  PrimExpr head_val_;
};

#undef NOT_IMPLEMENTED

PrimExpr TensorizeExpr(const PrimExpr& expr,
                       const Array<IterVar>& vars,
                       const std::string& name,
                       const Operation& like = Operation()) {
  auto clone = CloneIterVars(vars);
  PrimExpr new_expr = Substitute(expr, clone.second);
  Array<IterVar> new_vars = clone.first;
  Array<PrimExpr> values;
  for (const auto& iv : vars)
    values.push_back(iv->var);

  auto tag = like.get() ? like->tag : std::string();
  auto attrs = like.get() ? like->attrs : Map<std::string, ObjectRef>();
  Array<PrimExpr> body;
  int value_index;

  const ReduceNode* reduce = new_expr.as<ReduceNode>();
  if (reduce && reduce->source.size() > 1) {
    for (size_t i = 0; i < reduce->source.size(); ++i)
      body.push_back(ReduceNode::make(
            reduce->combiner, reduce->source, reduce->axis, reduce->condition, i));
    value_index = reduce->value_index;
  } else {
    body.push_back(new_expr);
    value_index = 0;
  }

  Operation op = ComputeOpNode::make(name, tag, attrs, new_vars, body);
  return CallNode::make(new_expr.dtype(), name, values, CallNode::Halide, op, value_index);
}

std::pair<PrimExpr, Array<PrimExpr>> CalcOneByOneDerivative(
                                        int idx,
                                        const Tensor& input,
                                        const Array<IterVar>& input_iv,
                                        const Array<PrimExpr>& args,
                                        const arith::IntConstraintsTransform& tf,
                                        const Tensor& output,
                                        const PrimExpr& head_val) {
  PrimExpr sub_result;
  Array<PrimExpr> cond;

  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op);
  CHECK_EQ(args.size(), tf->src->relations.size());
  CHECK_EQ(input_iv.size(), tf->src->relations.size());
  CHECK_EQ(op->axis.size() + op->reduce_axis.size(), tf->src->variables.size());

  PrimExpr expr = op->body[output->value_index];
  const ReduceNode* reduce = expr.as<ReduceNode>();
  CHECK(reduce);

  PrimExpr sub_cond;
  for (const auto& iv : reduce->axis) {
    PrimExpr e = NENode::make(tf->src_to_dst[iv->var], iv->var);
    sub_cond = sub_cond.get() ? AndNode::make(sub_cond, e) : e;
  }
  for (const auto& rel : tf->dst->relations)
    cond.push_back(rel);

  TensorArgsReplacer replacer(input, args, tf->src_to_dst,
                              TensorArgsReplacer::kReducePartially, sub_cond);
  PrimExpr e = replacer.Replace(expr);

  Array<IterVar> new_axis = input_iv;
  for (const auto& it : tf->dst->ranges)
    new_axis.push_back(IterVarNode::make(it.second, it.first, IterVarType::kDataPar));
  PrimExpr part_expr = TensorizeExpr(e, new_axis,
          output->op->name + ".part" + std::to_string(idx), output->op);

  CommReducer combiner = reduce->combiner;
  Map<Var, PrimExpr> vmap;
  replacer = TensorArgsReplacer(input, args, tf->src_to_dst,
                                TensorArgsReplacer::kReduceSkip);
  e = Derivative(combiner->result[0], combiner->rhs[0]);
  vmap.Set(combiner->lhs[0], part_expr);
  vmap.Set(combiner->rhs[0], Substitute(reduce->source[0], tf->src_to_dst));
  sub_result = MulNode::make(Substitute(e, vmap), replacer.Replace(expr));

  sub_result = MulNode::make(sub_result, Substitute(head_val, tf->src_to_dst));
  sub_result = Simplify(sub_result);

  return std::make_pair(sub_result, cond);
}

std::pair<PrimExpr, Array<PrimExpr>> CalcConditionalDerivative(
                                        int idx,
                                        const Tensor& input,
                                        const Array<IterVar>& axis,
                                        const Array<PrimExpr>& args,
                                        const arith::IntConstraintsTransform& tf,
                                        const Tensor& output,
                                        const PrimExpr& head_val) {
  PrimExpr sub_result;
  Array<PrimExpr> cond;

  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op);
  CHECK_EQ(args.size(), tf->src->relations.size());
  CHECK_EQ(op->axis.size(), tf->src->variables.size());

  PrimExpr expr = op->body[output->value_index];
  const ReduceNode* reduce = expr.as<ReduceNode>();
  CHECK(reduce);

  std::unordered_set<const VarNode*> vset;
  for (const auto& iv : reduce->axis)
    vset.insert(iv->var.get());

  PrimExpr sub_cond;
  for (const auto& rel : tf->dst->relations) {
    if (ExprUseVar(rel, vset))
      sub_cond = sub_cond.get() ? AndNode::make(sub_cond, rel) : rel;
    else
      cond.push_back(rel);
  }

  TensorArgsReplacer replacer(input, args, tf->src_to_dst,
                              TensorArgsReplacer::kReduceDerivative, sub_cond,
                              Substitute(head_val, tf->src_to_dst));
  sub_result = replacer.Replace(expr);

  Array<IterVar> new_axis = axis;
  for (const auto& it : tf->dst->ranges)
    new_axis.push_back(IterVarNode::make(it.second, it.first, IterVarType::kDataPar));
  sub_result = TensorizeExpr(sub_result, new_axis,
                             input->op->name + ".deri" + std::to_string(idx), input->op);

  return std::make_pair(sub_result, cond);
}

std::pair<PrimExpr, Array<PrimExpr>> CalcNormalDerivative(
                                        const Tensor& input,
                                        const Array<IterVar>& input_iv,
                                        const Array<PrimExpr>& args,
                                        const arith::IntConstraintsTransform& tf,
                                        const Tensor& output,
                                        const PrimExpr& head_val) {
  PrimExpr sub_result;
  Array<PrimExpr> cond;

  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op);
  CHECK_EQ(args.size(), tf->src->relations.size());
  CHECK_EQ(input_iv.size(), tf->src->relations.size());
  CHECK_EQ(op->axis.size(), tf->src->variables.size());
  CHECK_EQ(op->reduce_axis.size(), (size_t)0);

  PrimExpr expr = op->body[output->value_index];

  for (const auto& rel : tf->dst->relations)
    cond.push_back(rel);

  TensorArgsReplacer replacer(input, args, tf->src_to_dst);
  sub_result = replacer.Replace(expr);
  sub_result = MulNode::make(sub_result, Substitute(head_val, tf->src_to_dst));

  return std::make_pair(sub_result, cond);
}

bool ChooseDerivativeMethod(const arith::IntConstraintsTransform& tf,
                            const Tensor& output) {
  bool result = true;

  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op);
  CHECK_EQ(op->reduce_axis.size(), tf->src->variables.size());

  PrimExpr expr = op->body[output->value_index];
  const ReduceNode* reduce = expr.as<ReduceNode>();
  CHECK(reduce);
  if (reduce->source.size() > 1) // TODO: Remove this demand.
    return false;

  std::unordered_set<const VarNode*> vset;
  for (const auto& v : tf->dst->variables)
    vset.insert(v.get());
  for (const auto& iv : op->reduce_axis)
    result = result && !ExprUseVar(tf->src_to_dst[iv->var], vset);
  // TODO: Choose different methods for different axes,
  //       according to their respective flags.

  return result;
}

PrimExpr SumUpDerivatives(const PrimExpr& expr,
                          const Array<PrimExpr>& cond,
                          const arith::IntConstraints& dst,
                          int idx,
                          const Array<IterVar>& axis,
                          const Tensor& input) {
  PrimExpr outer_cond, inner_cond;
  std::unordered_set<const VarNode*> vset;
  for (const Var& var : dst->variables)
    vset.insert(var.get());
  for (const auto& rel : cond) {
    if (ExprUseVar(rel, vset))
      inner_cond = inner_cond.get() ? AndNode::make(inner_cond, rel) : rel;
    else
      outer_cond = outer_cond.get() ? AndNode::make(outer_cond, rel) : rel;
  }

  // TODO: Remove useless conditions (conditions which are always TRUE).
  PrimExpr result = expr;
  if (inner_cond.get())
    result = if_then_else(inner_cond, result, make_zero(result.dtype()));
  if (dst->variables.size() != 0) {
    Array<IterVar> sum_axis;
    Map<Var, PrimExpr> vmap;
    for (const auto& it : dst->ranges) {
      IterVar var = reduce_axis(it.second, it.first->name_hint);
      sum_axis.push_back(var);
      vmap.Set(it.first, var->var);
    }
    result = sum(Substitute(result, vmap), sum_axis);
  }
  if (outer_cond.get()) {
    if (result.as<ReduceNode>())
      result = TensorizeExpr(result, axis,
            input->op->name + ".grad" + std::to_string(idx), input->op);
    result = if_then_else(outer_cond, result, make_zero(result.dtype()));
  }

  return result;
}

} // namespace vjp

// TODO: Add comments.
Tensor VectorJacobianProductOptimized(const Tensor& output,
                                      const Tensor& input,
                                      const Tensor& head)
{
  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  if (!op) // not implemented
    return Tensor();

  CHECK_GE(head->shape.size(), output->shape.size())
      << "head->shape shouldn't have less elements than output->shape";

  Array<PrimExpr> shape;
  Array<IterVar> axis, input_axis;
  Array<PrimExpr> head_indices;
  size_t output_offset = head->shape.size() - output->shape.size();

  for (size_t i = 0; i < output_offset; ++i) {
    PrimExpr ext = head->shape[i];
    IterVar iv = IterVarNode::make(Range(0, ext),
        Var(head->op->name + "_i" + std::to_string(i)), IterVarType::kDataPar);

    shape.push_back(ext);
    axis.push_back(iv);
    head_indices.push_back(iv);
  }
  for (const IterVar& it : op->axis)
    head_indices.push_back(it->var);
  PrimExpr head_val = CallNode::make(head->dtype, head->op->name,
          head_indices, CallNode::Halide, head->op, head->value_index);

  for (size_t i = 0; i < input->shape.size(); ++i) {
    PrimExpr ext = input->shape[i];
    IterVar iv = IterVarNode::make(Range(0, ext),
        Var(input->op->name + "_i" + std::to_string(i)), IterVarType::kDataPar);

    shape.push_back(ext);
    axis.push_back(iv);
    input_axis.push_back(iv);
  }

  Array<Var> vars;
  Map<Var, Range> ranges;
  for (const IterVar& it : op->axis) {
    vars.push_back(it->var);
    ranges.Set(it->var, it->dom);
  }

  Array<Var> reduce_vars, all_vars = vars;
  Map<Var, Range> reduce_ranges, all_ranges = ranges;
  for (const IterVar& it : op->reduce_axis) {
    reduce_vars.push_back(it->var);
    all_vars.push_back(it->var);
    reduce_ranges.Set(it->var, it->dom);
    all_ranges.Set(it->var, it->dom);
  }

  PrimExpr expr = op->body[output->value_index];
  PrimExpr result;

  try {
    vjp::TensorArgsCollector collector(input);
    Array<Array<PrimExpr>> args = collector.Collect(expr);

    for (size_t idx = 0; idx < args.size(); ++idx) {
      Array<PrimExpr> arg = args[idx];
      CHECK_EQ(arg.size(), input_axis.size());

      Array<PrimExpr> eqs;
      for (size_t i = 0; i < arg.size(); ++i)
        eqs.push_back(EQNode::make(arg[i], input_axis[i]->var));

      PrimExpr sub_result;
      Array<PrimExpr> cond;
      arith::IntConstraints constraints;
      arith::IntConstraintsTransform tf;
      const ReduceNode* reduce = op->body[output->value_index].as<ReduceNode>();
      CHECK_EQ((bool)reduce, (bool)op->reduce_axis.size());

      if (reduce) {
        constraints = arith::IntConstraints(reduce_vars, reduce_ranges, eqs);
        tf = arith::SolveLinearEquations(constraints);
      }

      if (reduce && vjp::ChooseDerivativeMethod(tf, output)) {
        constraints = arith::IntConstraints(all_vars, all_ranges, eqs);
        tf = arith::SolveLinearEquations(constraints);
        std::tie(sub_result, cond) = vjp::CalcOneByOneDerivative(
                idx, input, input_axis, arg, tf, output, head_val);
      } else {
        constraints = arith::IntConstraints(vars, ranges, eqs);
        tf = arith::SolveLinearEquations(constraints);
        if (reduce)
          std::tie(sub_result, cond) = vjp::CalcConditionalDerivative(
                  idx, input, axis, arg, tf, output, head_val);
        else
          std::tie(sub_result, cond) = vjp::CalcNormalDerivative(
                    input, input_axis, arg, tf, output, head_val);
      }

      sub_result = vjp::SumUpDerivatives(sub_result, cond, tf->dst, idx, axis, input);

      if (result.as<ReduceNode>())
        result = vjp::TensorizeExpr(result, axis,
            input->op->name + ".grad" + std::to_string(idx - 1), input->op);
      if (result.get() && sub_result.as<ReduceNode>())
        sub_result = vjp::TensorizeExpr(sub_result, axis,
            input->op->name + ".grad" + std::to_string(idx), input->op);
      result = result.get() ? AddNode::make(result, sub_result) : sub_result;
    }
  } catch (vjp::NotImplementedError& err) {
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
