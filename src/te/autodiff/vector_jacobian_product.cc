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

/*
 * Given a tensor and an expression, suppose there are some calls made to
 * the tensor in the expression. This class will collect all these calls'
 * argument lists together to form a list. Thus the return value is a list
 * of lists of arguments (Array<Array<PrimExpr>>).
 */
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
      // Arguments like the first argument in tvm_if_then_else are skipped.
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

/*
 * Generally, this class will take the derivative of a given expression,
 * with respect to a given tensor with a given list of arguments. All
 * other tensors and the same tensor but with another list of arguments
 * will be treated as a constant.
 *
 * But for ReduceNode, things are different. The behavior depends on the
 * value reduce_type_, as following:
 *  kReduceDerivative:
 *    Keep the ReduceNode. When reduce_cond_ is true, take the derivative
 *    and multiply it by head_val_.
 *  kReducePartially:
 *    Keep the ReduceNode, but only perform reduction when reduce_cond_
 *    is true, and never take the derivative.
 *  kReduceSkip:
 *    Remove the ReduceNode, and take the derivative of its body directly.
 */
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
      reduce_cond_(reduce_cond), head_val_(head_val), replaced_(false) {}

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
      if (op->func.same_as(input_->op) && op->args.same_as(args_)) {
        if (replaced_) // TODO: Can this happen?
          throw NotImplementedError(FILE_POSITION);
        replaced_ = true;
        return FloatImm(op->dtype, 1.0);
      } else {
        return FloatImm(op->dtype, 0.0);
      }
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
      // Build a new combiner for taking the derivative.
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

    // Clone original reduction axes.
    Array<IterVar> axis;
    Map<Var, PrimExpr> vmap;
    for (const auto& iv : op->axis) {
      IterVar new_iv = reduce_axis(iv->dom, iv->var->name_hint);
      axis.push_back(new_iv);
      vmap.Set(iv->var, new_iv->var);
    }

    Array<PrimExpr> source;
    if (reduce_type_ == kReduceDerivative) {
      // Take the derivative of bodies when reduce_cond_ is true.
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

    // Substitute old axes by new axes.
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
  bool replaced_;
};

#undef NOT_IMPLEMENTED

/*
 * Reductions are only allowed at the top level of compute. So for farther
 * composition, we must create another tensor. This is a helper function to
 * do such thing.
 */
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
    // We have to copy the ReduceNode several times as new bodies.
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

/*
 * Now consider how to take the derivative of ReduceNode. Suppose we have
 * the input tensor's axis i, the output tensor's axis j and the reduction's
 * axis k. Equations are built between i and the input tensor's given argument
 * in the output tensor (which is a function of j and k).
 *
 * A first approach is to fix i, and solve equations for j and k. Sum across
 * free variables, and we need only focus on how to take the derivate when i,
 * j, k are all given. This method is implemented in CalcOneByOneDerivative.
 *
 * A second approach is to fix i and always perform original reduction on k.
 * For fixed i and k, we solve equations for j. We will take the derivative
 * in the reduction once there is such j that satisfies those equations. Summing
 * across free variables is still needed outside the reduction. This method
 * is implemented in CalcConditionalDerivative.
 */

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
  CHECK_EQ(reduce->source.size(), (size_t)1);

  /*
   * Suppose the reduction's body is evaluated along its axis as a_1, ..., a_n.
   * And denote its combiner as the function f(x, y). Let f(x, y, z) be f(f(x, y), z)
   * and so on.
   *
   * Now we are going to take the derivative with respect to some tensor in a_k.
   * Because the combiner is commutative, we can write:
   *    f(a_1, ..., a_n) = f(a_1, ..., a_{k-1}, a_{k+1}, ..., a_n, a_k)
   *                     = f(f(a_1, ..., a_{k-1}, a_{k+1}, ..., a_n), a_k)
   * We need only calculate:
   *   1. f(a_1, ..., a_{k-1}, a_{k+1}, ..., a_n)
   *   2. a_k
   *   3. the derivative of f with respect to x and y
   *   4. the derivative of a_k with respect to the specified tensor
   * The final result can be easily carried out from above expressions, after
   * some simple substitution and multiplication.
   */
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

/*
 * Take the derivative when there is no reduction.
 */
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

/*
 * When there is a ReduceNode, this function will tell us which method
 * (one-by-one derivative or conditional derivative) should be chosen
 * to take the derivative.
 *
 * Suppose axes i, j, k are defined same as before. Fix i and j, then solve
 * equations for k. If there is at most one k that satisfies those equations,
 * we'll choose one-by-one derivative method, otherwise we choose conditional
 * derivative method.
 *
 * The reason is clear for one-dimensional situation. If such k is unique when
 * i and j are fixed, one-by-one derivative method can avoid much unnecessary
 * work compared with conditional derivative method, especially when the
 * reduction is a sum. But if such k is not unique, it is likely that k can
 * take all values over its range. In that case, one-by-one derivative method
 * will involve much duplicative work, compared with conditional derivative
 * method.
 *
 * TODO: In high-dimensional situation, it seems to be possible and it would
 * work much better if we can choose different methods for different reduction
 * axes.
 */
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

  return result;
}

/*
 * The equation solver can generate some free variables and some conditions.
 * This function will sum up a given expression across those free variables,
 * and restrict the expression according to those conditions.
 */
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
  // Split conditions into two parts. The inner part depends on free variables
  // across which we are going to sum, while the outer part doesn't.
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

Tensor VectorJacobianProductOptimized(const Tensor& output,
                                      const Tensor& input,
                                      const Tensor& head)
{
  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  if (!op) // not implemented
    return Tensor();

  CHECK_GE(head->shape.size(), output->shape.size())
      << "head->shape shouldn't have less elements than output->shape";

  // Do some preparation, calculate the result tensor's shape, axis, and so on.
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
  // head_val = head(head_indices)
  // Note that the final expression shouldn't have variables of output's axis (op->axis),
  // but head_val does have those variables now. So they must be substituted later.
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

  // Calculate several lists of variables and ranges, which will be used
  // when solving equations later.
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

    /*
     * The input tensor may occur several times in the output tensor with
     * different lists of arguments. (Here the word "different" means their
     * memory addresses, instead of their contents, are different.)
     *
     * We simply take the derivative of each of them (in the below for-loop)
     * respectively and add the derivatives up to get the final result.
     */
    for (size_t idx = 0; idx < args.size(); ++idx) {
      Array<PrimExpr> arg = args[idx];
      CHECK_EQ(arg.size(), input_axis.size());

      // These equations will be solved for variables of output axis and/or reduce axis.
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

      // Choose and perform a suitable method to calculate the derivative.
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

      // Add sub_result into result.
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
