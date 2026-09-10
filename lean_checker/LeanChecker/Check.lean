import Mathlib.Analysis.InnerProductSpace.Basic

open scoped RealInnerProductSpace

variable {V : Type*} [InnerProductSpace ℝ V]
variable (x y : V)

example : inner (x + y) (x + y) = inner x x + 2 * inner x y + inner y y := by
  calc
    inner (x + y) (x + y)
        = inner x x + inner x y + inner y x + inner y y := by
          simpa [inner_add_left, inner_add_right, add_comm, add_left_comm, add_assoc]
    _ = inner x x + 2 * inner x y + inner y y := by
          simpa [inner_symm x y, two_mul, mul_comm]
