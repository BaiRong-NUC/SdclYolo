// Native translations of the pinned VisDrone toolkit's four hot-loop helpers.
// Matching order, tie handling, ignore flags and arithmetic follow the .m files.
#include <octave/oct.h>
#include <octave/parse.h>
#include <algorithm>
#include <numeric>
#include <vector>

static Matrix overlaps(const Matrix& dt, const Matrix& gt, const ColumnVector& ig) {
  Matrix result(dt.rows(), gt.rows(), 0.0);
  for (octave_idx_type d = 0; d < dt.rows(); ++d) {
    for (octave_idx_type g = 0; g < gt.rows(); ++g) {
      double w = std::min(dt(d,0)+dt(d,2), gt(g,0)+gt(g,2))
                 - std::max(dt(d,0), gt(g,0));
      if (w <= 0) continue;
      double h = std::min(dt(d,1)+dt(d,3), gt(g,1)+gt(g,3))
                 - std::max(dt(d,1), gt(g,1));
      if (h <= 0) continue;
      double area = w*h;
      double denominator = dt(d,2)*dt(d,3);
      if (!ig(g)) denominator = denominator + gt(g,2)*gt(g,3) - area;
      result(d,g) = area/denominator;
    }
  }
  return result;
}

DEFUN_DLD(createIntImg, args, , "Native exact-order toolkit integral image") {
  if (args.length() != 1) print_usage();
  Matrix out = args(0).matrix_value();
  octave_idx_type h = out.rows(), w = out.columns();
  if (!h || !w) return octave_value(out);
  for (octave_idx_type i = 1; i < h; ++i) out(i,0) += out(i-1,0);
  for (octave_idx_type j = 1; j < w; ++j) out(0,j) += out(0,j-1);
  for (octave_idx_type i = 1; i < h; ++i)
    for (octave_idx_type j = 1; j < w; ++j)
      out(i,j) = out(i,j) + out(i-1,j) + out(i,j-1) - out(i-1,j-1);
  return octave_value(out);
}

DEFUN_DLD(compOas, args, , "Native toolkit overlap calculation") {
  if (args.length() < 2 || args.length() > 3) print_usage();
  Matrix dt = args(0).matrix_value(), gt = args(1).matrix_value();
  ColumnVector ig(gt.rows(), 0.0);
  if (args.length() == 3) ig = args(2).column_vector_value();
  return octave_value(overlaps(dt, gt, ig));
}

DEFUN_DLD(evalRes, args, , "Native numeric-array toolkit matching") {
  if (args.length() < 2 || args.length() > 4) print_usage();
  Matrix in_gt = args(0).matrix_value(), in_dt = args(1).matrix_value();
  if (!in_gt.numel()) in_gt = Matrix(0,5);
  if (!in_dt.numel()) in_dt = Matrix(0,5);
  if (in_gt.columns() != 5 || in_dt.columns() != 5) error("Expected five columns.");
  double thr = args.length() >= 3 && !args(2).isempty() ? args(2).double_value() : .7;
  bool mul = args.length() >= 4 && !args(3).isempty() ? args(3).bool_value() : false;
  octave_idx_type ng = in_gt.rows(), nd = in_dt.rows();
  std::vector<octave_idx_type> gi(ng), di(nd);
  std::iota(gi.begin(), gi.end(), 0);
  std::iota(di.begin(), di.end(), 0);
  std::stable_sort(gi.begin(), gi.end(),
                   [&](auto a, auto b) { return in_gt(a,4) < in_gt(b,4); });
  std::stable_sort(di.begin(), di.end(),
                   [&](auto a, auto b) { return in_dt(a,4) > in_dt(b,4); });
  Matrix gt(ng,5), dt(nd,6,0.0), gb(ng,4), db(nd,4);
  ColumnVector ig(ng);
  for (octave_idx_type g = 0; g < ng; ++g) {
    for (int c = 0; c < 5; ++c) gt(g,c) = in_gt(gi[g],c);
    gt(g,4) = -gt(g,4);
    ig(g) = gt(g,4) == -1;
    for (int c = 0; c < 4; ++c) gb(g,c) = gt(g,c);
  }
  for (octave_idx_type d = 0; d < nd; ++d) {
    for (int c = 0; c < 5; ++c) dt(d,c) = in_dt(di[d],c);
    for (int c = 0; c < 4; ++c) db(d,c) = dt(d,c);
  }
  Matrix oa = overlaps(db, gb, ig);
  for (octave_idx_type d = 0; d < nd; ++d) {
    double best = thr;
    octave_idx_type best_g = -1;
    int best_m = 0;
    for (octave_idx_type g = 0; g < ng; ++g) {
      double m = gt(g,4);
      if (m == 1 && !mul) continue;
      if (best_m != 0 && m == -1) break;
      if (oa(d,g) < best) continue;
      best = oa(d,g);
      best_g = g;
      best_m = m == 0 ? 1 : -1;
    }
    if (best_m == -1) dt(d,5) = -1;
    else if (best_m == 1) {
      gt(best_g,4) = 1;
      dt(d,5) = 1;
    }
  }
  return ovl(gt, dt);
}

DEFUN_DLD(VOCap, args, , "Native toolkit precision envelope, Octave sum") {
  if (args.length() != 2) print_usage();
  ColumnVector rec = args(0).column_vector_value();
  ColumnVector prec = args(1).column_vector_value();
  if (rec.numel() != prec.numel()) error("Recall/precision length mismatch.");
  octave_idx_type n = rec.numel();
  ColumnVector r(n+2), p(n+2);
  r(0) = 0; p(0) = 0; r(n+1) = 1; p(n+1) = 0;
  for (octave_idx_type i = 0; i < n; ++i) { r(i+1) = rec(i); p(i+1) = prec(i); }
  for (octave_idx_type i = n; i >= 0; --i) p(i) = std::max(p(i), p(i+1));
  std::vector<double> products;
  for (octave_idx_type i = 1; i < n+2; ++i)
    if (r(i) != r(i-1)) products.push_back((r(i)-r(i-1))*p(i));
  ColumnVector values(products.size());
  for (octave_idx_type i = 0; i < values.numel(); ++i) values(i) = products[i];
  return octave::feval("sum", ovl(values), 1);
}
