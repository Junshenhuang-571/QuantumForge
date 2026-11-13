use quantumforge::analytic::{Assumptions, ExactComplex, Rational, ScalarExpr, Simplifier};

#[test]
fn eval_complex_basic() {
    let x = ScalarExpr::symbol("x");
    let expr = ScalarExpr::mul([
        ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap())),
        x.clone(),
    ])
    .unwrap();
    let mut vals = std::collections::HashMap::new();
    vals.insert("x".to_string(), quantumforge::Complex64::new(3.0, 0.0));
    let result = expr.eval_complex(&vals).unwrap();
    assert!((result.re - 6.0).abs() < 1e-10);
    assert!(result.im.abs() < 1e-10);
}

#[test]
fn eval_f64_basic() {
    let x = ScalarExpr::symbol("x");
    let expr = ScalarExpr::mul([
        ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap())),
        x.clone(),
    ])
    .unwrap();
    let mut vals = std::collections::HashMap::new();
    vals.insert("x".to_string(), 3.0);
    let result = expr.eval_f64(&vals).unwrap();
    assert!((result - 6.0).abs() < 1e-10);
}

#[test]
fn eval_f64_complex_error() {
    let x = ScalarExpr::symbol("x");
    let expr = ScalarExpr::mul([
        ScalarExpr::exact(ExactComplex {
            re: Rational::new(1, 1).unwrap(),
            im: Rational::new(1, 1).unwrap(),
        }),
        x.clone(),
    ])
    .unwrap();
    let mut vals = std::collections::HashMap::new();
    vals.insert("x".to_string(), 1.0);
    let result = expr.eval_f64(&vals);
    assert!(result.is_err());
}

#[test]
fn taylor_series_exp() {
    let x = ScalarExpr::symbol("x");
    let expr = ScalarExpr::exp(x.clone());
    let ts = expr.taylor_series("x", &ScalarExpr::zero(), 4).unwrap();
    let s = ts.to_string();
    assert!(s.contains("1") || s.contains("x"));
}

#[test]
fn taylor_series_sin() {
    let x = ScalarExpr::symbol("x");
    let expr = ScalarExpr::sin(x.clone());
    let ts = expr.taylor_series("x", &ScalarExpr::zero(), 5).unwrap();
    let s = ts.to_string();
    assert!(s.contains("x"));
}

#[test]
fn factor_quadratic() {
    let x = ScalarExpr::symbol("x");
    let a = ScalarExpr::exact(ExactComplex::real(Rational::new(1, 1).unwrap()));
    let b = ScalarExpr::exact(ExactComplex::real(Rational::new(-5, 1).unwrap()));
    let c = ScalarExpr::exact(ExactComplex::real(Rational::new(6, 1).unwrap()));
    let expr = ScalarExpr::add([
        ScalarExpr::mul([a, ScalarExpr::pow(x.clone(), 2).unwrap()]).unwrap(),
        ScalarExpr::mul([b, x.clone()]).unwrap(),
        c,
    ])
    .unwrap();
    let factored = expr.factor("x");
    let s = factored.to_string();
    assert!(s.contains("x") || s.contains("Sqrt"));
}

#[test]
fn factor_diff_squares() {
    let x = ScalarExpr::symbol("x");
    let y = ScalarExpr::symbol("y");
    let expr = ScalarExpr::add([
        ScalarExpr::pow(x.clone(), 2).unwrap(),
        ScalarExpr::mul([
            ScalarExpr::exact(ExactComplex::real(Rational::MINUS_ONE)),
            ScalarExpr::pow(y.clone(), 2).unwrap(),
        ])
        .unwrap(),
    ])
    .unwrap();
    let factored = expr.factor("x");
    let s = factored.to_string();
    assert!(s.contains("x") && s.contains("y"));
}

#[test]
fn solve_linear() {
    let x = ScalarExpr::symbol("x");
    let two = ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap()));
    let three = ScalarExpr::exact(ExactComplex::real(Rational::new(3, 1).unwrap()));
    let expr = ScalarExpr::add([ScalarExpr::mul([two, x.clone()]).unwrap(), three]).unwrap();
    let sol = expr.solve("x");
    assert_eq!(sol.len(), 1);
}

#[test]
fn solve_quadratic() {
    let x = ScalarExpr::symbol("x");
    let a = ScalarExpr::exact(ExactComplex::real(Rational::new(1, 1).unwrap()));
    let b = ScalarExpr::exact(ExactComplex::real(Rational::new(-5, 1).unwrap()));
    let c = ScalarExpr::exact(ExactComplex::real(Rational::new(6, 1).unwrap()));
    let expr = ScalarExpr::add([
        ScalarExpr::mul([a, ScalarExpr::pow(x.clone(), 2).unwrap()]).unwrap(),
        ScalarExpr::mul([b, x.clone()]).unwrap(),
        c,
    ])
    .unwrap();
    let sol = expr.solve("x");
    assert_eq!(sol.len(), 2);
}

#[test]
fn is_linear_quadratic() {
    let x = ScalarExpr::symbol("x");
    let linear = ScalarExpr::add([
        ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap())),
        x.clone(),
    ])
    .unwrap();
    let quadratic = ScalarExpr::add([ScalarExpr::pow(x.clone(), 2).unwrap(), x.clone()]).unwrap();
    let cubic = ScalarExpr::pow(x.clone(), 3).unwrap();

    assert!(linear.is_linear("x"));
    assert!(linear.is_quadratic("x"));
    assert!(quadratic.is_quadratic("x"));
    assert!(!cubic.is_quadratic("x"));
}

#[test]
fn is_constant() {
    let x = ScalarExpr::symbol("x");
    let constant = ScalarExpr::exact(ExactComplex::real(Rational::new(5, 1).unwrap()));
    let non_const = ScalarExpr::add([constant.clone(), x.clone()]).unwrap();

    assert!(constant.is_constant());
    assert!(!non_const.is_constant());
}

#[test]
fn collect_symbols() {
    let x = ScalarExpr::symbol("x");
    let y = ScalarExpr::symbol("y");
    let expr = ScalarExpr::add([
        ScalarExpr::mul([x.clone(), y.clone()]).unwrap(),
        ScalarExpr::pow(x.clone(), 2).unwrap(),
    ])
    .unwrap();
    let syms = expr.collect_symbols();
    assert!(syms.contains(&"x".to_string()));
    assert!(syms.contains(&"y".to_string()));
}

#[test]
fn degree_test() {
    let x = ScalarExpr::symbol("x");
    let y = ScalarExpr::symbol("y");
    let constant = ScalarExpr::exact(ExactComplex::real(Rational::new(5, 1).unwrap()));
    let linear = x.clone();
    let quadratic = ScalarExpr::pow(x.clone(), 2).unwrap();
    let mixed = ScalarExpr::mul([x.clone(), y.clone()]).unwrap();

    assert_eq!(constant.degree("x"), Some(0));
    assert_eq!(linear.degree("x"), Some(1));
    assert_eq!(quadratic.degree("x"), Some(2));
    assert_eq!(mixed.degree("x"), Some(1));
    assert_eq!(mixed.degree("y"), Some(1));
}

#[test]
fn leading_coefficient_test() {
    let x = ScalarExpr::symbol("x");
    let two = ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap()));
    let expr = ScalarExpr::mul([two, ScalarExpr::pow(x.clone(), 2).unwrap()]).unwrap();
    let lc = expr.leading_coefficient("x").unwrap();
    assert_eq!(lc.to_string(), "2");
}

#[test]
fn content_test() {
    let x = ScalarExpr::symbol("x");
    let two = ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap()));
    let four = ScalarExpr::exact(ExactComplex::real(Rational::new(4, 1).unwrap()));
    let expr = ScalarExpr::add([
        ScalarExpr::mul([two.clone(), x.clone()]).unwrap(),
        ScalarExpr::mul([four, ScalarExpr::pow(x.clone(), 2).unwrap()]).unwrap(),
    ])
    .unwrap();
    let c = expr.content("x");
    assert_eq!(c.re.numerator(), 2);
    assert_eq!(c.re.denominator(), 1);
}

#[test]
fn primitive_part_test() {
    let x = ScalarExpr::symbol("x");
    let two = ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap()));
    let four = ScalarExpr::exact(ExactComplex::real(Rational::new(4, 1).unwrap()));
    let expr = ScalarExpr::add([
        ScalarExpr::mul([two.clone(), x.clone()]).unwrap(),
        ScalarExpr::mul([four, ScalarExpr::pow(x.clone(), 2).unwrap()]).unwrap(),
    ])
    .unwrap();
    let pp = expr.primitive_part("x");
    let s = pp.to_string();
    assert!(s.contains("x"));
}

#[test]
fn compose_test() {
    let x = ScalarExpr::symbol("x");
    let y = ScalarExpr::symbol("y");
    let expr = ScalarExpr::sin(x.clone());
    let composed = expr.compose("x", &y).unwrap();
    assert_eq!(composed.to_string(), "Sin[y]");
}

#[test]
fn integrate_basic() {
    let x = ScalarExpr::symbol("x");
    let expr = ScalarExpr::mul([
        ScalarExpr::exact(ExactComplex::real(Rational::new(2, 1).unwrap())),
        x.clone(),
    ])
    .unwrap();
    let integrated = expr.integrate("x").unwrap();
    let deriv = integrated.differentiate("x").unwrap();
    let simp = Simplifier::new(Assumptions::default());
    let simplified = simp.simplify(&deriv).unwrap();
    assert_eq!(simplified.to_string(), "2 * x");
}

#[test]
fn integrate_sin() {
    let x = ScalarExpr::symbol("x");
    let expr = ScalarExpr::sin(x.clone());
    let integrated = expr.integrate("x").unwrap();
    let s = integrated.to_string();
    assert!(s.contains("Cos"));
}
