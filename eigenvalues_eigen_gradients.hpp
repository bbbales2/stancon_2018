// If this is templated on vars, then we need to set up the precomputed gradients
Eigen::Matrix<var, Eigen::Dynamic, 1>
build_output(const Eigen::MatrixXd& K_unscaled_double,
                const var& k,
                const Eigen::VectorXd& evals,
                const Eigen::MatrixXd& evecs) {
  Eigen::Matrix<var, Eigen::Dynamic, 1> output(evals.size());
  
  // This is shared computation between all the outputs
  Eigen::MatrixXd K_unscaled_times_evecs = K_unscaled_double * evecs;
  
  // For each output we need to build a special var
  for(int i = 0; i < evals.size(); i++) {
    // Create a new var which holds a vari which holds
    //   1. The value of the output
    //   2. A pointer to the parameter on which this value depends
    //   3. The partial derivative of this output with respect to that value
    //
    // The vari is allocated with new, but we don't need to worry about
    // cleaning up the memory. It is allocated in a special place that
    // Stan handles
    output(i) =
      var(new precomp_v_vari(evals(i),
                             k.vi_,
                             evecs.col(i).transpose() *
                               K_unscaled_times_evecs.col(i)));
  }
  
  return output;
}

// If this is templated on type double, there is no autodiff, so just return the evals
Eigen::Matrix<double, Eigen::Dynamic, 1>
build_output(const Eigen::MatrixXd& K_unscaled_double,
                const double& k,
                const Eigen::VectorXd& evals,
                const Eigen::MatrixXd& evecs) {
  return evals;
}

template <typename T0__, typename T1__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__, T1__>::type,
              Eigen::Dynamic,1>
eigenvalues_sym_external_gradients
  (const Eigen::Matrix<T0__, Eigen::Dynamic,Eigen::Dynamic>& K_unscaled,
   const T1__& k, std::ostream* pstream__) {
  // We don't want to use autodiff, so use the stan function value_of to
  // extract values from the autodiff vars before computing the eigenvalues.
  // If we don't, the autodiff will build the evalulation tree even though
  // we don't need it.
  Eigen::MatrixXd K_unscaled_double = value_of(K_unscaled);
  Eigen::MatrixXd K = value_of(k) * K_unscaled_double;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(K);
  Eigen::VectorXd evals = solver.eigenvalues();
  Eigen::MatrixXd evecs = solver.eigenvectors();
  
  // This function will need to 
  return build_output(K_unscaled_double, k, evals, evecs);
}