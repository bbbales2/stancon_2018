template <typename T0__>
Eigen::Matrix<typename boost::math::tools::promote_args<T0__>::type, Eigen::Dynamic,1>
eigenvalues_sym_custom(const Eigen::Matrix<T0__, Eigen::Dynamic,Eigen::Dynamic>& K, std::ostream* pstream__) {
  return Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T0__, Eigen::Dynamic,Eigen::Dynamic> >(K).eigenvalues();
}