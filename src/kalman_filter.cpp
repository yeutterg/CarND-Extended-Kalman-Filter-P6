#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // Predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Update the state by using Kalman Filter equations

  // Kalman filter equations
  VectorXd y = z - H_ * x_;
  MatrixXd H_transpose = H_.transpose();
  MatrixXd S = H_ * P_ * H_transpose + R_;
  MatrixXd K = P_ * H_transpose * S.inverse();

  // Update
  x_ = x_ + K * y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Update the state by using Extended Kalman Filter equations

  // Calculate px^2 and py^2
  x0_2 = x_(0) * x_(0);
  x1_2 = x_(1) * x_(1);

  // Get rho, phi, and rho dot
  float rho = sqrt(x0_2 * x1_2);
  float phi = atan2(x_(1), x_(0));
  float rhodot = 0;
  if (rho >= 0.0001 || rho <= -0.0001) {
    pxvx = x_(0) * x_(2);
    pyvy = x_(1) * x_(3);
    rhodot = (pxvx + pyvy) / rho;
  }

  // Extended Kalman Filter Equations
  VectorXd zp(3);
  zp << rho, phi, rhodot;
  VectorXd y = z - zp;
  MatrixXd H_transpose = H_.transpose();
  MatrixXd S = H_ * P_ * H_transpose + R_;
  MatrixXd K = P_ * H_transpose * S.inverse();

  // Update
  x_ = x_ + K * y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
