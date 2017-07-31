#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  ekf_.F_ << MatrixXd(4, 4);
  ekf_.P_ << MatrixXd(4, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  //project 2D to 4D space
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //jacobian
  Hj_ << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1;

  //state transition matrix
  ekf_.F_ << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1;

  //covariance matrix
  ekf_.P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000; 
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho, phi, rhodot;
      rho = measurement_pack.raw_measurements_(0);
      phi = measurement_pack.raw_measurements_(1);
      rhodot = measurement_pack.raw_measurements_(2);

      ekf_.x_(0) = rho * cos(phi);
      ekf_.x_(1) = rho * sin(phi);
      ekf_.x_(2) = rhodot * cos(phi);
      ekf_.x_(3) = rhodot * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_(0) = measurement_pack.raw_measurements_(0);
      ekf_.x_(1) = measurement_pack.raw_measurements_(1);
    }

    //update the timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

   // Get the elapsed time in seconds
   float delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

   // Update the previous measurement to now
   previous_timestamp_ = measurement_pack.timestamp_;

   // Compute the powers of delta_t
   float delta_t_2 = delta_t * delta_t;
   float delta_t_3 = delta_t * delta_t_2;
   float delta_t_4 = delta_t * delta_t_3;

   // Update the state transition matrix with the new elapsed time
   ekf_.F_(1, 3) = delta_t;
   ekf_.F_(0, 2) = delta_t;

   // Noise acceleration values for Q matrix
   float noise_ax = 9;
   float noise_ay = 9;

   // Initialize and update the process noise covariance matrix
   ekf_.Q_ = MatrixXd(4, 4);
   ekf_.Q_ << delta_t_4 / 4 * noise_ax, 0, delta_t_3 / 2 * noise_ax, 0,
              0, delta_t_4 / 4 * noise_ay, 0, delta_t_3 / 2 * noise_ay,
              delta_t_3 / 2 * noise_ax, 0, delta_t_2 * noise_ax, 0,
              0, delta_t_3 / 2 * noise_ay, 0, delta_t_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;

    Tools tools;
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
