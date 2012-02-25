#include "AlphaBetaTracker.h"
#include <cstdio>

AlphaBetaTracker::AlphaBetaTracker()
{
    count = 0;
    alpha = 0.5;
    beta = 0.25;
}

bool AlphaBetaTracker::Ready()
{
    if(count == 0) {
        return false;
    }

    return true;
}

void AlphaBetaTracker::Reset()
{
    count = 0;
}

void AlphaBetaTracker::SetAlphaBeta(double a, double b)
{
    alpha = a;
    beta = b;
}

void AlphaBetaTracker::SetState(double x, double y, double z, double yaw, double pitch, double roll)
{
    if(count == 0) {
        est_x = x;
        est_y = y;
        est_z = z;

        est_yaw = yaw;
        est_pitch = pitch;
        est_roll = roll;

        est_dx = 0;
        est_dy = 0;
        est_dz = 0;
        est_dyaw = 0;
        est_dpitch = 0;
        est_droll = 0;

        count++;
    }
    else {
		// Our predicted state
		double pred_x = est_x + est_dx;
		double pred_y = est_y + est_dy;
		double pred_z = est_z + est_dz;

		double pred_yaw = est_yaw + est_dyaw;
		double pred_pitch = est_pitch + est_dpitch;
		double pred_roll = est_roll + est_droll;

		//printf("Predicted: %f %f %f\n", x, y, z);

        // residual
        double r_x = x - pred_x;
        double r_y = y - pred_y;
        double r_z = z - pred_z;

        double r_yaw = yaw - pred_yaw;
        double r_pitch = pitch - pred_pitch;
        double r_roll = roll - pred_roll;

		//printf("Residual: %f %f %f\n", r_x, r_y, r_z);

		// correction
        pred_x += alpha*r_x;
        pred_y += alpha*r_y;
        pred_z += alpha*r_z;
        pred_yaw += alpha*r_yaw;
        pred_pitch += alpha*r_pitch;
        pred_roll += alpha*r_roll;

		est_x = pred_x;
		est_y = pred_y;
		est_z = pred_z;

		est_yaw = pred_yaw;
		est_pitch = pred_pitch;
		est_roll = pred_roll;

        est_dx = est_dx + beta*r_x;
        est_dy = est_dy + beta*r_y;
        est_dz = est_dz + beta*r_z;
        est_dyaw = est_dyaw + beta*r_yaw;
        est_dpitch = est_dpitch + beta*r_pitch;
        est_droll = est_droll + beta*r_roll;
    }
}

void AlphaBetaTracker::GetCorrectedState(double *x, double *y, double *z, double *yaw, double *pitch, double *roll)
{
    *x = est_x;
    *y = est_y;
    *z = est_z;

    *yaw = est_yaw;
    *pitch = est_pitch;
    *roll = est_roll;
}

void AlphaBetaTracker::GetPredictedState(double *x, double *y, double *z, double *yaw, double *pitch, double *roll)
{
/*
    // Disable prediction for now

    *x = est_x;
    *y = est_y;
    *z = est_z;
    *yaw = est_yaw;
    *pitch = est_pitch;
    *roll = est_roll;
*/

    *x = est_x + est_dx;
    *y = est_y + est_dy;
    *z = est_z + est_dz;
    *yaw = est_yaw;// + est_dyaw;
    *pitch = est_pitch;// + est_dpitch;
    *roll = est_roll;// + est_droll;
}

void AlphaBetaTracker::GetVelocity(double *dx, double *dy, double *dz, double *dyaw, double *dpitch, double *droll)
{
    *dx = est_dx;
    *dy = est_dy;
    *dz = est_dz;
    *dyaw = est_dyaw;
    *dpitch = est_dpitch;
    *droll = est_droll;
}
